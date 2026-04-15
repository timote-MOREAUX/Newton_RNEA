from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING

import numpy as np
import torch
import warp as wp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv       # type: ignore[import-untyped]
    from isaaclab.managers import SceneEntityCfg       # type: ignore[import-untyped]

from .enums import InputMode, JointType
from .kernels import (
    eval_rigid_fk,
    eval_rigid_id,
    eval_rigid_tau,
    compute_spatial_inertia,
    compute_com_transforms,
    extract_root_compensation,
)


# --------------------------------------------------------------------------- #
# USD topology extraction                                                       #
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class _Topology:
    """All static kinematic data needed by the RNEA kernels, as Warp arrays."""
    joint_type:         wp.array   # (num_joints,)      int32
    joint_parent:       wp.array   # (num_joints,)      int32  — -1 = world
    joint_child:        wp.array   # (num_joints,)      int32
    joint_X_p:          wp.array   # (num_joints,)      wp.transform
    joint_X_c:          wp.array   # (num_joints,)      wp.transform
    joint_axis:         wp.array   # (total_qd,)        wp.vec3
    joint_dof_dim:      wp.array   # (num_joints, 2)    int32  — [lin, ang]
    joint_q_start:      wp.array   # (num_joints,)      int32
    joint_qd_start:     wp.array   # (num_joints,)      int32
    articulation_start: wp.array   # (2,)               int32  — [0, num_joints]
    # body_world[i] = index into the gravity array for body i.
    # We pass a single-element zero-gravity array, so every body must index 0.
    body_world:         wp.array   # (num_bodies,)      int32  — all zeros
    total_q:            int        # total position coords  (len of joint_q vector)
    total_qd:           int        # total velocity dofs    (len of joint_qd vector)
    num_joints:         int
    is_fixed_base:      bool
    # IsaacLab body indices actually used (sorted).  Used to slice inertial data.
    body_ids:           list[int]
    # IsaacLab dof indices of the selected joints.  Used to slice joint_pos/vel.
    joint_ids:          list[int] | slice


def _gf_quat_to_xyzw(q) -> tuple[float, float, float, float]:
    """Gf.Quatf / Gf.Quatd (real, imaginary) → (x, y, z, w) for wp.quat."""
    i = q.GetImaginary()
    return float(i[0]), float(i[1]), float(i[2]), float(q.GetReal())


def _axis_token_to_vec3(token: str) -> tuple[float, float, float]:
    return {"X": (1., 0., 0.), "Y": (0., 1., 0.), "Z": (0., 0., 1.)}[token.upper()]


def _build_topology(
    articulation,
    device: str,
    entity_cfg=None,
) -> _Topology:
    """
    Parse the USD stage once to extract the kinematic structure needed by the
    Newton RNEA kernels and return them as Warp arrays on ``device``.

    The stage is obtained from ``omni.usd`` internally.  Joint selection is
    resolved via ``articulation.find_joints`` so that ``entity_cfg.joint_names``
    regex patterns are handled by IsaacLab rather than hand-rolled logic.

    Kinematic structure (parent/child relationships, joint frame offsets, joint
    axes) is not exposed by the ``Articulation`` class and must be read from
    USD.  All runtime state (joint positions/velocities, inertia) comes from
    ``Articulation.data`` in the caller.

    Args:
        articulation:  IsaacLab ``Articulation`` object (already resolved).
        device:        Warp / torch device string.
        entity_cfg:    ``SceneEntityCfg`` (already resolved).  ``None`` → use
                       all joints.
    """
    import omni.usd  # type: ignore[import-untyped]
    from pxr import UsdPhysics, UsdGeom, Gf, Usd

    stage = omni.usd.get_context().get_stage()

    all_body_names = list(articulation.body_names)

    # Full body-name → IsaacLab body index (used during USD prim lookup).
    full_body_idx = {name: i for i, name in enumerate(all_body_names)}

    # Resolve which joints to include using IsaacLab's find_joints so that
    # regex patterns in entity_cfg.joint_names are handled natively.
    if entity_cfg is None or entity_cfg.joint_names is None:
        joint_ids:   list[int] | slice = slice(None)
        joint_names: list[str]         = list(articulation.joint_names)
    else:
        joint_ids, joint_names = articulation.find_joints(entity_cfg.joint_names)

    # Articulation root prim path — used only for error messages.
    art_path = articulation.root_physx_view.prim_paths[0]

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))

    # ------------------------------------------------------------------ #
    # Collect all USD joint prims for this articulation.                   #
    # In Isaac Sim joints can live anywhere in the stage — not necessarily  #
    # under the articulation root prim.  Search the whole stage and keep   #
    # prims whose name appears in articulation.joint_names, which IsaacLab #
    # has already resolved from the PhysX articulation.                    #
    # ------------------------------------------------------------------ #
    known_joint_names: set[str] = set(articulation.joint_names)

    usd_joints: dict[str, Usd.Prim] = {}
    is_fixed_base = False
    for prim in stage.TraverseAll():
        if prim.IsA(UsdPhysics.RevoluteJoint) or \
                prim.IsA(UsdPhysics.PrismaticJoint) or \
                prim.IsA(UsdPhysics.SphericalJoint) or \
                prim.IsA(UsdPhysics.FixedJoint):
            name = prim.GetName()
            if name in known_joint_names:
                usd_joints[name] = prim
            # A FixedJoint with no body0 relationship is welded to the world.
            if prim.IsA(UsdPhysics.FixedJoint):
                if not prim.GetRelationship("physics:body0").GetTargets():
                    is_fixed_base = True

    # ------------------------------------------------------------------ #
    # Parse USD joints → collect raw (full IsaacLab) parent/child indices. #
    # We store full indices first, then remap to contiguous Newton indices  #
    # once we know which bodies are actually used.                          #
    # ------------------------------------------------------------------ #
    jtype_raw:    list[int]               = []
    parent_raw:   list[int]               = []  # full IsaacLab body indices
    child_raw:    list[int]               = []  # full IsaacLab body indices
    X_p_raw:      list[wp.transform]      = []
    X_c_raw:      list[wp.transform]      = []
    dof_dim_raw:  list[list[int]]         = []
    axis_entries: list[tuple[int, tuple]] = []  # (dof_idx, (x,y,z))

    q_cursor = qd_cursor = 0
    q_starts:  list[int] = []
    qd_starts: list[int] = []

    def _register(jtype: JointType, lin: int, ang: int,
                  par_full: int, chi_full: int,
                  X_p: wp.transform, X_c: wp.transform,
                  axes: list[tuple[float, float, float]]):
        nonlocal q_cursor, qd_cursor
        jtype_raw.append(int(jtype))
        parent_raw.append(par_full)
        child_raw.append(chi_full)
        X_p_raw.append(X_p)
        X_c_raw.append(X_c)
        dof_dim_raw.append([lin, ang])
        q_starts.append(q_cursor)
        qd_starts.append(qd_cursor)
        for k, ax in enumerate(axes):
            axis_entries.append((qd_cursor + k, ax))
        dof_qd, dof_q = jtype.dof_count(lin + ang)
        q_cursor  += dof_q
        qd_cursor += dof_qd

    # --- Synthetic root FREE joint (floating-base only) ------------------ #
    if not is_fixed_base:
        # Body 0 is always the root in IsaacLab's body ordering.
        # FREE joint axes are hardcoded in the kernel; no axis_entries needed.
        _register(JointType.FREE, 3, 3, -1, 0, wp.transform(), wp.transform(), [])

    # --- USD arm joints -------------------------------------------------- #
    def _get_transform(prim, pos_attr: str, rot_attr: str) -> wp.transform:
        pos = prim.GetAttribute(pos_attr).Get() or Gf.Vec3f(0.)
        rot = (prim.GetAttribute(rot_attr).Get() or Gf.Quatf(1.)).GetNormalized()
        s = meters_per_unit
        p = wp.vec3(float(pos[0]) * s, float(pos[1]) * s, float(pos[2]) * s)
        q = wp.quat(*_gf_quat_to_xyzw(rot))
        return wp.transform(p, q)

    for jname in joint_names:
        prim = usd_joints.get(jname)
        if prim is None:
            # Partial-name fallback (handles prefix/suffix differences).
            matches = [p for n, p in usd_joints.items()
                       if n.endswith(jname) or jname.endswith(n)]
            if len(matches) == 1:
                prim = matches[0]
            else:
                raise RuntimeError(
                    f"Joint '{jname}' not found in stage for articulation '{art_path}'. "
                    f"Available USD joints: {list(usd_joints.keys())}"
                )

        # Joint type
        if prim.IsA(UsdPhysics.RevoluteJoint):
            jt, lin, ang = JointType.REVOLUTE,  0, 1
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            jt, lin, ang = JointType.PRISMATIC, 1, 0
        elif prim.IsA(UsdPhysics.FixedJoint):
            jt, lin, ang = JointType.FIXED,     0, 0
        elif prim.IsA(UsdPhysics.SphericalJoint):
            jt, lin, ang = JointType.BALL,      0, 3
        else:
            raise RuntimeError(f"Unsupported USD joint type for '{jname}'")

        # Parent / child — full IsaacLab body indices.
        b0 = prim.GetRelationship("physics:body0").GetTargets()
        b1 = prim.GetRelationship("physics:body1").GetTargets()
        b0_name = str(b0[0]).split("/")[-1] if b0 else None
        b1_name = str(b1[0]).split("/")[-1]
        par_full = full_body_idx.get(b0_name, -1) if b0_name else -1
        chi_full = full_body_idx[b1_name]

        X_p = _get_transform(prim, "physics:localPos0", "physics:localRot0")
        X_c = _get_transform(prim, "physics:localPos1", "physics:localRot1")

        axes: list[tuple[float, float, float]] = []
        if jt in (JointType.REVOLUTE, JointType.PRISMATIC):
            tok = prim.GetAttribute("physics:axis").Get() or "X"
            axes = [_axis_token_to_vec3(tok)]

        _register(jt, lin, ang, par_full, chi_full, X_p, X_c, axes)

    # ------------------------------------------------------------------ #
    # Body set: root (always) + all parents + all children referenced by  #
    # the selected joints.  Bodies not in this set are excluded from the   #
    # Newton model (their inertia, buffers, etc. are not allocated).       #
    # ------------------------------------------------------------------ #
    body_set: set[int] = {0}  # root body is always index 0 in IsaacLab
    for p, c in zip(parent_raw, child_raw):
        if p >= 0:
            body_set.add(p)
        body_set.add(c)

    body_ids = sorted(body_set)                              # IsaacLab indices, ascending
    body_remap = {old: new for new, old in enumerate(body_ids)}  # full → Newton index

    # Remap parent / child to contiguous Newton body indices.
    parent_list = [body_remap.get(p, -1) if p >= 0 else -1 for p in parent_raw]
    child_list  = [body_remap[c] for c in child_raw]
    num_bodies_used = len(body_ids)

    # ------------------------------------------------------------------ #
    # Assemble joint_axis indexed by DoF position.                         #
    # ------------------------------------------------------------------ #
    axis_np = np.zeros((qd_cursor, 3), dtype=np.float32)
    for dof_idx, (ax0, ax1, ax2) in axis_entries:
        axis_np[dof_idx] = [ax0, ax1, ax2]

    # ------------------------------------------------------------------ #
    # Convert to Warp arrays.                                              #
    # ------------------------------------------------------------------ #
    def _wi(lst: list[int]) -> wp.array:
        return wp.array(np.array(lst, dtype=np.int32), dtype=wp.int32, device=device)

    num_joints_total = len(jtype_raw)

    return _Topology(
        joint_type         = _wi(jtype_raw),
        joint_parent       = _wi(parent_list),
        joint_child        = _wi(child_list),
        joint_X_p          = wp.array(X_p_raw,  dtype=wp.transform, device=device),
        joint_X_c          = wp.array(X_c_raw,  dtype=wp.transform, device=device),
        joint_axis         = wp.array([wp.vec3(*r) for r in axis_np],
                                      dtype=wp.vec3, device=device),
        joint_dof_dim      = wp.array(np.array(dof_dim_raw, dtype=np.int32),
                                      dtype=wp.int32, ndim=2, device=device),
        joint_q_start      = _wi(q_starts),
        joint_qd_start     = _wi(qd_starts),
        articulation_start = _wi([0, num_joints_total]),
        # All bodies index into gravity[0] = zero vector.
        body_world         = wp.zeros((num_bodies_used,), dtype=wp.int32, device=device),
        total_q            = q_cursor,
        total_qd           = qd_cursor,
        num_joints         = num_joints_total,
        is_fixed_base      = is_fixed_base,
        body_ids           = body_ids,
        joint_ids          = joint_ids,
    )


# --------------------------------------------------------------------------- #
# Predictor                                                                     #
# --------------------------------------------------------------------------- #

class ArmWrenchPredictor:
    """
    Computes the 6D wrench the arm exerts on the drone root body given planned
    joint motion.  No gravity, no contacts, no integration — pure RNEA.

    The articulation and joint subset are specified via an IsaacLab
    ``SceneEntityCfg``, exactly as in MDP action / observation terms.  This
    lets you target only the arm joints of a larger articulation (e.g. a
    drone that also has propeller or leg joints).

    Example
    -------
    .. code-block:: python

        from isaaclab.managers import SceneEntityCfg
        from src import ArmWrenchPredictor, InputMode

        entity_cfg = SceneEntityCfg("robot", joint_names=["arm_joint_.*"])
        predictor = ArmWrenchPredictor(env, entity_cfg)

        wrench = predictor.compute_root_wrench(pos_target, mode=InputMode.POS)
        feedforward = -wp.to_torch(wrench)[:6]

    Topology (joint types, parent/child tree, axes, frame offsets) is parsed
    from the USD stage once at construction.  Runtime state is read from
    ``Articulation.data`` each call through persistent, pre-allocated Warp
    views — zero per-call allocation.

    Quaternion convention
    ---------------------
    Isaac Lab uses ``(w, x, y, z)``; Newton/Warp kernels use ``(x, y, z, w)``.
    The root pose is reordered in-place before each kernel launch.
    """

    def __init__(
        self,
        env: ManagerBasedRLEnv,
        entity_cfg: SceneEntityCfg,
        joint_limit_ke: float = 0.0,
        joint_limit_kd: float = 0.0,
        gravity: tuple[float, float, float] = (0.0, 0.0, -9.81),
    ):
        """
        Args:
            env:             ``ManagerBasedRLEnv`` — provides device and physics dt.
            entity_cfg:         ``SceneEntityCfg`` naming the articulation and
                             optionally filtering joints/bodies with regex patterns
                             (same convention as IsaacLab MDP action/obs terms).
                             The cfg is resolved here; pass it *before* or
                             *after* calling ``resolve()`` — both work.
            joint_limit_ke:  Spring stiffness for joint-limit enforcement in the RNEA
                             backward pass (default 0 = no limit forces).
            joint_limit_kd:  Damping for joint-limit enforcement (default 0).
            gravity:         Gravity vector in world frame (m/s²).  Default
                             ``(0, 0, -9.81)`` matches IsaacLab's standard gravity
                             and enables full ``g(q) + C(q,q̇)q̇`` compensation.
                             Pass ``(0, 0, 0)`` for Coriolis-only compensation.
        """
        self._device: str   = env.device
        self._dt:     float = env.physics_dt
        device              = self._device

        # ------------------------------------------------------------------ #
        # Resolve cfg → articulation, then build kinematic topology.          #
        # ------------------------------------------------------------------ #
        entity_cfg.resolve(env.scene)
        articulation = env.scene[entity_cfg.name]
        self.articulation = articulation

        data = articulation.data

        # _build_topology resolves joint names via articulation.find_joints
        # and stores the resulting dof indices in topo.joint_ids.
        topo  = _build_topology(articulation, device, entity_cfg)
        self._topo      = topo
        self._joint_ids = topo.joint_ids   # list[int] | slice — runtime slicing

        num_bodies = len(topo.body_ids)   # only the bodies used by the arm
        root_dofs  = 0 if topo.is_fixed_base else 6

        # ------------------------------------------------------------------ #
        # Pre-allocated joint state work tensors (filled in-place per call).  #
        # ------------------------------------------------------------------ #
        self._q_work  = torch.zeros(topo.total_q,  dtype=torch.float32, device=device)
        self._qd_work = torch.zeros(topo.total_qd, dtype=torch.float32, device=device)

        # Persistent Warp views — no wp.from_torch inside compute_root_wrench.
        # Passed only to kernel inputs (read-only); IsaacLab tensors are never
        # written by the RNEA kernels.
        self._joint_q_wp  = wp.from_torch(self._q_work,  dtype=wp.float32)
        self._joint_qd_wp = wp.from_torch(self._qd_work, dtype=wp.float32)

        # Live references to IsaacLab buffers (updated in-place each physics step).
        self._joint_pos_buf = data.joint_pos   # (num_envs, total_dofs)
        self._joint_vel_buf = data.joint_vel   # (num_envs, total_dofs)
        if not topo.is_fixed_base:
            self._root_pose_buf = data.root_link_pose_w  # (num_envs, 7) pos+quat(w,x,y,z)
            self._root_vel_buf  = data.root_link_vel_w   # (num_envs, 6) lin+ang

        # ------------------------------------------------------------------ #
        # Static inertia tensors — slice to the arm bodies only.              #
        # topo.body_ids holds the IsaacLab body indices actually used.        #
        # default_mass   : (num_envs, all_bodies)                             #
        # default_inertia: (num_envs, all_bodies, 9)  — row-major 3×3        #
        # body_com_pose_b: (num_envs, all_bodies, 7)                         #
        # ------------------------------------------------------------------ #
        bids = topo.body_ids  # IsaacLab indices of the arm bodies

        mass_wp = wp.from_torch(
            data.default_mass[0, bids].to(device).contiguous(), dtype=wp.float32
        )
        inertia_wp = wp.from_torch(
            data.default_inertia[0, bids].reshape(num_bodies, 3, 3).to(device).contiguous(),
            dtype=wp.mat33,
        )
        self.body_I_m = wp.empty((num_bodies,), dtype=wp.spatial_matrix, device=device)
        wp.launch(
            compute_spatial_inertia,
            num_bodies,
            inputs=[inertia_wp, mass_wp],
            outputs=[self.body_I_m],
            device=device,
        )

        com_pos_wp = wp.from_torch(
            data.body_com_pose_b[0, bids, :3].to(device).contiguous(), dtype=wp.vec3
        )
        self.body_X_com = wp.empty((num_bodies,), dtype=wp.transform, device=device)
        wp.launch(
            compute_com_transforms,
            num_bodies,
            inputs=[com_pos_wp],
            outputs=[self.body_X_com],
            device=device,
        )

        # ------------------------------------------------------------------ #
        # Joint limits from IsaacLab.  ke/kd are not in the USD schema and   #
        # default to 0 (no limit-spring forces in the RNEA backward pass).   #
        # data.joint_pos_limits: (num_envs, total_dofs, 2)                   #
        # ------------------------------------------------------------------ #
        limits_np = np.empty((topo.total_qd, 2), dtype=np.float32)
        if not topo.is_fixed_base:
            limits_np[:root_dofs, 0] = -np.inf
            limits_np[:root_dofs, 1] =  np.inf
            limits_np[root_dofs:]    = data.joint_pos_limits[0, self._joint_ids].cpu().numpy()
        else:
            limits_np[:] = data.joint_pos_limits[0, self._joint_ids].cpu().numpy()

        self._joint_limit_lower = wp.array(limits_np[:, 0], dtype=wp.float32, device=device)
        self._joint_limit_upper = wp.array(limits_np[:, 1], dtype=wp.float32, device=device)
        self._joint_limit_ke = wp.array(
            np.full(topo.total_qd, joint_limit_ke, dtype=np.float32), dtype=wp.float32, device=device
        )
        self._joint_limit_kd = wp.array(
            np.full(topo.total_qd, joint_limit_kd, dtype=np.float32), dtype=wp.float32, device=device
        )

        # ------------------------------------------------------------------ #
        # Working buffers (reused every kernel call).                         #
        # ------------------------------------------------------------------ #
        self.joint_S_s      = wp.empty((topo.total_qd,), dtype=wp.spatial_vector, device=device)
        self.body_I_s       = wp.empty((num_bodies,),    dtype=wp.spatial_matrix, device=device)
        self.body_v_s       = wp.empty((num_bodies,),    dtype=wp.spatial_vector, device=device)
        self.body_f_s       = wp.zeros((num_bodies,),    dtype=wp.spatial_vector, device=device)
        self.body_a_s       = wp.empty((num_bodies,),    dtype=wp.spatial_vector, device=device)
        self.body_ft_s      = wp.zeros((num_bodies,),    dtype=wp.spatial_vector, device=device)
        self.body_q_com     = wp.empty((num_bodies,),    dtype=wp.transform,      device=device)
        self.body_q         = wp.empty((num_bodies,),    dtype=wp.transform,      device=device)
        self.joint_tau      = wp.empty((topo.total_qd,), dtype=wp.float32,        device=device)
        # eval_rigid_tau reads body_f_ext unconditionally — must not be None.
        self.body_f_ext_zero = wp.zeros((num_bodies,),   dtype=wp.spatial_vector, device=device)

        # Single-element gravity arrays.  Must be wp.vec3 dtype because the
        # kernel does `gravity[world_idx]` → wp.vec3.  gravity_zero is kept
        # for reference; gravity_wp is what the kernels actually use.
        self.gravity_zero      = wp.zeros((1,),             dtype=wp.vec3,    device=device)
        self.gravity_wp        = wp.array(
            [wp.vec3(gravity[0], gravity[1], gravity[2])],
            dtype=wp.vec3, device=device,
        )
        self.joint_f_zero      = wp.zeros((topo.total_qd,), dtype=wp.float32, device=device)
        self.joint_target_zero = wp.zeros((topo.total_qd,), dtype=wp.float32, device=device)
        self.joint_ke_zero     = wp.zeros((topo.total_qd,), dtype=wp.float32, device=device)

        self._root_dofs = root_dofs
        self._num_envs  = env.num_envs

        # Pre-build tiled topology + working buffers for all-env batch launch.
        self._setup_batched()

    # ---------------------------------------------------------------------- #

    def _setup_batched(self) -> None:
        """
        Tile the single-env topology arrays into batched equivalents so that
        ``compute_all_wrenches`` can run one kernel launch across all
        environments instead of looping.

        Naming convention:
          *_b  — batched Warp arrays (size = num_envs × single-env size)
        """
        topo       = self._topo
        device     = self._device
        E          = self._num_envs
        NJ         = topo.num_joints
        NB         = len(topo.body_ids)
        total_q    = topo.total_q
        total_qd   = topo.total_qd

        # ---- tile scalar-per-joint arrays --------------------------------- #
        # joint_type: same for every env
        jtype_np = topo.joint_type.numpy()                         # (NJ,)
        self._jtype_b = wp.array(np.tile(jtype_np, E),
                                 dtype=wp.int32, device=device)

        # joint_parent / joint_child: body indices must be offset by e*NB.
        # Parent == -1 means "world" (floating root); keep -1 intact.
        jp_np = topo.joint_parent.numpy()                          # (NJ,)
        jc_np = topo.joint_child.numpy()                           # (NJ,)
        jp_blocks = [np.where(jp_np < 0, jp_np, jp_np + e * NB) for e in range(E)]
        jc_blocks = [jc_np + e * NB                               for e in range(E)]
        self._jparent_b = wp.array(np.concatenate(jp_blocks),
                                   dtype=wp.int32, device=device)
        self._jchild_b  = wp.array(np.concatenate(jc_blocks),
                                   dtype=wp.int32, device=device)

        # joint_X_p / joint_X_c: frame transforms are env-independent
        jXp_np = topo.joint_X_p.numpy()                           # (NJ, 7)
        jXc_np = topo.joint_X_c.numpy()                           # (NJ, 7)
        self._jXp_b = wp.array(np.tile(jXp_np, (E, 1)).reshape(-1, 7),
                                dtype=wp.transform, device=device)
        self._jXc_b = wp.array(np.tile(jXc_np, (E, 1)).reshape(-1, 7),
                                dtype=wp.transform, device=device)

        # joint_axis: (total_qd, 3) — DoF-indexed, tile with qd offset bookkeeping
        jaxis_np = topo.joint_axis.numpy()                         # (total_qd, 3)
        self._jaxis_b = wp.array(np.tile(jaxis_np, (E, 1)).reshape(-1, 3),
                                 dtype=wp.vec3, device=device)

        # joint_dof_dim: (NJ, 2) — same for every env
        jdof_np = topo.joint_dof_dim.numpy()                      # (NJ, 2)
        self._jdof_b = wp.array(np.tile(jdof_np, (E, 1)).reshape(-1, 2),
                                dtype=wp.int32, ndim=2, device=device)

        # joint_q_start / joint_qd_start: offset by e*total_q / e*total_qd
        jqs_np  = topo.joint_q_start.numpy()                      # (NJ,)
        jqds_np = topo.joint_qd_start.numpy()                     # (NJ,)
        jqs_blocks  = [jqs_np  + e * total_q  for e in range(E)]
        jqds_blocks = [jqds_np + e * total_qd for e in range(E)]
        self._jqs_b  = wp.array(np.concatenate(jqs_blocks),
                                dtype=wp.int32, device=device)
        self._jqds_b = wp.array(np.concatenate(jqds_blocks),
                                dtype=wp.int32, device=device)

        # articulation_start: [0, NJ, 2*NJ, ..., E*NJ]
        self._art_start_b = wp.array(
            np.arange(E + 1, dtype=np.int32) * NJ,
            dtype=wp.int32, device=device,
        )

        # body_world: all zeros (all bodies share gravity[0] = zero)
        self._bworld_b = wp.zeros(E * NB, dtype=wp.int32, device=device)

        # ---- tile inertial arrays (env-independent — use env-0 values) ---- #
        bIm_np = self.body_I_m.numpy()                             # (NB, 6, 6)
        self._bIm_b = wp.array(
            np.tile(bIm_np, (E, 1, 1)).reshape(-1, 6, 6),
            dtype=wp.spatial_matrix, device=device,
        )

        bXcom_np = self.body_X_com.numpy()                         # (NB, 7)
        self._bXcom_b = wp.array(
            np.tile(bXcom_np, (E, 1)).reshape(-1, 7),
            dtype=wp.transform, device=device,
        )

        # ---- tile limit / spring arrays ----------------------------------- #
        self._jlim_lo_b = wp.array(
            np.tile(self._joint_limit_lower.numpy(), E),
            dtype=wp.float32, device=device,
        )
        self._jlim_hi_b = wp.array(
            np.tile(self._joint_limit_upper.numpy(), E),
            dtype=wp.float32, device=device,
        )
        self._jlim_ke_b = wp.array(
            np.tile(self._joint_limit_ke.numpy(), E),
            dtype=wp.float32, device=device,
        )
        self._jlim_kd_b = wp.array(
            np.tile(self._joint_limit_kd.numpy(), E),
            dtype=wp.float32, device=device,
        )

        # ---- batched joint-state work tensors ----------------------------- #
        # Flat 1-D tensors; 2-D views share storage for easy env-wise filling.
        self._q_work_b  = torch.zeros(E * total_q,  dtype=torch.float32, device=self._device)
        self._qd_work_b = torch.zeros(E * total_qd, dtype=torch.float32, device=self._device)
        self._q_work_b2d  = self._q_work_b.view(E, total_q)
        self._qd_work_b2d = self._qd_work_b.view(E, total_qd)

        # Persistent Warp views — updated in-place by compute_all_wrenches.
        self._jq_b_wp  = wp.from_torch(self._q_work_b,  dtype=wp.float32)
        self._jqd_b_wp = wp.from_torch(self._qd_work_b, dtype=wp.float32)

        # ---- batched working buffers (one slot per env) ------------------- #
        self._jS_b    = wp.empty((E * total_qd,), dtype=wp.spatial_vector, device=device)
        self._bIs_b   = wp.empty((E * NB,),       dtype=wp.spatial_matrix, device=device)
        self._bv_b    = wp.empty((E * NB,),       dtype=wp.spatial_vector, device=device)
        self._bf_b    = wp.zeros((E * NB,),       dtype=wp.spatial_vector, device=device)
        self._ba_b    = wp.empty((E * NB,),       dtype=wp.spatial_vector, device=device)
        self._bft_b   = wp.zeros((E * NB,),       dtype=wp.spatial_vector, device=device)
        self._bq_b    = wp.empty((E * NB,),       dtype=wp.transform,      device=device)
        self._bqcom_b = wp.empty((E * NB,),       dtype=wp.transform,      device=device)
        self._jtau_b        = wp.empty((E * total_qd,), dtype=wp.float32, device=device)
        self._jtau_gravity_b = wp.empty((E * total_qd,), dtype=wp.float32, device=device)

        self._jf_zero_b      = wp.zeros((E * total_qd,), dtype=wp.float32,        device=device)
        self._jtgt_zero_b    = wp.zeros((E * total_qd,), dtype=wp.float32,        device=device)
        self._jke_zero_b     = wp.zeros((E * total_qd,), dtype=wp.float32,        device=device)
        # eval_rigid_tau reads body_f_ext unconditionally — must not be None.
        self._bfext_zero_b   = wp.zeros((E * NB,),       dtype=wp.spatial_vector, device=device)

        # ---- wrench composer integration ---------------------------------- #
        # Buffers written by extract_root_compensation and consumed directly by
        # the wrench composer — no PyTorch intermediate, no extra synchronize.
        self._wrench_composer  = self.articulation.instantaneous_wrench_composer
        self._comp_force_wp    = wp.zeros((E, 1), dtype=wp.vec3f, device=device)
        self._comp_torque_wp   = wp.zeros((E, 1), dtype=wp.vec3f, device=device)
        # body_ids=[0] — root body is always the first body in the articulation.
        self._root_body_id_wp  = wp.array([0], dtype=wp.int32, device=device)

    # ---------------------------------------------------------------------- #

    def compute_root_wrench(
        self,
        target: torch.Tensor | wp.array,
        mode: InputMode = InputMode.ACCEL,
        env_idx: int = 0,
    ) -> wp.array:
        """
        Compute the 6D wrench the arm exerts on the drone root body.

        Joint state is read from ``Articulation.data`` (sliced to the joints
        selected by ``SceneEntityCfg``) and written into pre-allocated work
        tensors — no allocation per call.

        Args:
            target:   Desired arm-joint motion.  Shape ``(num_arm_joints,)`` or
                      ``(num_envs, num_arm_joints)`` where ``num_arm_joints``
                      matches the joints resolved by ``SceneEntityCfg``.
                      Accepts ``torch.Tensor`` for all modes; ``wp.array`` for
                      ``ACCEL`` only.
            mode:     Interpretation of ``target`` (default ``ACCEL``):

                      * ``ACCEL`` — explicit joint accelerations.
                      * ``VEL``   — desired joint velocities →
                        ``qdd ≈ (vel_target − vel_curr) / dt``.
                      * ``POS``   — desired joint positions →
                        ``qdd = 2·(pos_target − pos_curr − vel·dt) / dt²``.
            env_idx:  Which environment instance to read state from (default 0).

        Returns:
            ``joint_tau`` — Warp array of length ``total_qd``.
            Elements ``[0:6]`` are the 6D wrench at the root free-joint for
            floating-base systems.  Negate to obtain the feedforward
            compensation wrench.

        Note:
            The RNEA kernels compute the Coriolis / centrifugal bias only.
            The inertial ``M·qdd`` term is not yet wired into the kernel;
            ``joint_qdd`` derived from ``target`` is computed but not forwarded.
        """
        topo   = self._topo
        device = self._device

        # Slice to only the joints selected by SceneEntityCfg.
        arm_q  = self._joint_pos_buf[env_idx, self._joint_ids]
        arm_qd = self._joint_vel_buf[env_idx, self._joint_ids]

        # ---- Assemble full joint state in-place (no allocation) ----------- #
        if not topo.is_fixed_base:
            pose = self._root_pose_buf[env_idx]  # (7,) pos + quat(w,x,y,z)
            vel  = self._root_vel_buf[env_idx]   # (6,) lin + ang

            # Use local origin (0,0,0) for root position.
            # IsaacLab places environments hundreds of metres from the world
            # origin; feeding world coordinates into RNEA triggers massive
            # Steiner-parallel-axis terms (m·r²) that corrupt Coriolis forces.
            # Coriolis/centrifugal forces are position-invariant, so (0,0,0) is
            # always correct here.
            self._q_work[0:3] = 0.0
            # quat: IsaacLab (w,x,y,z) → Newton/Warp (x,y,z,w)
            self._q_work[3]   = pose[4]
            self._q_work[4]   = pose[5]
            self._q_work[5]   = pose[6]
            self._q_work[6]   = pose[3]
            self._q_work[7:]  = arm_q

            self._qd_work[0:6] = vel
            self._qd_work[6:]  = arm_qd
        else:
            self._q_work[:]  = arm_q
            self._qd_work[:] = arm_qd

        # ---- Derive arm-joint qdd from the chosen input mode -------------- #
        dt = self._dt

        def _pick(t: torch.Tensor) -> torch.Tensor:
            return t[env_idx] if t.dim() == 2 else t

        if mode is InputMode.ACCEL:
            arm_qdd: torch.Tensor | wp.array = (
                _pick(target) if isinstance(target, torch.Tensor) else target
            )
        elif mode is InputMode.VEL:
            arm_qdd = (_pick(target) - arm_qd) / dt
        elif mode is InputMode.POS:
            t = _pick(target)
            arm_qdd = 2.0 * (t - arm_q - arm_qd * dt) / (dt * dt)
        else:
            raise ValueError(f"Unknown InputMode: {mode}")

        del arm_qdd  # not consumed by current kernels; reserved for M·qdd extension

        # ---- 1. Forward kinematics --------------------------------------- #
        wp.launch(
            eval_rigid_fk,
            dim=topo.articulation_start.shape[0] - 1,
            inputs=[
                topo.articulation_start,
                topo.joint_type,
                topo.joint_parent,
                topo.joint_child,
                topo.joint_q_start,
                topo.joint_qd_start,
                self._joint_q_wp,
                topo.joint_X_p,
                topo.joint_X_c,
                self.body_X_com,
                topo.joint_axis,
                topo.joint_dof_dim,
            ],
            outputs=[self.body_q, self.body_q_com],
            device=device,
        )

        # ---- 2. RNEA forward pass (Coriolis / centrifugal bias) ---------- #
        self.body_f_s.zero_()
        wp.launch(
            eval_rigid_id,
            dim=topo.articulation_start.shape[0] - 1,
            inputs=[
                topo.articulation_start,
                topo.joint_type,
                topo.joint_parent,
                topo.joint_child,
                topo.joint_qd_start,
                self._joint_qd_wp,
                topo.joint_axis,
                topo.joint_dof_dim,
                self.body_I_m,
                self.body_q,
                self.body_q_com,
                topo.joint_X_p,
                topo.body_world,
                self.gravity_wp,
            ],
            outputs=[
                self.joint_S_s,
                self.body_I_s,
                self.body_v_s,
                self.body_f_s,
                self.body_a_s,
            ],
            device=device,
        )

        # ---- 3. RNEA backward pass --------------------------------------- #
        self.body_ft_s.zero_()
        wp.launch(
            eval_rigid_tau,
            dim=topo.articulation_start.shape[0] - 1,
            inputs=[
                topo.articulation_start,
                topo.joint_type,
                topo.joint_parent,
                topo.joint_child,
                topo.joint_q_start,
                topo.joint_qd_start,
                topo.joint_dof_dim,
                self.joint_target_zero,
                self.joint_target_zero,
                self._joint_q_wp,
                self._joint_qd_wp,
                self.joint_f_zero,
                self.joint_ke_zero,
                self.joint_ke_zero,
                self._joint_limit_lower,
                self._joint_limit_upper,
                self._joint_limit_ke,
                self._joint_limit_kd,
                self.joint_S_s,
                self.body_f_s,
                self.body_f_ext_zero,
            ],
            outputs=[
                self.body_ft_s,
                self.joint_tau,
            ],
            device=device,
        )

        wp.synchronize_device(device)
        return self.joint_tau  # [0:6] = root free-joint = drone wrench

    # ---------------------------------------------------------------------- #

    def _run_rnea_all(self, target: torch.Tensor, mode: InputMode) -> None:
        """Fill ``_jtau_b`` with the RNEA output for all environments.

        Reads joint state from ``Articulation.data``, fills the batched work
        tensors in-place, and launches FK → ID → tau kernels on Warp's stream.
        Does **not** synchronize — callers are responsible for synchronization
        before reading ``_jtau_b`` from a different CUDA stream.
        """
        topo   = self._topo
        device = self._device
        E      = self._num_envs
        dt     = self._dt

        arm_q_all  = self._joint_pos_buf[:, self._joint_ids]
        arm_qd_all = self._joint_vel_buf[:, self._joint_ids]

        if target.dim() == 1:
            target = target.unsqueeze(0).expand(E, -1)

        if mode is InputMode.ACCEL:
            pass
        elif mode is InputMode.VEL:
            target = (target - arm_qd_all) / dt
        elif mode is InputMode.POS:
            target = 2.0 * (target - arm_q_all - arm_qd_all * dt) / (dt * dt)
        else:
            raise ValueError(f"Unknown InputMode: {mode}")

        q2d  = self._q_work_b2d
        qd2d = self._qd_work_b2d

        if not topo.is_fixed_base:
            root_pose = self._root_pose_buf
            root_vel  = self._root_vel_buf
            # Use local origin (0,0,0) — world position causes huge Steiner terms.
            q2d[:, 0:3] = 0.0
            q2d[:, 3:6] = root_pose[:, 4:7]   # quat xyz  (IsaacLab w,x,y,z → x,y,z,w)
            q2d[:, 6]   = root_pose[:, 3]      # quat w
            q2d[:, 7:]  = arm_q_all
            qd2d[:, 0:6] = root_vel
            qd2d[:, 6:]  = arm_qd_all
        else:
            q2d[:]  = arm_q_all
            qd2d[:] = arm_qd_all

        wp.launch(
            eval_rigid_fk,
            dim=E,
            inputs=[
                self._art_start_b, self._jtype_b, self._jparent_b, self._jchild_b,
                self._jqs_b, self._jqds_b, self._jq_b_wp,
                self._jXp_b, self._jXc_b, self._bXcom_b,
                self._jaxis_b, self._jdof_b,
            ],
            outputs=[self._bq_b, self._bqcom_b],
            device=device,
        )

        self._bf_b.zero_()
        wp.launch(
            eval_rigid_id,
            dim=E,
            inputs=[
                self._art_start_b, self._jtype_b, self._jparent_b, self._jchild_b,
                self._jqds_b, self._jqd_b_wp,
                self._jaxis_b, self._jdof_b,
                self._bIm_b, self._bq_b, self._bqcom_b,
                self._jXp_b, self._bworld_b, self.gravity_zero,
            ],
            outputs=[self._jS_b, self._bIs_b, self._bv_b, self._bf_b, self._ba_b],
            device=device,
        )

        self._bft_b.zero_()
        wp.launch(
            eval_rigid_tau,
            dim=E,
            inputs=[
                self._art_start_b, self._jtype_b, self._jparent_b, self._jchild_b,
                self._jqs_b, self._jqds_b, self._jdof_b,
                self._jtgt_zero_b, self._jtgt_zero_b,
                self._jq_b_wp, self._jqd_b_wp,
                self._jf_zero_b, self._jke_zero_b, self._jke_zero_b,
                self._jlim_lo_b, self._jlim_hi_b, self._jlim_ke_b, self._jlim_kd_b,
                self._jS_b, self._bf_b, self._bfext_zero_b,
            ],
            outputs=[self._bft_b, self._jtau_b],
            device=device,
        )

    # ---------------------------------------------------------------------- #

    def apply_compensation(
        self,
        target: torch.Tensor,
        mode: InputMode = InputMode.ACCEL,
    ) -> None:
        """Run RNEA and apply feedforward compensation via the wrench composer.

        Equivalent to calling ``compute_all_wrenches``, negating the root
        wrench, and passing it to ``instantaneous_wrench_composer``, but
        without any PyTorch intermediate tensor or explicit synchronization —
        everything stays on Warp's CUDA stream.

        Args:
            target: Desired arm-joint motion for all environments (same
                    semantics as ``compute_all_wrenches``).
            mode:   Interpretation of ``target``.
        """
        self._run_rnea_all(target, mode)

        # Negate and NaN-guard the root wrench → (E, 1) vec3f Warp buffers.
        wp.launch(
            extract_root_compensation,
            dim=self._num_envs,
            inputs=[self._jtau_b, self._topo.total_qd],
            outputs=[self._comp_force_wp, self._comp_torque_wp],
            device=self._device,
        )

        # Pass Warp arrays directly — no clone, no sync needed.
        # The composer launches its own kernel on the same Warp stream, so
        # ordering is guaranteed without an explicit wp.synchronize_device.
        self._wrench_composer.add_forces_and_torques(
            forces=self._comp_force_wp,
            torques=self._comp_torque_wp,
            body_ids=self._root_body_id_wp,
            is_global=True,
        )

    # ---------------------------------------------------------------------- #

    def compute_all_wrenches(
        self,
        target: torch.Tensor,
        mode: InputMode = InputMode.ACCEL,
    ) -> torch.Tensor:
        """
        Compute the 6D wrench for **all** environments in a single GPU kernel
        launch by tiling the topology arrays across environments.

        Args:
            target: Desired arm-joint motion for all environments.
                    Shape ``(num_envs, num_arm_joints)`` for ``ACCEL``/``VEL``/``POS``,
                    or ``(num_arm_joints,)`` to broadcast the same target to
                    every environment.
            mode:   Interpretation of ``target`` (same as ``compute_root_wrench``).

        Returns:
            ``torch.Tensor`` of shape ``(num_envs, total_qd)``.
            Column ``[:, 0:6]`` is the 6D wrench at the root free-joint for
            each environment.  Negate to obtain the feedforward compensation.
        """
        topo   = self._topo
        E      = self._num_envs

        self._run_rnea_all(target, mode)

        # Sync before handing Warp buffer to PyTorch/PhysX stream.
        wp.synchronize_device(self._device)

        result = wp.to_torch(self._jtau_b).view(E, topo.total_qd).clone()

        # Zero out rows where physics has already diverged so NaN forces don't
        # cascade to neighbouring envs.
        bad_rows = torch.isnan(result).any(dim=1) | torch.isinf(result).any(dim=1)
        if bad_rows.any():
            result[bad_rows] = 0.0

        return result

    # ---------------------------------------------------------------------- #

    def debug_compensation(
        self,
        target: torch.Tensor,
        mode: InputMode = InputMode.ACCEL,
        print_interval: int = 10,
        _s: dict = {},
    ) -> None:
        """Apply compensation and log a comparison of gravity vs. Coriolis contributions.

        Runs RNEA twice (with and without gravity) and reads back the forces
        currently staged in the wrench composer, then stores per-step snapshots.
        Call this **instead of** ``apply_compensation`` when debugging.

        Args:
            target:         Same as ``apply_compensation``.
            mode:           Same as ``apply_compensation``.
            print_interval: Print a summary every N steps (default 10).

        Stored state (accessible via ``debug_last``):
            ``comp_gravity``    (E, 6) — with-gravity compensation (negated wrench)
            ``comp_coriolis``   (E, 6) — Coriolis-only compensation (no gravity)
            ``gravity_term``    (E, 6) — difference = pure gravity contribution
            ``applied``         (E, 6) — forces+torques actually in composer buffer
                                         (written by the *previous* call to this method)
        """
        device = self._device
        E      = self._num_envs
        total_qd = self._topo.total_qd
        step = _s.get('step', 0)

        # One-time sanity check on the gravity vector.
        if step == 0:
            print(f"[RNEA debug] gravity_wp = {self.gravity_wp.numpy()}")

        # ---- Read the compensation we applied last step -------------------- #
        # The instantaneous_wrench_composer is reset by IsaacLab before
        # apply_actions, so reading the composer buffer here always gives zeros.
        # We save our own copy at the end of each call instead.
        applied = _s.get('prev_applied', torch.zeros(E, 6, device=device))

        # ---- With gravity (current setting) -------------------------------- #
        self._run_rnea_all(target, mode)
        # Save result within Warp's stream before overwriting with pass 2.
        # Using wp.copy avoids a race condition: if we cloned via PyTorch here,
        # the clone (PyTorch stream) and the next _run_rnea_all (Warp stream)
        # would race on _jtau_b.  wp.copy stays on Warp's stream — ordered.
        wp.copy(self._jtau_gravity_b, self._jtau_b)

        # ---- Coriolis-only (swap to gravity_zero for one pass) ------------- #
        _orig = self.gravity_wp
        self.gravity_wp = self.gravity_zero
        self._run_rnea_all(target, mode)
        self.gravity_wp = _orig

        # Single sync — both Warp passes are done, safe to read both buffers.
        wp.synchronize_device(device)
        tau_g = wp.to_torch(self._jtau_gravity_b).view(E, total_qd)[:, :6].clone()
        tau_c = wp.to_torch(self._jtau_b).view(E, total_qd)[:, :6].clone()
        comp_gravity  = -tau_g   # (E, 6)
        comp_coriolis = -tau_c   # (E, 6)

        gravity_term = comp_gravity - comp_coriolis   # (E, 6)

        # ---- Apply compensation using the gravity result already in _jtau_gravity_b #
        # Re-run extract on the saved gravity buffer (no extra RNEA pass needed).
        wp.launch(
            extract_root_compensation,
            dim=E,
            inputs=[self._jtau_gravity_b, total_qd],
            outputs=[self._comp_force_wp, self._comp_torque_wp],
            device=device,
        )
        self._wrench_composer.add_forces_and_torques(
            forces=self._comp_force_wp,
            torques=self._comp_torque_wp,
            body_ids=self._root_body_id_wp,
            is_global=True,
        )

        # ---- Persist for external inspection ------------------------------- #
        self.debug_last = {
            'step':          step,
            'comp_gravity':  comp_gravity,
            'comp_coriolis': comp_coriolis,
            'gravity_term':  gravity_term,
            'applied':       applied,
        }
        # Save this step's applied compensation so next call can report it.
        _s['prev_applied'] = comp_gravity.clone()
        _s['step'] = step + 1

        # ---- Periodic console summary -------------------------------------- #
        if step % print_interval == 0:
            def _stats(t: torch.Tensor) -> str:
                """mean ± std of per-env L2 norm, env-0 value."""
                norms = t.norm(dim=-1)          # (E,) or (E,)
                return (f"norm mean={norms.mean():.4f} max={norms.max():.4f} "
                        f"| env0={t[0].tolist()}")

            print(f"\n[RNEA debug step={step}]")
            print(f"  comp_gravity  (F+T): {_stats(comp_gravity)}")
            print(f"  comp_coriolis (F+T): {_stats(comp_coriolis)}")
            print(f"  gravity_term  (F+T): {_stats(gravity_term)}")
            print(f"  applied prev  (F+T): {_stats(applied)}")
            # Ratio: how much of the compensation is gravitational?
            g_frac = gravity_term.norm(dim=-1) / (comp_gravity.norm(dim=-1) + 1e-8)
            print(f"  gravity fraction: mean={g_frac.mean():.2%}  max={g_frac.max():.2%}")
