# Newton RNEA

Feedforward wrench prediction for a robot arm mounted on a drone, using the
Recursive Newton-Euler Algorithm (RNEA) implemented in [Warp](https://github.com/NVIDIA/warp).

The predictor reads live joint state directly from an
[IsaacLab](https://github.com/isaac-sim/IsaacLab) `Articulation` and parses
kinematic structure (joint types, axes, frame offsets, parent/child tree) from
the USD stage at initialisation — no external model object required.

---

## What it does

Given the current joint positions and velocities of the arm (read automatically
from IsaacLab), and a desired future arm motion, `ArmWrenchPredictor` computes
the **6D wrench** (force + torque) that the arm exerts on the drone root body.
Negating this gives the feedforward compensation term the drone controller
needs to reject the arm's inertial disturbance.

The computation is pure RNEA — no gravity, no contacts, no integration:

```
tau = C(q, qd) * qd          # Coriolis / centrifugal bias only
wrench = tau[0:6]             # root free-joint = drone attachment point
```

---

## Requirements

- [IsaacLab](https://github.com/isaac-sim/IsaacLab) (tested on Isaac Sim 5.x)
- [Warp](https://github.com/NVIDIA/warp) (`pip install warp-lang`)
- [PyTorch](https://pytorch.org/)
- [OpenUSD](https://openusd.org/) (`pxr` — bundled with Isaac Sim)

---

## Installation

```bash
git clone <repo>
cd Newton_RNEA
pip install -e .
```

---

## Usage

### Basic setup

```python
from src import ArmWrenchPredictor, InputMode

predictor = ArmWrenchPredictor(
    env=env,                   # isaaclab.envs.ManagerBasedRLEnv
    articulation=env.scene["robot"],  # isaaclab.assets.Articulation
    joint_limit_ke=0.0,        # limit spring stiffness (0 = no limit forces)
    joint_limit_kd=0.0,        # limit spring damping
)
```

`ArmWrenchPredictor.__init__` traverses the USD stage once to extract the full
kinematic tree. All subsequent calls are allocation-free.

### Predicting from joint accelerations

```python
import torch

# Planned arm joint accelerations — shape (num_arm_joints,)
joint_qdd = torch.zeros(6, device=env.device)

wrench = predictor.compute_root_wrench(joint_qdd, mode=InputMode.ACCEL)

# wrench is a Warp array of length total_qd.
# wrench[0:6] = [fx, fy, fz, tx, ty, tz] at the drone root joint.
feedforward_compensation = -wp.to_torch(wrench)[:6]
```

### Predicting from a velocity target

```python
vel_target = torch.zeros(6, device=env.device)   # desired arm joint velocities

wrench = predictor.compute_root_wrench(vel_target, mode=InputMode.VEL)
# Internally computes: qdd ≈ (vel_target − vel_current) / dt
```

### Predicting from a position target

```python
pos_target = torch.zeros(6, device=env.device)   # desired arm joint positions

wrench = predictor.compute_root_wrench(pos_target, mode=InputMode.POS)
# Internally computes: qdd = 2·(pos_target − pos − vel·dt) / dt²
```

### Batched environments

Pass `env_idx` to select which parallel environment to use:

```python
wrench = predictor.compute_root_wrench(joint_qdd, env_idx=3)
```

---

## How it works

### Initialisation (once)

| Step | Source |
|------|--------|
| Joint types, axes, parent/child tree, frame offsets (`X_p`, `X_c`) | USD stage via `pxr.UsdPhysics` |
| Body masses and inertia tensors | `articulation.data.default_mass / default_inertia` |
| Centre-of-mass offsets | `articulation.data.body_com_pose_b` |
| Joint position limits | `articulation.data.joint_pos_limits` |
| Physics timestep | `env.physics_dt` |

For floating-base robots (drone + arm) a synthetic **FREE joint** is prepended
as joint 0 (world → drone root), since PhysX does not expose the root 6-DOF as
a joint prim in USD.

### Per-call (`compute_root_wrench`)

1. Root pose and arm joint state are copied in-place into pre-allocated torch
   tensors (`_q_work`, `_qd_work`). Persistent Warp views over those tensors
   are passed to the kernels — no `wp.from_torch` call per step.
2. **FK** (`eval_rigid_fk`) — computes world-frame body transforms from joint
   positions.
3. **RNEA forward pass** (`eval_rigid_id`) — propagates spatial velocities and
   computes Coriolis/centrifugal bias forces. Gravity is set to zero.
4. **RNEA backward pass** (`eval_rigid_tau`) — propagates wrenches back to
   joint torques. `joint_tau[0:6]` is the wrench at the root free-joint.

### Quaternion convention

| System | Order |
|--------|-------|
| IsaacLab | `(w, x, y, z)` |
| Newton / Warp | `(x, y, z, w)` |

The root quaternion is reordered in-place inside `compute_root_wrench` before
the kernel launches.

---

## Project structure

```
Newton_RNEA/
├── src/
│   ├── __init__.py       # Public exports
│   ├── predictor.py      # ArmWrenchPredictor + USD topology parser
│   ├── kernels.py        # Warp RNEA kernels (FK, ID forward, ID backward)
│   └── enums.py          # JointType, InputMode
└── README.md
```
