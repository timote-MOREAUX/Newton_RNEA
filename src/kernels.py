import warp as wp

from .enums import JointType

@wp.func
def compute_2d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    q0: float,
    q1: float,
    qd0: float,
    qd1: float,
):
    """
    Computes the rotation quaternion and 3D angular velocity given the joint axes, coordinates and velocities.
    """
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, wp.cross(axis_0, axis_1)))

    # body local axes
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))

    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, q0)

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, q1)

    rot = q_1 * q_0

    vel = axis_0 * qd0 + axis_1 * qd1

    return rot, vel

@wp.func
def compute_3d_rotational_dofs(
    axis_0: wp.vec3,
    axis_1: wp.vec3,
    axis_2: wp.vec3,
    q0: float,
    q1: float,
    q2: float,
    qd0: float,
    qd1: float,
    qd2: float,
):
    """
    Computes the rotation quaternion and 3D angular velocity given the joint axes, coordinates and velocities.
    """
    q_off = wp.quat_from_matrix(wp.matrix_from_cols(axis_0, axis_1, axis_2))

    # body local axes
    local_0 = wp.quat_rotate(q_off, wp.vec3(1.0, 0.0, 0.0))
    local_1 = wp.quat_rotate(q_off, wp.vec3(0.0, 1.0, 0.0))
    local_2 = wp.quat_rotate(q_off, wp.vec3(0.0, 0.0, 1.0))

    # reconstruct rotation axes
    axis_0 = local_0
    q_0 = wp.quat_from_axis_angle(axis_0, q0)

    axis_1 = wp.quat_rotate(q_0, local_1)
    q_1 = wp.quat_from_axis_angle(axis_1, q1)

    axis_2 = wp.quat_rotate(q_1 * q_0, local_2)
    q_2 = wp.quat_from_axis_angle(axis_2, q2)

    rot = q_2 * q_1 * q_0
    vel = axis_0 * qd0 + axis_1 * qd1 + axis_2 * qd2

    return rot, vel

@wp.func
def jcalc_transform(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    axis_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    joint_q: wp.array(dtype=float),
    q_start: int,
):
    if type == JointType.PRISMATIC:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(axis * q, wp.quat_identity())
        return X_jc

    if type == JointType.REVOLUTE:
        q = joint_q[q_start]
        axis = joint_axis[axis_start]
        X_jc = wp.transform(wp.vec3(), wp.quat_from_axis_angle(axis, q))
        return X_jc

    if type == JointType.BALL:
        qx = joint_q[q_start + 0]
        qy = joint_q[q_start + 1]
        qz = joint_q[q_start + 2]
        qw = joint_q[q_start + 3]

        X_jc = wp.transform(wp.vec3(), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.FIXED:
        X_jc = wp.transform_identity()
        return X_jc

    if type == JointType.FREE or type == JointType.DISTANCE:
        px = joint_q[q_start + 0]
        py = joint_q[q_start + 1]
        pz = joint_q[q_start + 2]

        qx = joint_q[q_start + 3]
        qy = joint_q[q_start + 4]
        qz = joint_q[q_start + 5]
        qw = joint_q[q_start + 6]

        X_jc = wp.transform(wp.vec3(px, py, pz), wp.quat(qx, qy, qz, qw))
        return X_jc

    if type == JointType.D6:
        pos = wp.vec3(0.0)
        rot = wp.quat_identity()

        # unroll for loop to ensure joint actions remain differentiable
        # (since differentiating through a for loop that updates a local variable is not supported)

        if lin_axis_count > 0:
            axis = joint_axis[axis_start + 0]
            pos += axis * joint_q[q_start + 0]
        if lin_axis_count > 1:
            axis = joint_axis[axis_start + 1]
            pos += axis * joint_q[q_start + 1]
        if lin_axis_count > 2:
            axis = joint_axis[axis_start + 2]
            pos += axis * joint_q[q_start + 2]

        ia = axis_start + lin_axis_count
        iq = q_start + lin_axis_count
        if ang_axis_count == 1:
            axis = joint_axis[ia]
            rot = wp.quat_from_axis_angle(axis, joint_q[iq])
        if ang_axis_count == 2:
            rot, _ = compute_2d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_q[iq + 0],
                joint_q[iq + 1],
                0.0,
                0.0,
            )
        if ang_axis_count == 3:
            rot, _ = compute_3d_rotational_dofs(
                joint_axis[ia + 0],
                joint_axis[ia + 1],
                joint_axis[ia + 2],
                joint_q[iq + 0],
                joint_q[iq + 1],
                joint_q[iq + 2],
                0.0,
                0.0,
                0.0,
            )

        X_jc = wp.transform(pos, rot)
        return X_jc

    # default case
    return wp.transform_identity()

@wp.func
def compute_link_transform(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # parent transform
    parent = joint_parent[i]
    child = joint_child[i]

    # parent transform in spatial coordinates
    X_pj = joint_X_p[i]
    X_cj = joint_X_c[i]
    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    type = joint_type[i]
    qd_start = joint_qd_start[i]
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    coord_start = joint_q_start[i]

    # compute transform across joint
    X_j = jcalc_transform(type, joint_axis, qd_start, lin_axis_count, ang_axis_count, joint_q, coord_start)

    # transform from world to joint anchor frame at child body
    X_wcj = X_wpj * X_j
    # transform from world to child body frame
    X_wc = X_wcj * wp.transform_inverse(X_cj)

    # compute transform of center of mass
    X_cm = body_X_com[child]
    X_sm = X_wc * X_cm

    # store geometry transforms
    body_q[child] = X_wc
    body_q_com[child] = X_sm

@wp.kernel
def eval_rigid_fk(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_q: wp.array(dtype=float),
    joint_X_p: wp.array(dtype=wp.transform),
    joint_X_c: wp.array(dtype=wp.transform),
    body_X_com: wp.array(dtype=wp.transform),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    # outputs
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
):
    # one thread per joint
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    for i in range(start, end):
        compute_link_transform(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_q_start,
            joint_qd_start,
            joint_q,
            joint_X_p,
            joint_X_c,
            body_X_com,
            joint_axis,
            joint_dof_dim,
            body_q,
            body_q_com,
        )

@wp.func
def transform_twist(t: wp.transform, x: wp.spatial_vector) -> wp.spatial_vector:
    """Transform a spatial twist between coordinate frames.

    This applies Warp's twist transform while preserving Newton's spatial
    layout. Newton stores twists as :math:`x = (v, \\omega)` (linear,
    angular), while Warp's low-level helper expects :math:`(\\omega, v)`.

    For rigid transform :math:`t = (R, p)` from source to destination:

    .. math::
       \\omega' = R\\omega,\\quad v' = Rv + p \\times \\omega'

    Args:
        t: Rigid transform from source frame to destination frame.
        x: Spatial twist in Newton layout ``(linear, angular)``.

    Returns:
        wp.spatial_vector: Transformed twist in Newton layout
        ``(linear, angular)``.
    """
    # Reorder Newton layout (linear, angular) → Warp layout (angular, linear).
    omega = wp.spatial_bottom(x)
    v     = wp.spatial_top(x)
    # Apply rigid transform: only the rotation acts on both components;
    # the translation introduces a coupling term on the linear part.
    p         = wp.transform_get_translation(t)
    omega_new = wp.transform_vector(t, omega)
    v_new     = wp.transform_vector(t, v) + wp.cross(p, omega_new)
    # Reorder back to Newton layout (linear, angular).
    return wp.spatial_vector(v_new, omega_new)

# compute motion subspace and velocity for a joint
@wp.func
def jcalc_motion(
    type: int,
    joint_axis: wp.array(dtype=wp.vec3),
    lin_axis_count: int,
    ang_axis_count: int,
    X_sc: wp.transform,
    joint_qd: wp.array(dtype=float),
    qd_start: int,
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
):
    if type == JointType.PRISMATIC:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.REVOLUTE:
        axis = joint_axis[qd_start]
        S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
        v_j_s = S_s * joint_qd[qd_start]
        joint_S_s[qd_start] = S_s
        return v_j_s

    if type == JointType.D6:
        v_j_s = wp.spatial_vector()
        if lin_axis_count > 0:
            axis = joint_axis[qd_start + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 0]
            joint_S_s[qd_start + 0] = S_s
        if lin_axis_count > 1:
            axis = joint_axis[qd_start + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 1]
            joint_S_s[qd_start + 1] = S_s
        if lin_axis_count > 2:
            axis = joint_axis[qd_start + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(axis, wp.vec3()))
            v_j_s += S_s * joint_qd[qd_start + 2]
            joint_S_s[qd_start + 2] = S_s
        if ang_axis_count > 0:
            axis = joint_axis[qd_start + lin_axis_count + 0]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 0]
            joint_S_s[qd_start + lin_axis_count + 0] = S_s
        if ang_axis_count > 1:
            axis = joint_axis[qd_start + lin_axis_count + 1]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 1]
            joint_S_s[qd_start + lin_axis_count + 1] = S_s
        if ang_axis_count > 2:
            axis = joint_axis[qd_start + lin_axis_count + 2]
            S_s = transform_twist(X_sc, wp.spatial_vector(wp.vec3(), axis))
            v_j_s += S_s * joint_qd[qd_start + lin_axis_count + 2]
            joint_S_s[qd_start + lin_axis_count + 2] = S_s

        return v_j_s

    if type == JointType.BALL:
        S_0 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        S_1 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        S_2 = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        joint_S_s[qd_start + 0] = S_0
        joint_S_s[qd_start + 1] = S_1
        joint_S_s[qd_start + 2] = S_2

        return S_0 * joint_qd[qd_start + 0] + S_1 * joint_qd[qd_start + 1] + S_2 * joint_qd[qd_start + 2]

    if type == JointType.FIXED:
        return wp.spatial_vector()

    if type == JointType.FREE or type == JointType.DISTANCE:
        v_j_s = transform_twist(
            X_sc,
            wp.spatial_vector(
                joint_qd[qd_start + 0],
                joint_qd[qd_start + 1],
                joint_qd[qd_start + 2],
                joint_qd[qd_start + 3],
                joint_qd[qd_start + 4],
                joint_qd[qd_start + 5],
            ),
        )

        joint_S_s[qd_start + 0] = transform_twist(X_sc, wp.spatial_vector(1.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 1] = transform_twist(X_sc, wp.spatial_vector(0.0, 1.0, 0.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 2] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 1.0, 0.0, 0.0, 0.0))
        joint_S_s[qd_start + 3] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 1.0, 0.0, 0.0))
        joint_S_s[qd_start + 4] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        joint_S_s[qd_start + 5] = transform_twist(X_sc, wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 1.0))

        return v_j_s

    wp.printf("jcalc_motion not implemented for joint type %d\n", type)

    # default case
    return wp.spatial_vector()

@wp.func
def spatial_cross(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b)
    v = wp.cross(w_a, v_b) + wp.cross(v_a, w_b)

    return wp.spatial_vector(v, w)

@wp.func
def transform_spatial_inertia(t: wp.transform, I: wp.spatial_matrix):
    """
    Transform a spatial inertia tensor to a new coordinate frame.

    This computes the change of coordinates for a spatial inertia tensor under a rigid-body
    transformation `t`. The result is mathematically equivalent to:

        adj_t^-T * I * adj_t^-1

    where `adj_t` is the adjoint transformation matrix of `t`, and `I` is the spatial inertia
    tensor in the original frame. This operation is described in Frank & Park, "Modern Robotics",
    Section 8.2.3 (pg. 290).

    Args:
        t (wp.transform): The rigid-body transform (destination ← source).
        I (wp.spatial_matrix): The spatial inertia tensor in the source frame.

    Returns:
        wp.spatial_matrix: The spatial inertia tensor expressed in the destination frame.
    """
    t_inv = wp.transform_inverse(t)

    q = wp.transform_get_rotation(t_inv)
    p = wp.transform_get_translation(t_inv)

    r1 = wp.quat_rotate(q, wp.vec3(1.0, 0.0, 0.0))
    r2 = wp.quat_rotate(q, wp.vec3(0.0, 1.0, 0.0))
    r3 = wp.quat_rotate(q, wp.vec3(0.0, 0.0, 1.0))

    R = wp.matrix_from_cols(r1, r2, r3)
    S = wp.skew(p) @ R

    T = wp.spatial_matrix(
        R[0, 0],
        R[0, 1],
        R[0, 2],
        S[0, 0],
        S[0, 1],
        S[0, 2],
        R[1, 0],
        R[1, 1],
        R[1, 2],
        S[1, 0],
        S[1, 1],
        S[1, 2],
        R[2, 0],
        R[2, 1],
        R[2, 2],
        S[2, 0],
        S[2, 1],
        S[2, 2],
        0.0,
        0.0,
        0.0,
        R[0, 0],
        R[0, 1],
        R[0, 2],
        0.0,
        0.0,
        0.0,
        R[1, 0],
        R[1, 1],
        R[1, 2],
        0.0,
        0.0,
        0.0,
        R[2, 0],
        R[2, 1],
        R[2, 2],
    )

    return wp.mul(wp.mul(wp.transpose(T), I), T)


@wp.func
def spatial_cross_dual(a: wp.spatial_vector, b: wp.spatial_vector):
    w_a = wp.spatial_bottom(a)
    v_a = wp.spatial_top(a)

    w_b = wp.spatial_bottom(b)
    v_b = wp.spatial_top(b)

    w = wp.cross(w_a, w_b) + wp.cross(v_a, v_b)
    v = wp.cross(w_a, v_b)

    return wp.spatial_vector(v, w)


@wp.func
def dense_index(stride: int, i: int, j: int):
    return i * stride + j

@wp.func
def compute_link_velocity(
    i: int,
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    body_world: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    type = joint_type[i]
    child = joint_child[i]
    parent = joint_parent[i]
    qd_start = joint_qd_start[i]

    X_pj = joint_X_p[i]
    # X_cj = joint_X_c[i]

    # parent anchor frame in world space
    X_wpj = X_pj
    if parent >= 0:
        X_wp = body_q[parent]
        X_wpj = X_wp * X_wpj

    # compute motion subspace and velocity across the joint (also stores S_s to global memory)
    lin_axis_count = joint_dof_dim[i, 0]
    ang_axis_count = joint_dof_dim[i, 1]
    v_j_s = jcalc_motion(
        type,
        joint_axis,
        lin_axis_count,
        ang_axis_count,
        X_wpj,
        joint_qd,
        qd_start,
        joint_S_s,
    )

    # parent velocity
    v_parent_s = wp.spatial_vector()
    a_parent_s = wp.spatial_vector()

    if parent >= 0:
        v_parent_s = body_v_s[parent]
        a_parent_s = body_a_s[parent]

    # body velocity, acceleration
    v_s = v_parent_s + v_j_s
    a_s = a_parent_s + spatial_cross(v_s, v_j_s)  # + joint_S_s[i]*self.joint_qdd[i]

    # compute body forces
    X_sm = body_q_com[child]
    I_m = body_I_m[child]

    # gravity and external forces (expressed in frame aligned with s but centered at body mass)
    m = I_m[0, 0]

    world_idx = body_world[child]
    world_g = gravity[wp.max(world_idx, 0)]
    f_g = m * world_g
    r_com = wp.transform_get_translation(X_sm)
    f_g_s = wp.spatial_vector(f_g, wp.cross(r_com, f_g))

    # body forces
    I_s = transform_spatial_inertia(X_sm, I_m)

    f_b_s = I_s * a_s + spatial_cross_dual(v_s, I_s * v_s)

    body_v_s[child] = v_s
    body_a_s[child] = a_s
    body_f_s[child] = f_b_s - f_g_s
    body_I_s[child] = I_s

# Inverse dynamics via Recursive Newton-Euler algorithm (Featherstone Table 5.1)
@wp.kernel
def eval_rigid_id(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_qd: wp.array(dtype=float),
    joint_axis: wp.array(dtype=wp.vec3),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    body_I_m: wp.array(dtype=wp.spatial_matrix),
    body_q: wp.array(dtype=wp.transform),
    body_q_com: wp.array(dtype=wp.transform),
    joint_X_p: wp.array(dtype=wp.transform),
    body_world: wp.array(dtype=wp.int32),
    gravity: wp.array(dtype=wp.vec3),
    # outputs
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_I_s: wp.array(dtype=wp.spatial_matrix),
    body_v_s: wp.array(dtype=wp.spatial_vector),
    body_f_s: wp.array(dtype=wp.spatial_vector),
    body_a_s: wp.array(dtype=wp.spatial_vector),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]

    # compute link velocities and coriolis forces
    for i in range(start, end):
        compute_link_velocity(
            i,
            joint_type,
            joint_parent,
            joint_child,
            joint_qd_start,
            joint_qd,
            joint_axis,
            joint_dof_dim,
            body_I_m,
            body_q,
            body_q_com,
            joint_X_p,
            body_world,
            gravity,
            joint_S_s,
            body_I_s,
            body_v_s,
            body_f_s,
            body_a_s,
        )

@wp.func
def joint_force(
    q: float,
    qd: float,
    joint_target_pos: float,
    joint_target_vel: float,
    target_ke: float,
    target_kd: float,
    limit_lower: float,
    limit_upper: float,
    limit_ke: float,
    limit_kd: float,
) -> float:
    """Joint force evaluation for a single degree of freedom."""

    limit_f = 0.0
    damping_f = 0.0
    target_f = 0.0

    target_f = target_ke * (joint_target_pos - q) + target_kd * (joint_target_vel - qd)

    # When limit violated: apply limit restoration forces and disable target control
    if q < limit_lower:
        limit_f = limit_ke * (limit_lower - q)
        damping_f = -limit_kd * qd
        target_f = 0.0
    elif q > limit_upper:
        limit_f = limit_ke * (limit_upper - q)
        damping_f = -limit_kd * qd
        target_f = 0.0

    return limit_f + damping_f + target_f

# computes joint space forces/torques in tau
@wp.func
def jcalc_tau(
    type: int,
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_f: wp.array(dtype=float),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    coord_start: int,
    dof_start: int,
    lin_axis_count: int,
    ang_axis_count: int,
    body_f_s: wp.spatial_vector,
    # outputs
    tau: wp.array(dtype=float),
):
    if type == JointType.BALL:
        # target_ke = joint_target_ke[dof_start]
        # target_kd = joint_target_kd[dof_start]

        for i in range(3):
            S_s = joint_S_s[dof_start + i]

            # w = joint_qd[dof_start + i]
            # r = joint_q[coord_start + i]

            tau[dof_start + i] = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]
            # tau -= w * target_kd - r * target_ke

        return

    if type == JointType.FREE or type == JointType.DISTANCE:
        for i in range(6):
            S_s = joint_S_s[dof_start + i]
            tau[dof_start + i] = -wp.dot(S_s, body_f_s) + joint_f[dof_start + i]

        return

    if type == JointType.PRISMATIC or type == JointType.REVOLUTE or type == JointType.D6:
        axis_count = lin_axis_count + ang_axis_count

        for i in range(axis_count):
            j = dof_start + i
            S_s = joint_S_s[j]

            q = joint_q[coord_start + i]
            qd = joint_qd[j]

            lower = joint_limit_lower[j]
            upper = joint_limit_upper[j]
            limit_ke = joint_limit_ke[j]
            limit_kd = joint_limit_kd[j]
            target_ke = joint_target_ke[j]
            target_kd = joint_target_kd[j]
            target_pos = joint_target_pos[j]
            target_vel = joint_target_vel[j]

            drive_f = joint_force(q, qd, target_pos, target_vel, target_ke, target_kd, lower, upper, limit_ke, limit_kd)

            # total torque / force on the joint
            t = -wp.dot(S_s, body_f_s) + drive_f + joint_f[j]

            tau[j] = t

        return

@wp.kernel
def eval_rigid_tau(
    articulation_start: wp.array(dtype=int),
    joint_type: wp.array(dtype=int),
    joint_parent: wp.array(dtype=int),
    joint_child: wp.array(dtype=int),
    joint_q_start: wp.array(dtype=int),
    joint_qd_start: wp.array(dtype=int),
    joint_dof_dim: wp.array(dtype=wp.int32, ndim=2),
    joint_target_pos: wp.array(dtype=float),
    joint_target_vel: wp.array(dtype=float),
    joint_q: wp.array(dtype=float),
    joint_qd: wp.array(dtype=float),
    joint_f: wp.array(dtype=float),
    joint_target_ke: wp.array(dtype=float),
    joint_target_kd: wp.array(dtype=float),
    joint_limit_lower: wp.array(dtype=float),
    joint_limit_upper: wp.array(dtype=float),
    joint_limit_ke: wp.array(dtype=float),
    joint_limit_kd: wp.array(dtype=float),
    joint_S_s: wp.array(dtype=wp.spatial_vector),
    body_fb_s: wp.array(dtype=wp.spatial_vector),
    body_f_ext: wp.array(dtype=wp.spatial_vector),
    # outputs
    body_ft_s: wp.array(dtype=wp.spatial_vector),
    tau: wp.array(dtype=float),
):
    # one thread per-articulation
    index = wp.tid()

    start = articulation_start[index]
    end = articulation_start[index + 1]
    count = end - start

    # compute joint forces
    for offset in range(count):
        # for backwards traversal
        i = end - offset - 1

        type = joint_type[i]
        parent = joint_parent[i]
        child = joint_child[i]
        dof_start = joint_qd_start[i]
        coord_start = joint_q_start[i]
        lin_axis_count = joint_dof_dim[i, 0]
        ang_axis_count = joint_dof_dim[i, 1]

        # total forces on body
        f_b_s = body_fb_s[child]
        f_t_s = body_ft_s[child]
        f_ext = body_f_ext[child]
        f_s = f_b_s + f_t_s + f_ext

        # compute joint-space forces, writes out tau
        jcalc_tau(
            type,
            joint_target_ke,
            joint_target_kd,
            joint_limit_ke,
            joint_limit_kd,
            joint_S_s,
            joint_q,
            joint_qd,
            joint_f,
            joint_target_pos,
            joint_target_vel,
            joint_limit_lower,
            joint_limit_upper,
            coord_start,
            dof_start,
            lin_axis_count,
            ang_axis_count,
            f_s,
            tau,
        )

        # update parent forces, todo: check that this is valid for the backwards pass
        if parent >= 0:
            wp.atomic_add(body_ft_s, parent, f_s)

@wp.kernel
def compute_spatial_inertia(
    body_inertia: wp.array(dtype=wp.mat33),
    body_mass: wp.array(dtype=float),
    # outputs
    body_I_m: wp.array(dtype=wp.spatial_matrix),
):
    tid = wp.tid()
    I = body_inertia[tid]
    m = body_mass[tid]
    # fmt: off
    body_I_m[tid] = wp.spatial_matrix(
        m,   0.0, 0.0, 0.0,     0.0,     0.0,
        0.0, m,   0.0, 0.0,     0.0,     0.0,
        0.0, 0.0, m,   0.0,     0.0,     0.0,
        0.0, 0.0, 0.0, I[0, 0], I[0, 1], I[0, 2],
        0.0, 0.0, 0.0, I[1, 0], I[1, 1], I[1, 2],
        0.0, 0.0, 0.0, I[2, 0], I[2, 1], I[2, 2],
    )
    # fmt: on


@wp.kernel
def compute_com_transforms(
    body_com: wp.array(dtype=wp.vec3),
    # outputs
    body_X_com: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    com = body_com[tid]
    body_X_com[tid] = wp.transform(com, wp.quat_identity())