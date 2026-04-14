from enum import IntEnum, Enum

# Types of joints linking rigid bodies
class JointType(IntEnum):
    """
    Enumeration of joint types supported in Newton.
    """

    PRISMATIC = 0
    """Prismatic joint: allows translation along a single axis (1 DoF)."""

    REVOLUTE = 1
    """Revolute joint: allows rotation about a single axis (1 DoF)."""

    BALL = 2
    """Ball joint: allows rotation about all three axes (3 DoF, quaternion parameterization)."""

    FIXED = 3
    """Fixed joint: locks all relative motion (0 DoF)."""

    FREE = 4
    """Free joint: allows full 6-DoF motion (translation and rotation, 7 coordinates)."""

    DISTANCE = 5
    """Distance joint: keeps two bodies at a distance within its joint limits (6 DoF, 7 coordinates)."""

    D6 = 6
    """6-DoF joint: Generic joint with up to 3 translational and 3 rotational degrees of freedom."""

    CABLE = 7
    """Cable joint: one linear (stretch) and one angular (isotropic bend/twist) DoF."""

    def dof_count(self, num_axes: int) -> tuple[int, int]:
        """
        Returns the number of degrees of freedom (DoF) in velocity and the number of coordinates
        in position for this joint type.

        Args:
            num_axes (int): The number of axes for the joint.

        Returns:
            tuple[int, int]: A tuple (dof_count, coord_count) where:
                - dof_count: Number of velocity degrees of freedom for the joint.
                - coord_count: Number of position coordinates for the joint.

        Notes:
            - For PRISMATIC and REVOLUTE joints, both values are 1 (single axis).
            - For BALL joints, dof_count is 3 (angular velocity), coord_count is 4 (quaternion).
            - For FREE and DISTANCE joints, dof_count is 6 (3 translation + 3 rotation), coord_count is 7 (3 position + 4 quaternion).
            - For FIXED joints, both values are 0.
        """
        dof_count = num_axes
        coord_count = num_axes
        if self == JointType.BALL:
            dof_count = 3
            coord_count = 4
        elif self == JointType.FREE or self == JointType.DISTANCE:
            dof_count = 6
            coord_count = 7
        elif self == JointType.FIXED:
            dof_count = 0
            coord_count = 0
        return dof_count, coord_count

    def constraint_count(self, num_axes: int) -> int:
        """
        Returns the number of velocity-level bilateral kinematic constraints for this joint type.

        Args:
            num_axes (int): The number of DoF axes for the joint.

        Returns:
            int: The number of bilateral kinematic constraints for the joint.

        Notes:
            - For PRISMATIC and REVOLUTE joints, this equals 5 (single DoF axis).
            - For FREE and DISTANCE joints, `cts_count = 0` since it yields no constraints.
            - For FIXED joints, `cts_count = 6` since it fully constrains the associated bodies.
        """
        cts_count = 6 - num_axes
        if self == JointType.BALL:
            cts_count = 3
        elif self == JointType.FREE or self == JointType.DISTANCE:
            cts_count = 0
        elif self == JointType.FIXED:
            cts_count = 6
        return cts_count


class InputMode(Enum):
    """How the desired arm motion is specified when calling ``compute_root_wrench``.

    ACCEL — joint accelerations provided directly (most accurate).
    VEL   — target joint velocities; qdd ≈ (vel_target − vel_curr) / dt.
    POS   — target joint positions;  qdd ≈ 2·(pos_target − pos_curr − vel·dt) / dt²
             (derived from the constant-acceleration kinematic identity
             pos_target = pos + vel·dt + ½·qdd·dt²).
    """
    ACCEL = "qdd"
    VEL   = "vel"
    POS   = "pos"
