from copy import deepcopy
from typing import Iterable
import numpy as np
import roboticstoolbox as rtb
from spatialmath import SE3, UnitQuaternion

class FrankaEasyIK():
    def __init__(self):
        self.robot = rtb.models.Panda()  # Load Panda robot model
        self.last_q = None  # Store the last joint configuration

    def __call__(self, p: Iterable[float], q: Iterable[float] = [1., 0., 0., 0.], verbose=False) -> Iterable[float]:
        """Perform custom inverse kinematics

        Args:
            p (Float[3]): Cartesian Position
            q (Float[4]): Absolute Quaternion Orientation w.r.t. robot base
                - Quaternion notation: [x, y, z, w]
            verbose (bool): Print results

        Raises:
            Exception: When IK solution is not found

        Returns:
            Float[7]: 7 DoF robot joint configuration
        """
        assert len(p) == 3, f"Position length: {len(p)} != 3"
        assert len(q) == 4, f"Quaternion length: {len(q)} != 4"

        # Deep copy to avoid side effects
        p = deepcopy(p)
        q = deepcopy(q)

        # Convert quaternion from [x, y, z, w] to [w, x, y, z] (if needed by the library)
        q = [q[3], q[0], q[1], q[2]]

        # Compute desired SE3 transformation
        desired_transform = SE3.Trans(*p) * UnitQuaternion(np.array(q)).SE3()

        # Perform IK using Levenberg-Marquardt method
        ik_result = self.robot.ikine_LM(desired_transform, q0=self.last_q)

        # Extract attributes from IKSolution object
        q1 = ik_result.q  # Joint configuration
        succ = ik_result.success  # Success flag
        reason = ik_result.reason  # Reason for failure (if any)

        # Raise exception if IK solution was not successful
        if not succ:
            raise Exception(f"IK not found because: {reason}")

        # Update last joint configuration
        if verbose:
            print("last q before: ", self.last_q)
        self.last_q = q1
        if verbose:
            print("last q: ", self.last_q)

        return q1
