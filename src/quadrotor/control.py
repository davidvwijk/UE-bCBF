"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Uncertainty Estimators for Robust Backup Control Barrier Functions.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module containing control laws.

"""

import numpy as np


class Control:
    def setupControl(
        self,
    ) -> None:

        # Control limits
        self.F_max = 20  # [N]
        self.M_max = 20  # [Nm]
        self.u_bounds = (
            (0, self.F_max),
            (-self.M_max, self.M_max),
        )

        # Backup control gains
        self.Kp = 1
        self.Kd = 1.01

    def primaryControl(self, x_curr, t):
        """
        Primary controller producing desired control at each step.

        """
        u_des = np.zeros(2)
        return u_des

    def backupControl(self, x):
        """
        Backup controller, u_b = [thrust (N), moment (Nm)].

        """
        u_b = np.array([self.F_max, self.Kp * x[2] + self.Kd * x[5]])
        return u_b
