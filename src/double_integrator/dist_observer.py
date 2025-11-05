"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Uncertainty Estimators for Robust Backup Control Barrier Functions.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module containing functions for disturbance observer.

"""

import numpy as np


class DOB:
    def setupDistOb(
        self,
    ) -> None:

        # Observer gain
        self.Lambda = 3 * np.eye(len(self.x0))
        self.normLambda = np.linalg.norm(self.Lambda, ord=2)

        # Initialization
        self.dhat0 = np.zeros(len(self.x0))
        self.xi0 = self.x0

        # Error bounds
        self.lam_min = min(np.linalg.eigvals(self.Lambda))

        self.delta_d = self.dw_max
        self.delta_v = self.dw_max * self.d_omega

    def e_bar(self, t):
        """
        Theoretical error bounds as a function of global time t.

        """
        return np.exp(-self.lam_min * t) * self.delta_d + (
            self.delta_v / self.lam_min
        ) * (1 - np.exp(-self.lam_min * t))

    def e_bar_dot(self, t):
        """
        Global time derivative of e_bar.

        """
        return (self.delta_v - self.lam_min * self.delta_d) * np.exp(-self.lam_min * t)

    def delta_max(self, tau, t):
        """
        Flow bound using Gronwall-Bellman Inequality.

        """
        return (self.delta_v / self.L_cl**2 + self.e_bar(t) / self.L_cl) * (
            np.exp(self.L_cl * (tau - t)) - 1
        ) - (self.delta_v / self.L_cl) * (tau - t)

    def delta_max_dot(self, tau, t):
        """
        Global time derivative of GW delta_max bound.

        """
        return (self.e_bar_dot(t) / self.L_cl) * (np.exp(self.L_cl * (tau - t)) - 1)
