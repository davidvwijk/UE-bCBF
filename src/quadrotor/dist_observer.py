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
        self.Lambda = 15 * np.eye(self.lenx)
        self.normLambda = np.linalg.norm(self.Lambda, ord=2)

        # Initialization
        self.dhat0 = np.zeros(self.lenx)
        self.xi0 = self.x0

        # Error bounds
        self.lam_min = min(np.linalg.eigvals(self.Lambda))

        self.delta_d = self.dw_max
        if self.tv_dist:
            self.delta_v = self.d_a1 * self.d_omega
        else:
            self.delta_v = 0.1  # Small, non-zero

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
            np.exp(self.L_cl * tau) - 1
        ) - (self.delta_v / self.L_cl) * tau

    def delta_max_dot(self, tau, t):
        """
        Global time derivative of Gronwall-Bellman delta_max bound.

        """
        return (self.e_bar_dot(t) / self.L_cl) * (np.exp(self.L_cl * (tau)) - 1)

    def delta_max_b(self, tau, t):
        """
        Flow bound using Gronwall-Bellman Inequality and one-sided Lipschitz constant.

        """
        b = self.muFcl
        return (self.delta_v / b**2 + self.e_bar(t) / b) * (np.exp(b * (tau)) - 1) - (
            self.delta_v / b
        ) * (tau)

    def delta_max_dot_b(self, tau, t):
        """
        Global time derivative of Gronwall-Bellman delta_max bound using one-sided Lipschitz constant.

        """
        b = self.muFcl
        return (self.e_bar_dot(t) / b) * (np.exp(b * (tau)) - 1)
