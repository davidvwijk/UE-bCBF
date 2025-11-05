"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Uncertainty Estimators for Robust Backup Control Barrier Functions.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module contains functions required for propagating dynamics of double integrator.

"""

import numpy as np
from scipy.integrate import solve_ivp


class Dynamics:
    def setupDynamics(
        self,
    ) -> None:

        # Constants
        self.g = 9.81  # [m/s^2]
        self.m = 1  # [kg]
        self.J = 0.25  # [kgm^2]
        self.Jinv = 1 / self.J

        # Integration options
        self.int_options = {"rtol": 1e-6, "atol": 1e-6}

        # Simulation data
        self.del_t = 0.02  # [s]
        self.total_steps = int(3.25 / self.del_t) + 1
        self.curr_step = 0

        # Initial conditions [x, z, theta, xdot, zdot, thetadot]
        self.x0 = np.array([0, 4.25, -np.pi / 2, 0, 0, 0])
        self.lenx = len(self.x0)

        # Disturbances
        if self.tv_dist:
            self.d_a1 = 0.5
            self.d_2 = -1
            self.d_omega = 0.3
            self.dw_max = np.sqrt(self.d_a1**2 + self.d_2**2)  # [m/s^2]
        else:
            self.dw_max = 1  # [m/s^2]

        # Constant
        self.sup_fcl = 1

    def computeJacobianCL(self, x):
        """
        Compute Jacobian of closed-loop backup dynamics.
        """
        jac = np.zeros((self.lenx, self.lenx))
        jac[:3, 3:6] = np.eye(3)
        jac[3, 2] = (self.F_max / self.m) * np.cos(x[2])
        jac[4, 2] = -(self.F_max / self.m) * np.sin(x[2])
        jac[5, 2] = -(self.Kp / self.J)
        jac[5, 5] = -(self.Kd / self.J)
        return jac

    def f_x(self, x):
        """
        Function f(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        f = np.zeros_like(x)
        f[:3] = x[3:6]
        f[4] = -self.g
        return f

    def g_x(self, x):
        """
        Function g(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        g = np.zeros((self.lenx, 2))
        g[3:, :] = np.array(
            [[np.sin(x[2]) / self.m, 0], [np.cos(x[2]) / self.m, 0], [0, -self.Jinv]]
        )
        return g

    def propMainBackup_DOB(self, t, x, d_hat_t, args):
        """
        Propagation function for backup dynamics, STM and Theta using the most recent disturbance estimate.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = (
            self.f_x(x[:lenx])
            + self.g_x(x[:lenx]) @ self.backupControl(x[:lenx])
            + d_hat_t
        )

        # Construct F
        F = self.computeJacobianCL(x[:lenx])

        # Extract STM & reshape
        STM = x[lenx : lenx + lenx**2].reshape(lenx, lenx)
        dSTM = F @ STM
        dx[lenx : lenx + lenx**2] = dSTM.reshape(lenx**2)

        # Extract Theta & reshape
        Theta = x[lenx + lenx**2 : lenx + 2 * lenx**2].reshape(lenx, lenx)
        dTheta = F @ Theta + np.eye(lenx)
        dx[lenx + lenx**2 : lenx + 2 * lenx**2] = dTheta.reshape(lenx**2)

        return dx

    def propMainBackup_truth(self, t, x, args):
        """
        Propagation function for backup dynamics.
        Uses the true disturbance signal. ONLY TO BE USED FOR VALIDATION.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = (
            self.f_x(x[:lenx])
            + self.g_x(x[:lenx]) @ self.backupControl(x[:lenx])
            + self.disturbanceFun(t)
        )
        return dx

    def propMainBackup_vanilla(self, t, x, args):
        """
        Propagation function for backup dynamics and STM if applicable.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) @ self.backupControl(
            x[:lenx]
        )

        # Construct F
        F = self.computeJacobianCL(x[:lenx])

        # Extract STM & reshape
        STM = x[lenx : lenx + lenx**2].reshape(lenx, lenx)
        dSTM = F @ STM
        dx[lenx : lenx + lenx**2] = dSTM.reshape(lenx**2)

        return dx

    def integrateStateBackup_truth(self, x, tspan_b, options):
        """
        Propagate backup flow over the backup horizon. Evaluate at discrete points.
        Uses the true disturbance signal. ONLY TO BE USED FOR VALIDATION.

        """
        t_step = (tspan_b[0], tspan_b[-1])
        args = {}
        soltn = solve_ivp(
            lambda t, x: self.propMainBackup_truth(t, x, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
            t_eval=tspan_b,
        )
        x = soltn.y[:, :]
        return x

    def integrateStateBackup_DOB(self, x, tspan_b, d_hat_t, options):
        """
        Propagate backup flow over the backup horizon. Evaluate at discrete points.
        Use the current estimate of the disturbance to improve accuracy of flow estimate.

        """
        t_step = (0.0, tspan_b[-1])
        args = {}
        soltn = solve_ivp(
            lambda t, x: self.propMainBackup_DOB(t, x, d_hat_t, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
            t_eval=tspan_b,
        )
        x = soltn.y[:, :]
        return x

    def integrateStateBackup_vanilla(self, x, tspan_b, options):
        """
        Propagate backup flow over the backup horizon. Evaluate at discrete points. Used for vanilla bCBF.

        """
        t_step = (0.0, tspan_b[-1])
        args = {}
        soltn = solve_ivp(
            lambda t, x: self.propMainBackup_vanilla(t, x, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
            t_eval=tspan_b,
        )
        x = soltn.y[:, :]
        return x

    ############# Disturbance observer #############

    def propStateDOB(self, t, x, u):
        """
        Propagation function for dynamics with disturbance and observer auxiliary state.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) @ u + self.disturbanceFun(t)
        d_hat = self.Lambda @ (x[:lenx] - x[lenx:])
        dx[lenx:] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) @ u + d_hat

        return dx

    def integrateStateDOB(self, x, u, t_curr, t_step, options):
        """
        Propagate state and observer auxiliary state.

        """
        t_int = (t_curr, t_curr + t_step)
        soltn = solve_ivp(
            lambda t, x: self.propStateDOB(t, x, u),
            t_int,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
        )
        x = soltn.y[:, -1]
        return x

    def disturbanceFun(self, t):
        """
        Process disturbance function, norm bounded by delta_d.

        """
        dist_t = np.zeros(self.lenx)
        if self.tv_dist:
            dist_t[3:5] = np.array(
                [-self.d_2, self.d_a1 * np.sin(self.d_omega * t - np.pi / 3)]
            )
        else:
            dist_t[3:5] = np.array([1, -1])
            dist_t = (dist_t / (np.linalg.norm(dist_t))) * self.dw_max
        return dist_t
