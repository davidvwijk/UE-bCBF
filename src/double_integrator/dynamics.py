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

        # A and B matrices constant
        self.A = np.array([[0, 1], [0, 0]])
        self.B = np.array([0, 1])

        # Integration options
        self.int_options = {"rtol": 1e-6, "atol": 1e-6}

        # Simulation data
        self.del_t = 0.02  # [sec]
        self.total_steps = int(6 / self.del_t) + 2
        self.curr_step = 0

        # Initial conditions
        self.x0 = np.array([-4, 1.2])
        self.lenx = len(self.x0)

        # Disturbances
        self.dw_max = 0.08
        self.d_omega = 0.2

        # Constant
        self.sup_fcl = 1

    def propMain(self, t, x, u, dist, args):
        """
        Propagation function for dynamics with disturbance and STM if applicable.
        Could be optimized (linear system).

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) * u + dist
        if len(x) > lenx:
            # Construct F
            F = self.computeJacobianSTM(x[:lenx])

            # Extract STM & reshape
            STM = x[lenx:].reshape(lenx, lenx)
            dSTM = F @ STM

            # Reshape back to column
            dSTM = dSTM.reshape(lenx**2)
            dx[lenx:] = dSTM

        return dx

    def computeJacobianSTM(self, x):
        """
        Compute Jacobian of dynamics.
        """
        jac = self.A
        return jac

    def f_x(self, x):
        """
        Function f(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        f = self.A @ x
        return f

    def g_x(self, x):
        """
        Function g(x) for control affine dynamics, x_dot = f(x) + g(x)u.

        """
        g = self.B
        return g

    def integrateState(self, x, u, t_step, dist, options):
        """
        State integrator using propagation function.

        """
        t_step = (0.0, t_step)
        args = {}
        soltn = solve_ivp(
            lambda t, x: self.propMain(t, x, u, dist, args),
            t_step,
            x,
            method="RK45",
            rtol=options["rtol"],
            atol=options["atol"],
        )
        x = soltn.y[:, -1]
        return x

    def propMainBackup(self, t, x, args):
        """
        Propagation function for backup dynamics and STM if applicable.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.A @ x[:lenx] + self.B * self.backupControl(x[:lenx])

        # Construct F
        F = self.A

        # Extract STM & reshape
        STM = x[lenx:].reshape(lenx, lenx)
        dSTM = F @ STM

        # Reshape back to column
        dSTM = dSTM.reshape(lenx**2)
        dx[lenx:] = dSTM

        return dx

    def propMainBackup_DOB(self, t, x, d_hat_t, args):
        """
        Propagation function for backup dynamics, STM and Theta.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.A @ x[:lenx] + self.B * self.backupControl(x[:lenx]) + d_hat_t

        # Construct F
        F = self.A

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
        Propagation function for backup dynamics and STM if applicable.
        Uses the true disturbance signal. ONLY TO BE USED FOR VALIDATION.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = (
            self.A @ x[:lenx]
            + self.B * self.backupControl(x[:lenx])
            + self.disturbanceFun(t)
        )

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

    def integrateStateBackup(self, x, tspan_b, options):
        """
        Propagate backup flow over the backup horizon. Evaluate at discrete points.

        """
        t_step = (0.0, tspan_b[-1])
        args = {}
        soltn = solve_ivp(
            lambda t, x: self.propMainBackup(t, x, args),
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

    ############# Disturbance observer #############

    def propStateDOB(self, t, x, u):
        """
        Propagation function for dynamics with disturbance and observer auxiliary state.

        """
        lenx = len(self.x0)
        dx = np.zeros_like(x)
        dx[:lenx] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) * u + self.disturbanceFun(t)
        d_hat = self.Lambda @ (x[:lenx] - x[lenx:])
        dx[lenx:] = self.f_x(x[:lenx]) + self.g_x(x[:lenx]) * u + d_hat

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
        Process disturbance function, norm bounded by dw_max.

        """
        dist_t = np.array(
            [np.sin(t * self.d_omega + np.pi / 4), np.cos(t * self.d_omega + np.pi / 4)]
        )
        return self.dw_max * dist_t
