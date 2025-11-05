"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Uncertainty Estimators for Robust Backup Control Barrier Functions.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module contains functions required for safety-critical control using control barrier functions.

"""

import numpy as np
import time
import quadprog


class Constraint:
    def alpha(self, x):
        """
        Strengthening function.

        """
        return 10 * x + x**3

    def alpha_b(self, x):
        """
        Strengthening function for reachability constraint.

        """
        return 10 * x

    def h1_x(self, x):
        """
        Safety constraint.

        """
        h = x[1] - self.z_min
        return h

    def grad_h1(self, x):
        """
        Gradient of safety constraint.

        """
        g = np.array([0, 1, 0, 0, 0, 0])
        return g

    def hb1_x(self, x):
        """
        Reachability constraint. Enforces zdot >= 0.

        """
        hb = x[4]
        return hb

    def grad_hb1(self, x):
        """
        Gradient of reachability constraint.

        """
        gb = np.array([0, 0, 0, 0, 1, 0])
        return gb

    def hb2_x(self, x):
        """
        Reachability constraint. Enforces |theta| <= theta_max

        """
        hb = self.theta_max**2 - x[2] ** 2
        return hb

    def grad_hb2(self, x):
        """
        Gradient of reachability constraint.

        """
        gb = np.array([0, 0, -2 * x[2], 0, 0, 0])
        return gb

    def hb3_x(self, x):
        """
        Reachability constraint. Enforces |thetaDot| <= thetaDot_max

        """
        hb = self.thetaDot_max**2 - x[5] ** 2
        return hb

    def grad_hb3(self, x):
        """
        Gradient of reachability constraint.

        """
        gb = np.array([0, 0, 0, 0, 0, -2 * x[5]])
        return gb

    def hb_softmin(self, x, hbfuns, kappa):
        """
        Soft-min constraint.
        x: [6x1] current state
        handles: array of function handles for constraints

        """
        exp_arg = 0
        for i in range(len(hbfuns)):
            hb_fun = hbfuns[i]
            exp_arg += np.exp(-kappa * hb_fun(x))

        return -np.log(exp_arg) / kappa

    def grad_hb_softmin(self, x, hbfuns, gradfuns, kappa):
        """
        Gradient of soft min for multiple constraints.

        """
        w_grad = np.zeros(self.lenx)
        w_hb = 0
        for i in range(len(hbfuns)):
            gradhb_fun = gradfuns[i]
            hb_fun = hbfuns[i]
            w_grad += np.exp(-kappa * hb_fun(x)) * gradhb_fun(x)
            w_hb += np.exp(-kappa * hb_fun(x))

        return w_grad / w_hb


class ASIF(Constraint):
    def setupASIF(
        self,
    ) -> None:

        # Backup properties
        self.backupTime = 0.4  # [sec] (total backup time)
        self.tspan_b = np.arange(0, self.backupTime + self.del_t, self.del_t)
        self.backupTrajs = []
        self.backupTrajs_d = []
        self.backup_save_N = 5  # saves every N backup trajectory (for plotting)
        self.delta_array = []

        # Safety constant
        self.z_min = 1  # [m]
        self.theta_max = np.deg2rad(55)  # [rad]
        self.thetaDot_max = 3  # [rad/sec]

        # Functions used in soft-min for hb(x) defining C_B
        self.kappa = 5
        self.hb_funs = [
            lambda x: self.hb1_x(x),
            lambda x: self.hb2_x(x),
            lambda x: self.hb3_x(x),
        ]
        self.gradhb_funs = [
            lambda x: self.grad_hb1(x),
            lambda x: self.grad_hb2(x),
            lambda x: self.grad_hb3(x),
        ]

        # Lipschitz constants
        self.Lh_const = 1
        self.Lhb_const = np.max([1, 2 * self.theta_max, 2 * self.thetaDot_max])
        self.L_cl = self.compute_Lcl()

        # Log norm of closed-loop Jacobian
        self.muFcl = self.compute_muFcl()

    def UEbCBF(self, x, u_des, t_curr):
        """
        Disturbance observer backup CBF based QP safety filter. Solves for safe control solution.
        x = [state; xi]

        """

        # QP objective function
        M = np.eye(2)
        q = u_des

        # QP actuation constraints
        G = np.vstack((np.eye(2), -np.eye(2)))
        h = np.zeros(4)
        h[1] = -self.M_max
        h[2] = -self.F_max
        h[3] = -self.M_max

        # Backup trajectory points
        rtapoints = len(self.tspan_b)

        # State tracking array
        lenx = len(self.x0)
        phi = np.zeros((rtapoints, lenx))
        phi[0, :] = x[:lenx]  # Extract current state

        # Estimated disturbance at time t
        d_hat_t = self.Lambda @ (x[:lenx] - x[lenx:])

        # Upper bound on the error at time t
        e_bar_t = self.e_bar(t_curr)

        # State transition matrix tracking array (dphi/dx)
        STM = np.zeros((lenx, lenx, rtapoints))
        STM[:, :, 0] = np.eye(lenx)

        # Theta matrix tracking array (dphi/ddhat)
        Theta = np.zeros((lenx, lenx, rtapoints))
        Theta[:, :, 0] = np.zeros(lenx)

        # Simulate flow under backup control law
        new_x = np.concatenate(
            (phi[0, :], STM[:, :, 0].flatten(), Theta[:, :, 0].flatten())
        )
        backupFlow = self.integrateStateBackup_DOB(
            new_x,
            self.tspan_b,
            d_hat_t,
            self.int_options,
        )

        # Extract phi, STM and Theta
        phi[:, :] = backupFlow[:lenx, :].T
        STM[:, :, :] = backupFlow[lenx : lenx + lenx**2, :].reshape(
            lenx, lenx, rtapoints
        )
        Theta[:, :, :] = backupFlow[lenx + lenx**2 : lenx + 2 * lenx**2, :].reshape(
            lenx, lenx, rtapoints
        )

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        fx_0 = self.f_x(phi[0, :])
        gx_0 = self.g_x(phi[0, :])

        delta_max_ar = np.zeros((1, rtapoints))

        # Construct barrier constraint for each point along trajectory
        for i in range(
            1, rtapoints
        ):  # Skip first point because of relative degree issue

            h_phi = self.h1_x(phi[i, :])
            gradh_phi = self.grad_h1(phi[i, :])
            g_temp_i = gradh_phi.T @ STM[:, :, i] @ gx_0

            tau = self.del_t * i

            # UE-bCBF bound
            delta_max = self.delta_max_b(tau, t_curr)
            delta_max_ar[:, i] = delta_max

            # Tightening epsilon
            epsilon = self.Lh_const * delta_max

            # Discretization tightening constant
            epsilon_d = (self.del_t / 2) * self.Lh_const * (self.sup_fcl + self.delta_d)

            robust_term = (
                np.linalg.norm(
                    gradh_phi @ (STM[:, :, i] + Theta[:, :, i] @ self.Lambda)
                )
                * e_bar_t
            )

            # Derivative of delta_max wrt. time t
            ddel_dt = self.delta_max_dot_b(tau, t_curr)

            h_temp_i = (
                -(
                    gradh_phi @ STM[:, :, i] @ (fx_0 + d_hat_t)
                    + self.alpha(h_phi - epsilon - epsilon_d)
                )
                + robust_term
                + self.Lh_const * ddel_dt
            )

            # Append constraint
            if i == 1:
                g_temp = g_temp_i
                h_temp = h_temp_i
            else:
                g_temp = np.vstack([g_temp, g_temp_i])
                h_temp = np.vstack([h_temp, h_temp_i])

            # Terminal condition
            if i == rtapoints - 1:
                hb_phi = self.hb_softmin(phi[i, :], self.hb_funs, self.kappa)
                gradhb_phi = self.grad_hb_softmin(
                    phi[i, :], self.hb_funs, self.gradhb_funs, self.kappa
                )

                # Tightening epsilon
                epsilonT = self.Lhb_const * delta_max

                # Robustness term
                robust_term_b = (
                    np.linalg.norm(
                        gradhb_phi @ (STM[:, :, i] + Theta[:, :, i] @ self.Lambda)
                    )
                    * e_bar_t
                )

                h_temp_i = (
                    -(
                        gradhb_phi @ STM[:, :, i] @ (fx_0 + d_hat_t)
                        + self.alpha_b(hb_phi - epsilonT)
                    )
                    + robust_term_b
                    + self.Lhb_const * ddel_dt
                )
                g_temp_i = gradhb_phi.T @ STM[:, :, i] @ gx_0

                # Append constraint
                g_temp = np.vstack([g_temp, g_temp_i])
                h_temp = np.vstack([h_temp, h_temp_i])

        # Store backup deltas for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.delta_array.append(delta_max_ar)

        # Append constraints
        G = np.vstack([G, g_temp])
        h = np.vstack([h.reshape((-1, 1)), h_temp])

        # Solve QP
        d = h.reshape((len(h),))
        try:
            tic = time.perf_counter()
            sltn = quadprog.solve_qp(M, q, G.T, d, 0)
            toc = time.perf_counter()
            u_act = sltn[0]
            active_constraint = sltn[5]
            solver_dt = toc - tic
            u_act = u_act
        except:
            u_act = self.backupControl(x[:lenx])
            solver_dt = None
            if self.verbose:
                print("no soltn")

        # If safe action is different the desired action, RTA is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, solver_dt

    def compute_muFcl(self):
        """
        Compute 2 log norm of closed-loop Jacobian.

        """

        # Closed-loop Jacobian only depends on theta
        N_samples = 1000
        theta = np.linspace(-np.pi / 2, np.pi / 2, N_samples)
        mu = -np.inf

        for i in range(N_samples):
            Fcl = self.computeJacobianCL(np.array([0, 0, theta[i], 0, 0, 0]))
            max_eig = max(np.linalg.eigvals((Fcl + Fcl.T) / 2))
            if max_eig > mu:
                mu = max_eig

        return mu

    def compute_Lcl(self):
        """
        Compute closed-loop Lipschitz constant.

        """

        # Closed-loop Jacobian only depends on theta
        N_samples = 1000
        theta = np.linspace(-np.pi / 2, np.pi / 2, N_samples)
        L_cl = -np.inf

        for i in range(N_samples):
            Fcl = self.computeJacobianCL(np.array([0, 0, theta[i], 0, 0, 0]))
            norm_Fcl = np.linalg.norm(Fcl, ord=2)
            if norm_Fcl > L_cl:
                L_cl = norm_Fcl

        return L_cl
