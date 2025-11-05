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
        h = -x[0]
        return h

    def grad_h1(self, x):
        """
        Gradient of safety constraint.

        """
        g = np.array([-1, 0])
        return g

    def hb_x(self, x):
        """
        Reachability constraint.

        """
        hb = -x[1]
        return hb

    def grad_hb(self, x):
        """
        Gradient of reachability constraint.

        """
        gb = np.array([0, -1])
        return gb


class ASIF(Constraint):
    def setupASIF(
        self,
    ) -> None:

        # Backup properties
        self.backupTime = 2  # [sec] (total backup time)
        self.tspan_b = np.arange(0, self.backupTime + self.del_t, self.del_t)
        self.backupTrajs = []
        self.backupTrajs_d = []
        self.backup_save_N = 5  # saves every N backup trajectory (for plotting)
        self.delta_array = []

        # Lipschitz constants
        self.Lh_const = 1
        self.Lhb_const = 1
        self.L_cl = 1

    def UEbCBF(self, x, u_des, t_curr):
        """
        Disturbance observer backup CBF based QP safety filter. Solves for safe control solution.
        x = [state; xi]

        """

        # QP objective function
        M = np.eye(2)
        q = np.array(
            [u_des, 0.0]
        )  # Need to append the control with 0 to get at least 2 dimensions

        # Control constraints
        G = [[1.0, 0.0], [-1.0, 0.0]]
        h = [-self.u_max, -self.u_max]

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

            tau = self.del_t * i + t_curr

            # UE-bCBF bound
            delta_max = self.delta_max(tau, t_curr)
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
            ddel_dt = self.delta_max_dot(tau, t_curr)

            h_temp_i = (
                -(
                    gradh_phi @ STM[:, :, i] @ (fx_0 + d_hat_t)
                    + self.alpha(h_phi - epsilon - epsilon_d)
                )
                + robust_term
                + self.Lh_const * ddel_dt
            )

            # Append constraint
            G.append([g_temp_i, 0])
            h.append(h_temp_i)

            # Terminal condition
            if i == rtapoints - 1:

                hb_phi = self.hb_x(phi[i, :])
                gradhb_phi = self.grad_hb(phi[i, :])

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
                G.append([g_temp_i, 0])
                h.append(h_temp_i)

        # Store backup deltas for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.delta_array.append(delta_max_ar)

        # Solve QP
        try:
            tic = time.perf_counter()
            sltn = quadprog.solve_qp(M, q, np.array(G).T, np.array(h), 0)
            u_act = sltn[0]
            active_constraint = sltn[5]
            toc = time.perf_counter()
            solver_dt = toc - tic
            u_act = u_act[0]  # Only extract scalar we need
        except:
            u_act = -1
            solver_dt = None
            if self.verbose:
                print("no soltn")

        # If safe action is different the desired action, RTA is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, solver_dt

    def DRbCBF(self, x, u_des):
        """
        Disturbance robust backup CBF based QP safety filter. Solves for safe control solution.

        """

        # QP objective function
        M = np.eye(2)
        q = np.array(
            [u_des, 0.0]
        )  # Need to append the control with 0 to get at least 2 dimensions

        # Control constraints
        G = [[1.0, 0.0], [-1.0, 0.0]]
        h = [-self.u_max, -self.u_max]

        # Backup trajectory points
        rtapoints = len(self.tspan_b)

        # State tracking array
        lenx = len(self.x0)
        phi = np.zeros((rtapoints, lenx))
        phi[0, :] = x[:lenx]  # Extract state

        # State transition matrix tracking array (dphi/dx)
        STM = np.zeros((lenx, lenx, rtapoints))
        STM[:, :, 0] = np.eye(lenx)

        # Simulate flow under backup control law
        new_x = np.concatenate((phi[0, :], STM[:, :, 0].flatten()))
        backupFlow = self.integrateStateBackup(
            new_x,
            self.tspan_b,
            self.int_options,
        )

        # Extract phi, STM and Theta
        phi[:, :] = backupFlow[:lenx, :].T
        STM[:, :, :] = backupFlow[lenx : lenx + lenx**2, :].reshape(
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

            # DR-bCBF Gronwall bound
            t = self.del_t * i
            delta_max = (self.delta_d / self.L_cl) * (np.exp(self.L_cl * t) - 1)
            delta_max_ar[:, i] = delta_max

            # Tightening epsilon
            epsilon = self.Lh_const * delta_max

            # Discretization tightening constant
            epsilon_d = (self.del_t / 2) * self.Lh_const * (self.sup_fcl + self.delta_d)

            robust_term = np.linalg.norm(gradh_phi @ STM[:, :, i]) * self.delta_d

            h_temp_i = (
                -(
                    gradh_phi @ STM[:, :, i] @ fx_0
                    + self.alpha(h_phi - epsilon - epsilon_d)
                )
                + robust_term
            )

            # Append constraint
            G.append([g_temp_i, 0])
            h.append(h_temp_i)

            # Terminal condition
            if i == rtapoints - 1:

                hb_phi = self.hb_x(phi[i, :])
                gradhb_phi = self.grad_hb(phi[i, :])

                # Tightening epsilon
                epsilonT = self.Lhb_const * delta_max

                # Robustness term
                robust_term_b = np.linalg.norm(gradhb_phi @ STM[:, :, i]) * self.delta_d

                h_temp_i = (
                    -(
                        gradhb_phi @ STM[:, :, i] @ fx_0
                        + self.alpha_b(hb_phi - epsilonT)
                    )
                    + robust_term_b
                )
                g_temp_i = gradhb_phi.T @ STM[:, :, i] @ gx_0

                # Append constraint
                G.append([g_temp_i, 0])
                h.append(h_temp_i)

        # Store backup deltas for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.delta_array.append(delta_max_ar)

        # Solve QP
        try:
            tic = time.perf_counter()
            sltn = quadprog.solve_qp(M, q, np.array(G).T, np.array(h), 0)
            u_act = sltn[0]
            active_constraint = sltn[5]
            toc = time.perf_counter()
            solver_dt = toc - tic
            u_act = u_act[0]  # Only extract scalar we need
        except:
            u_act = -1
            solver_dt = None
            if self.verbose:
                print("no soltn")

        # If safe action is different the desired action, RTA is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, solver_dt

    def vanillabCBF(self, x, u_des):
        """
        Standard backup CBF based QP safety filter. Solves for safe control solution.

        """

        # QP objective function
        M = np.eye(2)
        q = np.array(
            [u_des, 0.0]
        )  # Need to append the control with 0 to get at least 2 dimensions

        # Control constraints
        G = [[1.0, 0.0], [-1.0, 0.0]]
        h = [-self.u_max, -self.u_max]

        # Backup trajectory points
        rtapoints = len(self.tspan_b)

        # State tracking array
        lenx = len(self.x0)
        phi = np.zeros((rtapoints, lenx))
        phi[0, :] = x[:lenx]  # Extract state

        # State transition matrix tracking array (dphi/dx)
        STM = np.zeros((lenx, lenx, rtapoints))
        STM[:, :, 0] = np.eye(lenx)

        # Simulate flow under backup control law
        new_x = np.concatenate((phi[0, :], STM[:, :, 0].flatten()))
        backupFlow = self.integrateStateBackup(
            new_x,
            self.tspan_b,
            self.int_options,
        )

        # Extract phi, STM and Theta
        phi[:, :] = backupFlow[:lenx, :].T
        STM[:, :, :] = backupFlow[lenx : lenx + lenx**2, :].reshape(
            lenx, lenx, rtapoints
        )

        # Store backup trajectories for plotting
        if self.curr_step % self.backup_save_N == 0:
            self.backupTrajs.append(phi)

        fx_0 = self.f_x(phi[0, :])
        gx_0 = self.g_x(phi[0, :])

        # Construct barrier constraint for each point along trajectory
        for i in range(
            1, rtapoints
        ):  # Skip first point because of relative degree issue

            h_phi = self.h1_x(phi[i, :])
            gradh_phi = self.grad_h1(phi[i, :])
            g_temp_i = gradh_phi.T @ STM[:, :, i] @ gx_0

            # Discretization tightening constant
            epsilon_d = (self.del_t / 2) * self.Lh_const * (self.sup_fcl)

            h_temp_i = -(
                gradh_phi @ STM[:, :, i] @ fx_0 + self.alpha(h_phi - epsilon_d)
            )

            # Append constraint
            G.append([g_temp_i, 0])
            h.append(h_temp_i)

            # Terminal condition
            if i == rtapoints - 1:

                hb_phi = self.hb_x(phi[i, :])
                gradhb_phi = self.grad_hb(phi[i, :])

                h_temp_i = -(gradhb_phi @ STM[:, :, i] @ fx_0 + self.alpha_b(hb_phi))
                g_temp_i = gradhb_phi.T @ STM[:, :, i] @ gx_0

                # Append constraint
                G.append([g_temp_i, 0])
                h.append(h_temp_i)

        # Solve QP
        try:
            tic = time.perf_counter()
            sltn = quadprog.solve_qp(M, q, np.array(G).T, np.array(h), 0)
            u_act = sltn[0]
            active_constraint = sltn[5]
            toc = time.perf_counter()
            solver_dt = toc - tic
            u_act = u_act[0]  # Only extract scalar we need
        except:
            u_act = -1
            solver_dt = None
            if self.verbose:
                print("no soltn")

        # If safe action is different the desired action, RTA is intervening
        if np.linalg.norm(u_act - u_des) >= 0.0001:
            intervening = True
        else:
            intervening = False

        return u_act, intervening, solver_dt
