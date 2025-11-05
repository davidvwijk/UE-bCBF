"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Uncertainty Estimators for Robust Backup Control Barrier Functions.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module runs full simulation for double integrator example.

"""

import os
from pathlib import Path
import numpy as np
from safety import ASIF
from control import Control
from dynamics import Dynamics
from dist_observer import DOB
from plotting_paper import PaperPlotter


class Simulation(ASIF, Control, Dynamics, DOB):
    def __init__(
        self,
        safety_flag=True,
        verbose=True,
        tv_dist=False,
        ue_flag=True,
    ) -> None:

        self.verbose = verbose
        self.safety_flag = safety_flag
        self.tv_dist = tv_dist
        self.ue_flag = ue_flag

        self.setupDynamics()
        self.setupControl()
        self.setupASIF()
        self.setupDistOb()
        self.checkConstants()

    def checkViolation(self, x_curr):
        """
        Check for safety violation.

        """
        h_funs = [lambda x: self.h1_x(x)]
        for i in range(len(h_funs)):
            h_x = h_funs[i]
            if h_x(x_curr) < 0:
                if self.verbose:
                    print(f"Safety violation, constraint {i+1}")

    def checkConstants(self):
        """
        Ensure backup controller gains and limits satisfy robust forward invariance criteria.

        """
        cond1 = self.Kd**2 > 4 * self.J * self.Kp  # Overdamped
        cond2 = self.Kp * self.theta_max + self.Kd * self.thetaDot_max <= self.M_max
        cond3 = self.F_max >= (self.m * (self.g + self.delta_d)) / np.cos(
            self.theta_max
        )
        if not (cond1 and cond2 and cond3):
            raise Exception("Please choose appropriate backup control gains.")

    def sim(self):
        """
        Simulates trajectory for pre-specified number of timesteps and performs
        point-wise safety-critical control modifications if applicable.

        """
        x0 = self.x0
        total_steps = self.total_steps
        lenx = self.lenx

        # Tracking variables
        x_full = np.zeros((lenx, total_steps))
        d_full = np.zeros((lenx, total_steps))
        dhat_full = np.zeros((lenx, total_steps))
        u_act_full = np.zeros((len(self.u_bounds), total_steps))
        u_des_full = np.zeros((len(self.u_bounds), total_steps))
        solver_times, avg_solver_t, max_solver_t, intervened = [], [], [], []
        x_full[:, 0] = x0
        d_full[:, 0] = self.disturbanceFun(0)

        x_curr = np.hstack([x0, self.xi0])

        # Main loop
        for i in range(1, total_steps):
            t = self.curr_step * self.del_t

            # Generate desired control
            u_des = self.primaryControl(x_curr[:lenx], t)
            u_des_full[:, i] = u_des

            # If safety check on, monitor control using active set invariance filter
            if self.safety_flag:
                if self.ue_flag:
                    u, boolean, sdt = self.UEbCBF(x_curr, u_des, t)
                    solver_times.append(sdt)
                if boolean:
                    intervened.append(i)

                # Store backup trajectory under true disturbance for plotting
                if self.curr_step % self.backup_save_N == 0:
                    backupFlow_d = self.integrateStateBackup_truth(
                        x_curr,
                        t + self.tspan_b,
                        self.int_options,
                    )
                    self.backupTrajs_d.append(backupFlow_d[:lenx, :].T)
            else:
                u = u_des

            u_act_full[:, i] = u

            # Propagate states and auxiliary states
            x_curr = self.integrateStateDOB(
                x_curr,
                u,
                t,
                self.del_t,
                self.int_options,
            )
            x_full[:, i] = x_curr[:lenx]
            d_full[:, i] = self.disturbanceFun(t + self.del_t)
            dhat_full[:, i] = self.Lambda @ (x_curr[:lenx] - x_curr[lenx:])

            self.curr_step += 1
            if self.h1_x(x_curr[:lenx]) < 0 and self.verbose:
                print("Crashed")

        if self.verbose and self.safety_flag:
            solver_times = [i for i in solver_times if i is not None]
            if solver_times:
                avg_solver_t = 1000 * np.average(solver_times)  # in ms
                max_solver_t = 1000 * np.max(solver_times)  # in ms
                print(f"Average solver time: {avg_solver_t:0.4f} ms")
                print(f"Maximum single solver time: {max_solver_t:0.4f} ms")

        return (
            x_full,
            d_full,
            dhat_full,
            total_steps,
            u_des_full,
            u_act_full,
            intervened,
            avg_solver_t,
            max_solver_t,
        )


if __name__ == "__main__":

    show_plots = False
    save_plots = True

    if save_plots:
        base_dir = os.path.dirname(__file__)
        folder = Path(base_dir, "plots")
        folder.mkdir(parents=True, exist_ok=True)

    env = Simulation(
        safety_flag=True,
        verbose=True,
        tv_dist=True,
        ue_flag=True,
    )
    print(
        "Running simulation with parameters:",
        "Safety:",
        env.safety_flag,
        "| Time-varying process dist:",
        env.tv_dist,
        "| Dist observer:",
        env.ue_flag,
    )

    (
        x_full,
        d_full,
        dhat_full,
        total_steps,
        u_des_full,
        u_act_full,
        intervened,
        avg_solver_t,
        max_solver_t,
    ) = env.sim()

    PaperPlotter.plotter(
        x_full,
        u_act_full,
        intervened,
        u_des_full,
        d_full,
        dhat_full,
        env,
        save_plots=save_plots,
        show_plots=show_plots,
    )
