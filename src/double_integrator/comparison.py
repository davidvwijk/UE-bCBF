"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Uncertainty Estimators for Robust Backup Control Barrier Functions.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Calls Simulation class for the standard bCBF approach, the DR-bCBF approach and our new UE-bCBF approach, running each and comparing the three.

"""

from main_sim import Simulation
from plotting_paper_comparison import PaperPlotter
import os
from pathlib import Path

if __name__ == "__main__":

    show_plots = False
    save_plots = True

    if save_plots:
        base_dir = os.path.dirname(__file__)
        folder = Path(base_dir, "plots")
        folder.mkdir(parents=True, exist_ok=True)

    # Run UE-bCBF (ours)
    print("Running simulation with UE-bCBF")
    env1 = Simulation(
        safety_flag=True,
        verbose=False,
        dw_bool=True,
        ue_flag=True,
        drbcbf_flag=False,
    )
    (
        x_full,
        d_full,
        dhat_full,
        _,
        u_des_full,
        u_act_full,
        _,
        _,
        _,
    ) = env1.sim()

    # Run vanilla backup CBF
    print("Running simulation with bCBF")
    env2 = Simulation(
        safety_flag=True,
        verbose=False,
        dw_bool=True,
        ue_flag=False,
        drbcbf_flag=False,
    )
    (
        xbCBF,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = env2.sim()

    # Run DR-bCBF
    print("Running simulation with DR-bCBF")
    env3 = Simulation(
        safety_flag=True,
        verbose=False,
        dw_bool=True,
        ue_flag=False,
        drbcbf_flag=True,
    )
    (
        xDRbCBF,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = env3.sim()

    PaperPlotter.plotter(
        x_full,
        u_act_full,
        _,
        u_des_full,
        d_full,
        dhat_full,
        env1,
        xbCBF=xbCBF,
        xDRbCBF=xDRbCBF,
        phase_plot=True,
        quad_plot=True,
        double_plot=True,
        latex_plots=True,
        save_plots=save_plots,
        show_plots=show_plots,
    )
