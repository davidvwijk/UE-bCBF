"""
--------------------------------------------------------------------------

van Wijk, David
California Institute of Technology

Codebase for Uncertainty Estimators for Robust Backup Control Barrier Functions.

© 2025 David van Wijk <vanwijk@caltech.edu>

---------------------------------------------------------------------------

Module contains plotting functions.

"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection
import numpy as np


class PaperPlotter:
    def plotter(
        x,
        u_act,
        intervening,
        u_p,
        d_full,
        dhat_full,
        env,
        xbCBF=None,
        xDRbCBF=None,
        phase_plot=True,
        quad_plot=True,
        double_plot=True,
        latex_plots=False,
        save_plots=False,
        show_plots=True,
    ):
        if latex_plots:
            plt.rcParams.update(
                {
                    "text.usetex": True,
                    "font.family": "serif",
                }
            )
            plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

        alpha_set = 0.4
        xaxis_sz, legend_sz, ticks_sz = 27, 23, 25
        lblsize = legend_sz * 1.35
        lwp_sets = 2
        x1 = x[0, :]
        x2 = x[1, :]

        def setupPhasePlot():
            # Plot limits
            x_max = 0.65
            x_min = -4.07
            y_max = 2.05
            y_min = -0.65
            x_c = np.linspace(x_min, 0, 1000)
            plt.figure(figsize=(12.5, 8.5), dpi=100)

            plt.fill_between(
                [0, x_max],
                y_min,
                y_max,
                color=[255 / 255, 204 / 255, 204 / 255],
                alpha=alpha_set,
            )
            plt.vlines(
                x=0.0,
                ymin=0,
                ymax=y_max,
                color=[255 / 255, 0 / 255, 0 / 255],
                linewidth=lwp_sets,
            )
            plt.hlines(
                y=0.0,
                xmin=x_min,
                xmax=0,
                color=[0 / 255, 176 / 255, 240 / 255],
                linewidth=lwp_sets,
                alpha=1,
            )

            plt.vlines(
                x=0.0,
                ymin=y_min,
                ymax=0,
                color=[0 / 255, 176 / 255, 240 / 255],
                alpha=1,
                linewidth=lwp_sets,
            )

            plt.fill_between(
                x_c,
                0,
                y_min,
                color=[193 / 255, 229 / 255, 245 / 255],
                alpha=alpha_set,
            )
            plt.fill_between(
                [x_min, 0],
                0,
                y_max,
                color=[217 / 255, 242 / 255, 208 / 255],
                alpha=alpha_set,
            )
            plt.text(-0.74, -0.45, "$\mathcal{C}_{\\rm B}$", fontsize=lblsize)
            plt.text(
                -0.6,
                1.62,
                "$\mathcal{C}_{\\rm S} \\backslash \mathcal{C}_{\\rm B}$",
                fontsize=lblsize,
            )
            plt.text(
                0.11,
                # 1.2,
                # 1.62,
                0.8,
                "$\mathcal{X} \\backslash \mathcal{C}_{\\rm S}$",
                fontsize=lblsize,
            )

            plt.axis("equal")
            plt.xlabel(r"$x_1$", fontsize=xaxis_sz, labelpad=0)
            plt.ylabel(r"$x_2$", fontsize=xaxis_sz, labelpad=0)
            plt.xlim([x_min, x_max])
            plt.ylim([y_min, y_max])
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            ax = plt.gca()
            ax.set_aspect("equal", adjustable="box")

        ##################################################################################
        ##################################################################################

        estimate_b_color = "#196b24"

        gw_edge_color = "#bababa"
        lwp = 2.4
        max_back_num = 17
        border_lw = 1.5

        # Legend stuff
        bp = 0.25  # Border pad
        frame_bool = True
        shadow_bool = False
        fancy_bool = True
        txtpd = 0.4

        lw_backup = 1.8

        if phase_plot:
            setupPhasePlot()
            ax = plt.gca()
            ax.plot(
                x1[0],
                x2[0],
                "k*",
                markersize=9,
                # label=r"$\boldsymbol{x}_0$",
                label=None,
                zorder=800,
            )

            if xbCBF is not None and xDRbCBF is not None:
                plt.plot(
                    xbCBF[0, :],
                    xbCBF[1, :],
                    "-",
                    # color="#D2042D",
                    # color=colors_quad[1],
                    color="#ff595e",
                    linewidth=lwp,
                    label="bCBF",
                )
                arrow_indices = np.arange(8, len(xbCBF[0, :]), 20)
                for i in arrow_indices:
                    ax.annotate(
                        "",
                        xy=(xbCBF[0, :][i + 1], xbCBF[1, :][i + 1]),
                        xytext=(xbCBF[0, :][i], xbCBF[1, :][i]),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="#ff595e",
                            lw=lwp * 1.1,
                        ),
                    )
                plt.plot(
                    xDRbCBF[0, :],
                    xDRbCBF[1, :],
                    "-",
                    # color="#FF69B4",
                    # color=colors_quad[0],
                    color="#ff924c",
                    linewidth=lwp,
                    label="DR-bCBF",
                    zorder=1,
                )
                arrow_indices = np.arange(8, len(xDRbCBF[0, :]), 20)
                for i in arrow_indices:
                    ax.annotate(
                        "",
                        xy=(xDRbCBF[0, :][i + 1], xDRbCBF[1, :][i + 1]),
                        xytext=(xDRbCBF[0, :][i], xDRbCBF[1, :][i]),
                        arrowprops=dict(
                            arrowstyle="->",
                            color="#ff924c",
                            lw=lwp * 1.1,
                        ),
                    )

            # label = r"$\boldsymbol{\phi}^{d}(t,\boldsymbol{x}_0)" + "$"
            label = "UE-bCBF"
            plt.plot(
                x1, x2, "-", color="#7030a0", linewidth=lwp, label=label, zorder=999
            )
            arrow_indices = np.arange(8, len(x1), 20)
            for i in arrow_indices:
                ax.annotate(
                    "",
                    xy=(x1[i + 1], x2[i + 1]),
                    xytext=(x1[i], x2[i]),
                    arrowprops=dict(
                        arrowstyle="->",
                        color="#7030a0",
                        lw=lwp * 1.1,
                    ),
                    zorder=800,
                )

            e_tstep = 3
            if env.backupTrajs:
                rta_points = len(env.backupTrajs[0])
                max_numBackup = max_back_num  # len(env.backupTrajs)
                for i, xy in enumerate(zip(env.backupTrajs, env.backupTrajs_d)):
                    if i == 0:
                        label = (
                            r"$\boldsymbol{\phi}_{\rm b}^{\hat{d}}(\tau,\boldsymbol{x})"
                            + "$"
                        )
                        label2 = (
                            r"$\boldsymbol{\phi}_{\rm b}^{d}(\tau,\boldsymbol{x})" + "$"
                        )
                    else:
                        label = None
                        label2 = None

                    if i < max_numBackup:
                        plt.plot(
                            xy[0][:, 0],
                            xy[0][:, 1],
                            color=estimate_b_color,
                            linewidth=lw_backup,
                            label=label,
                            zorder=2,
                            # marker=".",
                            # markersize=5,
                            # markevery=e_tstep,
                        )
                        plt.plot(
                            xy[1][:, 0],
                            xy[1][:, 1],
                            "--",
                            color="black",
                            linewidth=lw_backup,
                            label=label2,
                            zorder=1,
                            # marker=".",
                            # markersize=5,
                            # markevery=e_tstep,
                        )
                        circ = []
                        # if env.robust:
                        if env.delta_array:
                            for j in np.arange(0, rta_points, e_tstep):
                                r_t = env.delta_array[i][0, j]
                                cp = patches.Circle(
                                    (xy[0][j, 0], xy[0][j, 1]),
                                    r_t,
                                    color=gw_edge_color,
                                    fill=False,
                                    linestyle="--",
                                    label="$\delta_{\\rm max}(\\tau,t)$",
                                )
                                if i == 0 and j == 0:
                                    ax.add_patch(cp)
                                circ.append(cp)
                            coll = PatchCollection(
                                circ,
                                zorder=100,
                                facecolors=("none",),
                                edgecolors=(gw_edge_color,),
                                linewidths=(1,),
                                linestyle=("--",),
                            )
                            ax.add_collection(coll)

            for spine in ax.spines.values():
                spine.set_linewidth(border_lw)

            # ax.xaxis.set_tick_params(pad=0)
            # ax.yaxis.set_tick_params(pad=1)
            # ax.legend(fontsize=legend_sz, loc="upper center")
            ax.legend(
                fontsize=legend_sz,
                loc="upper center",
                # bbox_to_anchor=(0.5, 1.15),
                bbox_to_anchor=(0.47, 1.15),
                framealpha=1,
                # ncol=5,
                ncol=6,
                columnspacing=0.4,
                handletextpad=0.2,
                handlelength=1.5,
                fancybox=fancy_bool,
                frameon=frame_bool,
                borderpad=bp,
                shadow=shadow_bool,
            )

            if save_plots:
                plt.savefig(
                    "plots/comparison_phase_plot_omega=" + str(env.d_omega) + ".pdf",
                    # "plots/comparison_phase_plot_omega=" + str(env.d_omega) + ".png",
                    dpi=500,
                    bbox_inches="tight",
                )

        if show_plots:
            plt.show()
