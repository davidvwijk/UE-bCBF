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
import numpy as np
from matplotlib.ticker import FuncFormatter


class PaperPlotter:
    def plotter(
        x,
        u_act,
        intervening,
        u_p,
        d_full,
        dhat_full,
        env,
        latex_plots=True,
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

        def format_sig_figs(value, pos):
            return f"{value:.1g}"  # '3g' ensures 3 significant figures

        dpi = 200
        n_fig_size = 7
        n = n_fig_size

        # Define constants and extract values
        xaxis_sz, legend_sz, ticks_sz = 25, 23, 25
        x1 = x[0, :]
        x2 = x[1, :]
        N = len(x1)
        t_span = np.arange(N) * env.del_t
        t_span_u = np.arange(u_act.shape[1] - 1) * env.del_t
        lblsize = legend_sz * 1.35
        alpha_set = 0.4
        lwp = 2.5
        lwp_sets = 2

        lwp_main = 2.8

        plt.rcParams.update(
            {
                "text.usetex": True,
                "font.family": "serif",
            }
        )
        plt.rc("text.latex", preamble=r"\usepackage{amsmath}")

        ############### ANIMATION FUNCTIONS ###############

        plt.figure(figsize=(2 * n, 0.9 * n), dpi=dpi)

        ##################### 2D PLOT #####################
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=2)
        ax1.axis("equal")
        ax1.set_xlabel(r"$x \text{ (m)}$", fontsize=xaxis_sz)
        ax1.set_ylabel(r"$z \text{ (m)}$", fontsize=xaxis_sz)
        ax1.xaxis.set_tick_params(labelsize=ticks_sz)
        ax1.yaxis.set_tick_params(labelsize=ticks_sz)

        ax1.set_aspect("equal", adjustable="box")
        buf = 1.06
        ax1.fill_between(
            [-50, 50],
            -10,
            env.z_min,
            color=[255 / 255, 204 / 255, 204 / 255],
            alpha=alpha_set,
        )
        ax1.hlines(
            y=env.z_min,
            xmin=-50,
            xmax=50,
            color=[255 / 255, 0 / 255, 0 / 255],
            linewidth=lwp_sets,
        )
        ax1.fill_between(
            [-50, 50],
            env.z_min,
            20,
            color=[217 / 255, 242 / 255, 208 / 255],
            alpha=alpha_set,
        )
        ax1.set_ylim([0.5, buf * max(x2)])
        ax1.text(
            2,
            2.9,
            "$\mathcal{C}_{\\rm S}$",
            fontsize=lblsize,
        )
        ax1.text(
            -1.5,
            0.79,
            "$z_{\\rm min}$",
            fontsize=lblsize,
            color=[255 / 255, 0 / 255, 0 / 255],
        )
        ax1.set_xlim([-1.8, 2.85])

        # Plot orientations of quadrotor
        L = 0.3
        o_step_freq = 2
        o_freq = env.backup_save_N * o_step_freq
        h = 0.05
        L_blade = 0.03
        tstampBufy = 0.99

        for i in range(0, N, o_freq):
            # Compute end points of the orientation line
            x_start = x[0, i] - L / 2 * np.cos(x[2, i])
            z_start = x[1, i] + L / 2 * np.sin(x[2, i])
            x_end = x[0, i] + L / 2 * np.cos(x[2, i])
            z_end = x[1, i] - L / 2 * np.sin(x[2, i])

            # Draw orientation line
            plt.plot([x_start, x_end], [z_start, z_end], "k-", linewidth=2)
            plt.scatter(x[0, i], x[1, i], color="black", zorder=5)

            if i == 0:
                ax1.text(
                    x[0, i] + 0.12,
                    x[1, i] * tstampBufy,
                    "t \!=\! 0",
                    fontsize=lblsize / 1.5,
                )

            # Draw orientation of props
            x_prop_del = h * np.sin(x[2, i])
            z_prop_del = h * np.cos(x[2, i])
            plt.plot(
                [x_start, x_start + x_prop_del],
                [z_start, z_start + z_prop_del],
                "k-",
                linewidth=2,
            )
            plt.plot(
                [x_end, x_end + x_prop_del],
                [z_end, z_end + z_prop_del],
                "k-",
                linewidth=2,
            )

            # Draw prop blades
            x_blade_del = L_blade / 2 * np.cos(x[2, i])
            z_blade_del = L_blade / 2 * np.sin(x[2, i])

            x_bladeL = x_start + x_prop_del
            z_bladeL = z_start + z_prop_del
            x_bladeR = x_end + x_prop_del
            z_bladeR = z_end + z_prop_del
            plt.plot(
                [x_bladeL - x_blade_del, x_bladeL + x_blade_del],
                [z_bladeL + z_blade_del, z_bladeL - z_blade_del],
                "k-",
                linewidth=2,
            )
            plt.plot(
                [x_bladeR - x_blade_del, x_bladeR + x_blade_del],
                [z_bladeR + z_blade_del, z_bladeR - z_blade_del],
                "k-",
                linewidth=2,
            )

        plt.plot(
            x[0, :],
            x[1, :],
            # color="magenta",
            color="#7030a0",
            linewidth=lwp_main,
            linestyle="-",
        )

        lw_backup = 2
        estimate_b_color = "#196b24"

        # Plot backup trajectories
        if env.backupTrajs:
            max_x_phi = []
            min_x_phi = []
            max_numBackup = len(env.backupTrajs)
            for i, xy in enumerate(
                zip(env.backupTrajs[::o_step_freq], env.backupTrajs_d[::o_step_freq])
            ):
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
                    )
                    plt.plot(
                        xy[1][:, 0],
                        xy[1][:, 1],
                        "--",
                        color="black",
                        linewidth=lw_backup,
                        label=label2,
                        zorder=1,
                    )
                    max_x_phi.append(max(xy[0][:, 0]))
                    min_x_phi.append(min(xy[0][:, 0]))

        leg1 = ax1.legend(
            loc="upper right",
            ncol=1,
            fontsize=legend_sz,
            columnspacing=0.4,
            handletextpad=0.2,
            handlelength=1.5,
            fancybox=True,
            shadow=True,
        )
        leg1.get_title().set_multialignment("center")
        plt.tight_layout()

        N_tick_bins = 7

        ##################### DISTURBANCE PLOT #####################
        ax2 = plt.subplot2grid((2, 2), (0, 1), colspan=1)
        plt.tight_layout()
        colors_quad = ["#ff595e", "#ff924c", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93"]

        # Legend stuff
        frame_bool = True
        shadow_bool = False
        fancy_bool = True

        for i in range(np.shape(d_full)[0]):
            ax2.set_xlim([0, t_span[-1]])  # Adjust the x-axis limits
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            color = colors_quad[i]

            if i == 3 or i == 4:
                ax2.plot(
                    t_span,
                    d_full[i, :],
                    "-",
                    color=color,
                    label="$d_{\\rm" + f"{i+1}" + "}$",
                    linewidth=lwp_main,
                )
                ax2.plot(
                    t_span,
                    dhat_full[i, :],
                    "--",
                    color=color,
                    label="$\\hat{d}_{\\rm" + f"{i+1}" + "}$",
                    linewidth=lwp_main,
                )
        ax2.yaxis.tick_right()
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax2.set_ylim([-0.5, 1.1])  # hardcoded
        plt.tight_layout()

        ax2.legend(
            fontsize=legend_sz,
            loc="center",
            columnspacing=0.4,
            handletextpad=0.2,
            handlelength=1.5,
            fancybox=fancy_bool,
            frameon=frame_bool,
            shadow=shadow_bool,
            ncol=4,
        )
        ax2.yaxis.set_major_formatter(FuncFormatter(format_sig_figs))
        plt.tight_layout()
        ax2.set_xlim([0, t_span[-1]])
        ax2.yaxis.set_major_locator(plt.MaxNLocator(N_tick_bins))

        ##################### CONTROL PLOT #####################
        ax3 = plt.subplot2grid((2, 2), (1, 1), colspan=1)
        ax3.yaxis.tick_right()
        plt.xticks(fontsize=ticks_sz)
        plt.yticks(fontsize=ticks_sz)
        ax3.set_xlim([0, t_span[-1]])
        ax3.set_ylim([-20, 20])  # hardcoded
        for i in range(np.shape(u_p)[0]):
            ax3.grid(False)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            ax3.set_xlim([0, t_span[-1]])  # Adjust the x-axis limits

            if i == 0:
                color = "#231942"
                label = "$F"
            elif i == 1:
                color = "#9F86C0"
                label = "$M"

            ax3.plot(
                t_span_u,
                u_p[i, 1:],
                "--",
                color=color,
                label=label + "_{\\rm p}$",
                linewidth=lwp,
            )
            ax3.plot(
                t_span_u,
                u_act[i, 1:],
                "-",
                color=color,
                label=label + "_{\\rm safe}$",
                linewidth=lwp,
            )

        ax3.legend(
            fontsize=legend_sz,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.35),
            columnspacing=0.4,
            handletextpad=0.2,
            handlelength=1.5,
            fancybox=fancy_bool,
            frameon=frame_bool,
            shadow=shadow_bool,
            ncol=4,
        )
        plt.xlabel("time (s)", fontsize=xaxis_sz)
        plt.tight_layout()

        border_lw = 1.5

        for a in [ax1, ax2, ax3]:
            for spine in a.spines.values():
                spine.set_linewidth(border_lw)

        plt.subplots_adjust(wspace=0.05)
        if show_plots:
            plt.show()

        if save_plots:
            plt.savefig(
                "plots/quadrotor_xz_d_u.pdf",
                dpi=100,
                bbox_inches="tight",
                pad_inches=0,
            )
