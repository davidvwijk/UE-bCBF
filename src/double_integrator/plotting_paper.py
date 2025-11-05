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
        phase_plot=True,
        quad_plot=True,
        tri_plot_lateral=False,
        bi_plot=True,
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
            y_max = 2.0
            y_min = -0.6
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
                -0.45,
                "$\mathcal{X} \\backslash \mathcal{C}_{\\rm S}$",
                fontsize=lblsize,
            )

            plt.axis("equal")
            plt.xlabel(r"$x_1$", fontsize=xaxis_sz)
            plt.ylabel(r"$x_2$", fontsize=xaxis_sz)
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
                label=r"$\boldsymbol{x}_0$",
                zorder=800,
            )
            label = r"$\boldsymbol{\phi}^{d}(t,\boldsymbol{x}_0)" + "$"
            plt.plot(x1, x2, "-", color="#7030a0", linewidth=lwp, label=label)
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
                                    label="$\delta_{\\rm max}(\\tau)$",
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

            # ax.legend(fontsize=legend_sz, loc="upper center")
            ax.legend(
                fontsize=legend_sz,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.15),
                framealpha=1,
                ncol=5,
                columnspacing=0.4,
                handletextpad=0.2,
                # handlelength=1.5,
                fancybox=fancy_bool,
                frameon=frame_bool,
                borderpad=bp,
                shadow=shadow_bool,
            )

            if save_plots:
                plt.savefig(
                    "plots/phase_plot_omega=" + str(env.d_omega) + ".pdf",
                    dpi=100,
                    bbox_inches="tight",
                )

        ####################################################################################################################
        ######################## Quad plot #################################################################################

        xaxis_sz, legend_sz, ticks_sz = 23, 20, 22

        def format_sig_figs(value, pos):
            return f"{value:.1g}"  # '3g' ensures 3 significant figures

        delta_t = env.del_t
        t_span_u = np.arange(u_act.shape[1] - 1) * delta_t
        t_span = np.arange(len(x1)) * delta_t

        # Legend stuff
        bp = 0.22  # Border pad
        frame_bool = True
        shadow_bool = False
        fancy_bool = True
        txtpd = 0.4
        lwp_quad = 2.8

        axes_ar = []
        border_lw = 1.5

        # Colors
        colors_quad = ["#ff595e", "#ff924c", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93"]
        # colors_quad = ["#264653", "#287271", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

        if quad_plot:
            fig = plt.figure(figsize=(10, 4.5), dpi=100)
            # plt.tight_layout()

            ################### Disturbance plot ###################
            ax = fig.add_subplot(2, 2, 1)
            axes_ar.append(ax)
            for i in range(np.shape(d_full)[0]):
                ax.set_xlim([0, t_span[-1]])  # Adjust the x-axis limits
                plt.xticks(fontsize=ticks_sz)
                plt.yticks(fontsize=ticks_sz)
                color = colors_quad[i]

                ax.plot(
                    t_span,
                    d_full[i, :],
                    "-",
                    color=color,
                    label="$d_{\\rm" + f"{i+1}" + "}$",
                    linewidth=lwp_quad,
                )
                ax.plot(
                    t_span,
                    dhat_full[i, :],
                    "--",
                    color=color,
                    label="$\\hat{d}_{\\rm" + f"{i+1}" + "}$",
                    linewidth=lwp_quad,
                )
            # ax.legend(fontsize=legend_sz, loc="lower right")
            ax.legend(
                fontsize=legend_sz,
                # loc="upper center",
                loc="center",
                # bbox_to_anchor=(0.5, 1.35),
                # bbox_to_anchor=(0.5, 0.4),
                bbox_to_anchor=(0.5, 0.45),
                columnspacing=0.4,
                handletextpad=0.2,
                handlelength=1.5,
                fancybox=fancy_bool,
                frameon=frame_bool,
                shadow=shadow_bool,
                ncol=4,
            )
            ax.yaxis.set_major_formatter(FuncFormatter(format_sig_figs))
            ax.set_xticklabels([])
            ##################### Control plot #####################

            ax = fig.add_subplot(2, 2, 3)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            ax.set_xlim([0, t_span[-1]])
            ax.plot(
                t_span_u,
                u_p[0][1:],
                "--",
                color=colors_quad[4],
                # color="red",
                label="$u_{\\rm p}$",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act[0][1:],
                "-",
                color=colors_quad[3],
                # color="green",
                label="$u_{\\rm safe}$",
                linewidth=lwp_quad,
            )
            # ax.set_ylabel("u", fontsize=xaxis_sz)
            ax.legend(
                fontsize=legend_sz,
                loc="center right",
                frameon=frame_bool,
                borderpad=bp,
                fancybox=fancy_bool,
                shadow=shadow_bool,
                handletextpad=txtpd,
            )
            # ax.set_xticklabels([])
            plt.xlabel("time (s)", fontsize=xaxis_sz)

            ##################### Error plot #####################
            ax = fig.add_subplot(2, 2, 2)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            ax.set_xlim([0, t_span[-1]])
            ax.plot(
                t_span,
                np.linalg.norm(d_full - dhat_full, axis=0),
                "-",
                # color=colors_quad[2],
                # color="black",
                color="deeppink",
                label=r"$\|\boldsymbol{e}\|$",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span,
                env.e_bar(t_span),
                "--",
                # color=colors_quad[2],
                # color="deeppink",
                color="black",
                label=r"$\bar{e}$",
                linewidth=lwp_quad,
            )
            ax.legend(
                fontsize=legend_sz,
                loc="upper right",
                frameon=frame_bool,
                borderpad=bp,
                fancybox=fancy_bool,
                shadow=shadow_bool,
                handletextpad=txtpd,
            )
            ax.set_xticklabels([])
            # plt.xlabel("time (s)", fontsize=xaxis_sz)

            ##################### h(x) plot ######################
            ax = fig.add_subplot(2, 2, 4)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            ax.set_xlim([0, t_span[-1]])
            ax.plot(
                t_span,
                env.h1_x(x),
                "-",
                color=colors_quad[5],
                # label=r"$h(\boldsymbol{x})$",
                label=r"$h$",
                linewidth=lwp_quad,
                zorder=999,
            )
            plt.axhline(
                0,
                color="k",
                linestyle="--",
                linewidth=lwp_quad,
            )
            ax.legend(
                fontsize=legend_sz,
                loc="upper right",
                frameon=frame_bool,
                borderpad=bp,
                fancybox=fancy_bool,
                shadow=shadow_bool,
                handletextpad=txtpd,
            )
            plt.xlabel("time (s)", fontsize=xaxis_sz)

            for a in axes_ar:
                for spine in a.spines.values():
                    spine.set_linewidth(border_lw)

            if save_plots:
                plt.savefig(
                    "plots/quad_plot_omega=" + str(env.d_omega) + ".pdf",
                    dpi=100,
                    bbox_inches="tight",
                )

        ####################################################################################################
        # TRI PLOT

        def format_sig_figs(value, pos):
            return f"{value:.1g}"  # '3g' ensures 3 significant figures

        if tri_plot_lateral:
            fig = plt.figure(figsize=(10 * (3 / 2), 4.5 / 3), dpi=100)
            # plt.tight_layout()
            # fig = plt.figure(figsize=(10, 4.5 / 3), dpi=100)

            ################### Disturbance plot ###################
            ax = fig.add_subplot(1, 3, 1)
            axes_ar.append(ax)
            for i in range(np.shape(d_full)[0]):
                ax.set_xlim([0, t_span[-1]])  # Adjust the x-axis limits
                plt.xticks(fontsize=ticks_sz)
                plt.yticks(fontsize=ticks_sz)
                color = colors_quad[i]

                ax.plot(
                    t_span,
                    d_full[i, :],
                    "-",
                    color=color,
                    label="$d_{\\rm" + f"{i+1}" + "}$",
                    linewidth=lwp_quad,
                )
                ax.plot(
                    t_span,
                    dhat_full[i, :],
                    "--",
                    color=color,
                    label="$\\hat{d}_{\\rm" + f"{i+1}" + "}$",
                    linewidth=lwp_quad,
                )
            # ax.legend(fontsize=legend_sz, loc="lower right")
            ax.legend(
                fontsize=legend_sz,
                # loc="upper center",
                loc="center",
                # bbox_to_anchor=(0.5, 1.35),
                # bbox_to_anchor=(0.5, 0.4),
                bbox_to_anchor=(0.5, 0.45),
                columnspacing=0.4,
                handletextpad=0.2,
                handlelength=1.5,
                fancybox=fancy_bool,
                frameon=frame_bool,
                shadow=shadow_bool,
                ncol=4,
            )
            ax.yaxis.set_major_formatter(FuncFormatter(format_sig_figs))
            # ax.set_xticklabels([])
            plt.xlabel("time (s)", fontsize=xaxis_sz)
            ##################### Control plot #####################

            ax = fig.add_subplot(1, 3, 3)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            ax.set_xlim([0, t_span[-1]])
            ax.plot(
                t_span_u,
                u_p[0][1:],
                "--",
                color=colors_quad[4],
                # color="red",
                label="$u_{\\rm p}$",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act[0][1:],
                "-",
                color=colors_quad[3],
                # color="green",
                label="$u_{\\rm safe}$",
                linewidth=lwp_quad,
            )
            # ax.set_ylabel("u", fontsize=xaxis_sz)
            ax.legend(
                fontsize=legend_sz,
                loc="center right",
                frameon=frame_bool,
                borderpad=bp,
                fancybox=fancy_bool,
                shadow=shadow_bool,
                handletextpad=txtpd,
            )
            # ax.set_xticklabels([])
            plt.xlabel("time (s)", fontsize=xaxis_sz)

            ##################### Error plot #####################
            ax = fig.add_subplot(1, 3, 2)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            ax.set_xlim([0, t_span[-1]])
            ax.plot(
                t_span,
                np.linalg.norm(d_full - dhat_full, axis=0),
                "-",
                # color=colors_quad[2],
                # color="black",
                color="deeppink",
                label=r"$\|\boldsymbol{e}\|$",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span,
                env.e_bar(t_span),
                "--",
                # color=colors_quad[2],
                # color="deeppink",
                color="black",
                label=r"$\bar{e}$",
                linewidth=lwp_quad,
            )
            ax.legend(
                fontsize=legend_sz,
                loc="upper right",
                frameon=frame_bool,
                borderpad=bp,
                fancybox=fancy_bool,
                shadow=shadow_bool,
                handletextpad=txtpd,
            )
            # ax.set_xticklabels([])
            plt.xlabel("time (s)", fontsize=xaxis_sz)
            ax.yaxis.set_major_formatter(FuncFormatter(format_sig_figs))

            for a in axes_ar:
                for spine in a.spines.values():
                    spine.set_linewidth(border_lw)

            # plt.subplots_adjust(hspace=0.3)
            if save_plots:
                plt.savefig(
                    "plots/tri_plot_omega=" + str(env.d_omega) + ".png",
                    dpi=500,
                    bbox_inches="tight",
                    pad_inches=0,
                )

        ####################################################################################################################
        ######################## Biplot #################################################################################

        xaxis_sz, legend_sz, ticks_sz = 23, 20, 22

        def format_sig_figs(value, pos):
            return f"{value:.1g}"  # '3g' ensures 3 significant figures

        delta_t = env.del_t
        t_span_u = np.arange(u_act.shape[1] - 1) * delta_t
        t_span = np.arange(len(x1)) * delta_t

        # Legend stuff
        bp = 0.22  # Border pad
        frame_bool = True
        shadow_bool = False
        fancy_bool = True
        txtpd = 0.4
        lwp_quad = 2.8

        axes_ar = []
        border_lw = 1.5

        # Colors
        colors_quad = ["#ff595e", "#ff924c", "#ffca3a", "#8ac926", "#1982c4", "#6a4c93"]
        # colors_quad = ["#264653", "#287271", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]

        if bi_plot:
            fig = plt.figure(figsize=(10, 4.5 / 3), dpi=100)
            # plt.tight_layout()

            ################### Disturbance plot ###################
            ax = fig.add_subplot(1, 2, 1)
            axes_ar.append(ax)
            for i in range(np.shape(d_full)[0]):
                ax.set_xlim([0, t_span[-1]])  # Adjust the x-axis limits
                plt.xticks(fontsize=ticks_sz)
                plt.yticks(fontsize=ticks_sz)
                color = colors_quad[i]

                ax.plot(
                    t_span,
                    d_full[i, :],
                    "-",
                    color=color,
                    label="$d_{\\rm" + f"{i+1}" + "}$",
                    linewidth=lwp_quad,
                )
                ax.plot(
                    t_span,
                    dhat_full[i, :],
                    "--",
                    color=color,
                    label="$\\hat{d}_{\\rm" + f"{i+1}" + "}$",
                    linewidth=lwp_quad,
                )
            # ax.legend(fontsize=legend_sz, loc="lower right")
            ax.legend(
                fontsize=legend_sz,
                loc="upper center",
                # bbox_to_anchor=(0.5, 1.35),
                # bbox_to_anchor=(0.5, 0.4),
                bbox_to_anchor=(0.5, 0.6),
                columnspacing=0.4,
                handletextpad=0.2,
                handlelength=1.5,
                fancybox=fancy_bool,
                frameon=frame_bool,
                shadow=shadow_bool,
                ncol=4,
            )
            ax.yaxis.set_major_formatter(FuncFormatter(format_sig_figs))
            # ax.set_xticklabels([])
            plt.xlabel("time (s)", fontsize=xaxis_sz)

            ##################### Control plot #####################

            ax = fig.add_subplot(1, 2, 2)
            axes_ar.append(ax)
            plt.xticks(fontsize=ticks_sz)
            plt.yticks(fontsize=ticks_sz)
            ax.set_xlim([0, t_span[-1]])
            ax.plot(
                t_span_u,
                u_p[0][1:],
                "--",
                color=colors_quad[4],
                # color="red",
                label="$u_{\\rm p}$",
                linewidth=lwp_quad,
            )
            ax.plot(
                t_span_u,
                u_act[0][1:],
                "-",
                color=colors_quad[3],
                # color="green",
                label="$u_{\\rm safe}$",
                linewidth=lwp_quad,
            )
            # ax.set_ylabel("u", fontsize=xaxis_sz)
            ax.legend(
                fontsize=legend_sz,
                loc="center right",
                frameon=frame_bool,
                borderpad=bp,
                fancybox=fancy_bool,
                shadow=shadow_bool,
                handletextpad=txtpd,
            )
            # ax.set_xticklabels([])
            plt.xlabel("time (s)", fontsize=xaxis_sz)
            plt.subplots_adjust(wspace=0.21)

            for a in axes_ar:
                for spine in a.spines.values():
                    spine.set_linewidth(border_lw)

            if save_plots:
                plt.savefig(
                    "plots/bi_plot_omega=" + str(env.d_omega) + ".pdf",
                    # "plots/bi_plot_omega=" + str(env.d_omega) + ".png",
                    dpi=500,
                    bbox_inches="tight",
                    pad_inches=0,
                )

        if show_plots:
            plt.show()
