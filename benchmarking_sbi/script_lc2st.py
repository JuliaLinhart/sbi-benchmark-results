import sbibm
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import random

from sbibm.utils.io import get_tensor_from_csv, save_float_to_csv

import os

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run L-C2ST on sbi-benchmarking example."
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default="multirun/2023-02-19/12-11-37",
        help='Experiment name: "multirun/yyyy-mm-dd/hh-mm-ss"',
    )
    parser.add_argument(
        "--num_observation",
        "-no",
        type=int,
        default=1,
        help="Observation number between 1 and 10.",
    )
    parser.add_argument(
        "--task", "-t", type=str, default="gaussian_mixture", help="sbibm task name"
    )
    parser.add_argument(
        "--test_size",
        "-ts",
        type=int,
        default=10000,
        help="number of base-dist samples for l-c2st evaluation.",
    )
    parser.add_argument(
        "--n_trials_null",
        "-nt",
        type=int,
        default=100,
        help="number of trials under the null hypothesis.",
    )
    parser.add_argument(
        "--n_ensemble",
        "-ne",
        type=int,
        default=10,
        help="number of models for ensemble-prediction.",
    )

    parser.add_argument(
        "--run_htest",
        "-r",
        action="store_true",
        help="Whether to run the lc2st or just generate plots.",
    )

    parser.add_argument(
        "--correct_posterior",
        "-c",
        action="store_true",
        help="Whether to run posterior corretion or not.",
    )

    parser.add_argument(
        "--z_space",
        "-z",
        action="store_true",
        help="Whether to run lc2st in z-space or not.",
    )

    args = parser.parse_args()

    random.seed(1)

    PATH_EXPERIMENT = Path.cwd() / args.experiment / f"{args.num_observation - 1}"

    task = sbibm.get_task(args.task)
    prior = task.get_prior()
    simulator = task.get_simulator()
    observation = task.get_observation(num_observation=args.num_observation)

    cal_set = torch.load(PATH_EXPERIMENT / "calibration_dataset.pkl")
    posterior_samples = task.get_reference_posterior_samples(args.num_observation)[
        : len(cal_set["x"]), :
    ]

    posterior_est = torch.load(PATH_EXPERIMENT / "posterior_est.pkl")
    inv_flow_samples_cal = torch.load(PATH_EXPERIMENT / "inv_flow_samples.pkl")
    base_dist_samples_cal = torch.load(PATH_EXPERIMENT / "base_dist_samples.pkl")
    flow_posterior_samples_cal = torch.load(
        PATH_EXPERIMENT / "flow_posterior_samples_cal.pkl"
    ).detach()
    algorithm_posterior_samples = get_tensor_from_csv(
        PATH_EXPERIMENT / "posterior_samples.csv.bz2"
    )[: task.num_posterior_samples, :]

    # =============== Reference plots ==================

    # True conditional distributions: norm, inv-flow
    from valdiags.localC2ST import flow_vs_reference_distribution

    # embedding not intergrated in transform method (includes standardize)
    observation_emb = posterior_est.flow.net._embedding_net(observation)

    thetas, xs = posterior_est.flow._match_theta_and_x_batch_shapes(
        posterior_samples, observation_emb
    )
    inv_flow_samples_ref = posterior_est.flow.net._transform(thetas, xs)[0].detach()

    dim = thetas.shape[-1]
    if dim <= 2:
        flow_vs_reference_distribution(
            samples_ref=base_dist_samples_cal,
            samples_flow=inv_flow_samples_ref,
            z_space=True,
            dim=dim,
            hist=False,
        )
        plt.savefig(PATH_EXPERIMENT / "z_space_reference.pdf")
        plt.show()

    from valdiags.plot_utils import multi_corner_plots

    samples_list = [base_dist_samples_cal, inv_flow_samples_ref]
    legends = [
        r"Ref: $\mathcal{N}(0,1)$",
        r"NPE: $T_{\phi}^{-1}(\Theta;x_0) \mid x_0$",
    ]
    colors = ["blue", "orange"]
    multi_corner_plots(
        samples_list,
        legends,
        colors,
        title=r"Base-Distribution vs. Inverse Flow-Transformation (of $\Theta \mid x_0$)",
        labels=[r"$z$" f"_{i}" for i in range(dim)],
        # domain=(torch.tensor([-5, -5]), torch.tensor([5, 5])),
    )
    plt.savefig(PATH_EXPERIMENT / "z_space_reference_corner.pdf")
    plt.show()

    # True vs. estimated posterior samples
    if dim <= 2:
        flow_vs_reference_distribution(
            samples_ref=posterior_samples,
            samples_flow=algorithm_posterior_samples,
            z_space=False,
            dim=dim,
            hist=False,
        )
        plt.savefig(PATH_EXPERIMENT / "theta_space_reference.pdf")
        plt.show()

    samples_list = [posterior_samples, algorithm_posterior_samples]
    legends = [r"Ref: $p(\Theta \mid x_0)$", r"NPE: $p(T_{\phi}(Z;x_0))$"]
    colors = ["blue", "orange"]
    multi_corner_plots(
        samples_list,
        legends,
        colors,
        title=r"True vs. Estimated distributions at $x_0$",
        labels=[r"$\theta$" + f"_{i}" for i in range(dim)],
        # domain=(torch.tensor([-15, -15]), torch.tensor([5, 5])),
    )
    plt.savefig(PATH_EXPERIMENT / "theta_space_reference_corner.pdf")
    plt.show()

    plt.close("all")

    # =============== Hypothesis test and ensemble probas ================
    metrics = ["probas_mean", "w_dist", "TV"]

    if args.run_htest:
        print("Running L-C2ST...")
        from valdiags.localC2ST import lc2st_htest_sbibm

        test_size = args.test_size

        if args.z_space:
            P = base_dist_samples_cal
            Q = inv_flow_samples_cal
            P_eval = posterior_est.flow.net._distribution.sample(test_size).detach()
            null_dist = posterior_est.flow.net._distribution
            null_samples_list = None
            clf_kwargs = {"alpha": 0, "max_iter": 25000}
        else:
            P = flow_posterior_samples_cal
            Q = cal_set["theta"]
            P_eval = algorithm_posterior_samples[:test_size]
            null_dist = None
            null_samples_list = []
            for t in range(args.n_trials_null):
                samples_z = posterior_est.flow.net._distribution.sample(
                    len(cal_set["x"])
                ).detach()
                observation_emb = posterior_est.flow.net._embedding_net(observation)
                zs, xs = posterior_est.flow._match_theta_and_x_batch_shapes(
                    samples_z, observation_emb
                )
                null_samples_list.append(
                    posterior_est.flow.net._transform.inverse(zs, xs)[0].detach()
                )
            clf_kwargs = None

        path_trained_clfs = Path.cwd() / "trained_clfs_lc2st" / f"{args.task}"
        if os.path.exists(
            path_trained_clfs / f"lc2st_probas_null_z_{args.z_space}.pkl"
        ):
            probas_null = torch.load(
                path_trained_clfs / f"lc2st_probas_null_z_{args.z_space}.pkl"
            )
        else:
            probas_null = []

        if os.path.exists(path_trained_clfs / f"trained_clfs_z_{args.z_space}.pkl"):
            trained_clfs = torch.load(
                path_trained_clfs / f"trained_clfs_z_{args.z_space}.pkl"
            )
        else:
            trained_clfs = []

        (
            p_values,
            test_stats,
            probas_ens,
            probas_null,
            t_stats_null,
            trained_clfs,
            run_time,
        ) = lc2st_htest_sbibm(
            P,
            Q,
            cal_set["x"],
            P_eval=P_eval,
            x_eval=observation,
            null_dist=null_dist,
            null_samples_list=null_samples_list,
            test_stats=metrics,
            n_trials_null=args.n_trials_null,
            n_ensemble=args.n_ensemble,
            clf_kwargs=None,
            probas_null=probas_null,
            trained_clfs=trained_clfs,
        )
        lc2st_htest_results = {
            "p_values": p_values,
            "test_stats": test_stats,
            "t_stats_null": t_stats_null,
        }
        torch.save(
            lc2st_htest_results,
            PATH_EXPERIMENT / f"lc2st_htest_results_z_{args.z_space}.pkl",
        )
        torch.save(
            probas_ens, PATH_EXPERIMENT / f"lc2st_probas_ensemble_z_{args.z_space}.pkl"
        )

        torch.save(P_eval, PATH_EXPERIMENT / f"P_eval_z_{args.z_space}.pkl")

        torch.save(
            probas_null,
            Path.cwd()
            / "trained_clfs_lc2st"
            / f"{args.task}"
            / f"lc2st_probas_null_z_{args.z_space}.pkl",
        )
        torch.save(
            trained_clfs,
            Path.cwd()
            / "trained_clfs_lc2st"
            / f"{args.task}"
            / f"trained_clfs_z_{args.z_space}.pkl",
        )

        if run_time != 0:
            save_float_to_csv(
                Path.cwd()
                / "trained_clfs_lc2st"
                / f"{args.task}"
                / f"run_time_1_clf_z_{args.z_space}.csv",
                run_time,
            )

        print("Finished running L-C2ST.")

    P_eval = torch.load(PATH_EXPERIMENT / f"P_eval_z_{args.z_space}.pkl")
    probas_ens = torch.load(
        PATH_EXPERIMENT / f"lc2st_probas_ensemble_z_{args.z_space}.pkl"
    )

    probas_null = torch.load(
        Path.cwd()
        / "trained_clfs_lc2st"
        / f"{args.task}"
        / f"lc2st_probas_null_z_{args.z_space}.pkl"
    )

    lc2st_htest_results = torch.load(
        PATH_EXPERIMENT / f"lc2st_htest_results_z_{args.z_space}.pkl"
    )
    test_stats = lc2st_htest_results["test_stats"]
    t_stats_null = lc2st_htest_results["t_stats_null"]
    p_values = lc2st_htest_results["p_values"]

    # # =============== Result plots ===============

    from valdiags.localC2ST import box_plot_lc2st

    for m in metrics:
        box_plot_lc2st(
            [test_stats[m]],
            t_stats_null[m],
            labels=["NPE"],
            colors=["red"],
        )
        plt.savefig(
            PATH_EXPERIMENT / f"lc2st_htest_box_plot_metric_{m}_z_{args.z_space}.pdf"
        )
        plt.show()
        print(f"pvalues {m}:", p_values[m])

    from valdiags.localC2ST import pp_plot_lc2st

    pp_plot_lc2st([probas_ens], probas_null, labels=["NPE"], colors=["red"])
    plt.savefig(PATH_EXPERIMENT / f"lc2st_pp_plot_z_{args.z_space}.pdf")
    plt.show()

    # =============== Interpretability plots ==================
    if dim <= 2:
        # High / Low probability regions
        from valdiags.localC2ST import (
            z_space_with_proba_intensity,
            eval_space_with_proba_intensity,
        )

        eval_space_with_proba_intensity(
            probas_ens, probas_null, P_eval, dim=dim, z_space=args.z_space
        )
        plt.savefig(
            PATH_EXPERIMENT / f"eval_with_lc2st_proba_intensity_z_{args.z_space}.pdf"
        )
        plt.show()

    # =============== Posterior correction ==================

    if args.correct_posterior:
        d = probas_ens
        r = (1 - d) / d

        resample_idx = torch.multinomial(
            torch.tensor(r),
            len(P_eval),
            replacement=True,
        )

        if args.z_space:

            samples_list = [
                P_eval.numpy(),
                P_eval[resample_idx].numpy(),
                inv_flow_samples_ref.numpy(),
            ]
            legends = [
                "normal: Z",
                "corrected normal",
                r"Reference: $T_{\phi}^{-1}(\Theta, x_0)\mid x_0$",
            ]
            colors = ["grey", "red", "orange"]
            multi_corner_plots(
                samples_list,
                legends,
                colors,
                title="correction in z-space",
                labels=[r"$z$" + f"_{i}" for i in range(dim)]
                # domain=(torch.tensor([-5, -5]), torch.tensor([5, 5])),
            )
            plt.savefig(PATH_EXPERIMENT / "z_space_correction_corner.pdf")
            plt.show()

            zs, xs = posterior_est.flow._match_theta_and_x_batch_shapes(
                P_eval, observation_emb
            )
            flow_thetas = posterior_est.flow.net._transform.inverse(zs, xs)[0]

            corrected_zs, xs = posterior_est.flow._match_theta_and_x_batch_shapes(
                P_eval[resample_idx], observation_emb
            )
            corrected_thetas = posterior_est.flow.net._transform.inverse(
                corrected_zs, xs
            )[0]

            samples_list = [
                flow_thetas.detach().numpy(),
                corrected_thetas.detach().numpy(),
                posterior_samples,
            ]
            legends = [
                r"NPE: $T_{\phi}(Z;x_0)$",
                r"corrected NPE: $T_{\phi}(Z_r;x_0)$",
                r"Reference: $\Theta \mid x_0$",
            ]
            colors = ["orange", "red", "blue"]
            multi_corner_plots(
                samples_list,
                legends,
                colors,
                title="correction in z-space",
                labels=[r"$\theta$" + f"_{i}" for i in range(dim)]
                # domain=(torch.tensor([-15, -15]), torch.tensor([5, 5])),
            )
            plt.savefig(PATH_EXPERIMENT / "posterior_correction_z_space_corner.pdf")
            plt.show()

            # samples_list = [
            #     posterior_samples,
            #     algorithm_posterior_samples,
            #     flow_thetas.detach().numpy(),
            # ]
            # legends = [
            #     r"Ref: $p(\Theta \mid x_0)$",
            #     r"NPE: $p(T_{\phi}(Z;x_0))$",
            #     r"NPE on z-eval: $T_{\phi}(Z_eval;x_0)$",
            # ]
            # colors = ["blue", "orange", "red"]
            # multi_corner_plots(
            #     samples_list,
            #     legends,
            #     colors,
            #     title=r"True vs. Estimated distributions at $x_0$",
            #     labels=[r"$\theta$" + f"_{i}" for i in range(dim)],
            #     # domain=(torch.tensor([-15, -15]), torch.tensor([5, 5])),
            # )
            # # plt.savefig(PATH_EXPERIMENT / "theta_space_reference_corner.pdf")
            # plt.show()

        else:
            samples_list = [
                P_eval.numpy(),
                P_eval[resample_idx].numpy(),
                posterior_samples,
            ]
            legends = [
                r"NPE: $\Theta \sim q_{\phi}(\theta \mid x_0)$",
                "corrected NPE",
                r"Reference: $\Theta \sim p(\theta \mid x_0)$",
            ]
            colors = ["orange", "red", "blue"]
            multi_corner_plots(
                samples_list,
                legends,
                colors,
                title="correction in parameter space",
                labels=[r"$\theta$" + f"_{i}" for i in range(dim)]
                # domain=(torch.tensor([-5, -5]), torch.tensor([5, 5])),
            )
            plt.savefig(PATH_EXPERIMENT / "posterior_correction_theta_space_corner.pdf")
            plt.show()
