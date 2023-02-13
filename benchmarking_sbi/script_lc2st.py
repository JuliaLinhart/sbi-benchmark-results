import sbibm
import torch
from pathlib import Path
import matplotlib.pyplot as plt
import random

from sbibm.utils.io import get_tensor_from_csv

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run L-C2ST on sbi-benchmarking example."
    )
    parser.add_argument(
        "--experiment",
        "-e",
        type=str,
        default="multirun/2023-02-11/14-42-12/0",
        help='Experiment name: "multirun/yyyy-mm-dd/hh-mm-ss/int"',
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
        default=1000,
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

    args = parser.parse_args()

    random.seed(1)

    # EXPERIMENT = "multirun/2023-02-11/14-42-12/5"
    # NUM_OBSERVATION = 5
    PATH_EXPERIMENT = Path.cwd() / args.experiment

    task = sbibm.get_task("gaussian_mixture")
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

    # =============== Hypothesis test and ensemble probas ================
    metrics = ["probas_mean", "w_dist", "TV"]

    if args.run_htest:
        print("Running L-C2ST...")
        from valdiags.localC2ST import lc2st_htest_sbibm

        test_size = args.test_size
        z_space_eval = posterior_est.flow.net._distribution.sample(test_size)

        n_trials = args.n_trials_null
        n_ensemble = args.n_ensemble

        p_values, test_stats, probas_ens, probas_null, t_stats_null = lc2st_htest_sbibm(
            base_dist_samples_cal,
            inv_flow_samples_cal,
            cal_set["x"],
            P_eval=z_space_eval,
            x_eval=observation,
            null_dist=posterior_est.flow.net._distribution,
            test_stats=metrics,
            n_trials_null=n_trials,
            n_ensemble=n_ensemble,
        )
        lc2st_htest_results = {
            "p_values": p_values,
            "test_stats": test_stats,
            "t_stats_null": t_stats_null,
        }
        torch.save(lc2st_htest_results, PATH_EXPERIMENT / "lc2st_htest_results.pkl")
        torch.save(probas_ens, PATH_EXPERIMENT / "lc2st_probas_ensemble.pkl")
        torch.save(probas_null, PATH_EXPERIMENT / "lc2st_probas_null.pkl")
        torch.save(z_space_eval, PATH_EXPERIMENT / "z_space_eval.pkl")
        print("Finished running L-C2ST.")

    lc2st_htest_results = torch.load(PATH_EXPERIMENT / "lc2st_htest_results.pkl")
    test_stats = lc2st_htest_results["test_stats"]
    t_stats_null = lc2st_htest_results["t_stats_null"]
    p_values = lc2st_htest_results["p_values"]

    probas_ens = torch.load(PATH_EXPERIMENT / "lc2st_probas_ensemble.pkl")
    probas_null = torch.load(PATH_EXPERIMENT / "lc2st_probas_null.pkl")
    z_space_eval = torch.load(PATH_EXPERIMENT / "z_space_eval.pkl")

    # # =============== Result plots ===============
    from valdiags.localC2ST import box_plot_lc2st

    for m in metrics:
        box_plot_lc2st(
            [test_stats[m]],
            t_stats_null[m],
            labels=["NPE"],
            colors=["red"],
        )
        plt.savefig(PATH_EXPERIMENT / f"lc2st_htest_box_plot_metric_{m}.pdf")
        plt.show()
        print(f"pvalues {m}:", p_values[m])

    from valdiags.localC2ST import pp_plot_lc2st

    pp_plot_lc2st([probas_ens], probas_null, labels=["NPE"], colors=["red"])
    plt.savefig(PATH_EXPERIMENT / "lc2st_pp_plot.pdf")
    plt.show()

    # =============== Interpretability plots ==================

    # Hist of the true conditional distributions: norm, inv-flow
    from valdiags.localC2ST import flow_vs_reference_distribution

    # embedding not intergrated in transform method (includes standardize)
    observation_emb = posterior_est.flow.net._embedding_net(observation)

    thetas, xs = posterior_est.flow._match_theta_and_x_batch_shapes(
        posterior_samples, observation_emb
    )
    inv_flow_samples_ref = posterior_est.flow.net._transform(thetas, xs)[0].detach()

    dim = thetas.shape[-1]

    for hist in [True, False]:
        flow_vs_reference_distribution(
            samples_ref=base_dist_samples_cal,
            samples_flow=inv_flow_samples_ref,
            z_space=True,
            dim=dim,
            hist=hist,
        )
        if hist:
            plt.savefig(PATH_EXPERIMENT / "hist_z_space_reference.pdf")
        else:
            plt.savefig(PATH_EXPERIMENT / "z_space_reference.pdf")
        plt.show()

    # High / Low probability regions
    from valdiags.localC2ST import z_space_with_proba_intensity

    z_space_with_proba_intensity(probas_ens, probas_null, z_space_eval, dim=dim)
    plt.savefig(PATH_EXPERIMENT / "z_space_with_lc2st_proba_intensity.pdf")
    plt.show()

    _, xs = posterior_est.flow._match_theta_and_x_batch_shapes(
        z_space_eval, observation_emb
    )
    theta_space_eval = posterior_est.flow.net._transform.inverse(z_space_eval, xs)[
        0
    ].detach()
    z_space_with_proba_intensity(
        probas_ens, probas_null, z_space_eval, theta_space=theta_space_eval, dim=2
    )
    plt.savefig(PATH_EXPERIMENT / "theta_space_with_lc2st_proba_intensity.pdf")
    plt.show()

    # True vs. estimated posterior samples
    algorithm_posterior_samples = get_tensor_from_csv(
        PATH_EXPERIMENT / "posterior_samples.csv.bz2"
    )[: task.num_posterior_samples, :]
    for hist in [True, False]:
        flow_vs_reference_distribution(
            samples_ref=posterior_samples,
            samples_flow=algorithm_posterior_samples,
            z_space=False,
            dim=dim,
            hist=hist,
        )
        if hist:
            plt.savefig(PATH_EXPERIMENT / "hist_theta_space_reference.pdf")
        else:
            plt.savefig(PATH_EXPERIMENT / "theta_space_reference.pdf")
        plt.show()
