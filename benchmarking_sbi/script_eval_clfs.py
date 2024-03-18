from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
import torch.distributions as D

import pandas as pd
import sbibm
import random
from pathlib import Path
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import time

from valdiags.vanillaC2ST import c2st_scores


def eval_classifier_for_lc2st(
    x_samples,
    ref_samples,
    shifted_samples,
    shifts,
    clf_class,
    clf_kwargs,
    metrics=["accuracy", "probas_mean"],
    n_folds=10,
    single_class_eval=False,
):
    shift_list = []
    scores = {}
    accuracies = []
    for m in metrics:
        scores[m] = []
    times = []
    for s_samples, s in zip(shifted_samples, shifts):

        x_samples_shuffled = np.random.permutation(x_samples)

        joint_P_x = np.concatenate([ref_samples, x_samples], axis=1)
        joint_Q_x = np.concatenate([s_samples, x_samples_shuffled], axis=1)

        start = time.time()

        score = c2st_scores(
            P=joint_P_x,
            Q=joint_Q_x,
            metrics=metrics,
            clf_class=clf_class,
            clf_kwargs=clf_kwargs,
            cross_val=True,
            n_folds=n_folds,
            single_class_eval=single_class_eval,
        )

        for m in metrics:
            scores[m] = np.concatenate([scores[m], score[m]])

        accuracies = np.concatenate([accuracies, score["accuracy"]])

        total_cv_time = time.time() - start

        for _ in range(n_folds):
            shift_list.append(s)
            times.append(total_cv_time)
    return shift_list, scores, accuracies, times


def eval_null_lc2st(
    x_samples,
    null_dist_samples,
    test_stats=["probas_mean"],
    clf_class=MLPClassifier,
    clf_kwargs={"alpha": 0, "max_iter": 25000},
    clf_name="mlp_base",
    n_samples=1000,
    n_folds=10,
    single_class_eval=True,
):

    scores = {}
    for m in test_stats:
        scores[m] = []

    P, Q = null_dist_samples
    x_samples_shuffled = np.random.permutation(x_samples)

    joint_P_x = np.concatenate([P, x_samples], axis=1)
    joint_Q_x = np.concatenate([Q, x_samples_shuffled], axis=1)

    start = time.time()

    scores = c2st_scores(
        P=joint_P_x,
        Q=joint_Q_x,
        metrics=test_stats + ["probas_std"],
        clf_class=clf_class,
        clf_kwargs=clf_kwargs,
        cross_val=True,
        n_folds=n_folds,
        single_class_eval=single_class_eval,
    )
    total_cv_time = time.time() - start

    times = [total_cv_time] * n_folds
    n_samples_list = [n_samples] * n_folds
    classifier = [clf_name] * n_folds

    df = pd.DataFrame(
        {
            f"nb_samples": n_samples_list,
            "total_cv_time": times,
            "classifier": classifier,
        }
    )
    for m in test_stats + ["probas_std"]:
        df[m] = scores[m]

    return df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run eval classifiers for L-C2ST on sbi-benchmarking example."
    )
    parser.add_argument(
        "--task", "-t", type=str, default="gaussian_mixture", help="sbibm task name"
    )

    parser.add_argument(
        "--n_samples", "-n", type=int, default=10000, help="number of samples"
    )

    parser.add_argument(
        "--shift", "-s", type=str, default="mean", help="mean or scale shift experiment"
    )
    parser.add_argument(
        "--null_hyp", "-nh", action="store_true", help="do null precision experiment"
    )
    parser.add_argument(
        "--theta_space",
        "-ts",
        action="store_true",
        help="do null precision experiment in theta_space.",
    )

    parser.add_argument(
        "--clf_names",
        "-cn",
        type=str,
        help="names of classifiers to use in experiments.",
        nargs="+",
    )

    args = parser.parse_args()

    # Seeding
    torch.manual_seed(13)
    random.seed(13)
    np.random.seed(13)

    task = sbibm.get_task(args.task)
    prior = task.get_prior()
    simulator = task.get_simulator()
    theta = prior(num_samples=args.n_samples)
    x = simulator(theta)

    PATH = Path.cwd() / "eval_clfs_lc2st" / f"{args.task}"

    # Models
    ndim = theta.shape[-1] + x.shape[-1]
    clf_classes = {
        "mlp_base": MLPClassifier,
        "mlp_sbi": MLPClassifier,
        "rf": RandomForestClassifier,
        "lda": LinearDiscriminantAnalysis,
        "qda": QuadraticDiscriminantAnalysis,
    }
    clf_kwargs_dict = {
        "mlp_base": {"alpha": 0, "max_iter": 25000},
        "mlp_sbi": {
            "activation": "relu",
            "hidden_layer_sizes": (10 * ndim, 10 * ndim),
            "max_iter": 1000,
            "solver": "adam",
            "early_stopping": True,
            "n_iter_no_change": 50,
        },
        "rf": {},
        "lda": {},
        "qda": {},
    }

    dim = theta.shape[-1]

    if args.null_hyp:

        N_LIST = [1000, 2000, 5000, 10000]

        x_samples = {}
        theta_samples = {}
        for n in N_LIST:
            theta_samples[n] = prior(num_samples=n)
            x_samples[n] = simulator(theta_samples[n])

        null_dist = D.MultivariateNormal(
            loc=torch.zeros(dim), covariance_matrix=torch.eye(dim)
        )
        if args.clf_names is None:
            clf_names = ["mlp_base", "mlp_sbi", "rf", "lda", "qda"]
        else:
            clf_names = args.clf_names
        dfs = []
        for clf in clf_names:
            for n in N_LIST:
                if args.theta_space:
                    # if args.task == "gaussian_mixture":
                    #     null_dist_samples = [
                    #         task._sample_reference_posterior(
                    #             1, observation=x_samples[n]
                    #         ),
                    #         task._sample_reference_posterior(
                    #             1, observation=x_samples[n]
                    #         ),
                    #     ]
                    if args.task == "two_moons":
                        null_samples_1 = []
                        null_samples_2 = []
                        for x in x_samples[n]:
                            x = x[None, :]
                            null_samples_1.append(
                                task._sample_reference_posterior(
                                    1, num_observation=100, observation=x
                                )[0]
                            )
                            null_samples_2.append(
                                task._sample_reference_posterior(
                                    1, num_observation=100, observation=x
                                )[0]
                            )
                        null_dist_samples = [
                            torch.stack(null_samples_1),
                            torch.stack(null_samples_2),
                        ]
                    else:  # args.task == "gaussian_linear":
                        null_samples_1 = []
                        null_samples_2 = []
                        for x in x_samples[n]:
                            x = x[None, :]
                            null_samples_1.append(
                                task._sample_reference_posterior(1, observation=x)[0]
                            )
                            null_samples_2.append(
                                task._sample_reference_posterior(1, observation=x)[0]
                            )
                        null_dist_samples = [
                            torch.stack(null_samples_1),
                            torch.stack(null_samples_2),
                        ]
                    # else:
                    #     print("Task not available for this experiment.")

                else:
                    null_dist_samples = [null_dist.sample((n,)), null_dist.sample((n,))]
                print(f"Null-eval for n = {n}...")
                df = eval_null_lc2st(
                    x_samples,
                    null_dist_samples,
                    test_stats=["w_dist", "TV"],
                    n=n,
                    n_folds=50,
                    clf_class=clf_classes[clf],
                    clf_kwargs=clf_kwargs_dict[clf],
                    clf_name=clf,
                )
                dfs.append(df)

        df = pd.concat(dfs, ignore_index=True)
        torch.save(df, PATH / f"df_null_hyp_lc2st_ts_{args.theta_space}")

        for T, y1, y2 in zip(["TV", "w_dist"], [0.023, 0.08], [0.012, 0.04]):
            g = sns.relplot(
                data=df,
                x="nb_samples",
                y=T,
                hue="classifier",
                style="classifier",
                kind="line",
            )
            g.map(
                plt.axhline,
                y=y1,
                color=".7",
                dashes=(2, 1),
                zorder=0,
                label="norm with std 0.1",
            )
            g.map(
                plt.axhline,
                y=y2,
                color=".5",
                dashes=(2, 1),
                zorder=0,
                label="norm with std 0.05",
            )
            plt.legend()
            plt.savefig(PATH / f"lc2st_null_hyp_{T}_ts_{args.theta_space}.pdf")
            plt.show()
    else:

        # shifted gaussian samples for class 1
        mean_shifts = np.array([0, 0.3, 0.6, 1, 1.5, 2, 2.5, 3, 5, 10])
        mean_shifts = np.concatenate([-1 * mean_shifts, mean_shifts[1:]])
        scale_shifts = np.linspace(0.01, 10, 20)
        mean_shifted_samples = {}
        scale_shifted_samples = {}

        mean_shifted_samples = [
            D.MultivariateNormal(
                loc=torch.FloatTensor([m] * dim), covariance_matrix=torch.eye(dim)
            ).rsample((args.n_samples,))
            for m in mean_shifts
        ]

        scale_shifted_samples = [
            D.MultivariateNormal(
                loc=torch.zeros(dim), covariance_matrix=torch.eye(dim) * s
            ).rsample((args.n_samples,))
            for s in scale_shifts
        ]

        # norm samples
        norm_samples = D.MultivariateNormal(
            loc=torch.zeros(dim), covariance_matrix=torch.eye(dim)
        ).rsample((args.n_samples,))

        if args.shift == "mean":
            if args.clf_names is None:
                clf_names = ["lda", "mlp_sbi", "mlp_base", "rf"]
            else:
                clf_names = args.clf_names

            # if os.path.exists(PATH / f"df_mean_exp_lc2st_n_{args.n_samples}.pkl"):
            #     df_mean = torch.load(PATH / f"df_mean_exp_lc2st_n_{args.n_samples}.pkl")
            #     df_mean = df_mean.query(f"classifier in {args.clf_names}")
            # else:
            dfs = []
            for b in [True, False]:
                for clf_name in clf_names:
                    (
                        shift_list,
                        scores,
                        accuracies,
                        times,
                    ) = eval_classifier_for_lc2st(
                        x,
                        norm_samples,
                        shifted_samples=mean_shifted_samples,
                        shifts=mean_shifts,
                        clf_class=clf_classes[clf_name],
                        clf_kwargs=clf_kwargs_dict[clf_name],
                        metrics=["accuracy", "probas_mean", "div"],
                        single_class_eval=b,
                    )
                    if b:
                        clf_method = [clf_name] * len(shift_list)
                    else:
                        clf_method = [clf_name + f"_ref"] * len(shift_list)

                    dfs.append(
                        pd.DataFrame(
                            {
                                "mean_shift": shift_list,
                                "accuracy": accuracies,
                                "probas_mean": scores["probas_mean"],
                                "div": scores["div"],
                                "total_cv_time": times,
                                "classifier": clf_method,
                            }
                        )
                    )
            df_mean = pd.concat(dfs, ignore_index=True)

            torch.save(
                df_mean, PATH / f"df_mean_exp_lc2st_n_{args.n_samples}.pkl",
            )

            sns.relplot(
                data=df_mean,
                x="mean_shift",
                y="probas_mean",
                hue="classifier",
                style="classifier",
                kind="line",
            )
            plt.savefig(PATH / f"lc2st_mean_shift_n_{args.n_samples}_probas_mean.pdf")
            plt.show()

            sns.relplot(
                data=df_mean,
                x="mean_shift",
                y="accuracy",
                hue="classifier",
                style="classifier",
                kind="line",
            )
            plt.savefig(PATH / f"lc2st_mean_shift_n_{args.n_samples}_accuracy.pdf")
            plt.show()

            sns.relplot(
                data=df_mean,
                x="mean_shift",
                y="div",
                hue="classifier",
                style="classifier",
                kind="line",
            )
            plt.savefig(PATH / f"lc2st_mean_shift_n_{args.n_samples}_div.pdf")
            plt.show()

        elif args.shift == "scale":
            # if os.path.exists(PATH / f"df_scale_exp_lc2st_n_{args.n_samples}.pkl"):
            #     df_scale = torch.load(
            #         PATH / f"df_scale_exp_lc2st_n_{args.n_samples}.pkl"
            #     )
            #     df_scale = df_scale.query(f"classifier in {args.clf_names}")
            # else:
            if args.clf_names is None:
                clf_names = ["qda", "mlp_sbi", "mlp_base", "rf"]
            else:
                clf_names = args.clf_names

            dfs = []
            for b in [True, False]:
                for clf_name in clf_names:
                    (
                        shift_list,
                        scores,
                        accuracies,
                        times,
                    ) = eval_classifier_for_lc2st(
                        x,
                        norm_samples,
                        shifted_samples=scale_shifted_samples,
                        shifts=scale_shifts,
                        clf_class=clf_classes[clf_name],
                        clf_kwargs=clf_kwargs_dict[clf_name],
                        metrics=["accuracy", "probas_mean", "div"],
                        single_class_eval=b,
                    )
                    if b:
                        clf_method = [clf_name] * len(shift_list)
                    else:
                        clf_method = [clf_name + "_ref"] * len(shift_list)
                    dfs.append(
                        pd.DataFrame(
                            {
                                "scale_shift": shift_list,
                                "accuracy": accuracies,
                                "probas_mean": scores["probas_mean"],
                                "div": scores["div"],
                                "total_cv_time": times,
                                "classifier": clf_method,
                            }
                        )
                    )
            df_scale = pd.concat(dfs, ignore_index=True)

            torch.save(
                df_scale, PATH / f"df_scale_exp_lc2st_n_{args.n_samples}.pkl",
            )

            sns.relplot(
                data=df_scale,
                x="scale_shift",
                y="probas_mean",
                hue="classifier",
                style="classifier",
                kind="line",
            )
            plt.savefig(PATH / f"lc2st_scale_shift_n_{args.n_samples}_probas_mean.pdf")
            plt.show()

            sns.relplot(
                data=df_scale,
                x="scale_shift",
                y="accuracy",
                hue="classifier",
                style="classifier",
                kind="line",
            )
            plt.savefig(PATH / f"lc2st_scale_shift_n_{args.n_samples}_accuracy.pdf")
            plt.show()

            sns.relplot(
                data=df_scale,
                x="scale_shift",
                y="div",
                hue="classifier",
                style="classifier",
                kind="line",
            )
            plt.savefig(PATH / f"lc2st_scale_shift_n_{args.n_samples}_div.pdf")
            plt.show()

        else:
            print("Invalid experiment.")
