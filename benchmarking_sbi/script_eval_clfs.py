from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import (
    LinearDiscriminantAnalysis,
    QuadraticDiscriminantAnalysis,
)
import torch.distributions as D

from valdiags.localC2ST import eval_classifier_for_lc2st
import pandas as pd
import sbibm
import random
from pathlib import Path
import torch
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import os

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

    args = parser.parse_args()

    random.seed(1)

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
        "logreg": LogisticRegression,
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
        "logreg": {},
        "lda": {},
        "qda": {},
    }

    # shifted gaussian samples for class 1
    dim = theta.shape[-1]
    mean_shifts = [0, 0.3, 0.6, 1, 1.5, 2, 2.5, 3, 5, 10]
    scale_shifts = np.linspace(1, 20, 10)
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
        if os.path.exists(PATH / f"df_mean_exp_lc2st_n_{args.n_samples}.pkl"):
            df_mean = torch.load(PATH / f"df_mean_exp_lc2st_n_{args.n_samples}.pkl")
        else:
            clf_names = ["lda", "mlp_sbi", "mlp_base", "rf", "logreg"]

            dfs = []
            for clf_name in clf_names:
                shift_list, scores, accuracies, times = eval_classifier_for_lc2st(
                    x,
                    norm_samples,
                    shifted_samples=mean_shifted_samples,
                    shifts=mean_shifts,
                    clf_class=clf_classes[clf_name],
                    clf_kwargs=clf_kwargs_dict[clf_name],
                    metrics=["probas_mean"],
                )
                clf_method = [clf_name] * len(shift_list)
                dfs.append(
                    pd.DataFrame(
                        {
                            "mean_shift": shift_list,
                            "accuracy": accuracies,
                            "probas_mean": scores["probas_mean"],
                            "total_cv_time": times,
                            "classifier": clf_method,
                        }
                    )
                )
            df_mean = pd.concat(dfs, ignore_index=True)

            torch.save(
                df_mean,
                PATH / f"df_mean_exp_lc2st_n_{args.n_samples}.pkl",
            )

        sns.relplot(
            data=df_mean,
            x="mean_shift",
            y="probas_mean",
            hue="classifier",
            style="classifier",
            kind="line",
        )
        plt.savefig(PATH / f"lc2st_mean_shift_n_{args.n_samples}.pdf")
        plt.show()

    elif args.shift == "scale":
        if os.path.exists(PATH / f"df_scale_exp_lc2st_n_{args.n_samples}.pkl"):
            df_scale = torch.load(PATH / f"df_scale_exp_lc2st_n_{args.n_samples}.pkl")
        else:
            clf_names = ["qda", "mlp_sbi", "mlp_base", "rf", "logreg"]

            dfs = []
            for clf_name in clf_names:
                shift_list, scores, accuracies, times = eval_classifier_for_lc2st(
                    x,
                    norm_samples,
                    shifted_samples=scale_shifted_samples,
                    shifts=scale_shifts,
                    clf_class=clf_classes[clf_name],
                    clf_kwargs=clf_kwargs_dict[clf_name],
                    metrics=["probas_mean"],
                )
                clf_method = [clf_name] * len(shift_list)
                dfs.append(
                    pd.DataFrame(
                        {
                            "scale_shift": shift_list,
                            "accuracy": accuracies,
                            "probas_mean": scores["probas_mean"],
                            "total_cv_time": times,
                            "classifier": clf_method,
                        }
                    )
                )
            df_scale = pd.concat(dfs, ignore_index=True)

            torch.save(
                df_scale,
                PATH / f"df_scale_exp_lc2st_n_{args.n_samples}.pkl",
            )

        sns.relplot(
            data=df_scale,
            x="scale_shift",
            y="probas_mean",
            hue="classifier",
            style="classifier",
            kind="line",
        )
        plt.savefig(PATH / f"lc2st_scale_shift_n_{args.n_samples}.pdf")
        plt.show()

    else:
        print("Invalid experiment.")
