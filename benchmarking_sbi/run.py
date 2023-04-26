import importlib
import logging
import random
import socket
import sys
import time

import hydra
import numpy as np
import pandas as pd
import sbibm
import torch
import yaml
from omegaconf import DictConfig, OmegaConf
from sbibm.utils.debug import pdb_hook
from sbibm.utils.io import (
    get_float_from_csv,
    get_tensor_from_csv,
    save_float_to_csv,
    save_tensor_to_csv,
)
from sbibm.utils.nflows import FlowWrapper

from utils import fwd_flow_transform_obs, inv_flow_transform_obs
import copy


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    log.info(f"sbibm version: {sbibm.__version__}")
    log.info(f"Hostname: {socket.gethostname()}")
    if cfg.seed is None:
        log.info("Seed not specified, generating random seed for replicability")
        cfg.seed = int(torch.randint(low=1, high=2**32 - 1, size=(1,))[0])
        log.info(f"Random seed: {cfg.seed}")
    save_config(cfg)

    # Seeding
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Devices
    gpu = True if cfg.device != "cpu" else False
    if gpu:
        torch.cuda.set_device(0)
        torch.set_default_tensor_type(
            "torch.cuda.FloatTensor" if gpu else "torch.FloatTensor"
        )

    # Paths
    path_samples = "posterior_samples.csv.bz2"
    path_runtime = "runtime.csv"
    path_log_prob_true_parameters = "log_prob_true_parameters.csv"
    path_num_simulations_simulator = "num_simulations_simulator.csv"
    path_predictive_samples = "predictive_samples.csv.bz2"

    # Run
    task = sbibm.get_task(cfg.task.name)
    t0 = time.time()
    parts = cfg.algorithm.run.split(".")
    module_name = ".".join(["sbibm", "algorithms"] + parts[:-1])
    run_fn = getattr(importlib.import_module(module_name), parts[-1])
    # print(run_fn)
    algorithm_params = cfg.algorithm.params if "params" in cfg.algorithm else {}
    log.info("Start run")
    outputs = run_fn(
        task,
        num_observation=cfg.task.num_observation,
        num_samples=cfg.task.num_posterior_samples,  # N_cal + N_v
        num_simulations=cfg.task.num_simulations,  # N_train
        **algorithm_params,
    )
    # print(outputs)
    runtime = time.time() - t0
    log.info("Finished run")

    # Store outputs
    if type(outputs) == torch.Tensor:
        samples = outputs
        num_simulations_simulator = float("nan")
        log_prob_true_parameters = float("nan")
        posterior_est = float("nan")
    elif type(outputs) == tuple and len(outputs) == 4:
        samples = outputs[0]
        num_simulations_simulator = float(outputs[1])
        log_prob_true_parameters = (
            float(outputs[2]) if outputs[2] is not None else float("nan")
        )
        posterior_est = outputs[-1]
    else:
        raise NotImplementedError
    save_tensor_to_csv(path_samples, samples, columns=task.get_labels_parameters())
    save_float_to_csv(path_runtime, runtime)
    save_float_to_csv(path_num_simulations_simulator, num_simulations_simulator)
    save_float_to_csv(path_log_prob_true_parameters, log_prob_true_parameters)
    torch.save(posterior_est, f"posterior_est.pkl")

    # Predictive samples
    log.info("Draw posterior predictive samples")
    simulator = task.get_simulator()
    predictive_samples = []
    batch_size = 1_000
    for idx in range(int(samples.shape[0] / batch_size)):
        try:
            predictive_samples.append(
                simulator(samples[(idx * batch_size) : ((idx + 1) * batch_size), :])
            )
        except:
            predictive_samples.append(
                float("nan") * torch.ones((batch_size, task.dim_data))
            )
    predictive_samples = torch.cat(predictive_samples, dim=0)
    save_tensor_to_csv(
        path_predictive_samples, predictive_samples, task.get_labels_data()
    )

    # Compute metrics
    if cfg.compute_metrics:
        df_metrics = compute_metrics_df(
            task_name=cfg.task.name,
            num_observation=cfg.task.num_observation,
            path_samples=path_samples,
            path_runtime=path_runtime,
            path_predictive_samples=path_predictive_samples,
            path_log_prob_true_parameters=path_log_prob_true_parameters,
            log=log,
            posterior_est=posterior_est,
        )
        df_metrics.to_csv("metrics.csv", index=False)
        log.info(f"Metrics:\n{df_metrics.transpose().to_string(header=False)}")


def save_config(cfg: DictConfig, filename: str = "run.yaml") -> None:
    """Saves config as yaml

    Args:
        cfg: Config to store
        filename: Filename
    """
    with open(filename, "w") as fh:
        yaml.dump(
            OmegaConf.to_container(cfg, resolve=True), fh, default_flow_style=False
        )


def compute_metrics_df(
    task_name: str,
    num_observation: int,
    path_samples: str,
    path_runtime: str,
    path_predictive_samples: str,
    path_log_prob_true_parameters: str,
    log: logging.Logger = logging.getLogger(__name__),
    posterior_est: FlowWrapper = None,
) -> pd.DataFrame:
    """Compute all metrics, returns dataframe

    Args:
        task_name: Task
        num_observation: Observation
        path_samples: Path to posterior samples
        path_runtime: Path to runtime file
        path_predictive_samples: Path to predictive samples
        path_log_prob_true_parameters: Path to NLTP
        log: Logger

    Returns:
        Dataframe with results
    """
    log.info(f"Compute all metrics")

    # Load task
    task = sbibm.get_task(task_name)
    # Load observation
    observation = task.get_observation(num_observation=num_observation)  # noqa
    # get prior and simulator
    prior = task.get_prior()
    simulator = task.get_simulator()

    # Load estimated posterior samples
    algorithm_posterior_samples = get_tensor_from_csv(path_samples)

    # Dataset sizes
    N_cal = len(algorithm_posterior_samples) // 2
    N_v = len(algorithm_posterior_samples) - N_cal
    log.info(f"Calibration dataset size: {N_cal} / Validation dataset size: {N_v}")

    ## ==== C2ST calibration dataset ==== ##
    # class 0: P ~ p_est(\theta | x_0)
    algorithm_posterior_samples = algorithm_posterior_samples[:N_cal, :]
    # class 1: Q ~ p_ref(\theta | x_0)
    reference_posterior_samples = task._sample_reference_posterior(
        num_samples=N_cal, num_observation=num_observation
    )

    assert reference_posterior_samples.shape[0] == N_cal
    assert algorithm_posterior_samples.shape[0] == N_cal
    log.info(f"Loaded / generated {N_cal} samples from reference / algorithm")

    # # Load posterior predictive samples
    # predictive_samples = get_tensor_from_csv(path_predictive_samples)[
    #     : task.num_posterior_samples, :
    # ]
    # assert predictive_samples.shape[0] == task.num_posterior_samples

    # ==== L-C2ST calibration dataset ==== #
    theta_cal = prior(num_samples=N_cal)
    x_cal = simulator(theta_cal)
    cal_set = {"theta": theta_cal, "x": x_cal}  # D_cal
    log.info(f"Generated {N_cal} samples from the joint")

    # Compute flow-posterior samples from x_cal
    est = copy.deepcopy(posterior_est)
    flow_posterior_samples_cal = []
    for x in x_cal:
        x = x[None, :]
        est.flow.set_default_x(x)
        z = est.flow.posterior_estimator._distribution.sample(1)
        z_transformed = fwd_flow_transform_obs(z, x, est.flow)
        flow_posterior_samples_cal.append(z_transformed)
    flow_posterior_samples_cal = torch.stack(flow_posterior_samples_cal)[:, 0, :]
    log.info(f"Computed Flow-posterior samples from x_cal")

    torch.save(cal_set, "calibration_dataset.pkl")
    torch.save(flow_posterior_samples_cal, "flow_posterior_samples_cal.pkl")

    # Inverse transform for (L)-C2ST-NF
    # on reference distribution
    posterior_est.flow.set_default_x(observation)
    inv_flow_samples_ref = inv_flow_transform_obs(
        reference_posterior_samples, observation, posterior_est.flow
    )
    # on cal set
    est = copy.deepcopy(posterior_est)
    inv_flow_samples_cal = []
    for theta, x in zip(theta_cal, x_cal):
        theta, x = theta[None, :], x[None, :]
        est.flow.set_default_x(x)
        theta_transformed = inv_flow_transform_obs(theta, x, est.flow)
        inv_flow_samples_cal.append(theta_transformed)
    inv_flow_samples_cal = torch.stack(inv_flow_samples_cal)[:, 0, :]

    # Base distribution samples for L-C2ST-NF
    base_dist_samples = posterior_est.flow.posterior_estimator._distribution.sample(
        N_cal
    )
    torch.save(inv_flow_samples_ref, "inv_flow_samples_ref.pkl")
    torch.save(inv_flow_samples_cal, "inv_flow_samples_cal.pkl")
    torch.save(base_dist_samples, "base_dist_samples.pkl")
    log.info(
        f"(L)C2ST_NF Data : Generated samples from the flow base distribution"
        + "\n Computed inverse flow transform on the joint and reference posterior"
    )

    # ==== (L)-C2ST evaluation dataset ==== #
    # class 0: P_eval ~ p_est(\theta | x_0) - for C2ST and L-C2ST
    algorithm_posterior_samples_eval = algorithm_posterior_samples[N_v:, :]
    # class 1: Q_eval ~ p_ref(\theta | x_0) - only for C2ST
    reference_posterior_samples_eval = task._sample_reference_posterior(
        num_samples=N_v, num_observation=num_observation
    )
    log.info(f"Loaded {N_v} samples from reference and algorithm (for non cross-val).")

    # Inverse transform for (L)-C2ST-NF
    # class 0: P_eval ~ N(0, I) - for C2ST and L-C2ST
    base_dist_samples_eval = (
        posterior_est.flow.posterior_estimator._distribution.sample(N_v)
    )
    # class 1: Q_eval ~ p(T^{-1}(\theta;x_0) | x_0) - only for C2ST-NF
    posterior_est.flow.set_default_x(observation)
    inv_flow_samples_ref_eval = inv_flow_transform_obs(
        reference_posterior_samples_eval, observation, posterior_est.flow
    )
    log.info(
        f"Computed the inverse flow transform on the reference posterior samples (for non cross-val)."
    )

    # base mlp classifier
    from sklearn.neural_network import MLPClassifier

    mlp_base = MLPClassifier(alpha=0, max_iter=250000)

    # Get runtime info
    runtime_sec = torch.tensor(get_float_from_csv(path_runtime))  # noqa

    # Get log prob true parameters
    log_prob_true_parameters = torch.tensor(
        get_float_from_csv(path_log_prob_true_parameters)
    )  # noqa

    # Names of all metrics as keys, values are calls that are passed to eval
    # NOTE: Originally, we computed a large number of metrics, as reflected in the
    # dictionary below. Ultimately, we used 10k samples and z-scoring for C2ST but
    # not for MMD. If you were to adapt this code for your own pipeline of experiments,
    # the entries for C2ST_Z, MMD and RT would probably suffice (and save compute).
    _METRICS_ = {
        #
        # 10k samples
        #
        # "C2ST": "metrics.c2st(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=False)",
        # "C2ST_NF": "metrics.c2st(X=inv_flow_samples_ref, Y=base_dist_samples, z_score=False)",
        # sbibm reference metric - cross_val and z_score
        "C2ST_Z_CV": "metrics.c2st(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=True)",
        "C2ST_NF_Z_CV": "metrics.c2st(X=inv_flow_samples_ref, Y=base_dist_samples, z_score=True)",
        # vanilla c2st ensemble + out-of-sample, fixed eval_set - not cross_val
        "C2ST": "c2st_sbibm(P=reference_posterior_samples, Q=algorithm_posterior_samples, metric='accuracy', n_ensemble=10, cross_val=False, P_eval=reference_posterior_samples_eval, Q_eval=algorithm_posterior_samples_eval)",
        "C2ST_NF": "c2st_sbibm(P=inv_flow_samples_ref, Q=base_dist_samples, metric='accuracy', n_ensemble=10, cross_val=False, P_eval=inv_flow_samples_ref_eval, Q_eval=base_dist_samples_eval)",
        # reg-c2st ensemble + out-of-sample, fixed eval_set (instead of insample) - not cross_val
        "C2ST_REG": "c2st_sbibm(P=reference_posterior_samples, Q=algorithm_posterior_samples, metric='mse', n_ensemble=10, cross_val=False, P_eval=reference_posterior_samples_eval, Q_eval=algorithm_posterior_samples_eval)",
        "C2ST_REG_NF": "c2st_sbibm(P=inv_flow_samples_ref, Q=base_dist_samples, metric='mse', n_ensemble=10, cross_val=False, P_eval=inv_flow_samples_ref_eval, Q_eval=base_dist_samples_eval)",
        #
        "MMD": "metrics.mmd(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=False)",
        # "MMD_Z": "metrics.mmd(X=reference_posterior_samples, Y=algorithm_posterior_samples, z_score=True)",
        # "KSD_GAUSS": "metrics.ksd(task=task, num_observation=num_observation, samples=algorithm_posterior_samples, sig2=float(torch.median(torch.pdist(reference_posterior_samples))**2), log=False)",
        # "MEDDIST": "metrics.median_distance(predictive_samples, observation)",
        # lc2st cv
        "LC2ST_CV": "lc2st_sbibm(P=flow_posterior_samples_cal, Q=theta_cal, x_P=x_cal, x_Q=x_cal, x_eval=observation, metric='accuracy', P_eval=algorithm_posterior_samples, n_folds=10, cross_val=True)",
        "LC2ST_NF_CV": "lc2st_sbibm(P=base_dist_samples, Q=inv_flow_samples_cal, x_P=x_cal, x_Q=x_cal, x_eval=observation, metric='accuracy', P_eval=base_dist_samples_eval, n_folds=10, cross_val=True)",
        "LC2ST_REG_CV": "lc2st_sbibm(P=flow_posterior_samples_cal, Q=theta_cal, x_P=x_cal, x_Q=x_cal, x_eval=observation, metric='mse', P_eval=algorithm_posterior_samples, n_folds=10, cross_val=True)",
        "LC2ST_NF_REG_CV": "lc2st_sbibm(P=base_dist_samples, Q=inv_flow_samples_cal, x_P=x_cal, x_Q=x_cal, x_eval=observation, metric='mse', P_eval=base_dist_samples_eval, n_folds=10, cross_val=True)",
        # lc2st ensemble
        "LC2ST": "lc2st_sbibm(P=flow_posterior_samples_cal, Q=theta_cal, x_P=x_cal, x_Q=x_cal, x_eval=observation, metric='accuracy', P_eval=algorithm_posterior_samples, n_ensemble=10, cross_val=False)",
        "LC2ST_NF": "lc2st_sbibm(P=base_dist_samples, Q=inv_flow_samples_cal, x_P=x_cal, x_Q=x_cal, x_eval=observation, metric='accuracy', P_eval=base_dist_samples_eval, n_ensemble=10, cross_val=False)",
        "LC2ST_REG": "lc2st_sbibm(P=flow_posterior_samples_cal, Q=theta_cal, x_P=x_cal, x_Q=x_cal, x_eval=observation, metric='mse', P_eval=algorithm_posterior_samples, n_ensemble=10, cross_val=False)",
        "LC2ST_REG_NF": "lc2st_sbibm(P=base_dist_samples, Q=inv_flow_samples_cal, x_P=x_cal, x_Q=x_cal, x_eval=observation, metric='mse', P_eval=base_dist_samples_eval, n_ensemble=10, cross_val=False)",
        #
        # # 1K samples
        # #
        # "C2ST_1K": "metrics.c2st(X=reference_posterior_samples[:1000,:], Y=algorithm_posterior_samples[:1000,:], z_score=False)",
        # "C2ST_1K_Z": "metrics.c2st(X=reference_posterior_samples[:1000,:], Y=algorithm_posterior_samples[:1000, :], z_score=True)",
        # "MMD_1K": "metrics.mmd(X=reference_posterior_samples[:1000,:], Y=algorithm_posterior_samples[:1000, :], z_score=False)",
        # "MMD_1K_Z": "metrics.mmd(X=reference_posterior_samples[:1000,:], Y=algorithm_posterior_samples[:1000, :], z_score=True)",
        # "KSD_GAUSS_1K": "metrics.ksd(task=task, num_observation=num_observation, samples=algorithm_posterior_samples[:1000, :], sig2=float(torch.median(torch.pdist(reference_posterior_samples))**2), log=False)",
        # "MEDDIST_1K": "metrics.median_distance(predictive_samples[:1000,:], observation)",
        # #
        # # Not based on samples
        # #
        # "NLTP": "-1. * log_prob_true_parameters",
        "RT": "runtime_sec",
    }

    import sbibm.metrics as metrics  # noqa
    from valdiags.localC2ST_old import lc2st_sbibm as lc2st_sbibm_old
    from valdiags.vanillaC2ST import c2st_sbibm
    from valdiags.localC2ST import lc2st_sbibm

    metrics_dict = {}
    for metric, eval_cmd in _METRICS_.items():
        log.info(f"Computing {metric}")
        t0 = time.time()
        try:
            metrics_dict[metric] = eval(eval_cmd).cpu().numpy().astype(np.float32)
            log.info(f"{metric}: {metrics_dict[metric]}")
        except:
            metrics_dict[metric] = float("nan")
        runtime = time.time() - t0
        log.info(f"Runtime {metric}: {runtime}")
        metrics_dict["runtime_" + metric] = runtime
    return pd.DataFrame(metrics_dict)


def cli():
    if "--debug" in sys.argv:
        sys.excepthook = pdb_hook
    main()


if __name__ == "__main__":
    cli()
