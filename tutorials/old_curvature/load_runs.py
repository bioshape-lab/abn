"""Load wandb runs and their attributes."""

import json
import os

import pandas as pd

CONFIG_DIR = os.path.join(os.getcwd(), "results", "configs")
CURVATURE_PROFILES_DIR = os.path.join(os.getcwd(), "results", "curvature_profiles")


class AttrDict(dict):
    """Convert a dict into an object where attributes are accessed with "."

    This is needed for the utils.load() function.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def config_from_run_id(run_id):
    """Load config object associated to a wandb run.

    Parameters
    ----------
    run_id : str
        ID of the run.

    Returns
    -------
    config : AttrDict
        Config object.
    """
    for config_file in os.listdir(CONFIG_DIR):
        if run_id in config_file:
            # print(f"Found: {config_file}")
            run_id_config_file = config_file
    with open(os.path.join(CONFIG_DIR, run_id_config_file)) as f:
        config_dict = json.load(f)
    return AttrDict(config_dict)


def curvature_profiles_from_run_id(run_id, config):
    """Load curvature profiles associated to a wandb run.

    Parameters
    ----------
    run_id : str
        ID of the run.

    Returns
    -------
    learned_profile : pd.DataFrame
        Learned curvature profile (norms).
    true_profile : pd.DataFrame
        True curvature profile (norms).
    """
    for curv_path in os.listdir(CURVATURE_PROFILES_DIR):
        if (run_id in curv_path) and ("learned" in curv_path or "true" in curv_path):
            pass
    learned_profile = pd.read_csv(
        os.path.join(
            CURVATURE_PROFILES_DIR,
            config.results_prefix + "_curv_norm_learned_profile.csv",
        )
    )
    true_profile = pd.read_csv(
        os.path.join(
            CURVATURE_PROFILES_DIR,
            config.results_prefix + "_curv_norm_true_profile.csv",
        )
    )
    return learned_profile, true_profile
