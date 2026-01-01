from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

import numpy as np
import argparse
import warnings
from gen_env import GeneralizedOvercooked
from agent_policy import DumbAgent

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")


# ARGUMENTS:
MAX_LEN_EPISODE = 400  # max number of frames per episode
agent_name = "ppo"
save_path = f"../saves/{agent_name}"
discount_gamma = 0.995
advantage_lambda = 0.95
epochs_per_episode = 5
epsilon_clip = 0.2
entropy_factor = 0.01  # TODO increase
lr_policy = 3e-4
lr_critic = 1e-3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train", nargs="?", help="path where to save trained models weights")
    args = parser.parse_args()

    if args.train:
        train_agent()
    else:
        test_agent()


if __name__ == "__main__":
    main()
