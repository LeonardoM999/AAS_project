import numpy as np
import argparse
from overcooked_ai_py.agents.agent import AgentPair
from overcooked_ai_py.visualization.state_visualizer import StateVisualizer
from overcooked_ai_py.agents.benchmarking import AgentEvaluator

# from human_aware_rl.imitation.behavior_cloning_tf2 import (    _get_base_ae, BehaviorCloningPolicy)
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld
from tqdm.auto import trange
import warnings

from agent_policy import dumbAgent

warnings.filterwarnings("ignore", category=UserWarning, module="pygame.pkgdata")


def normalize_reward(reward):
    warnings.warn("reward normalization not implemented")
    return reward


def clip_reward(reward):
    warnings.warni("reward clipping non implemented")
    return reward


# ARGUMENTS:
FRAMESKIP = 0  # number of skipped frames
MAX_LEN_EPISODE = 300  # max number of frames per episode
agent_name = "ppo"
save_path = f"../saves/{agent_name}"
lr = 2.3e-3
layout_name = "cramped_room"
n_steps = 250  # step = rollout+backprop
discount_gamma = 0.995
advantage_lambda = 0.95
epochs_per_episode = 3
epsilon_greedy = 0.02
val_Epsilon_greedy = 1.0
epsilon_clip = 0.2


# train agent
def train_agent():
    """"""
    base_mdp = OvercookedGridworld.from_layout_name(layout_name)
    base_env = OvercookedEnv.from_mdp(base_mdp, info_level=0, horizon=400)
    env = Overcooked(base_env=base_env, featurize_fn=base_env.featurize_state_mdp)

    print(env.action_space)
    my_agent = dumbAgent()

    score_history = np.zeros((MAX_LEN_EPISODE,))  #

    initial_state = env.reset()
    for step in trange(n_steps, colour="blue", desc="Training steps"):
        state = env.reset()
        # score = total_reward
        episode_over, score, n_steps = False, 0, 0

        for frame_n in trange(MAX_LEN_EPISODE):  # lenght of the episode is known/fixed
            action, prob, val = my_agent.act(state)
            next_state, reward, episode_over, _info = env.step(action)
            normalized_reward = normalize_reward(reward)
            clipped_reward = clip_reward(reward)

            n_steps += 1
            score += reward

            if frame_n % (FRAMESKIP + 1) == 0:
                # then don't skip this frame
                my_agent.store_step()
                if my_agent.buffer_full():
                    my_agent.learn()

            state = next_state

        # TODO add statistics

    my_agent.save()
    env.close()
    # print/plot statistics


# test agent
def test_agent():
    1


# test random agent
def test_random_agent():
    2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "train", nargs="?", help="path where to save trained models weights"
    )
    args = parser.parse_args()

    if args.train:
        train_agent()
    else:
        test_agent()


if __name__ == "__main__":
    main()
