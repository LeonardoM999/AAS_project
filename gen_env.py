from random import random
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv, Overcooked
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld

# from overcooked_ai_py.utils import lossless_state_encoding


class GeneralizedOvercooked:
    def __init__(
        self,
        layouts=["cramped_room", "asymmetric_advantages"],
        info_level=0,
        horizon=400,
    ):
        self.envs = []
        # self.base_envs = []
        for layout in layouts:
            base_mdp = OvercookedGridworld.from_layout_name(layout)
            base_env = OvercookedEnv.frommdp(
                base_mdp, info_level=info_level, horizon=horizon
            )
            env = Overcooked(base_env, featurize_fn=base_env.featurize_state_mdp)
            self.envs.append(env)
            self.base_envs.append(base_env)
        self.cur_env = self.envs[0]
        # self.cur_base_env = self.base_envs[0]

    def reset(self):
        # TODO non uniform environment sampling
        idx = random.randint(0, len(self.envs) - 1)
        self.cur_env = self.envs[idx]
        # self.cur_base_env = self.base_envs[idx]
        state = self.cur_env.reset()
        return (
            self.cur_env.featurize_fn(state["overcooked_state"]),
            state["both_agent_obs"],
        )

    def step(self, actions):
        next_state, reward, episode_over, _info = self.cur_env.step(actions)
        return (
            next_state["overcooked_state"],
            next_state["both_agent_obs"],
            reward,
            episode_over,
            _info,
        )

    def render(self, *args):
        next_state, reward, episode_over, _info = self.cur_env.render(*args)

    def close(self):
        for e in self.envs:
            e.close()
