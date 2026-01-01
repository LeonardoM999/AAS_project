from math import exp
import keras
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import time
import glob, os
from tqdm.auto import trange
import tensorflow_probability as tfp
from collections import deque

from gen_env import GeneralizedOvercooked

POLICY_HIDDEN_SIZES = (256, 128)
CRITIC_HIDDEN_SIZES = (128, 64)
DISCOUNT_GAMMA = 0.99  # 0.95
ADVANTAGE_LAMBDA = 0.98  # 0.95
# ACTIVATION="tanh"
ACTIVATION = "leaky_relu"
# ACTIVATION = "relu"
BATCH_SIZE = 256
N_EPOCHS = 5
PPO_EPS = 0.3
ENTROPY_FACTOR = 0.05  # 0.01 0.05
LR_POLICY = 3e-4
LR_CRITIC = 1e-3
CRITIC_COEFF = 0.5  # 0.01
MAX_GRAD_NORM = 0.5
USE_LR_DECAY = False
SCALE = 1.0  # 0.1 , 0.5, 1.0, 2.0, 5.0
STANDARDIZE_NN_INPUTS = False
STANDARDIZE_ADV = True


class TrainingLogger:
    def __init__(self, window_size=100):
        self.episode_rewards = deque(maxlen=window_size)
        self.policy_losses = []
        self.value_losses = []
        self.kl_divs = []
        self.entropy_values = []
        self.explained_variances = []

    def log_episode(self, reward):
        self.episode_rewards.append(reward)

    def log_training(self, policy_loss, value_loss, kl_div, entropy, explained_var):
        self.policy_losses.append(policy_loss)
        self.value_losses.append(value_loss)
        self.kl_divs.append(kl_div)
        self.entropy_values.append(entropy)
        self.explained_variances.append(explained_var)

    def print_stats(self, episode_n):
        if len(self.episode_rewards) == 0:
            return

        print(f"\n{'='*60}")
        print(f"Episode {episode_n}")
        print(
            f"Average Reward (last {len(self.episode_rewards)}): {np.mean(self.episode_rewards):.2f} ± {np.std(self.episode_rewards):.2f}"
        )
        print(f"Average Length: {np.mean(self.episode_lengths):.1f}")

        if len(self.policy_losses) > 0:
            print(f"Policy Loss: {np.mean(self.policy_losses[-10:]):.4f}")
            print(f"Value Loss: {np.mean(self.value_losses[-10:]):.4f}")
            print(f"KL Divergence: {np.mean(self.kl_divs[-10:]):.6f}")
            print(f"Entropy: {np.mean(self.entropy_values[-10:]):.4f}")
            print(f"Explained Variance: {np.mean(self.explained_variances[-10:]):.4f}")
        print(f"{'='*60}\n")


class MyBuffer:
    def __init__(
        self,
        buffer_size,
        obs_shape=96,
        state_shape=192,
        # device="/CPU:0",
        device="/GPU:0",
    ):
        """statically allocated gpu/cpu tensors where dim0 is batch/buffer size.
        use a cursor to tell up to where it's full
        assumption: (environment_state).shape == (agent_observation).shape"""

        assert device in ("/GPU:0", "/CPU:0"), "Wrong device name"
        if device == "/CPU:0":
            print("USING CPU memory for minibatch tensors")
        else:
            print("using GPU memory for minibatch tensors")

        self.buffer_size = buffer_size
        self.cursor = 0
        self.device = device

        # TODO controllare che anche reward debba essere per forza un tensore

        # use cpu memory
        # (cuda)malloc tensors
        # assume: state_shape = obs_shape
        # self.states = tf.Variable(
        #     tf.zeros((buffer_size, *obs_shape), dtype=tf.float32)
        # )
        # TODO check state shape is correct

        # batch:
        self.states = np.zeros((buffer_size, state_shape))
        self.action0s = np.zeros((buffer_size))
        self.action1s = np.zeros((buffer_size))
        self.obs0s = np.zeros((buffer_size, obs_shape))
        self.obs1s = np.zeros((buffer_size, obs_shape))
        self.logp0s = np.zeros((buffer_size,))
        self.logp1s = np.zeros((buffer_size,))
        self.advantages0 = np.zeros((buffer_size,))
        self.advantages1 = np.zeros((buffer_size,))
        self.expected_returns0 = np.zeros((buffer_size,))
        self.expected_returns1 = np.zeros((buffer_size,))

        # for adv/exp returns
        self.reward0s = np.zeros((buffer_size,))
        self.reward1s = np.zeros((buffer_size,))
        self.episode_overs = np.zeros((buffer_size,)) - 0.5
        self.values = np.zeros((buffer_size))

    def store_smth(self, state, obss, actions, rewards, episode_over, value, logps):
        i = self.cursor
        # self.states[i].assign(tf.convert_to_tensor(state, dtype=tf.float32))
        self.states[i] = state
        self.action0s[i] = actions[0]
        self.action1s[i] = actions[1]
        self.obs0s[i] = obss[0]
        self.obs1s[i] = obss[1]
        self.reward0s[i] = rewards[0]
        self.reward1s[i] = rewards[1]
        self.episode_overs[i] = episode_over
        self.values[i] = value
        self.logp0s[i] = logps[0]
        self.logp1s[i] = logps[1]

        self.cursor += 1

    def make_batches(self, batch_size):
        # concatenate data, keep everythong sync
        batch_data = [
            np.concatenate((a[: self.cursor], b[: self.cursor]))
            for (a, b) in [
                (self.states, self.states),
                (self.obs0s, self.obs1s),
                (self.action0s, self.action1s),
                (self.logp0s, self.logp1s),
                (self.advantages0, self.advantages1),
                (self.expected_returns0, self.expected_returns1),
            ]
        ]
        double_cursor = self.cursor * 2

        # (
        #    double_states,
        #    double_obss,
        #    double_actions,
        #    double_logps,
        #    double_advantages,
        #    double_expected_returns,
        # ) = batch_data

        if STANDARDIZE_ADV:
            # standardize advantages at batch level VS AT MINIBATCH LEVEL
            advs = batch_data[4]
            advs_std = (advs - np.average(advs)) / (np.std(advs) + 1e-10)
            batch_data[4] = advs_std

        if STANDARDIZE_NN_INPUTS:
            # standardize states
            states = batch_data[0]
            states_std = (states - np.average(states)) / (np.std(states) + 1e-10)
            batch_data[0] = states_std

            # standardize observations:
            obs = batch_data[1]
            obs_std = (obs - np.average(obs)) / (np.std(obs) + 1e-10)
            batch_data[1] = obs_std

        if True:
            # check all lenghts but allocates all the memory eagerly
            lens = np.array([len(l) for l in batch_data])
            assert np.all(lens == lens[0]), f"data is out of sync: lengths don't match: {lens}"
        # endif

        # i shuffle the trajiectories because i have already computed gae and i am not using any rnn/lstm
        idxs = np.arange(double_cursor)
        np.random.shuffle(idxs)
        idxs = [
            idxs[i * batch_size : (i + 1) * batch_size]
            for i in np.arange((double_cursor // batch_size) + (1 if double_cursor % batch_size else 0))
        ]
        # i have partitioned indices into groups, now i return a list of batches
        # each batch is a dict with the data of his indices group, data is put into tensors
        ## with tf.device(self.device):  # use gpu if possible
        ##    result = (
        ##        {
        ##            k: tf.convert_to_tensor(np.array(v)[idx_group], dtype=tf.float32)
        ##            for k, v, in zip(
        ##                ("state", "obs", "action", "logp", "advantage", "expected_return"),
        ##                batch_data,
        ##            )
        ##        }
        ##        for idx_group in idxs
        ##    )

        list_of_batches = [
            {
                k: np.array(v)[idx_group]
                for k, v, in zip(
                    ("state", "obs", "action", "logp", "advantage", "expected_return"),
                    batch_data,
                )
            }
            for idx_group in idxs
        ]
        # if STANDARDIZE_NN_INPUTS:
        #    # at minibatch level instead of batch level
        #    for batch in list_of_batches:
        #        avg_adv = np.average(batch["advantage"])
        #        std_adv = np.std(batch["advantage"])
        #        batch["advantage"] = (batch["advantage"] - avg_adv) / (std_adv + 1e-10)

        with tf.device(self.device):  # use gpu if possible
            # I hope i cuda memcpy the second batch while the first is training by using ()generator instead of []list
            result_tf = (
                {k: tf.convert_to_tensor(v, dtype=tf.float32) for k, v, in batch.items()}
                for batch in list_of_batches
            )
        # end with

        return result_tf

    def clear_memory(self):
        """no need to do anything with the memory, just fill from the beginning"""
        self.cursor = 0

    def is_full(self):
        return self.cursor == self.buffer_size  # - 1

    def compute_gae(self, gamma=DISCOUNT_GAMMA, lam=ADVANTAGE_LAMBDA):
        # I assume that the last element in the buffer is a terminal state
        assert self.episode_overs[self.cursor - 1] > 0, "Last step in the buffer is not a terminal state"
        buffersize = self.cursor
        gae0 = 0
        gae1 = 0
        for t in reversed(range(buffersize)):
            if self.episode_overs[t]:
                gae0, gae1 = 0, 0
                next_val = 0
            if t == buffersize - 1:
                # last next_val=0 because episode ends at buffer end
                next_val = 0
            else:
                # if episode terminates: next_val=0
                next_val = self.values[t + 1] * (1 - self.episode_overs[t])
            # TODO: check che sia corretto
            expected_return0 = self.reward0s[t] + gamma * next_val
            delta0 = expected_return0 - self.values[t]
            gae0 = delta0 + gamma * lam * gae0  # *(1 - self.episode_overs[t])
            self.advantages0[t] = gae0
            self.expected_returns0[t] = expected_return0  # gae0 + self.values[t]

            expected_return1 = self.reward1s[t] + gamma * next_val
            delta1 = expected_return1 - self.values[t]
            gae1 = delta1 + gamma * lam * gae1  # *(1 - self.episode_overs[t])
            self.advantages1[t] = gae1
            self.expected_returns1[t] = expected_return1  # gae1 + self.values[t]

    def print_std(self):
        pred_values = np.concatenate((self.values[: self.cursor], self.values[: self.cursor]))
        exp_returns = np.concatenate(
            (self.expected_returns0[: self.cursor], self.expected_returns1[: self.cursor])
        )
        # print(f"avg pred value {np.mean(pred_values)} avg exp return {np.mean(exp_returns)}")
        print(f"std pred value {np.std(pred_values)} std exp return {np.std(exp_returns)}")
        advantages = np.concatenate([self.advantages0[: self.cursor], self.advantages1[: self.cursor]])
        print(f"Advantage std: {np.std(advantages):.6f}")
        # print(f"Advantage mean: {np.mean(advantages):.6f}")
        # print(f"Advantage range: [{np.min(advantages):.4f}, {np.max(advantages):.4f}]")


class DensePolicyNN(Model):

    def __init__(self, hidden_sizes, n_actions, activation):
        assert (
            hidden_sizes is not None and len(hidden_sizes) > 0
        ), "policy network needs at least one hidden size"
        assert activation is not None, "missing activation"
        super().__init__()
        # self.flatten = layers.Flatten()
        self.hidden_sizes = hidden_sizes
        self.n_actions = n_actions
        self.activation = activation
        self.denses = [
            layers.Dense(
                hsize,
                activation=self.activation,
                kernel_initializer=keras.initializers.Orthogonal(np.sqrt(2)),
                bias_initializer=keras.initializers.Constant(0),
            )
            for hsize in self.hidden_sizes
        ]
        self.logits_layer = layers.Dense(
            self.n_actions,
            activation=None,
            dtype=tf.float32,
            kernel_initializer=keras.initializers.Orthogonal(0.01),
            bias_initializer=keras.initializers.Constant(0),
        )  # raw logits

    def call(self, inputs):
        # x = self.flatten(inputs)
        x = inputs
        for d in self.denses:
            x = d(x)
        return self.logits_layer(x)  # return logits

    def get_config(self):
        """Return config for model serialization"""
        return {"hidden_sizes": self.hidden_sizes, "n_actions": self.n_actions, "activation": self.activation}

    @classmethod
    def from_config(cls, config):
        """Create model from config"""
        return cls(**config)


class DenseValueNN(Model):
    def __init__(self, hidden_sizes, activation):
        assert (
            hidden_sizes is not None and len(hidden_sizes) > 0
        ), "Value/Critic network needs at least one hidden size"
        assert activation is not None, "missing activation"
        super().__init__()
        self.activation = activation
        self.hidden_sizes = hidden_sizes
        # self.flatten = layers.Flatten()
        self.denses = [
            layers.Dense(
                hsize,
                activation=self.activation,
                kernel_initializer=keras.initializers.Orthogonal(np.sqrt(2)),
                bias_initializer=keras.initializers.Constant(0),
            )
            for hsize in self.hidden_sizes
        ]
        self.value = layers.Dense(
            1,
            activation=None,
            kernel_initializer=keras.initializers.Orthogonal(1),
            bias_initializer=keras.initializers.Constant(0),
        )  # value scalar head

    def call(self, inputs):
        # x = self.flatten(inputs)
        x = inputs
        for d in self.denses:
            x = d(x)
        return self.value(x)  # shape (batch, 1)

    def get_config(self):
        """Return config for model serialization"""
        return {"hidden_sizes": self.hidden_sizes, "activation": self.activation}

    @classmethod
    def from_config(cls, config):
        """Create model from config"""
        return cls(**config)


# TODO remove
class ConvPolicyNN(Model):
    def __init__(
        self,
        sizes=((3, 16), (3, 16)),
        hidden_sizes=(64,),
        n_actions=6,
    ):
        super().__init__()
        self.input = layers.Input(shape=(None, None, 26))
        self.convs = [
            layers.Conv2D(
                filters=ff,
                kernel_size=(ks, ks),
                padding="same",
                activation="leaky_relu",
            )
            for (ks, ff) in sizes
        ]
        self.denses = [layers.Dense(hsize, activation="leaky_relu") for hsize in hidden_sizes]
        self.logits_layer = layers.Dense(n_actions)  # raw logits

    def call(self, inputs):
        x = inputs  # TODO shape
        for c in self.convs:
            x = c(x)
        x = layers.Flatten()(x)  # TODO:error: not constant shape for denses

        for d in self.denses:
            x = d(x)
        return self.logits_layer(x)  # return logits


class DumbAgent:
    """MAPPO agent"""

    def __init__(self, buffer_len, lr_policy=LR_POLICY, lr_critic=LR_CRITIC, decay=USE_LR_DECAY):
        self.buffer = MyBuffer(buffer_len)
        self.value_nn = DenseValueNN(hidden_sizes=CRITIC_HIDDEN_SIZES, activation=ACTIVATION)
        self.policy_nn = DensePolicyNN(hidden_sizes=POLICY_HIDDEN_SIZES, n_actions=6, activation=ACTIVATION)
        self.policy_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_policy, decay_steps=1000, decay_rate=0.9
        )
        self.critic_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=lr_critic, decay_steps=1000, decay_rate=0.9
        )
        self.policy_optimizer = tf.keras.optimizers.Adam(self.policy_schedule if decay else lr_policy)
        self.critic_optimizer = tf.keras.optimizers.Adam(self.critic_schedule if decay else lr_critic)
        self.logger = TrainingLogger()

    def store_step(self, state, obss, actions, rewards, episode_over, value, logprobs):
        self.buffer.store_smth(state, obss, actions, rewards, episode_over, value, logprobs)

    def save(self, path="weights"):
        t = time.time()
        ckpoint_id = int((t * 10) % 1e6)
        lt = time.localtime(t)
        self.policy_nn.save(f"{path}/policy_{lt[3]}_{lt[4]}_{lt[5]}_{ckpoint_id}.keras")
        self.value_nn.save(f"{path}/value_{lt[3]}_{lt[4]}_{lt[5]}_{ckpoint_id}.keras")

    def _get_ckpoint_id(self, path):
        name = os.path.splitext(os.path.basename(path))[0]  # remove type
        parts = name.split("_")
        return int(parts[-1])  # last token is ckpoint_id

    def load_recent(self, path="weights", ckpoint_id=None):
        """loads the most recent training checkpoint or the one specified with ckpoint_id"""
        # find all suitable files
        pol_files = glob.glob(f"{path}/policy_*.keras")
        val_files = glob.glob(f"{path}/value_*.keras")
        if pol_files is None or val_files is None:
            raise RuntimeError("\n no weight files in the path \n")
        if ckpoint_id is not None:  # load a specific ckpoint
            val_f = [f for f in val_files if self._get_ckpoint_id(f) == ckpoint_id][-1]
            pol_f = [f for f in pol_files if self._get_ckpoint_id(f) == ckpoint_id][-1]
        else:  # load the most recent
            pol_f = max(pol_files, key=self._get_ckpoint_id)
            val_f = max(val_files, key=self._get_ckpoint_id)
        print(pol_f)
        print(val_f)
        self.policy_nn = keras.models.load_model(pol_f, custom_objects={"DensePolicyNN": DensePolicyNN})
        self.value_nn = keras.models.load_model(val_f, custom_objects={"DenseValueNN": DenseValueNN})

    def sample_action(self, logits):
        # Sample directly from logits
        # logits = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        # action = tf.distributions.categorical(logits, 1).numpy().flatten()
        dist = tfp.distributions.Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)  # optional
        return action, log_prob

    def act(self, obs0, obs1, state):
        """
        Args:
            state: The current state in featurized encoding
            obs0,obs1: observations in featurized encoding

        Returns: Action, log prob of action, state value.
        """
        # make tensor of state
        state = np.expand_dims(state, axis=0)  # add dim for (mini)batchsize
        state = tf.convert_to_tensor(state, dtype=tf.float32)

        # get critic value
        critic_value = self.value_nn(state)

        # make tensor of obs
        obs0, obs1 = [np.expand_dims(o, axis=0) for o in (obs0, obs1)]  # add dim for (mini) batchsize
        obs0, obs1 = [tf.convert_to_tensor(o, dtype=tf.float32) for o in (obs0, obs1)]

        # get logits probs for actions
        logits0 = self.policy_nn(obs0)
        logits1 = self.policy_nn(obs1)
        action0, log_prob0 = self.sample_action(logits0)
        action1, log_prob1 = self.sample_action(logits1)

        return [tf.squeeze(a).numpy() for a in (action0, action1)], (log_prob0, log_prob1), critic_value

    # train step
    def learn(self, n_epochs=N_EPOCHS, ppo_eps=PPO_EPS, entropy_factor=ENTROPY_FACTOR):
        self.buffer.print_std()
        for _ in range(n_epochs):  # trange(n_epochs, desc="epoch n"):
            for batch in self.buffer.make_batches(batch_size=BATCH_SIZE):
                # Policy loss (in the batch there is data from both agents)
                with tf.GradientTape() as policy_tape:
                    new_logits = self.policy_nn(batch["obs"])
                    new_dist = tfp.distributions.Categorical(logits=new_logits)
                    new_logp = new_dist.log_prob(batch["action"])  # gpu tensor
                    ratio = tf.exp(new_logp - batch["logp"])  # gpu tensor
                    # print(f"new_logp {type(new_logp)}")
                    # print(f"ratio   {len(ratio)} {type(ratio)}")
                    # print(f"batch advantage  {len(batch['advantage'])} {(batch['advantage']).device}")

                    policy_loss = -tf.reduce_mean(  # minus because we maximize
                        tf.minimum(
                            ratio * batch["advantage"],
                            (tf.clip_by_value(ratio, 1 - ppo_eps, 1 + ppo_eps)) * batch["advantage"],
                        ),
                        axis=None,  # TODO check axis
                    )

                    # entropy loss:
                    # mean on batch dimension
                    entropy_loss = -tf.reduce_mean(new_dist.entropy())  # minus because we maximize
                    # addition (+) because both are already negated
                    policy_loss = policy_loss + entropy_factor * entropy_loss
                # apply policy loss:
                policy_grads = policy_tape.gradient(policy_loss, self.policy_nn.trainable_variables)
                pol_grad_norm = tf.linalg.global_norm(policy_grads)
                policy_grads, _ = tf.clip_by_global_norm(policy_grads, clip_norm=MAX_GRAD_NORM)
                self.policy_optimizer.apply_gradients(zip(policy_grads, self.policy_nn.trainable_variables))

                with tf.GradientTape() as critic_tape:
                    # Critic loss: (in the batch there is data from both agents)
                    value_preds = self.value_nn(batch["state"])
                    critic_loss = tf.keras.losses.MSE(value_preds, batch["expected_return"]) * CRITIC_COEFF
                # apply critic loss:
                critic_grads = critic_tape.gradient(critic_loss, self.value_nn.trainable_variables)
                critic_grads, _ = tf.clip_by_global_norm(critic_grads, clip_norm=MAX_GRAD_NORM)
                self.critic_optimizer.apply_gradients(zip(critic_grads, self.value_nn.trainable_variables))
                critic_grad_norm = tf.linalg.global_norm(critic_grads)

        return pol_grad_norm, critic_grad_norm

    # train agent
    def train(self, env: GeneralizedOvercooked, n_episodes=1000):
        epoch_n = 0
        score_history = []
        policy_gradient_norm_history = []
        critic_gradient_norm_history = []

        for episode_n in trange(n_episodes, colour="blue", desc="Episodes"):
            # while epoch_n < n_epochs:
            next_state, next_obss = env.reset()
            score = 0
            episode_over, score = False, 0

            while not episode_over:
                # for _ in range(MAX_LEN_EPISODE) #the episode len is always the same
                state, obss = next_state, next_obss
                if np.allclose(obss[0], obss[1], atol=0.01):
                    print(f"OBSERVATIONS ARE CLOSE")

                actions, logprobs, critic_val = self.act(obss[0], obss[1], state)
                next_state, next_obss, reward, episode_over, info = env.step(actions)
                score += reward
                # clipped_reward = clip_reward(reward)
                shaped_rewards = (
                    SCALE * np.array(info["shaped_r_by_agent"]) + reward
                )  # TODO anche senza + reward

                # if sum(shaped_rewards) != 0:
                #    print(f"{actions}  {shaped_rewards}")

                self.store_step(
                    state,
                    obss,
                    actions,
                    shaped_rewards,
                    # (reward, reward), #così anche peggio
                    episode_over,
                    critic_val,
                    logprobs,
                )
                if self.buffer.is_full():
                    epoch_n += 1
                    self.buffer.compute_gae()
                    pol_grad_norm, crit_grad_norm = self.learn()
                    self.buffer.clear_memory()
                    policy_gradient_norm_history.append(pol_grad_norm)
                    critic_gradient_norm_history.append(crit_grad_norm)
                    if pol_grad_norm > 0.49 or crit_grad_norm > 0.49:
                        print(f"pol grad norm {pol_grad_norm}   crit_grad norm {crit_grad_norm}")

            # end while not episode over
            score_history.append(score)
            if episode_n % 20 == 0 or score:
                print(f"EPISODE {episode_n}:\t SCORE = {score}")
            if 1 + episode_n % 100 == 0:
                self.save()
        # end for training steps
        if self.buffer.cursor >= 100:
            # make sure not to throw away good data
            policy_gradient_norm_history.append(pol_grad_norm)
            critic_gradient_norm_history.append(crit_grad_norm)

        # print(policy_gradient_norm_history)

        self.save()
        env.close()
        # print/plot statistics

    # test agent
    def test_agent():
        1
