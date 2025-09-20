from tensorflow.keras.models import Model  # type: ignore
from tensorflow.keras.layers import Dense, Dropout, Layer
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import warnings
import time
import glob, os


class MyBuffer:
    def __init__(
        self,
        obs_shape,
        action_shape,
        prob_shape,
        value_shape=1,
        buffer_size=2000,
        device="/CPU:0",
    ):
        """statically allocated gpu/cpu tensors where dim0 is batch/buffer size.
        use a cursor to tell up to where it's full
        assumption: (environment_state).shape == (agent_observation).shape"""

        assert (device in ("/GPU:0", "/CPU:0"), "Wrong device name")
        if device == "/CPU:0":
            print("USING CPU memory for buffer tensors")
        else:
            print("using GPU memory for buffer tensors")

        self.buffer_size = buffer_size
        self.cursor = 0
        self.device = device

        # TODO controllare che anche reward debba essere per forza un tensore
        # TODO controllare che values abbiano quella dim a 1 in più e che non siano come il reward (con una dim in meno)

        with tf.device(self.device):
            # (cuda)malloc tensors
            # assume: state_shape = obs_shape
            self.states = tf.Variable(tf.zeros((buffer_size, *obs_shape), dtype=tf.float32))
            self.obs1s = tf.Variable(tf.zeros((buffer_size, *obs_shape), dtype=tf.float32))
            self.obs2s = tf.Variable(tf.zeros((buffer_size, *obs_shape), dtype=tf.float32))

            self.action1s = tf.Variable(tf.zeros((buffer_size, action_shape), dtype=tf.int8))
            self.action2s = tf.Variable(tf.zeros((buffer_size, action_shape), dtype=tf.int8))

            self.prob1s = tf.Variable(tf.zeros((buffer_size, prob_shape), dtype=tf.float32))
            self.prob2s = tf.Variable(tf.zeros((buffer_size, prob_shape), dtype=tf.float32))

            self.value1s = tf.Variable(tf.zeros((buffer_size, value_shape), dtype=tf.float32))
            self.value2s = tf.Variable(tf.zeros((buffer_size, value_shape), dtype=tf.float32))

            self.normalized_rewards = tf.Variable(tf.zeros((buffer_size,), dtype=tf.float32))
            self.episode_overs = tf.Variable(tf.zeros((buffer_size,), dtype=tf.bool))

    def store_smth(
        self,
        state,
        obs1,
        obs2,
        action1,
        action2,
        prob1,
        prob2,
        value1,
        value2,
        reward,
        done,
    ):
        i = self.cursor
        with tf.device(self.device):
            # Convert everything to tensors at store time, less memory transfers
            # (and maybe non blocking cpu-gpu mem transfers)
            self.states[i].assign(tf.convert_to_tensor(state, dtype=tf.float32))
            self.obs1s[i].assign(tf.convert_to_tensor(obs1, dtype=tf.float32))
            self.obs2s[i].assign(tf.convert_to_tensor(obs2, dtype=tf.float32))

            self.action1s[i].assign(tf.convert_to_tensor(action1, dtype=tf.int8))
            self.action2s[i].assign(tf.convert_to_tensor(action2, dtype=tf.int8))

            self.prob1s[i].assign(tf.convert_to_tensor(prob1, dtype=tf.float32))
            self.prob2s[i].assign(tf.convert_to_tensor(prob2, dtype=tf.float32))

            self.value1s[i].assign(tf.convert_to_tensor(value1, dtype=tf.float32))
            self.value2s[i].assign(tf.convert_to_tensor(value2, dtype=tf.float32))

            self.normalized_rewards[i].assign(tf.convert_to_tensor(reward, dtype=tf.float32))
            self.episode_overs[i].assign(tf.convert_to_tensor(done, dtype=tf.bool))

        self.cursor += 1

    def make_batches(self, batch_size):
        # sample along dim 0: buffersiz->batchsize
        idxs = tf.random.uniform((batch_size,), minval=0, maxval=self.ptr, dtype=tf.int32)

        with tf.device(self.device):
            batch = {
                "state": tf.gather(self.state, idxs),
                "obs1": tf.gather(self.obs1, idxs),
                "obs2": tf.gather(self.obs2, idxs),
                "action1": tf.gather(self.action1, idxs),
                "action2": tf.gather(self.action2, idxs),
                "prob1": tf.gather(self.prob1, idxs),
                "prob2": tf.gather(self.prob2, idxs),
                "value1": tf.gather(self.value1, idxs),
                "value2": tf.gather(self.value2, idxs),
                "normalized_reward": tf.gather(self.normalized_reward, idxs),
                "episode_over": tf.gather(self.episode_over, idxs),
            }
        return batch

    def clear_memory(self):
        """no need to do anything with the memory, just fill from the beginning"""
        self.ptr = 0


class ActorCritic(Model):
    # TODO remove
    def __init__(self, input_dim=96, num_actions=6, hidden_sizes=(128, 64, 32)):
        super().__init__()
        # Shared backbone
        self.d1 = layers.Dense(hidden_sizes[0], activation="relu")
        self.d2 = layers.Dense(hidden_sizes[1], activation="relu")
        self.d3 = layers.Dense(hidden_sizes[2], activation="relu")

        # Policy head (logits for actions)
        self.policy_logits = layers.Dense(num_actions)

        # Value head (scalar)
        self.value = layers.Dense(1)

    def call(self, inputs):
        x = self.d1(inputs)
        x = self.d2(x)
        x = self.d3(x)

        logits = self.policy_logits(x)
        value = self.value(x)

        return logits, value


class PolicyNN(Model):
    def __init__(self, input_dim=96, n_actions=6, hidden_sizes=(128, 64, 32)):
        super().__init__()
        self.fc1 = layers.Dense(hidden_sizes[0], activation="relu")
        self.fc2 = layers.Dense(hidden_sizes[1], activation="relu")
        self.fc3 = layers.Dense(hidden_sizes[2], activation="relu")
        self.logits_layer = layers.Dense(n_actions)  # raw logits

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.logits_layer(x)  # return logits


class ValueNN(Model):
    def __init__(self, input_dim=96, hidden_sizes=(128, 64, 32)):
        super().__init__()
        self.fc1 = layers.Dense(hidden_sizes[0], activation="relu")
        self.fc2 = layers.Dense(hidden_sizes[1], activation="relu")
        self.fc3 = layers.Dense(hidden_sizes[2], activation="relu")
        self.value = layers.Dense(1)  # value scalar head

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.value(x)  # shape (batch, 1)


class dumbAgent:
    """MAPPO agent"""

    def __init__(self, buffer_len):
        self.buffer = MyBuffer(buffer_len)
        self.value_nn = 1
        self.policy_nn = 1

    def store_step(
        self,
        state,
        obs1,
        obs2,
        action1,
        action2,
        prob1,
        prob2,
        value1,
        value2,
        normalized_reward,
        episode_over,
    ):
        self.buffer.store_smth(
            state,
            obs1,
            obs2,
            action1,
            action2,
            prob1,
            prob2,
            value1,
            value2,
            normalized_reward,
            episode_over,
        )

    def save(self, path="./weights"):
        t = time.time()
        ckpoint_id = int((t * 10) % 1e6)
        lt = time.localtime(t)
        self.policy_nn.save_weights(f"{path}/policy_{lt[3]}_{lt[4]}_{lt[5]}_{ckpoint_id}.h5")
        self.value_nn.save_weights(f"{path}/value_{lt[3]}_{lt[4]}_{lt[5]}_{ckpoint_id}.h5")

    def _get_ckpoint_id(path):
        name = os.path.splitext(os.path.basename(path))[0]  # remove type
        parts = name.split("_")
        return int(parts[-1])  # last token is ckpoint_id

    def load_recent(self, path="weights", ckpoint_id=None):
        """loads the most recent training checkpoint or the one specified with ckpoint_id"""
        # find all suitable files
        pol_files = glob.glob(f"{path}/policy_*.h5")
        val_files = glob.glob(f"{path}/value_*.h5")
        if pol_files is None or val_files is None:
            raise RuntimeError("\n no weight files in the path \n")
        if ckpoint_id is not None:  # load a specific ckpoint
            val_f = [f for f in val_files if self._get_ckpoint_id(f) == ckpoint_id][-1]
            pol_f = [f for f in pol_files if self._get_ckpoint_id(f) == ckpoint_id][-1]
        else:  # load the most recent
            pol_f = max(pol_files, key=self._get_ckpoint_id)
            val_f = max(val_files, key=self._get_ckpoint_id)
        self.policy_nn.load_weights(pol_f)
        self.value_nn.load_weights(val_f)

    def sample_action(logits):
        # Sample directly from logits
        action = tf.squeeze(tf.random.categorical(logits, 1), axis=-1)
        probs = tf.nn.softmax(logits)  # optional
        log_probs = tf.nn.log_softmax(logits)  # optional
        return action, probs, log_probs

    def act(self, obs, state):
        """
        Args:
            state: The current state
            obs: an observation in featurized encoding

        Returns: Action, log prob of action, state value.
        """
        # make tensor of state
        state = np.expand_dims(state, axis=0)  # add dim for (mini)batchsize
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        # get critic value
        critic_value = self.value_nn(state)

        # make tensor of obs
        obs = np.expand_dims(obs, axis=0)  # add dim for (mini) batchsize
        obs = tf.convert_to_tensor(state, dtype=tf.float32)
        # get logits probs for actions
        logits = self.policy_nn(obs)
        # TODO
        warnings.warn("TODO implement log sum exp trick knowing how the policy nn works")

        action = 0
        log_prob = 0.5

        return action, log_prob, critic_value

    def learn(self, n_epochs=3):
        1  # TODO

    def __compute_gae(self):
        1  # TODO
