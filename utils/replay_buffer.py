import numpy as np
import os
from utils import load_pickle


class ReplayBufferLight:

    def __init__(self, act_dim, size, state_dim, sequence_len, batch_size):
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.size = size
        self.acts_buf = np.zeros([int(size), int(act_dim)], dtype=np.float32)
        self.done_buf = np.zeros(int(size), dtype=np.float32)
        self.rew_buf = np.zeros((size), dtype=np.float32)
        self.ep_start_buf = np.zeros(int(size), dtype=bool)
        self.state_buf = np.zeros([int(size), state_dim], dtype=np.float32)
        self.next_state_buf = np.zeros([int(size), state_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, int(size)

    def store(self, state, act, rew, next_state, done):
        self.acts_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.ptr = (self.ptr + 1) % self.max_size  # replace oldest entry from memory
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        idxs2 = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            if self.done_buf[idxs[i]] == 1.0:
                idxs[i] = idxs[i] - 1

        return dict(acts=self.acts_buf[idxs],
                    done=self.done_buf[idxs],
                    ep_start=self.ep_start_buf[idxs],
                    states=self.state_buf[idxs],
                    next_states=self.next_state_buf[idxs])

    def sample_sequence_batch(self, batch_size=32, sequence_len=1):
        idxs = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            for j in range(sequence_len):
                if self.done_buf[idxs[i] + j] == 1.0:
                    idxs[i] = idxs[i] - 1*(j+1)

        idxs_seq = np.zeros((sequence_len, batch_size)).astype('int32')

        for i in range(sequence_len):
            idxs_seq[i] = np.copy(idxs) + i

        # for i in range(sequence_len):
        #     self.state_buff_seq[i] = self.state_buf[idxs_seq[i]]
        #     self.next_state_buff_seq[i] = self.next_state_buf[idxs_seq[i]]

        return dict(acts=self.acts_buf[idxs_seq],
                    rews=self.rew_buf[idxs_seq],
                    done=self.done_buf[idxs_seq],
                    ep_start=self.ep_start_buf[idxs],
                    states=self.state_buf[idxs_seq],
                    next_states=self.next_state_buf[idxs_seq])

    def sample_sequence_batch_obs(self, batch_size=32, sequence_len=1):
        idxs = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            for j in range(sequence_len):
                if self.done_buf[idxs[i] + j] == 1.0:
                    idxs[i] = idxs[i] - 1*(j+1)

        idxs_seq = np.zeros((sequence_len, batch_size)).astype('int32')

        for i in range(sequence_len):
            idxs_seq[i] = np.copy(idxs) + i

        # for i in range(sequence_len):
        #     self.state_buff_seq[i] = self.state_buf[idxs_seq[i]]
        #     self.next_state_buff_seq[i] = self.next_state_buf[idxs_seq[i]]

        return dict(acts=self.acts_buf[idxs_seq],
                    rews=self.rew_buf[idxs_seq],
                    done=self.done_buf[idxs_seq],
                    ep_start=self.ep_start_buf[idxs],
                    states=self.state_buf[idxs_seq],
                    next_states=self.next_state_buf[idxs_seq])


    def sample_sequence(self, start_idx=0, seq_len=5):
        end_idx = start_idx + seq_len
        return dict(acts=self.acts_buf[np.arange(start_idx, end_idx)],
                    states=self.state_buf[np.arange(start_idx, end_idx)],
                    next_states=self.next_state_buf[np.arange(start_idx, end_idx)],
                    )

    def get_all_samples(self):
        return dict(acts=self.acts_buf[:self.size],
                    done=self.done_buf[:self.size])

    def clear_memory(self):
        self.__init__(self.act_dim, self.max_size, self.state_dim)



class ReplayBuffer:

    def __init__(self, obs_dim, act_dim, size, state_dim, sequence_len, batch_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.size = size
        self.obs_buf = np.zeros([int(size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])], dtype=np.float32)
        self.next_obs_buf = np.zeros([int(size), int(obs_dim[0]), int(obs_dim[1]), int(obs_dim[2])], dtype=np.float32)
        self.acts_buf = np.zeros([int(size), int(act_dim)], dtype=np.float32)
        self.done_buf = np.zeros(int(size), dtype=np.float32)
        self.rew_buf = np.zeros((size), dtype=np.float32)
        self.ep_start_buf = np.zeros(int(size), dtype=bool)
        self.state_buf = np.zeros([int(size), state_dim], dtype=np.float32)
        self.next_state_buf = np.zeros([int(size), state_dim], dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, int(size)

        # self.state_buff_seq = np.zeros([sequence_len, batch_size, state_dim], dtype=np.float32)
        # self.next_state_buff_seq = np.zeros([sequence_len, batch_size, state_dim], dtype=np.float32)


    def store(self, obs, act, rew, next_obs, done, state, next_state):
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.ptr = (self.ptr + 1) % self.max_size  # replace oldest entry from memory
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        idxs2 = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            if self.done_buf[idxs[i]] == 1.0:
                idxs[i] = idxs[i] - 1

        return dict(obs1=self.obs_buf[idxs],
                    obs2=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    done=self.done_buf[idxs],
                    ep_start=self.ep_start_buf[idxs],
                    obs3=self.next_obs_buf[idxs2],
                    states=self.state_buf[idxs],
                    next_states=self.next_state_buf[idxs])

    def sample_sequence_batch(self, batch_size=32, sequence_len=1):
        idxs = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            for j in range(sequence_len):
                if self.done_buf[idxs[i] + j] == 1.0:
                    idxs[i] = idxs[i] - 1*(j+1)

        idxs_seq = np.zeros((sequence_len, batch_size)).astype('int32')

        for i in range(sequence_len):
            idxs_seq[i] = np.copy(idxs) + i

        # for i in range(sequence_len):
        #     self.state_buff_seq[i] = self.state_buf[idxs_seq[i]]
        #     self.next_state_buff_seq[i] = self.next_state_buf[idxs_seq[i]]

        return dict(acts=self.acts_buf[idxs_seq],
                    rews=self.rew_buf[idxs_seq],
                    done=self.done_buf[idxs_seq],
                    ep_start=self.ep_start_buf[idxs],
                    states=self.state_buf[idxs_seq],
                    next_states=self.next_state_buf[idxs_seq])

    def sample_sequence_batch_obs(self, batch_size=32, sequence_len=1):
        idxs = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            for j in range(sequence_len):
                if self.done_buf[idxs[i] + j] == 1.0:
                    idxs[i] = idxs[i] - 1*(j+1)

        idxs_seq = np.zeros((sequence_len, batch_size)).astype('int32')

        for i in range(sequence_len):
            idxs_seq[i] = np.copy(idxs) + i

        # for i in range(sequence_len):
        #     self.state_buff_seq[i] = self.state_buf[idxs_seq[i]]
        #     self.next_state_buff_seq[i] = self.next_state_buf[idxs_seq[i]]

        return dict(obs1=self.obs_buf[idxs_seq],
                    obs2=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs_seq],
                    rews=self.rew_buf[idxs_seq],
                    done=self.done_buf[idxs_seq],
                    ep_start=self.ep_start_buf[idxs],
                    states=self.state_buf[idxs_seq],
                    next_states=self.next_state_buf[idxs_seq])


    def sample_sequence(self, start_idx=0, seq_len=5):
        end_idx = start_idx + seq_len
        return dict(obs1=self.obs_buf[np.arange(start_idx, end_idx)],
                    acts=self.acts_buf[np.arange(start_idx, end_idx)],
                    obs2=self.next_obs_buf[np.arange(start_idx, end_idx)],
                    states=self.state_buf[np.arange(start_idx, end_idx)],
                    next_states=self.next_state_buf[np.arange(start_idx, end_idx)],
                    )

    def get_all_samples(self):
        return dict(obs=self.obs_buf[:self.size],
                    next_obs=self.next_obs_buf[:self.size],
                    acts=self.acts_buf[:self.size],
                    done=self.done_buf[:self.size])

    def clear_memory(self):
        self.__init__(self.obs_dim, self.act_dim, self.max_size, self.state_dim)


class ReplayBufferBarrier:

    def __init__(self, act_dim, size, state_dim):
        self.act_dim = act_dim
        self.state_dim = state_dim
        self.size = size
        self.acts_buf = np.zeros([int(size), int(act_dim)], dtype=np.float32)
        self.done_buf = np.zeros(int(size), dtype=np.float32)
        self.rew_buf = np.zeros((size), dtype=np.float32)
        self.ep_start_buf = np.zeros(int(size), dtype=bool)
        self.state_buf = np.zeros([int(size), state_dim], dtype=np.float32)
        self.next_state_buf = np.zeros([int(size), state_dim], dtype=np.float32)
        self.cbfvalue_buf = np.zeros((size), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, int(size)

    def store(self, state, act, rew, next_state, done, cbf_value):
        self.acts_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.state_buf[self.ptr] = state
        self.next_state_buf[self.ptr] = next_state
        self.cbfvalue_buf[self.ptr] = cbf_value
        self.ptr = (self.ptr + 1) % self.max_size  # replace oldest entry from memory
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        #idxs2 = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            if self.done_buf[idxs[i]] == 1.0:
                idxs[i] = idxs[i] - 1

        return dict(acts=self.acts_buf[idxs],
                    done=self.done_buf[idxs],
                    ep_start=self.ep_start_buf[idxs],
                    states=self.state_buf[idxs],
                    next_states=self.next_state_buf[idxs])

    def sample_sequence_batch(self, batch_size=32, sequence_len=1):
        idxs = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            for j in range(sequence_len):
                if self.done_buf[idxs[i] + j] == 1.0:
                    idxs[i] = idxs[i] - 1*(j+1)

        idxs_seq = np.zeros((sequence_len, batch_size)).astype('int32')

        for i in range(sequence_len):
            idxs_seq[i] = np.copy(idxs) + i

        # for i in range(sequence_len):
        #     self.state_buff_seq[i] = self.state_buf[idxs_seq[i]]
        #     self.next_state_buff_seq[i] = self.next_state_buf[idxs_seq[i]]

        return dict(acts=self.acts_buf[idxs_seq],
                    rews=self.rew_buf[idxs_seq],
                    done=self.done_buf[idxs_seq],
                    ep_start=self.ep_start_buf[idxs],
                    states=self.state_buf[idxs_seq],
                    next_states=self.next_state_buf[idxs_seq],
                    cbf_values=self.cbfvalue_buf[idxs_seq])

    def sample_sequence_batch_obs(self, batch_size=32, sequence_len=1):
        idxs = np.random.randint(0, self.size, size=batch_size)

        for i in range(idxs.shape[0]):
            for j in range(sequence_len):
                if self.done_buf[idxs[i] + j] == 1.0:
                    idxs[i] = idxs[i] - 1*(j+1)

        idxs_seq = np.zeros((sequence_len, batch_size)).astype('int32')

        for i in range(sequence_len):
            idxs_seq[i] = np.copy(idxs) + i

        # for i in range(sequence_len):
        #     self.state_buff_seq[i] = self.state_buf[idxs_seq[i]]
        #     self.next_state_buff_seq[i] = self.next_state_buf[idxs_seq[i]]

        return dict(acts=self.acts_buf[idxs_seq],
                    rews=self.rew_buf[idxs_seq],
                    done=self.done_buf[idxs_seq],
                    ep_start=self.ep_start_buf[idxs],
                    states=self.state_buf[idxs_seq],
                    next_states=self.next_state_buf[idxs_seq],
                    cbf_values=self.cbfvalue_buf[idxs_seq])


    def sample_sequence(self, start_idx=0, seq_len=5):
        end_idx = start_idx + seq_len
        return dict(acts=self.acts_buf[np.arange(start_idx, end_idx)],
                    states=self.state_buf[np.arange(start_idx, end_idx)],
                    next_states=self.next_state_buf[np.arange(start_idx, end_idx)],
                    )

    def get_all_samples(self):
        return dict(acts=self.acts_buf[:self.size],
                    done=self.done_buf[:self.size])

    def clear_memory(self):
        self.__init__(self.act_dim, self.max_size, self.state_dim)

if __name__ == "__main__":

    testing_dataset = 'acrobot_state_test.pkl'

    # load data
    directory = os.path.dirname(os.path.abspath(__file__))

    folder_test = os.path.join(directory + '/Data', testing_dataset)

    data_test = load_pickle(folder_test)

    sequence_len = 15
    batch_size = 64
    test_loader = ReplayBuffer(obs_dim=(84, 84, 6), act_dim=1, size=len(data_test),
                               state_dim=6, sequence_len=sequence_len, batch_size=batch_size)
    counter_t = 0
    for dt in data_test:
        test_loader.store((dt[0] / 255).astype('float32'),
                          dt[1].astype('float32'),
                          (dt[3] / 255).astype('float32'),
                          dt[4],
                          dt[5].astype('float32'),
                          dt[6].astype('float32'))
        counter_t += 1
    print(counter_t)

    data = test_loader.sample_sequence_batch(batch_size=batch_size, sequence_len=sequence_len)

    print("done")