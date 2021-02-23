import numpy as np


class Memory:
    def __init__(self, memory_size, batch_size, transition_num):
        self.memory_list = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.transition_num = transition_num

    def store(self, transition):
        if self.memory_num >= self.memory_size:
            del self.memory_list[0]
        if len(transition) == 5:
            s, a, r, s_, t = transition
            self.memory_list.append([s, a, r, s_, t])
        if len(transition) == 4:
            s, a, r, s_ = transition
            self.memory_list.append([s, a, r, s_])

    def sample(self):
        assert self.memory_num >= self.batch_size
        if self.memory_num < self.memory_size:
            indices = np.random.choice(self.memory_num, size=self.batch_size)
        else:
            indices = np.random.choice(self.memory_size, self.batch_size)
        batch_states, batch_actions, batch_rewards, batch_states_, batch_terminal = [], [], [], [], []
        for i in indices:
            batch_states.append(self.memory_list[i][0])
            batch_actions.append(self.memory_list[i][1])
            batch_rewards.append(self.memory_list[i][2])
            batch_states_.append(self.memory_list[i][3])
            if self.transition_num == 5:
                batch_terminal.append(self.memory_list[i][4])

        batch_states = np.array(batch_states)
        batch_actions = np.array(batch_actions)
        batch_rewards = np.array(batch_rewards)
        batch_states_ = np.array(batch_states_)
        batch_rewards = batch_rewards[:, np.newaxis]
        if self.transition_num==5:
            batch_terminal = np.array(batch_terminal)
            batch_terminal = batch_terminal[:, np.newaxis]
            return batch_states, batch_actions, batch_rewards, batch_states_, batch_terminal
        if self.transition_num == 4:
            return batch_states, batch_actions, batch_rewards, batch_states_

    @property
    def memory_num(self):
        return len(self.memory_list)


