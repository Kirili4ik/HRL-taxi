import numpy as np
import gym
import copy
import matplotlib.pyplot as plt
import time


class Agent:
    def __init__(self, env, alpha, gamma):
        self.env = env
        nA = env.action_space.n + 5           # + goto * 2 + put + get + root
        nS = env.observation_space.n
        self.V = np.zeros((nA, nS))
        self.C = np.zeros((nA, nS, nA))
        self.V_copy = self.V.copy()
        self.graph = [
            set(),  # 0 s
            set(),  # 1 n
            set(),  # 2 e
            set(),  # 3 w
            set(),  # 4 pickup
            set(),  # 5 dropoff
            {0, 1, 2, 3},  # 6 gotoSource -> s, n, e, w
            {0, 1, 2, 3},  # 7 gotoDestination -> s, n, e, w
            {4, 6},  # 8 get -> pickup, gotoS
            {5, 7},  # 9 put -> dropoff, gotoD
            {8, 9},  # 10 root -> put, get
        ]
        self.alpha = alpha
        self.gamma = gamma
        self.r_sum = 0
        self.new_s = copy.copy(self.env.s)
        self.done = False
        self.num_of_ac = 0

    def is_terminal(self, a, done):
        RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]
        taxirow, taxicol, passidx, destidx = list(self.env.decode(self.env.s))
        taxiloc = (taxirow, taxicol)
        if done:
            return True
        elif a == 10:
            return done
        elif a == 9:
            return passidx < 4
        elif a == 8:
            return passidx >= 4
        elif a == 7:
            return passidx >= 4 and taxiloc == RGBY[destidx]
        elif a == 6:                                         # добавил наличие/отсутствие пассажира в условия
            return passidx < 4 and taxiloc == RGBY[passidx]
        elif a <= 5:
            return True

    def evaluate(self, i, s):
            if i <= 5:                          # primitive action
                return self.V_copy[i, s]
            else:
                for j in self.graph[i]:
                    self.V_copy[j, s] = self.evaluate(j, s)
                Q = np.arange(0)
                for a2 in self.graph[i]:
                    Q = np.concatenate((Q, [self.V_copy[a2, s]]))
                max_arg = np.argmax(Q)
                return self.V_copy[max_arg, s]

    # e-Greedy Approach
    def greed_act(self, i, s):
        e = 0.001
        Q = np.arange(0)
        possible_a = np.arange(0)
        for a2 in self.graph[i]:
            if a2 <= 5 or (not self.is_terminal(a2, self.done)):
                Q = np.concatenate((Q, [self.V[a2, s] + self.C[i, s, a2]]))
                possible_a = np.concatenate((possible_a, [a2]))
        max_arg = np.argmax(Q)
        if np.random.rand(1) < e:
            return np.random.choice(possible_a)
        else:
            return possible_a[max_arg]
        
    def MAXQ_0(self, i, s):
        if self.done:
            i = 13                  # to end
        self.done = False
        if i <= 5:                  # primitive action
            self.new_s, r, self.done, _ = copy.copy(self.env.step(i))
            self.r_sum += r
            self.num_of_ac += 1
            self.V[i, s] += self.alpha * (r - self.V[i, s])
            return 1
        elif i <= 10:
            count = 0
            while not self.is_terminal(i, self.done):
                a = self.greed_act(i, s)
                N = self.MAXQ_0(a, s)
                self.V_copy = self.V.copy()
                evaluate_res = self.evaluate(i, self.new_s)
                self.C[i, s, a] += self.alpha * (self.gamma ** N * evaluate_res - self.C[i, s, a])
                count += N
                s = self.new_s
            return count

    def reset(self):
        self.env.reset()
        self.r_sum = 0
        self.num_of_ac = 0
        self.done = False
        self.new_s = copy.copy(self.env.s)


alpha = 0.2
gamma = 1
env = gym.make('Taxi-v2').env
taxi = Agent(env, alpha, gamma)
episodes = 10000
sum_list = []
for j in range(episodes):
    taxi.reset()
    taxi.MAXQ_0(10, env.s)
    sum_list.append(taxi.r_sum)
    print(j)
plt.plot(sum_list)
plt.show()
