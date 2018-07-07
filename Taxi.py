import numpy as np
import gym
import copy
import matplotlib.pyplot as plt
import time     # for checks


class Agent:

    def __init__(self, env, alpha, gamma):
        self.env = env
        nA = env.action_space.n + 4
        nS = env.observation_space.n
        self.V = np.zeros((nA, nS))  # V for primitive and C for others, make dicts?
        self.C = np.zeros((nA, nS, nA))
        # s, n, e, w, pickup, dropoff, goto, put, get, root
        # 0, 1, 2, 3, 4,      5,       6,    7,   8,   9
        self.graph = [
            set(),
            set(),
            set(),
            set(),
            set(),
            set(),
            {0, 1, 2, 3},  # 6 goto -> s, n, e, w
            {5, 6},  # 7 put -> dropoff, goto
            {4, 6},  # 8 get -> pickup, goto
            {7, 8},  # 9 root -> put, get
        ]
        self.alpha = alpha
        self.gamma = gamma
        self.r_sum = 0
        self.new_s = copy.copy(self.env.s)
        self.done = False

    def is_incar(self, passidx):
        return passidx >= 4
    
    def is_primitive(self, i):
        return i <= 5
    
    def is_terminal(self, a, done):
        RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]         # from env
        taxirow, taxicol, passidx, destidx = list(self.env.decode(self.env.s))
        if done:
            return True
        elif a == 9:    # root
            return done
        elif a == 8:    # get
            return self.is_incar(passidx)
        elif a == 7:    # put
            return not self.is_incar(passidx)
        elif a == 6:    # goto
            return (not self.is_incar(passidx) and (taxirow, taxicol) == RGBY[passidx] or
                    self.is_incar(passidx) and (taxirow, taxicol) == RGBY[destidx])
        elif self.is_primitive(a):
            return True

    def evaluate(self, i, s):       # probably got problems
        if self.is_primitive(i):
            return self.V[i, s]
        else:
            for j in self.graph[i]:
                self.V[j, s] = self.evaluate(j, s)
            Q = np.arange(0)
            for a2 in self.graph[i]:
                Q = np.concatenate((Q, [self.V[a2, s] + self.C[i, s, a2]]))
            max_arg = np.argmax(Q)                                           
            return self.V[max_arg, s]

    # e-Greedy Approach
    def greed_act(self, i, s):
        e = 0.1
        Q = np.arange(0)
        for a2 in self.graph[i]:
            Q = np.concatenate((Q, [self.V[a2, s] + self.C[i, s, a2]]))
        max_arg = np.argmax(Q)
        possible_a = np.array(list(self.graph[i]))
        if np.random.rand(1) < e:
            return np.random.choice(possible_a)
        else:
            return possible_a[max_arg]
        
        # Another way
        '''Q = np.arange(0)                                                  
        for a2 in self.graph[i]:                                             
            Q = np.concatenate((Q, [self.V[a2, s] + self.C[i, s, a2]]))      
        max_arg = np.argmax(Q)                                              
        policy = np.zeros(len(Q)) + e / len(Q)
        policy[max_arg] += 1 - e
        possible_a = np.array(list(self.graph[i]))
        return np.random.choice(possible_a, p=policy)'''

    def MAXQ_0(self, i, s):
        if self.done:
            i = 10
        self.done = False
        if self.is_primitive(i):
            self.new_s, r, self.done, _ = copy.copy(self.env.step(i))
            self.r_sum += r
            self.V[i, s] += self.alpha * (r - self.V[i, s])
            # self.env.render()
            # time.sleep(0.7)
            return 1
        elif i <= 9:            # not primitive, but exists
            count = 0
            while not self.is_terminal(i, self.done):
                a = self.greed_act(i, s)
                N = self.MAXQ_0(a, s)
                # s' = new_s
                self.V[i, self.new_s] = self.evaluate(i, self.new_s)
                self.C[i, s, a] += self.alpha * (self.gamma ** N * self.V[i, self.new_s] - self.C[i, s, a])
                count += N
                s = self.new_s
            return count

    def reset(self):
        self.env.reset()
        self.r_sum = 0
        self.done = False
        self.new_s = copy.copy(self.env.s)

### MAIN PROGRAM
# at ~1100-1200 episodes ?

alpha = 0.1
gamma = 0.999
env = gym.make('Taxi-v2').env
taxi = Agent(env, alpha, gamma)
episodes = 2000
sum_list = []
for j in range(episodes):
    # if j == 1170:
        # print('go')
        # abcdef = input()
    taxi.reset()
    taxi.MAXQ_0(9, env.s)
    sum_list.append(taxi.r_sum)
    # print(j)
plt.plot(sum_list)
plt.show()
