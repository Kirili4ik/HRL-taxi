import numpy as np
import gym


class Agent:

    def __init__(self, env, alpha, gamma):
        self.env = env
        nA = env.action_space.n + 4         # +4 ?
        nS = env.observation_space.n
        self.V = np.zeros((nA, nS))         # V for primitive and C for others, dicts ?
        self.C = np.zeros((nA, nS, nA))
        # s, n, e, w, pickup, dropoff, goto, put, get, root
        # 0, 1, 2, 3, 4,      5,       6,    7,   8,   9
        self.graph = [
            set(),  # 0 s
            set(),  # 1 n
            set(),  # 2 e
            set(),  # 3 w
            set(),  # 4 pickup
            set(),  # 5 dropoff
            {0, 1, 2, 3},  # 6 goto -> s, n, e, w
            {5, 6},  # 7 put -> dropoff, goto
            {4, 6},  # 8 get -> pickup, goto
            {7, 8},  # 9 root -> put, get
        ]
        self.alpha = alpha
        self.gamma = gamma
        self.taken = False
        self.r_sum = 0

    def is_terminal(self, a, done):
        RGBY = [(0, 0), (0, 4), (4, 0), (4, 3)]
        taxirow, taxicol, passidx, destidx = list(self.env.decode(self.env.s))      # env.s == state now
        # if done or (a == 8 and passidx >= 4) or (a == 7 and passidx < 4) \
        #        or (a == 6 and (taxirow, taxicol) == RGBY[passidx] and (passidx < 4 or (taxirow, taxicol) == RGBY[destidx])):
        # goto AND taxiloc == passloc AND (not in car OR taxiloc == destloc)
        
                        # Terminal actions ?????
        
        if done:
            return True
        elif a == 8 and passidx >= 4:
            return True
        elif a == 7 and passidx < 4:
            return True
        elif a == 6 and (passidx < 4 and (taxirow, taxicol) == RGBY[passidx] or passidx >= 4 and (taxirow, taxicol) == RGBY[destidx]):
            return True
        elif a == 5 and (passidx < 4 or passidx >= 4 and (taxirow, taxicol) != RGBY[destidx]):
            return True
        elif a == 4 and (passidx >= 4 or passidx < 4 and (taxirow, taxicol) != RGBY[passidx]):
            return True
        elif a < 4 and (taxirow, taxicol) == RGBY[destidx]:
            return True
        else:
            return False

    def evaluate(self, i, s):
        if i <= 5:               # primitive action
            return self.V[i, s]
        else:
            for j in self.graph[i]:
                self.V[j, s] = self.evaluate(j, s)
            Q = np.arange(0)                                                 # |
            for a2 in self.graph[i]:                                         # |  def count_Q ?
                Q = np.concatenate((Q, [self.V[a2, s] + self.C[i, s, a2]]))  # |
            max_arg = np.argmax(Q)                                           # |
            return self.V[max_arg, s]

    # e-Greedy Approach
    def greed_act(self, i, s):
        e = 0.1
        Q = np.arange(0)                                                     # |
        for a2 in self.graph[i]:                                             # |  def count_Q ?
            Q = np.concatenate((Q, [self.V[a2, s] + self.C[i, s, a2]]))      # |
        max_arg = np.argmax(Q)                                               # |
        policy = np.zeros(len(Q)) + e / len(Q)
        policy[max_arg] += 1 - e
        possible_a = np.array(list(self.graph[i]))
        return np.random.choice(possible_a, p=policy)  # choose from children with probabilities for explor/exploit prob

    def MAXQ_0(self, i, s):
        observ = self.env.s
        done = False
        if i <= 5:                                          # primitive action
            observ, r, done, _ = self.env.step(i)
            self.r_sum += r
            self.V[i, s] += self.alpha * (r - self.V[i, s])
            return 1
        else:
            count = 0
            while not self.is_terminal(i, done):
                a = self.greed_act(i, s)
                N = self.MAXQ_0(a, s)
                self.V[i, observ] = self.evaluate(i, s)
                self.C[i, s, a] += self.alpha * (self.gamma ** N * self.V[i, observ] - self.C[i, s, a])
                count += N
                s = observ
            return count

    def reset(self, new_env):
        self.env = new_env

        
### MAIN PROGRAM
#For now: infinite loop :(

alpha = 0.1
gamma = 0.999
env = gym.make('Taxi-v2').env
taxi = Agent(env, alpha, gamma)
episodes = 100
for j in range(episodes):
    env.reset()
    taxi.reset(env)
    taxi.MAXQ_0(9, env.s)
    print(taxi.r_sum, end=" ")
