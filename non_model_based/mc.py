#coding=utf-8
import random
from collections import defaultdict
from env import Env

class OffPolicy(object):
    def __init__(self, actions):
        self.size = len(actions)
        self.actions = actions
        self.r = 1.0/self.size

    # 返回当前状态的行为
    def get_p(self, state, action):
        return self.r
    def get_a(self, state):
        return self.actions[random.randint(0, self.size-1)]

class OnPolicy(object):
    def __init__(self, actions, states):
        self.e = 0.1
        self.actions = actions
        self.size = len(self.actions)
        self.Pi = {}
        for a in actions:
            for s in states:
                self.Pi[(s, a)] = 1.0 / len(actions)

    # 按贪心返回最优策略
    def get_m(self, state):
        return  max([(x, self.Pi[(state, x)]) for x in self.actions], key=lambda x:x[1])[0]

    # 得到对应的行为
    def get_a(self, state):
        if random.random() < self.e:
            return self.actions[random.randint(0, self.size-1)]
        a = max([(x, self.Pi[(state, x)]) for x in self.actions], key=lambda x:x[1])[0]
        return a

    # 设置最优action
    def set_max(self, state, action, flag=True):
        v = self.e / len(self.actions)
        for a in self.actions:
            if a == action:
                v = 1 - self.e + v
            self.Pi[(state, a)] = v


def get_episode(env, policy):
    states = []
    s0 = env.init()
    while not env.is_t(s0):
        a = policy.get_a(s0)
        s1, r = env.step(a)
        states.append((s0, a, s1, r))
        s0 = s1
    return states


def off_mc():
    env = Env(6)
    policy = OffPolicy(env.actions())
    C = defaultdict(float)
    Q = defaultdict(float)
    Pi = {}
    for i in range(10000):
        G = 0
        W = 1.0
        n = 0
        states = get_episode(env, policy)
        for (s0, a, s1, r) in reversed(states):
            n += 1
            G = 0.9 * G + r
            C[(s0, a)] += W 
            Q[(s0, a)] += W/C[(s0, a)] * (G - Q[(s0, a)])
            Pi[s0] = max([(x, Q[(s0, x)]) for x in env.actions()], key=lambda x:x[1])[0]
            if a != Pi[s0]:
                break
            W = W / policy.get_p(s0, a)

    
    for t in env.get_t():
        Pi[t] = 'ter'
    env.render(Pi)

def on_mc():
    grid_size = 4
    env = Env(grid_size)
    policy = OnPolicy(env.actions(), range(grid_size**2))
    Q = defaultdict(float)
    R = defaultdict(list)
    for i in range(5000):
        G = 0
        states = get_episode(env, policy)
        for (s0, a, s1, r) in reversed(states):
            G = 0.9 * G + r
            R[(s0, a)].append(G)
            Q[(s0, a)] = sum(R[(s0, a)])/len(R[(s0, a)])

        for (s0, a, s1, r) in reversed(states):
            mm = [(x, Q[(s0, x)]) for x in env.actions()]
            action = max(mm, key=lambda x:x[1])[0]
            policy.set_max(s0, action)

    Pi = {}
    for i in range(grid_size**2):
        Pi[i] = policy.get_m(i)
    for t in env.get_t():
        Pi[t] = 'ter'

    env.render(Pi)

if __name__ == '__main__':
    off_mc()
