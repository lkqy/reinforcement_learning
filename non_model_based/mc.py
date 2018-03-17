#coding=utf-8
import random
from collections import defaultdict
from env import Env
from policy import RandomPolicy, EspionGreedyPolicy

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
    policy = RandomPolicy(env.actions())
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
    policy = EspionGreedyPolicy(env.actions(), range(grid_size**2))
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
