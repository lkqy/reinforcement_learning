#coding=utf-8
import random
from collections import defaultdict
from env import Env
from policy import RandomPolicy, EspionGreedyPolicy


def sarsa():
    grid_size = 4
    env = Env(grid_size)
    policy = EspionGreedyPolicy(env.actions(), range(grid_size**2))
    Q = defaultdict(float)
    for i in range(5000):
        s0 = env.init()
        if env.is_t(s0):
            continue
        a0 = policy.get_a(s0)
        while not env.is_t(s0):
            s, r = env.step(a0)
            a = policy.get_a(s)
            Q[(s0, a0)] += 0.9 * (r + 0.9 * Q[(s, a)] - Q[(s0, a0)])
            s0 = s
            a0 = a
            mm = [(x, Q[(s0, x)]) for x in env.actions()]
            action = max(mm, key=lambda x:x[1])[0]
            policy.set_max(s0, action)

    Pi = {}
    for i in range(grid_size**2):
        Pi[i] = policy.get_m(i)
    for t in env.get_t():
        Pi[t] = 'ter'

    env.render(Pi)

def qlearning():
    grid_size = 4
    env = Env(grid_size)
    policy = EspionGreedyPolicy(env.actions(), range(grid_size**2))
    Q = defaultdict(float)
    for i in range(5000):
        s0 = env.init()
        if env.is_t(s0):
            continue
        a0 = policy.get_a(s0)
        while not env.is_t(s0):
            s, r = env.step(a0)
            a = policy.get_a(s)
            max_a = max([(x, Q[(s, x)]) for x in env.actions()], key=lambda x:x[1])[0]
            Q[(s0, a0)] += 0.9 * (r + 0.9 * Q[(s, max_a)] - Q[(s0, a0)])
            s0 = s
            a0 = a
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
    sarsa()
    qlearning()
