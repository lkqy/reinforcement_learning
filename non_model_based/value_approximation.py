#coding=utf-8
from sklearn.linear_model import SGDRegressor
import random
from collections import defaultdict
from env import Env
from policy import RandomPolicy, EspionGreedyPolicy

class Sarsa(object):
    def __init__(self, size=4):
        self.grid_size = size
        self.env = Env(self.grid_size)
        self.a_id = dict([(a, i) for i, a in enumerate(self.env.actions())])
        self.policy = EspionGreedyPolicy(self.env.actions(), range(self.grid_size**2))
    def get_f(self, s, a):
        f = range(self.grid_size**2 + 4)
        f[s], f[self.a_id[a]] = 1, 1
        return f

    def sarsa(self):
        policy = self.policy
        Q = SGDRegressor()
        f = self.get_f(1, 'left')
        Q.fit([f], [1])
        for i in range(500):
            s0 = self.env.init()
            if self.env.is_t(s0):
                continue
            a0 = policy.get_a(s0)
            while not self.env.is_t(s0):
                s, r = self.env.step(a0)
                a = policy.get_a(s)
                f0 = self.get_f(s0, a0)
                f = self.get_f(s, a)
                target = Q.predict([f0])[0] + 0.9 * (r + 0.9 * Q.predict([f])[0] - Q.predict([f0])[0])
                Q.partial_fit([f], [target])
                s0 = s
                a0 = a
                mm = [(x, Q.predict([self.get_f(s0, x)])[0]) for x in self.env.actions()]
                action = max(mm, key=lambda x:x[1])[0]
                policy.set_max(s0, action)

        Pi = {}
        for i in range(self.grid_size**2):
            Pi[i] = policy.get_m(i)
        for t in self.env.get_t():
            Pi[t] = 'ter'

        self.env.render(Pi)

if __name__ == '__main__':
    s = Sarsa()
    s.sarsa()
