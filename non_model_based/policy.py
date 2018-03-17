#coding=utf-8
import random
from collections import defaultdict
from env import Env

class RandomPolicy(object):
    def __init__(self, actions):
        self.size = len(actions)
        self.actions = actions
        self.r = 1.0/self.size

    # 返回当前状态的行为
    def get_p(self, state, action):
        return self.r
    def get_a(self, state):
        return self.actions[random.randint(0, self.size-1)]

class EspionGreedyPolicy(object):
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
