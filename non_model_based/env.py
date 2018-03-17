#coding=utf-8
import random

class Env(object):
    def __init__(self, size):
        self.size = size
        self.m = range(0, self.size ** 2)
        '''
        0,  1, 2,  3,
        4,  5, 6,  7,
        8,  9, 10, 11,
        12, 13, 14, 15
        '''
        self.t = [0, self.size**2-1]

    def init(self):
        self.position = random.randint(1, self.size**2-2)
        return self.position

    def actions(self):
        return ['up', 'down', 'left', 'right']

    def get_t(self):
        return self.t
    def is_t(self, state):
        return state in self.t

    def render(self, pi):
        for i in range(self.size):
            ss = ['%5s' % pi[i * self.size + j] for j in range(self.size)]
            print ' '.join(ss)

    def step(self, action):
        new_p = -1
        if action == 'up':
            if self.position / self.size > 0:
                new_p = self.position - self.size
            else:
                new_p = -1
        if action == 'down':
            if self.position / self.size < self.size - 1:
                new_p = self.position + self.size
            else:
                new_p = -1
        if action == 'left':
            if self.position % self.size > 0:
                new_p = self.position - 1
            else:
                new_p = -1
        if action == 'right':
            if self.position % self.size < self.size - 1:
                new_p = self.position + 1
            else:
                new_p = -1
        if new_p < 0:
            return self.position, 0
        self.position = new_p
        if self.is_t(new_p):
            return new_p, 1
        return new_p, 0
