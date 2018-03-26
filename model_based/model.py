#coding=utf-8
import argparse
import hashlib
import random
import sys
import time
import json
'''
基于模型的策略迭代模型
Model: 输入状态空间，用户初始化转移概率，并控制终止state、state-action reward
Value: 值函数
Policy: 策略
'''

class Model(object):
    def __init__(self):
        self.raw_p = {} #转移概率
        self.p = {} #转移概率
        self.action_reward = {} #状态-动作回报
        self.states = set()
        self.actions = set()


    def remember(self, state, action, reward, next_state, count):
        self.states.add(state)
        self.states.add(next_state)
        self.actions.add(action)
        s_a = (state, action)
        if s_a not in self.raw_p:
            self.raw_p[s_a] = {next_state:count}
        elif next_state not in self.raw_p:
            self.raw_p[s_a][next_state] = count
        else:
            self.raw_p[s_a][next_state] += count
        if s_a not in self.action_reward:
            self.action_reward[s_a] = {'reward': reward * count, 'count': count}
        else:
            self.action_reward[s_a]['reward'] += reward * count
            self.action_reward[s_a]['count'] += count
        
    def finish_remember(self):
        self.p = {}
        for s_a, info in self.raw_p.items():
            all_count = sum(info.values())
            self.p[s_a] = {}
            for next_state, count in info.items():
                self.p[s_a][next_state] = float(count) / all_count

        temp = {}
        for k, info in self.action_reward.items():
            temp[k] = float(info['reward'] / info['count'])
        self.action_reward = temp

    # 所有状态空间
    def get_all_states(self):
        return self.states

    def get_next_states(self, state, action):
        d = self.p.get((state, action), {})
        return d.keys()

    # 所有动作空间
    def get_all_actions(self):
        return self.actions

    # 状态-动作回报
    def get_reward(self, state, action):
        return self.action_reward.get((state, action)) or 0

    # 模型
    def get_probability(self, state, action, next_state):
        # 零概率的状态为非法状态
        return self.p.get((state, action), {}).get(next_state) or 0

    def is_valid(self, state, action):
        return (state, action) in self.p

class Value(object):
    def __init__(self):
        self.v = {}
    # 值函数
    def get(self, state):
        return self.v.get(state) or 0
    # 
    def set(self, state, val):
        self.v[state] = val

# 平均策略
class AvgPolicy(object):
    def __init__(self):
        self.pi = {}

    # 策略，实际上就是pi(action|state), 返回归一化的概率值
    def get(self, state, action):
        return self.pi.get(state, {}).get(action) or 0

    # 返回是否有变动, 没有变动返回False
    def set(self, state, action_rewards):
        flag = False
        if state not in self.pi:
            self.pi[state] = {}
        # 平均策略
        for action in action_rewards.keys():
            p = 1.0/len(action_rewards)
            if abs(self.get(state, action) - p) > 0.000001:
                flag = True
            self.pi[state][action] = 1.0/len(action_rewards)
        return flag

# TODO add greed policy
class GreedyPolicy(AvgPolicy):
    def __init__(self):
        super(GreedyPolicy, self).__init__()
        self.geedy = 0.9

    # 返回是否有变动, 没有变动返回False
    def set(self, state, action_rewards):
        flag = False
        if state not in self.pi:
            self.pi[state] = {}
        # 贪心策略
        for i, (action, reward) in enumerate(sorted(action_rewards.items(), key=lambda x:x[1], reverse=True)):
            p = self.geedy
            if len(action_rewards) > 0:
                if i > 0:
                    p = (1-self.geedy)/(len(action_rewards) - 1)
            else:
                p = 1.0
            if abs(self.get(state, action) - p) > 0.000001:
                flag = True
            self.pi[state][action] = p
        return flag

class PolicyIteration(object):
    def __init__(self, model, value, policy, gamma=0.9, epsion=0.00001):
        self.model = model # 模型
        self.value = value # 值函数
        self.policy = policy # 策略
        self.gamama = gamma # 衰减系数
        self.epsion = epsion # 精度

    def one_step_ahead(self, state, action, log=False):
        valid = self.model.is_valid(state, action)
        p = self.policy.get(state, action)  if valid else 0
        r = self.model.get_reward(state, action)
        _tmp_reward = 0
        for _state in self.model.get_next_states(state, action):
            _v = self.value.get(_state)
            _p = self.model.get_probability(state, action, _state)
            _tmp_reward += _v * _p
        return  valid, r + self.gamama * _tmp_reward

    def policy_evaluation(self):
        states = list(self.model.get_all_states())
        actions = list(self.model.get_all_actions())
        loop = 0
        while True:
            loop += 1
            flag = True
            for state in states:
                new_v = 0
                for action in actions:
                    valid, tmp_reward = self.one_step_ahead(state, action)
                    if valid:
                        p = self.policy.get(state, action)
                        new_v += tmp_reward * p
                old_v = self.value.get(state)
                self.value.set(state, new_v) # 更新值
                if abs(old_v - new_v) > self.epsion:
                    flag = False
            if flag:
                break

    # 返回当天策略改进是否有变动，没有变动就返回False
    def policy_improvement(self):
        states = list(self.model.get_all_states())
        actions = list(self.model.get_all_actions())
        flag = False
        for state in states:
            rewards = {}
            for action in actions:
                valid, tmp_reward = self.one_step_ahead(state, action, log=True)
                if valid:
                    rewards[action] = tmp_reward
                
            v = self.policy.set(state, rewards)
            if v:
                flag = True
        
        return flag

            
    def train(self, loop=10):
        loop = 0
        while True:
            loop += 1
            self.policy_evaluation() # 策略评估
            if not self.policy_improvement(): # 策略优化
                break

    def output_policy(self):
        states = list(self.model.get_all_states())
        actions = list(self.model.get_all_actions())
        for state in states:
            rewards = {}
            for action in actions:
                rewards[action] = self.one_step_ahead(state, action)
            max_reward = max(rewards.items(), key=lambda x:x[1][1])
            yield state, max_reward[0], max_reward[1][1]
