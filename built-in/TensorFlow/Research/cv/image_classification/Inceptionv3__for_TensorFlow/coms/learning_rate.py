#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by learning_rate on 19-3-29

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

'''
三角策略
基于论文<<cyclical learning rates for training neural networks>>
实现三角学习速率方法调整策略
@:param ep_counter 一个批次训练过程中的计数值
@:param stepsize 训练一个批次需要的总次数
@:param base_lr 最小学习速率
@:param max_lr 最大学习速率
'''
def clr_base(ep_counter,stepsize,base_lr,max_lr):
    cycle = np.floor(1 + (ep_counter / (2 * stepsize)))
    x = np.abs(ep_counter/stepsize - 2*cycle + 1)
    lr = base_lr + (max_lr - base_lr) * np.maximum(0,(1-x))
    return lr


'''
三角策略指数衰减法
实现三角策略最大值指数衰减调整
当衰减后结果小于设定的最小lr时，将返回最小lr
'''
class CLR_EXP_RANGE():
    '''
    @:param INIT 是否初始化最大学习速率
    @:param MAX_LR 储存最大学习速率的设置值
    '''
    def __init__(self):
        self.INIT = False
        self.MAX_LR = 0.

    '''
    @:param gamma 衰减指数 用来衰减最大学习速率设置值
    其他参数和clr_base()意义一样
    '''
    def calc_lr(self,ep_counter,step_size,min_lr,max_lr,gamma):
        if self.INIT == False:
            self.MAX_LR = max_lr
            self.INIT = True
        lr = clr_base(ep_counter,step_size,min_lr,self.MAX_LR)
        self.MAX_LR *= gamma
        if lr <  min_lr:
            lr = min_lr
        return lr

    def test(self):
        minlr = 0.001
        maxlr = 0.006
        gamma = 0.9998
        lr_trend_exp_range = list()
        for ep in range(50):
            for iter in range(500):
                lr = self.calc_lr(iter, 500, minlr, maxlr, gamma)
                lr_trend_exp_range.append(lr)

        plt.plot(lr_trend_exp_range)
        plt.show()

'''
三角策略法最大值减半
每当一个批次结束时候,最大设定值变为波动值的一半 (波动值等于最大值加最小设定值)
'''
class CLR_TRI2():
    def __init__(self):
        self.INIT = False
        self.MAX_LR = 0.
        self.UPDATE = False

    def calc_lr(self,ep_counter,step_size,min_lr,max_lr):
        if self.INIT == False:
            self.MAX_LR = max_lr
            self.INIT = True
        lr = clr_base(ep_counter,step_size,min_lr,self.MAX_LR)
        if ep_counter == step_size-1:
            self.MAX_LR = (self.MAX_LR + min_lr) / 2.

        if lr < min_lr:
            lr = min_lr
        return lr

    def test(self):
        minlr = 0.001
        maxlr = 0.006
        lr_trend_exp_range = list()
        for ep in range(50):
            for iter in range(500):
                lr = self.calc_lr(iter, 500, minlr, maxlr)
                lr_trend_exp_range.append(lr)
        plt.plot(lr_trend_exp_range)
        plt.show()

if __name__ == '__main__':
    clr_exp = CLR_EXP_RANGE()
    clr_exp.test()

    clr2 = CLR_TRI2()
    clr2.test()