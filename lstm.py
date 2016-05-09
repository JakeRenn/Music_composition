#!/usr/bin/env python
# coding=utf-8
#========================================================================
#--> File Name: lstm.py
#--> Author: REN Chuangjie
#--> Mail: rencjviei@163.com
#--> Created Time: Sat Apr 16 01:10:47 2016
#========================================================================

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import numpy as np
import tensorflow as tf


def tf_variable(gate, input_size, hidden_size):
    with tf.device('/cpu:0'):
        with tf.variable_scope(gate):
            weight = tf.Variable(tf.random_normal([input_size, hidden_size],
                                                  mean=0,
                                                  stddev=1 / math.sqrt(input_size)),
                                 name='w',
                                 trainable=True)
            weight_u = tf.Variable(tf.random_normal([hidden_size, hidden_size],
                                                    mean=0,
                                                    stddev=1 / math.sqrt(hidden_size)),
                                   name='u',
                                   trainable=True)
            bias = tf.Variable(tf.zeros([hidden_size]), name='b',
                               trainable=True)
    return weight, weight_u, bias


class LSTM(object):
    'Long Short-Term Memory'

    def __init__(self, n_input, n_hidden):
        self.n_input = n_input
        self.n_hidden = n_hidden

        self.i_w, self.i_u, self.i_b = tf_variable('input',
                                                   self.n_input,
                                                   self.n_hidden)
        self.f_w, self.f_u, self.f_b = tf_variable('forget',
                                                   self.n_input,
                                                   self.n_hidden)
        self.o_w, self.o_u, self.o_b = tf_variable('output',
                                                   self.n_input,
                                                   self.n_hidden)
        self.c_w, self.c_u, self.c_b = tf_variable('cell',
                                                   self.n_input,
                                                   self.n_hidden)

        #self.params = [self.i_w, self.i_u, self.i_b,
                       #self.f_w, self.f_u, self.f_b,
                       #self.o_w, self.o_u, self.o_b,
                       #self.c_w, self.c_u, self.c_b]

    def feedforward(self, inputs, pre_hidden, pre_cell,
                    gate_activation=tf.sigmoid,
                    output_activation=tf.tanh):
        i_gate = gate_activation(tf.matmul(inputs, self.i_w) +
                                 tf.matmul(pre_hidden, self.i_u) +
                                 self.i_b)
        f_gate = gate_activation(tf.matmul(inputs, self.f_w) +
                                 tf.matmul(pre_hidden, self.f_u) +
                                 self.f_b)
        o_gate = gate_activation(tf.matmul(inputs, self.o_w) +
                                 tf.matmul(pre_hidden, self.o_u) +
                                 self.o_b)
        new_cell = output_activation(tf.matmul(inputs, self.c_w) +
                                     tf.matmul(pre_hidden, self.c_u) +
                                     self.c_b)
        cell = i_gate * new_cell + f_gate * pre_cell
        hidden = o_gate * output_activation(cell)
        return hidden, cell


class LSTMConfig(object):
    n_input = 100
    n_hidden = 200

if __name__ == "__main__":
    a = LSTM(100, 200)
    inputs = np.random.random([1, 100]).astype('float32')
    with tf.Session() as sess:
        tf.initialize_all_variables().run()
        print (len(a.params))
