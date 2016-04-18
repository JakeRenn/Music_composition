#!/usr/bin/env python
# coding=utf-8
#========================================================================
#--> File Name: lstm_rbm.py
#--> Author: REN Chuangjie
#--> Mail: rencjviei@163.com
#--> Created Time: Sat Apr 16 20:03:25 2016
#========================================================================

# standard libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

# third-party libraries
import numpy as np
import tensorflow as tf

# self-define libraries
import reader
import lstm

flags = tf.flags
logging = tf.logging

flags.DEFINE_string('data_path', None, 'data_path')

FLAGS = flags.FLAGS


def vectorize(inputs, size):
    with tf.device('/cpu:0'):
        embedding = get_embedding(size)
        return tf.nn.embedding_lookup(embedding, inputs)


def get_embedding(size):
    with tf.device('/cpu:0'):
        a = np.zeros([size, size]).astype('float32')
        for i in xrange(size):
            a[i][i] = 1
        a_variable = tf.Variable(a, name='embedding', trainable=False)
    return a_variable


class LSTM_RBM(object):
    'Long Short-Term Memory combined with Restricted Boltzman Machine'

    def __init__(self, config):
        self.batch_size = config.batch_size
        self.gibbs_steps = config.gibbs_steps
        self.num_steps = config.num_steps
        self.max_grad_norm = config.max_grad_norm
        self.max_len_outputs = config.max_len_outputs

        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                        self.global_step,
                                                        config.decay_steps,
                                                        config.decay_rate,
                                                        staircase=True)

        self.n_visible = config.n_visible
        self.n_hidden = config.n_hidden
        self.n_lstm_hidden = config.n_lstm_hidden

        self.inputs = tf.placeholder(tf.int32,
                                     [self.batch_size, self.num_steps])

        with tf.device('/cpu:0'):
            self.w_vh = tf.Variable(tf.random_normal([self.n_visible,
                                                      self.n_hidden],
                                                     stddev=1 / math.sqrt(self.n_visible)),
                                    name='w_vh')
            self.w_uh = tf.Variable(tf.random_normal([self.n_lstm_hidden,
                                                      self.n_hidden],
                                                     stddev=1 / math.sqrt(self.n_lstm_hidden)),
                                    name='w_uh')
            self.w_uv = tf.Variable(tf.random_normal([self.n_lstm_hidden,
                                                      self.n_visible],
                                                     stddev=1 / math.sqrt(self.n_lstm_hidden)),
                                    name='w_uv')
            self.b_h = tf.Variable(tf.zeros([self.batch_size,
                                             self.n_hidden]),
                                   name='b_h')
            self.b_v = tf.Variable(tf.zeros([self.batch_size,
                                             self.n_visible]),
                                   name='b_v')

        self.lstm = lstm.LSTM(self.n_visible, self.n_lstm_hidden)
        self.init_hidden_state = tf.zeros(
            [self.batch_size, self.n_lstm_hidden])
        self.init_cell_state = tf.zeros([self.batch_size, self.n_lstm_hidden])

        self.params = [self.w_vh, self.w_uh, self.w_uv,
                       self.b_h, self.b_v] + self.lstm.params

        vectorized_input = vectorize(self.inputs, self.n_visible)

        hidden_state = self.init_hidden_state
        cell_state = self.init_cell_state
        self.lstm_outputs = []
        self.lstm_outputs.append(hidden_state)
        with tf.variable_scope('recurrent'):
            for time_step in xrange(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (hidden_state, cell_state) = self.lstm.feedforward(
                    vectorized_input[:, time_step, :], hidden_state, cell_state
                )
                self.lstm_outputs.append(hidden_state)

        self.costs = 0
        self.losses = 0
        for time_step in xrange(self.num_steps):
            u = self.lstm_outputs[time_step]
            self.costs += (tf.reduce_mean(self.free_energy(vectorized_input[:, time_step, :], u)) -
                           tf.reduce_mean(self.free_energy(self.gibbs_vhv(vectorized_input[:, time_step, :], u), u)))
            self.losses = self.get_loss(vectorized_input[:, time_step, :], u)

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.costs, tvars),
                                          self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                  global_step=self.global_step)
        self.generate = self.get_generate(self.max_len_outputs).next()

        print ('model initialized')

    # def initialize_lstm_state(self):
        #self.init_hidden_state = tf.zeros([self.batch_size, self.n_lstm_hidden])
        #self.init_cell_state = tf.zeros([self.batch_size, self.n_lstm_hidden])

    def get_loss(self, inputs, u):
        return tf.reduce_sum((inputs - self.mean_v(inputs, u)) ** 2)

    def get_generate(self, len_outputs):
        hidden_state = self.init_hidden_state
        cell_state = self.init_cell_state
        with tf.variable_scope('lstm'):
            for time_step in xrange(len_outputs):
                tf.get_variable_scope().reuse_variables()
            inputs = self.mean_v(tf.zeros([self.batch_size,
                                           self.n_visible]), hidden_state)
            (hidden_state, cell_state) = self.lstm.feedforward(inputs,
                                                               hidden_state,
                                                               cell_state)
            yield tf.argmax(inputs, 1)

    def save_params(self, sess, filename='./params.ckpt'):
        'save params to file'
        saver = tf.train.Saver()
        save_path = saver.save(sess, filename)
        print ('Prams is saved in file: %s' % save_path)

    def load_params(self, sess, filename='./params.ckpt'):
        'if you saver to restore params, you do not have to initialize them beforehand'
        saver = tf.train.Saver()
        saver.restore(sess, filename)
        print ('Model restored.')

    def sample(self, inputs_mean, size):
        'Gibbs sampling'
        random_num = tf.random_uniform([size], minval=0, maxval=1)
        sample = inputs_mean > random_num
        inputs_sample = tf.to_float(sample)
        return inputs_sample

    def prop_up(self, v, u):
        return tf.nn.sigmoid(tf.matmul(v, self.w_vh) + tf.matmul(u, self.w_uh) +
                             self.b_h)

    def prop_down(self, h, u):
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.w_vh)) +
                             tf.matmul(u, self.w_uv) + self.b_v)

    def sample_h_given_v(self, v, u):
        temp = self.prop_up(v, u)
        return self.sample(temp, self.n_hidden)

    def sample_v_given_h(self, h, u):
        temp = self.prop_down(h, u)
        return self.sample(temp, self.n_visible)

    def gibbs_vhv(self, v, u):
        temp = self.sample_h_given_v(v, u)
        return self.sample_v_given_h(temp, u)

    def k_steps_gibbs_v(self, v, u, k):
        temp = v
        for i in xrange(k):
            temp = self.gibbs_vhv(temp, u)
        return temp

    def mean_v(self, v, u):
        temp = self.k_steps_gibbs_v(v, u, self.gibbs_steps)
        temp = self.sample_h_given_v(temp, u)
        return self.prop_down(temp, u)

    def free_energy(self, inputs, u):
        temp = tf.matmul(inputs, self.w_vh) + self.b_h + \
            tf.matmul(u, self.w_uh)
        b_v_term = tf.reduce_sum(tf.matmul(inputs,
                                           (self.b_v + tf.matmul(u, self.w_uv)),
                                           transpose_b=True), reduction_indices=1)
        hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(temp)),
                                    reduction_indices=1)
        return -b_v_term - hidden_term


def run_epoch(sess, model, data, verbose=False):
    'run the model on the given data'
    data = np.array(data)
    epoch_size = len(data) // model.num_steps
    for time_step in xrange(epoch_size):
        inputs = data[time_step * model.num_steps:
                      (time_step + 1) * model.num_steps].reshape([1, model.num_steps])
        _, losses, costs = sess.run([model.train_op, model.losses, model.costs],
                                    {
            model.inputs: inputs
        })
        if verbose and time_step == (epoch_size - 1):
            print ('losses:', losses, 'costs:', costs)


class Config():
    batch_size = 1
    gibbs_steps = 25
    num_steps = 50
    max_grad_norm = 1
    max_len_outputs = 5000
    max_epochs = 1

    learning_rate = 0.01
    decay_steps = 100
    decay_rate = 0.96

    n_visible = 100
    n_hidden = 2000
    n_lstm_hidden = 200

    new_data = False
    train = True

if __name__ == '__main__':
    config = Config()
    model = LSTM_RBM(config)

    # load data
    if config.new_data:
        pitches = reader.data2index('./pitches.pkl')
        config.n_visible = pitches[3]
        inputs_data = pitches[0]
        index_to_data = pitches[1]
        data_to_index = pitches[2]
        reader.save_data('pitches_i2d.pkl', pitches[1])
        reader.save_data('pitches_d2i.pkl', pitches[2])
        reader.save_data('pitches_len.pkl', pitches[3])
        print ('information of new data has been saved.')
    else:
        data_to_index = reader.load_data('./pitches_d2i.pkl')
        index_to_data = reader.load_data('./pitches_i2d.pkl')
        raw_data = reader.load_data('./pitches.pkl')
        inputs_data = reader.convert_to_index(raw_data, data_to_index)
        len_pitches = reader.load_data('./pitches_len.pkl')
        config.n_visible = len_pitches
        print ('information of needed data has been loaded.')

    outputs = []
    with tf.Session() as sess, tf.device('/cpu:0'):
        if config.new_data:
            sess.run(tf.initialize_all_variables())
            print ('check point: initialize variables')
        else:
            model.load_params(sess)
            print ('check point: load_params')

        if config.train:
            for i in xrange(config.max_epochs):
                for data in inputs_data:
                    run_epoch(sess, model, data, verbose=True)
                print ('epoch', i + 1)
            model.save_params(sess)

        if config.generate:
            for i in xrange(2000):
                temp = sess.run(model.generate)
                outputs.append(int(temp))
                if (i % 100 + 1) == 0:
                    print ('%d pitches have been generated' % (i + 1))

    outputs_i2d = reader.convert_to_data(temp, index_to_data)
    reader.save_data('./generated_pitches.pkl', outputs_i2d)
