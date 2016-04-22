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
import random

# third-party libraries
import numpy as np
import tensorflow as tf

# self-define libraries
import reader
import lstm

"""
Program Introduction
"""

flags = tf.flags
logging = tf.logging

flags.DEFINE_string('data_path', None, 'data_path')

FLAGS = flags.FLAGS

usual_prompt = '>>> '
exception_prompt = '!>> '


def vectorize(inputs, size):
    'turn a bunch of integer into a list of one-hot vector'
    with tf.device('/cpu:0'):
        embedding = get_embedding(size)
        return tf.nn.embedding_lookup(embedding, inputs)


def get_embedding(size):
    'create a bunch of one-hot vector for corresponding integer'
    with tf.device('/cpu:0'):
        a = np.zeros([size, size]).astype('float32')
        for i in xrange(size):
            a[i][i] = 1
        a_variable = tf.Variable(a, name='embedding', trainable=False)
    return a_variable


class LSTM_RBM(object):
    'Long Short-Term Memory combined with Restricted Boltzman Machine'

    def __init__(self, config):
        'assign the configuration and build the computation graph'
        # assign the corresponding configuration
        self.batch_size = config.batch_size
        self.gibbs_steps = config.gibbs_steps
        self.num_steps = config.num_steps
        self.max_grad_norm = config.max_grad_norm
        self.max_len_outputs = config.max_len_outputs

        # create Variable for global_step and exponential_decay_learning_rate
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(config.learning_rate,
                                                        self.global_step,
                                                        config.decay_steps,
                                                        config.decay_rate,
                                                        staircase=True)

        # where n_visible == n_visible_pitches + n_visible_notes
        self.n_visible = config.n_visible
        self.n_visible_pitches = config.n_visible_pitches
        self.n_visible_notes = config.n_visible_notes
        if self.n_visible == self.n_visible_pitches + self.n_visible_notes:
            print (usual_prompt + 'n_visible is configured correctly.')
        else:
            raise Exception(
                exception_prompt +
                'n_visible is not equal to n_visible_pitches plus n_visible_notes')
        self.n_hidden = config.n_hidden
        self.n_lstm_hidden = config.n_lstm_hidden

        # two placeholders for inputs_pitches and inputs_notes, which is a bunch of
        # integer
        self.inputs_pitches = tf.placeholder(tf.int32,
                                             [self.batch_size, self.num_steps])
        self.inputs_notes = tf.placeholder(tf.int32,
                                           [self.batch_size, self.num_steps])

        # create params in the cpu instead of gpu, which is said to be faster
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

        # create a lstm instance and define the initial hidden_state and
        # cell_state
        self.lstm = lstm.LSTM(self.n_visible, self.n_lstm_hidden)
        self.init_hidden_state = tf.zeros(
            [self.batch_size, self.n_lstm_hidden])
        self.init_cell_state = tf.zeros([self.batch_size, self.n_lstm_hidden])

        # self.params = [self.w_vh, self.w_uh, self.w_uv,
        # self.b_h, self.b_v] + self.lstm.params

        # vectorize both inputs_pitches and inputs_notes and combine them
        vectorized_inputs_pitches = vectorize(self.inputs_pitches,
                                              self.n_visible_pitches)
        vectorized_inputs_notes = vectorize(self.inputs_notes,
                                            self.n_visible_notes)
        vectorized_input = tf.concat(2, [vectorized_inputs_pitches,
                                         vectorized_inputs_notes])

        # start to compute the lstm hidden_state recurrently
        hidden_state = self.init_hidden_state
        cell_state = self.init_cell_state
        self.lstm_outputs = []
        self.lstm_outputs.append(hidden_state)
        with tf.variable_scope('lstm'):
            for time_step in xrange(self.num_steps):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                (hidden_state, cell_state) = self.lstm.feedforward(
                    vectorized_input[:, time_step, :], hidden_state, cell_state
                )
                self.lstm_outputs.append(hidden_state)

        # compute the average costs and losses through num_steps with lstm
        # hidden_state
        self.costs = 0
        self.losses = 0
        for time_step in xrange(self.num_steps):
            u = self.lstm_outputs[time_step]
            self.costs += (tf.reduce_mean(self.free_energy(vectorized_input[:, time_step, :], u)) -
                           tf.reduce_mean(self.free_energy(self.k_steps_gibbs_v(vectorized_input[:, time_step, :], u, self.gibbs_steps), u)))
            self.losses += self.get_loss(vectorized_input[:, time_step, :], u)

        # build the training of the model and generating of the model
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.costs, tvars),
                                          self.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars),
                                                  global_step=self.global_step)
        self.generate = self.get_generate(self.max_len_outputs).next()

        print (usual_prompt + 'model initialized')

    # def initialize_lstm_state(self):
        # self.init_hidden_state = tf.zeros([self.batch_size, self.n_lstm_hidden])
        # self.init_cell_state = tf.zeros([self.batch_size,
        # self.n_lstm_hidden])

    def get_loss(self, inputs, u):
        'the difference between inputs and sampled_inputs'
        return tf.reduce_sum((inputs - self.mean_v(inputs, u)) ** 2)

    def get_generate(self, len_outputs):
        'generate the pitch and note iteratively'
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
            inputs_pitches = inputs[:, :self.n_visible_pitches]
            inputs_notes = inputs[:, :self.n_visible_notes]
            yield (tf.argmax(inputs_pitches, 1), tf.argmax(inputs_notes, 1))

    def save_params(self, sess, filename='./params.ckpt'):
        'save params to file'
        saver = tf.train.Saver()
        save_path = saver.save(sess, filename)
        print (usual_prompt + 'Prams is saved in file: %s' % save_path)

    def load_params(self, sess, filename='./params.ckpt'):
        'if you saver to restore params, you do not have to initialize them beforehand'
        saver = tf.train.Saver()
        saver.restore(sess, filename)
        print (usual_prompt + 'Model restored.')

    def sample(self, inputs_mean, size):
        'Gibbs sampling'
        random_num = tf.random_uniform([size], minval=0, maxval=1)
        sample = inputs_mean > random_num
        inputs_sample = tf.to_float(sample)
        return inputs_sample

    def prop_up(self, v, u):
        'compute the probabilities of each hidden node to be 1'
        return tf.nn.sigmoid(tf.matmul(v, self.w_vh) + tf.matmul(u, self.w_uh) +
                             self.b_h)

    def prop_down(self, h, u):
        'compute the probabilities of each visible node to be 1'
        return tf.nn.sigmoid(tf.matmul(h, tf.transpose(self.w_vh)) +
                             tf.matmul(u, self.w_uv) + self.b_v)

    def sample_h_given_v(self, v, u):
        'given visible, sample hidden'
        temp = self.prop_up(v, u)
        return self.sample(temp, self.n_hidden)

    def sample_v_given_h(self, h, u):
        'given hidden, sample visible'
        temp = self.prop_down(h, u)
        return self.sample(temp, self.n_visible)

    def gibbs_vhv(self, v, u):
        'a step of gibbs sampling that sample h1 given v and then sample v1 given h1'
        temp = self.sample_h_given_v(v, u)
        return self.sample_v_given_h(temp, u)

    def k_steps_gibbs_v(self, v, u, k):
        'run k steps of gibbs_vhv'
        temp = v
        for i in xrange(k):
            temp = self.gibbs_vhv(temp, u)
        return temp

    def mean_v(self, v, u):
        'run self.gibbs_steps of gibbs_vhv, then sample sampled_h given v and compute the distribution of v_final given sampled_h'
        temp = self.k_steps_gibbs_v(v, u, self.gibbs_steps - 1)
        temp = self.sample_h_given_v(temp, u)
        return self.prop_down(temp, u)

    def free_energy(self, inputs, u):
        'this is used to evaluation the stability of the model'
        temp = tf.matmul(inputs, self.w_vh) + self.b_h + \
            tf.matmul(u, self.w_uh)
        b_v_term = tf.matmul(inputs,
                             (self.b_v + tf.matmul(u, self.w_uv)),
                             transpose_b=True)
        hidden_term = tf.reduce_sum(tf.log(1 + tf.exp(temp)),
                                    reduction_indices=1)
        return -b_v_term - hidden_term


def run_epoch(sess, model, pitches, notes, verbose=False):
    'run the model on the given pitches and notes'
    pitches = np.array(pitches)
    notes = np.array(notes)
    if len(pitches) == len(notes):
        epoch_size = len(pitches) // model.num_steps
    else:
        raise ValueError(
            exception_prompt + '''length of pitches is not equal to length of notes,
            please make sure they are equal to each other.''')
    for time_step in xrange(epoch_size):
        inputs_pitches = pitches[time_step * model.num_steps:
                                 (time_step + 1) * model.num_steps].reshape([1, model.num_steps])
        inputs_notes = notes[time_step * model.num_steps:
                             (time_step + 1) * model.num_steps].reshape([1, model.num_steps])
        _, losses, costs = sess.run([model.train_op, model.losses, model.costs],
                                    {
            model.inputs_pitches: inputs_pitches,
            model.inputs_notes: inputs_notes
        })
        if verbose and time_step == (epoch_size - 1):
            print (usual_prompt + 'losses: %10f' % losses, '   ',
                   'costs: %10f' % costs, '   ',
                   'global_step: %6d' % model.global_step.eval())


def get_random_indices(size):
    temp = range(size)
    random.shuffle(temp)
    return iter(temp)


class Config():
    'configuration, all hyperparameters are modified here'
    batch_size = 1
    gibbs_steps = 25
    num_steps = 50
    max_grad_norm = 5
    max_len_outputs = 50000
    generate_num = 20000
    max_epochs = 500

    learning_rate = 0.01
    decay_steps = 1000
    decay_rate = 0.96

    n_visible_pitches = 100
    n_visible_notes = 100
    n_visible = 199
    n_hidden = 2000
    n_lstm_hidden = 300

    new_data = True
    train = True
    generate = True


if __name__ == '__main__':
    config = Config()

    # load data
    if config.new_data:
        # read the data and make some transformations
        pitches = reader.data2index('./pitches.pkl')
        notes = reader.data2index('./notes.pkl')
        # modify the configuration
        config.n_visible_pitches = pitches[3]
        config.n_visible_notes = notes[3]
        config.n_visible = pitches[3] + notes[3]
        # get the input data of needed form
        inputs_data_pitches = pitches[0]
        inputs_data_notes = notes[0]
        # index_to_data of both pitches and notes
        index_to_data_pitches = pitches[1]
        index_to_data_notes = notes[1]
        # data_to_index of both pitches and notes
        data_to_index_pitches = pitches[2]
        data_to_index_notes = notes[2]
        # save data
        reader.save_data('pitches_i2d.pkl', pitches[1])
        reader.save_data('notes_i2d.pkl', notes[1])
        reader.save_data('pitches_d2i.pkl', pitches[2])
        reader.save_data('notes_d2i.pkl', notes[2])
        reader.save_data('pitches_len.pkl', pitches[3])
        reader.save_data('notes_len.pkl', notes[3])

        print (usual_prompt + 'information of new data has been saved.')
    else:
        # data_to_index and index_to_data of pitches
        data_to_index_pitches = reader.load_data('./pitches_d2i.pkl')
        index_to_data_pitches = reader.load_data('./pitches_i2d.pkl')
        # data_to_index and index_to_data of notes
        data_to_index_notes = reader.load_data('./notes_d2i.pkl')
        index_to_data_notes = reader.load_data('./notes_i2d.pkl')
        # len of pitches and notes
        len_pitches = reader.load_data('./pitches_len.pkl')
        len_notes = reader.load_data('./notes_len.pkl')
        # load and convert data
        raw_data_pitches = reader.load_data('./pitches.pkl')
        raw_data_notes = reader.load_data('./notes.pkl')
        inputs_data_pitches = reader.convert_to_index(raw_data_pitches,
                                                      data_to_index_pitches)
        inputs_data_notes = reader.convert_to_index(raw_data_notes,
                                                    data_to_index_notes)
        # modify the configuration
        config.n_visible_pitches = len_pitches
        config.n_visible_notes = len_notes
        config.n_visible = len_pitches + len_notes

        print (usual_prompt + 'information of needed data has been loaded.')

    model = LSTM_RBM(config)

    outputs_pitches = []
    outputs_notes = []
    # create a session to run the computation graph
    with tf.Session() as sess, tf.device('/cpu:0'):
        if config.new_data:
            sess.run(tf.initialize_all_variables())
            print (usual_prompt + 'variables have been initialized.')
        else:
            model.load_params(sess)
            print (usual_prompt + 'params have been loaded.')

        if len(inputs_data_pitches) == len(inputs_data_notes):
            len_inputs_data = len(inputs_data_pitches)
        else:
            raise ValueError(
                exception_prompt + 'length of pitches data is not equal to length of notes data')
        # train the model
        if config.train:
            for i in xrange(config.max_epochs):
                temp_indices = get_random_indices(len_inputs_data)
                for index in temp_indices:
                    data_pitches = inputs_data_pitches[index]
                    data_notes = inputs_data_notes[index]
                    run_epoch(
                        sess,
                        model,
                        data_pitches,
                        data_notes,
                        verbose=True)
                print (usual_prompt + 'epoch', i + 1)
                print ('-' * 20)
            model.save_params(sess)

        # generate data from well trained model
        if config.generate:
            for i in xrange(config.generate_num):
                temp = sess.run(model.generate)
                temp_pitch = int(temp[0])
                temp_note = int(temp[1])
                outputs_pitches.append(temp_pitch)
                outputs_notes.append(temp_note)
                if ((i + 1) % 100) == 0:
                    print (
                        usual_prompt + '%d pitches have been generated' %
                        (i + 1))

    # convert indices of pitches and notes into corresponding data
    outputs_pitches_i2d = reader.convert_to_data(
        outputs_pitches, index_to_data_pitches)
    outputs_notes_i2d = reader.convert_to_data(
        outputs_notes, index_to_data_notes)
    # save generated data
    reader.save_data('./generated_pitches.pkl', outputs_pitches_i2d)
    reader.save_data('./generated_notes.pkl', outputs_notes_i2d)
