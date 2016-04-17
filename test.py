#!/usr/bin/env python
# coding=utf-8
 #========================================================================
	#--> File Name: test.py
	#--> Author: REN Chuangjie
	#--> Mail: rencjviei@163.com
	#--> Created Time: Sat Apr 16 10:44:48 2016
 #========================================================================
import cPickle

import tensorflow as tf

a = tf.constant(range(10))
b = tf.constant(range(10))

c = a * b

with tf.Session() as sess:
    print sess.run(c)
