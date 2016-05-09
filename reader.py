#!/usr/bin/env python
# coding=utf-8
 #========================================================================
	#--> File Name: reader.py
	#--> Author: REN Chuangjie
	#--> Mail: rencjviei@163.com
	#--> Created Time: Sat Apr 16 00:34:05 2016
 #========================================================================

# standard libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import cPickle

import numpy as np


def load_data(filename):
    with open(filename, 'rb') as f:
        data = cPickle.load(f)
    return data

def encode(raw_data):
    data = np.concatenate(raw_data)
    set_data = set(data)
    len_data = len(set_data)
    index_to_data = {index:data for index,data in enumerate(set_data)}
    data_to_index = {data:index for index,data in enumerate(set_data)}
    return (index_to_data, data_to_index, len_data)

def convert_to_index(data, data_to_index):
    result = []
    for item in data:
        temp_cell = []
        for i in item:
            temp_cell.append(data_to_index[i])
        result.append(np.array(temp_cell))
    return result

def convert_to_data(index, index_to_data):
    result = []
    for item in index:
        result.append(index_to_data[item])
    return result ## this is used to be np.array

def data2index(filename):
    raw_data = load_data(filename)
    index_to_data, data_to_index, len_data = encode(raw_data)
    converted_data =  convert_to_index(raw_data, data_to_index)
    return converted_data, index_to_data, data_to_index, len_data

def save_data(filename, val):
    with open(filename, 'wb') as f:
        cPickle.dump(val, f)


if __name__ == '__main__':
    notes = data2index('./notes.pkl')
    pitches = data2index('./pitches.pkl')
    #save_data('notes_converted.pkl', notes[0])
    #save_data('notes_i2d.pkl', notes[1])
    #save_data('notes_d2i.pkl', notes[2])
    #print (notes[3])
    #save_data('pitches_converted.pkl', pitches[0])
    #save_data('pitches_i2d.pkl', pitches[1])
    #save_data('pitches_d2i.pkl', pitches[2])
    #print (pitches[3])
    for item in pitches[0]:
        print (item)
