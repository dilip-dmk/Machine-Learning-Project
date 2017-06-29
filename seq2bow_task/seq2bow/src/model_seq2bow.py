#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: Yinan Xu
@contact: yinanxu@wezhuiyi.com
@file: model_seq2bow.py
@time: 5/5/17 10:38 AM
"""


from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import sys, os

# from model_basic import BasicModel
# To import modules in utils folder
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)
from layers import ProjectionLayer
from layers import AttentionLayer
from model_basic import BasicModel

class Seq2bow_RNN(BasicModel):

	def add_placeholders(self):
		dim1 = self.batch_size
		# dim1 = None
		dim2 = self.num_steps
		dim3 = self.config.answer_num_classes

		self.input = tf.placeholder(tf.int32, [dim1, dim2], name="seq_wordids")
		self.labels = tf.placeholder(tf.float32, [dim1, dim3], name="seq_labels")
		self.lengths = tf.placeholder(tf.int32, [dim1], name="seq_lengths")
		self.dropout = tf.placeholder(tf.float32, name="dropout")

	def add_model_variables(self):
		gru_fw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
		gru_bw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
		self.gru_fw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_fw_cell, output_keep_prob=self.dropout)
		self.gru_bw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_bw_cell, output_keep_prob=self.dropout)


	def run_rnn(self, inputs, lengths):
		bi_states, bi_outputs = tf.nn.bidirectional_dynamic_rnn(
			self.gru_fw_cell,
			self.gru_bw_cell,
			inputs,
			sequence_length=lengths,
			dtype=tf.float32
		)
		return bi_states, bi_outputs

	'''
	Conpute the outputs of the network
	'''
	def compute_states(self, inputs, lengths):

		bi_states, _ = self.run_rnn(inputs, lengths)

		fw_out, bw_out = bi_states
		rnn_outputs = tf.concat(2, [fw_out, bw_out])  # [batch_size, num_steps, 2*size]

		atn_layer = AttentionLayer(in_dim=2*self.hidden_size,
								   dim=self.config.atn_hidden_size,
								   num_steps=self.num_steps,
								   name="Attention_Layer")

		hidden_vector = self.hidden_vector = atn_layer.get_output(fan_in=rnn_outputs, name="hidden_vector")

		return hidden_vector

	'''
	Compute the loss of the whole network
	'''
	def compute_loss(self, inputs, lengths):

		output = self.compute_states(inputs, lengths)

		proj_layer = ProjectionLayer(in_dim=2*self.hidden_size,
								   dim=self.config.answer_num_classes,
								   name="Projection_Layer")

		logits = proj_layer.get_output(fan_in=output, name="wxb_logits")

		pred = tf.nn.sigmoid(logits, name="predict_probs")

		loss = -tf.reduce_mean(
                tf.reduce_sum(
                    self.labels * tf.log(tf.clip_by_value(pred, 1e-15, 1)) + 
                    (1-self.labels) * tf.log(tf.clip_by_value((1-pred), 1e-15, 1)),
                    axis=1)
            )

		# self.predicts = tf.nn.top_k(pred, k=self.config.num_k)[1]

		self.probabilities = pred
		self.predicts = tf.to_float(tf.greater(pred, 0.5))

		return loss

