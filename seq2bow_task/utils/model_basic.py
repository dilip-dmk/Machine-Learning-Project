from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
import sys, os

# To import modules in utils folder
base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)
from layers import ProjectionLayer


class BasicModel(object):

	def __init__(self, config):
		self.config = config

		self.num_steps = config.num_steps
		self.vocab_size = config.vocab_size
		self.wordvec_size = config.wordvec_size
		self.num_classes = config.num_classes
		self.num_layers = config.num_layers
		self.hidden_size = config.hidden_size
		self.batch_size = config.batch_size
		self.initializer = tf.random_uniform_initializer(-self.config.init_scale, self.config.init_scale)

		self.add_placeholders()
		self.add_model_variables()
		inputs = self.add_embedding()
		self.loss = self.compute_loss(inputs, self.lengths)
		self.train_op = self.add_train_op()

	def add_placeholders(self):
		# dim1 = self.batch_size
		dim1 = None
		dim2 = self.num_steps
		dim3 = self.num_classes

		self.input = tf.placeholder(tf.int32, [dim1, dim2], name="seq_wordids")
		self.labels = tf.placeholder(tf.int32, [dim1], name="seq_labels")
		self.lengths = tf.placeholder(tf.int32, [dim1], name="seq_lengths")
		self.dropout = tf.placeholder(tf.float32, name="dropout")

	def add_model_variables(self):
		gru_fw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
		gru_bw_cell = tf.nn.rnn_cell.GRUCell(self.hidden_size)
		self.gru_fw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_fw_cell, output_keep_prob=self.dropout)
		self.gru_bw_cell = tf.nn.rnn_cell.DropoutWrapper(gru_bw_cell, output_keep_prob=self.dropout)


	def add_embedding(self):
		with tf.variable_scope("Embed"):
			if self.config.embedding_random_flag:
				embedding = tf.get_variable(
					'embedding',
					[self.vocab_size, self.wordvec_size],
					initializer = tf.random_uniform_initializer(-0.05, 0.05),
					trainable = True,
					regularizer = None
				)
			else:
				# word_embedding = np.loadtxt(self.config.word_vec_path, dtype=np.float32)
				embedding = tf.get_variable("embedding", [self.vocab_size, self.wordvec_size])
				# embedding = embedding.assign(word_embedding)

			inputs = tf.nn.embedding_lookup(embedding, self.input)
			inputs = tf.nn.dropout(inputs, self.dropout)

		return inputs

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
		_, bi_outputs = self.run_rnn(inputs, lengths)
		fw_out, bw_out = bi_outputs
		hidden_vector = tf.concat(1, [fw_out, bw_out])
		return hidden_vector

	'''
	Compute the loss of the whole network
	'''
	def compute_loss(self, inputs, lengths):

		output = self.compute_states(inputs, lengths)

		proj_layer = ProjectionLayer(in_dim=2*self.hidden_size,
								   dim=self.num_classes,
								   name="Projection_Layer")

		logits = proj_layer.get_output(fan_in=output, name="wxb_logits")

		l1 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, self.labels)
		loss = tf.reduce_mean(l1, [0])

		pred = tf.nn.softmax(logits, name="predict_probs")
		self.probabilities = tf.nn.top_k(pred, k=self.config.num_k)[0]
		self.predicts = tf.nn.top_k(pred, k=self.config.num_k)[1]

		return loss


	def add_train_op(self):
		self.learning_rate = tf.Variable(self.config.learning_rate, trainable=False)
		optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
		tvars = tf.trainable_variables()
		grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), self.config.max_grad_norm)
		train_op = optimizer.apply_gradients(zip(grads, tvars))

		'''
		Train Ops using TF build-in function 'exponential_decay' for learning rate decay
		'''
		# batch = tf.Variable(0, trainable=False)
		# self.learning_rate = tf.train.exponential_decay(
		# 		self.config.learning_rate, # base learning rate
		# 		batch*self.batch_size,
		# 		self.config.train_size,
		# 		self.config.lr_decay,
		# 		staircase=True
		# 	)
		# train_op = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.loss, global_step=batch)

		return train_op
