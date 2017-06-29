
import numpy as np
import sys
import tensorflow as tf
from random import shuffle

class BasicEvaluator(object):
	def __init__(self):
		self.data = None
		self.label = None
		self.length = None
		self.data_idxs = None

	def _feed_raw_data(self, raw_data, shuffle_flag=True):
		self.data, self.label, self.length = raw_data
		data_idxs = range(len(self.data))
		if shuffle_flag:
			shuffle(data_idxs)
		self.data_idxs = data_idxs

	def _train_feed_batch_n_run(self, step, model, sess):
		batch_size = model.batch_size
		batch_idxs = self.data_idxs[step:step+batch_size]

		input_data = [self.data[ix] for ix in batch_idxs]
		label_data = [self.label[ix] for ix in batch_idxs]
		length_data = [self.length[ix] for ix in batch_idxs]

		loss, _ = sess.run(
			[model.loss, model.train_op],
			feed_dict = {
				model.input: input_data,
				model.labels: label_data,
				model.lengths: length_data,
				model.dropout: model.config.dropout
			}
		)

		return loss

	def train(self, raw_data, model, sess, shuffle_flag=True):
		self._feed_raw_data(raw_data, shuffle_flag=shuffle_flag)

		batch_size = model.batch_size
		output_iter = 0

		losses = []
		with open(model.config.log_train_loss_path, "a") as train_loss_fp: 
			for i in range(0, len(self.data), batch_size):
				if min(i+batch_size,len(self.data))-i < batch_size:
					break

				loss = self._train_feed_batch_n_run(i, model, sess)
				losses.append(loss)
				avg_loss = np.mean(losses)
				percent = (i+batch_size)*100.0/len(self.data)
				if percent//20 > output_iter:
					output_iter = percent//20
					print('avg loss: %.4f at %.2f%% of training set. \r' % (avg_loss, percent))

				train_loss_fp.write(("%.4f: %.4f\n"%(percent/100, loss)))

				sstr = 'avg loss: %.4f at %.2f%% of training set. \r' % (avg_loss, percent)
				sys.stdout.write(sstr)
				sys.stdout.flush()

		return np.mean(losses)

	def _evaluate_feed_batch_n_run(self, step, model, sess):
		batch_size = model.batch_size
		batch_idxs = self.data_idxs[step:step+batch_size]

		input_data = [self.data[ix] for ix in batch_idxs]
		label_data = [self.label[ix] for ix in batch_idxs]
		length_data = [self.length[ix] for ix in batch_idxs]

		pred_y, prob_y = sess.run(
			[model.predicts, model.probabilities],
			feed_dict={
				model.input: input_data,
				model.labels: label_data,
				model.lengths: length_data,
				model.dropout: 1.0
			}
		)

		return (pred_y, prob_y, label_data, input_data)

	def evaluate(self, raw_data, model, sess, prob_flag=False):
		self._feed_raw_data(raw_data, shuffle_flag=False)

		total_data = 0
		num_correct = [0]*model.config.num_k
		batch_size = model.config.batch_size

		for step in range(0, len(self.data), batch_size):
			if min(step+batch_size,len(self.data))-step < batch_size:
				break

			pred_y, prob_y, label_data, input_data = self._evaluate_feed_batch_n_run(
														step, model, sess)

			for i, v in enumerate(label_data):
				if prob_flag:
					print('x: ' + ','.join(['%d' % num for num in input_data[i]]))
					print('l: '+str(v))

				for j in range(model.config.num_k):
					if prob_flag:
						print('c: ' + str(j) + ' : ' + str(pred_y[i][j]) + ' : '+ str(prob_y[i][j]))
					if pred_y[i][j] == v:
						num_correct[j] += 1
			total_data += batch_size

		acc = [0.0]*model.config.num_k
		for i in range(model.config.num_k):
			acc[i] = float(sum(num_correct[:i+1]))/float(total_data)

		return acc
