	
import numpy as np
import sys, os
import tensorflow as tf

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)
from train_n_eval_basic import BasicEvaluator


class Evaluator(BasicEvaluator):

	# def _feed_raw_data(self, raw_data, shuffle_flag=True):
	# 	self.data, self.label, self.length = raw_data
	# 	data_idxs = range(len(self.data))
	# 	if shuffle_flag:
	# 		from random import shuffle
	# 		shuffle(data_idxs)
	# 	self.data_idxs = data_idxs

	# def _train_feed_batch_n_run(self, step, model, sess):
	# 	batch_size = model.batch_size
	# 	batch_idxs = self.data_idxs[step:step+batch_size]

	# 	input_data = [self.data[ix] for ix in batch_idxs]
	# 	label_data = [self.label[ix] for ix in batch_idxs]
	# 	length_data = [self.length[ix] for ix in batch_idxs]

	# 	print str(len(input_data)) + " x " +  str(len(input_data[0]))
	# 	print str(len(label_data)) + " x " +  str(len(label_data[0]))
	# 	print str(len(length_data))

	# 	loss, _ = sess.run(
	# 		[model.loss, model.train_op],
	# 		feed_dict = {
	# 			model.input: input_data,
	# 			model.labels: label_data,
	# 			model.lengths: length_data,
	# 			model.dropout: model.config.dropout
	# 		}
	# 	)

	# 	return loss

	# def train(self, raw_data, model, sess, shuffle_flag=True):
	# 	self._feed_raw_data(raw_data, shuffle_flag=shuffle_flag)

	# 	batch_size = model.batch_size
	# 	output_iter = 0

	# 	losses = []
	# 	with open(model.config.log_train_loss_path, "a") as train_loss_fp: 
	# 		for i in range(0, len(self.data), batch_size):
	# 			if min(i+batch_size,len(self.data))-i < batch_size:
	# 				break

	# 			loss = self._train_feed_batch_n_run(i, model, sess)
	# 			losses.append(loss)
	# 			avg_loss = np.mean(losses)
	# 			percent = (i+batch_size)*100.0/len(self.data)
	# 			if percent//20 > output_iter:
	# 				output_iter = percent//20
	# 				print('avg loss: %.4f at %.2f%% of training set. \r' % (avg_loss, percent))

	# 			train_loss_fp.write(("%.4f: %.4f\n"%(percent/100, loss)))

	# 			sstr = 'avg loss: %.4f at %.2f%% of training set. \r' % (avg_loss, percent)
	# 			sys.stdout.write(sstr)
	# 			sys.stdout.flush()

	# 	return np.mean(losses)

	def _evaluate_feed_batch_n_run(self, step, model, sess):
	 	batch_size = model.batch_size
	 	batch_idxs = self.data_idxs[step:step+batch_size]

	 	input_data = [self.data[ix] for ix in batch_idxs]
	 	label_data = [self.label[ix] for ix in batch_idxs]
	 	length_data = [self.length[ix] for ix in batch_idxs]

	 	pred_y, prob_y, hidden_vec = sess.run(
	 		[model.predicts, model.probabilities, model.hidden_vector],
	 		feed_dict={
	 			model.input: input_data,
	 			model.labels: label_data,
	 			model.lengths: length_data,
	 			model.dropout: 1.0
	 		}
	 	)

	 	return (pred_y, prob_y, label_data, input_data, hidden_vec)

	def evaluate(self, raw_data, model, sess, prob_flag=False):
		self._feed_raw_data(raw_data, shuffle_flag=False)

		total_data = 0
		num_correct = 0
		batch_size = model.config.batch_size

		for step in range(0, len(self.data), batch_size):
			if min(step+batch_size,len(self.data))-step < batch_size:
				break

			pred_y, prob_y, label_data, input_data, hidden_vec = self._evaluate_feed_batch_n_run(
														step, model, sess)

			for i, v in enumerate(label_data):
				if prob_flag:
					#print('x: ' + ','.join(['%d' % num for num in input_data[i]]))
					print("\t".join(map(str, hidden_vec[i])))
					# print('l: '+str(v))

				#num_correct += sum((v==pred_y[i]).astype(int))
				#total_data += len(v)
				num_correct += sum(v*pred_y[i])
				total_data += sum(v)

		acc = [0]
		acc[0] = float(num_correct)/float(total_data)

		return acc
