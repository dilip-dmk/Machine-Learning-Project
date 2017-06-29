from __future__ import absolute_import
from __future__ import division

import tensorflow as tf
import numpy as np
import time
import sys, os, glob

from config import *
import reader
from train_n_eval import Evaluator


def train(config, evaluator, restore=False):

	data, num_emb = reader.load_data(config)

	train_set, dev_set, test_set = data['train'], data['valid'], data['test']

	if not os.path.exists(config.model_dir):
		os.mkdir(config.model_dir)
	if not os.path.exists(config.log_dir):
		os.mkdir(config.log_dir)
	if not os.path.exists(config.log_train_dir):
		os.mkdir(config.log_train_dir)

	if restore==False:
		train_files = glob.glob(config.log_train_dir+'/*')
		for train_file in train_files:
			os.remove(train_file)

	if len(config.gpu_chosen)>0: 
		gpu_options = tf.GPUOptions(
			visible_device_list=",".join(map(str, config.gpu_chosen)),
			per_process_gpu_memory_fraction=config.gup_per_fraction
		)
	else:
		gpu_options = tf.GPUOptions(
			per_process_gpu_memory_fraction=config.gup_per_fraction
		)

	with tf.Graph().as_default(), \
		tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		# with tf.variable_scope("model", reuse=None):
		model = config.model_func(config)

		init = tf.global_variables_initializer()
		saver = tf.train.Saver()
		sess.run(init)

		if not config.DEBUG:
			word_embedding = np.loadtxt(config.word_vec_path, dtype=np.float32)
			# with tf.variable_scope("model", reuse=True):
			with tf.variable_scope("Embed", reuse=True):
				embedding = tf.get_variable("embedding", [config.vocab_size, config.wordvec_size])
				ea = embedding.assign(word_embedding)
				sess.run(ea)

		best_valid_score = 0.0
		best_valid_epoch = 0
		
		if restore:
			saver.restore(sess, config.model_path)

		with open(config.log_train_acc_path, "w") as train_acc_fp, \
			open(config.log_valid_acc_path, "w") as valid_acc_fp:
			for epoch in range(config.num_epoch):

				start_time = time.time()

				if epoch>config.decay_epoch:
					learning_rate = sess.run(model.learning_rate)
					lr_decay = config.lr_decay
					#learning_rate = config.learning_rate
					#lr_decay = config.lr_decay**max(epoch-config.decay_epoch, 0.0)
					sess.run(tf.assign(model.learning_rate, learning_rate*lr_decay))

				print('='*40)
				print(("Epoch %d, Learning rate: %.4f")%(epoch+1, sess.run(model.learning_rate)))
				avg_loss = evaluator.train(train_set, model, sess)
				print(('\ntrain loss: %.4f')%avg_loss)

				if (epoch+1)%5 == 0:
					train_score = evaluator.evaluate(train_set, model, sess)[0]
					print(('train top1 acc: %.4f')%train_score)
					train_acc_fp.write("%d: %.4f\n"%(epoch+1, train_score))

				valid_score = evaluator.evaluate(dev_set, model, sess)[0]
				print(('valid top1 acc: %.4f')%valid_score)
				valid_acc_fp.write("%d: %.4f\n"%(epoch+1, valid_score))

				if valid_score > best_valid_score:
					best_valid_score = valid_score
					best_valid_epoch = epoch
					if config.model_save_by_best_valid:
						saver.save(sess, config.model_path)

				if not config.model_save_by_best_valid and (epoch+1)%config.model_save_period==0:
					saver.save(sess, config.model_path)

				if config.model_save_by_best_valid and epoch-best_valid_epoch > config.early_stop_epoch:
					break

				print("time per epoch is %.2f min"%((time.time()-start_time)/60.0))
		
		if not config.model_save_by_best_valid:
			saver.save(sess, config.model_path)

		print(("\nbest valid top1 acc: %.4f")%best_valid_score)
		test_score = evaluator.evaluate(test_set, model, sess)[0]
		print(('*'*10 + 'test top1 acc: %.4f')%test_score)

def test(config, evaluator):

	data, _ = reader.load_data_for_test(config)
	data = sorted(data.items(), key=lambda x: x[0])
	
	#config.batch_size = 1

	if not os.path.exists(config.log_dir):
		os.mkdir(config.log_dir)

	if len(config.gpu_chosen)>0: 
		gpu_options = tf.GPUOptions(
			visible_device_list=",".join(map(str, config.gpu_chosen)),
			per_process_gpu_memory_fraction=config.gup_per_fraction
		)
	else:
		gpu_options = tf.GPUOptions(
			per_process_gpu_memory_fraction=config.gup_per_fraction
		)

	with tf.Graph().as_default(), \
		tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		# with tf.variable_scope("model", reuse=None):
		model = config.model_func(config)

		init=tf.global_variables_initializer()
		saver = tf.train.Saver()
		
		sess.run(init)
		saver.restore(sess, config.model_path)

		for variable in tf.trainable_variables():
			tensor = tf.constant(variable.eval())
			tf.assign(variable, tensor, name="freeze_weights")

		tf.train.write_graph(
			sess.graph.as_graph_def(), 
			config.model_dir, config.model_name+".pb",
			as_text=config.pb_as_text
		)

		with open(config.log_path, 'w') as fp:
			for key, value in data:

				output = sys.stdout
				outputfile = open(os.path.join(config.log_dir, ".".join([key, config.model_name, "log"])),'w')
				sys.stdout = outputfile

				scores = evaluator.evaluate(value, model, sess, True)

				outputfile.close()
				sys.stdout = output

				fp.write(key + ' score:\n')
				for i, score in enumerate(scores):
					fp.write(("top_"+str(i+1)+": %.4f\n")%score)
				fp.write('\n')

		print("Test Finished! Please check details in %s."%config.log_path)


if __name__ == "__main__":
	config = Configuration()
	evaluator = Evaluator()

	if len(sys.argv) > 1:
		if sys.argv[1]=="continue":
			restore=True
			train(config, evaluator, restore)		
		elif sys.argv[1]=="train":
			restore = False
			train(config, evaluator, restore)
		test(config, evaluator)

	else:
		restore=False
		train(config, evaluator, restore)
		test(config, evaluator)


