import string
import matplotlib.pyplot as plt  
import numpy as np

from config_basic import BasicConfiguration


def train_plot(config=None):
	if not config:
		return

	log_train_loss_path = config.log_train_loss_path
	log_train_acc_path = config.log_train_acc_path
	log_valid_acc_path = config.log_valid_acc_path

	with open(log_train_loss_path, 'r') as loss_fp, \
		open(log_train_acc_path, 'r') as train_acc_fp, \
		open(log_valid_acc_path, 'r') as valid_acc_fp:

		# for loss 
		loss_lines = loss_fp.readlines()
		x_loss = []
		y_loss = []
		percent_prev = 0
		iter_num = 0
		for line in loss_lines:
			elems = line.strip().split(':')
			precent = float(elems[0])
			loss = float(elems[1])
			if precent<percent_prev:
				iter_num += 1
			x_loss.append(precent+iter_num)
			y_loss.append(loss)
			percent_prev = precent

		# for train accuracy
		train_acc_lines = train_acc_fp.readlines()
		x_train_acc = []
		y_train_acc = []
		for line in train_acc_lines:
			elems = line.strip().split(':')
			epoch = float(elems[0])
			acc = float(elems[1])
			x_train_acc.append(epoch)
			y_train_acc.append(acc)


		# for valid accuracy
		valid_acc_lines = valid_acc_fp.readlines()
		x_valid_acc = []
		y_valid_acc = []
		for line in valid_acc_lines:
			elems = line.strip().split(':')
			epoch = float(elems[0])
			acc = float(elems[1])
			x_valid_acc.append(epoch)
			y_valid_acc.append(acc)
		
	fig = plt.figure()

	ax1= fig.add_subplot(111)
	loss_curve, = ax1.plot(x_loss, y_loss, 'r', label="train loss")
	ax1.set_ylabel('Loss')

	ax2 = ax1.twinx()
	train_acc_curve, = ax2.plot(x_train_acc, y_train_acc, 'g', label="train acc")
	valid_acc_curve, = ax2.plot(x_valid_acc, y_valid_acc, 'b', label="valid acc")
	ax2.set_ylabel('Accuracy')

	ax1.set_xlabel('Eopch')
	ax1.set_title('Training Curve')
	plt.legend(handles = [loss_curve, train_acc_curve, valid_acc_curve])
	plt.savefig(config.log_train_fig_path)
