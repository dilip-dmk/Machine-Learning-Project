

"""
This file is to define the useful functions for the project.

"""

from __future__ import absolute_import
from __future__ import division

import os
import numpy as np
import glob


def _read_words_from_line(line):
	elems = line.strip().split("##")
	result = []
	result.append(elems[0].split(" "))
	result.append(elems[1].split(" "))

	return result

def _build_vocab(filename, key=0, value=1, key_func=lambda x:x, value_func=lambda x:x):
	res_dict = {}
	fin = open(filename, "r")
	for line in fin:
		splits = line.strip().split("\t")
		res_dict[key_func(splits[key])] = value_func(splits[value])
	return res_dict


def _file_to_ids(filename, word_to_id, answer_word_to_id, config, read_weight=False):
	num_steps = config.num_steps
	num_classes = config.answer_num_classes

	file_fp = open(filename, "r")
	file_lines = file_fp.readlines()

	num_data = len(file_lines)

	data = np.zeros([num_data, num_steps], dtype=int)
	label = np.zeros([num_data, num_classes], dtype=float)
	length = np.zeros([num_data], dtype=int)

	if read_weight:
		weight_path = filename + ".weight"
		weight_fp = open(weight_path)
		weight_lines = weight_fp.readlines()

	for i, line in enumerate(file_lines):
		query, answer = _read_words_from_line(line)
		
		word_id = []

		for j, word in enumerate(query):
			if word_to_id.has_key(word) and j<num_steps:
				data[i][j] = word_to_id[word]

		length[i] = min(len(query)+1, num_steps)

		for word in answer:
			if answer_word_to_id.has_key(word):
				idx = answer_word_to_id[word]
				label[i][idx] = 1.0

	if read_weight:
		return (data, label, weight, length)

	return (data, label, length)


def load_data(config=None):

	if config:
		train_path = os.path.join(config.data_dir, config.train_file)
		valid_path = os.path.join(config.data_dir, config.valid_file)
		test_path = os.path.join(config.data_dir, config.test_file)
		vocab_path = os.path.join(config.data_dir, config.vocab_file)
		answer_vocab_path = os.path.join(config.data_dir, config.answer_vocab_file)
		docid_path = os.path.join(config.data_dir, config.docid_file)
		read_weight = config.weightRNN_flag
	else:
		data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data/")
		train_path = os.path.join(data_dir, "train.txt")
		valid_path = os.path.join(data_dir, "valid.txt")
		test_path = os.path.join(data_dir, "valid.txt")
		vocab_path = os.path.join(data_dir, "id_word")
		answer_vocab_path = os.path.join(data_dir, "answer_id_word")
		docid_path = os.path.join(data_dir, "id_docId")
		read_weight = False

	word_to_id = _build_vocab(vocab_path, key=1, value=0, value_func=lambda x:int(x))
	answer_word_to_id = _build_vocab(answer_vocab_path, key=1, value=0, value_func=lambda x:int(x))

	data = {}

	data["train"]  = _file_to_ids(train_path, word_to_id, answer_word_to_id, config, read_weight)
	data["valid"]  = _file_to_ids(valid_path, word_to_id, answer_word_to_id, config, read_weight)
	data["test"]  = _file_to_ids(test_path, word_to_id, answer_word_to_id, config, read_weight)

	return data, len(word_to_id)


def load_data_for_test(config=None, split=None):

	# Default Path for test
	data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "data/")
	test_dir = os.path.join(data_dir, "test")
	test_path_dict = {}
	test_path_dict["train"] = os.path.join(test_dir, "train.txt")
	test_path_dict["valid"] = os.path.join(test_dir, "valid.txt")
	test_path_dict["test"] = os.path.join(test_dir, "valid.txt")
	vocab_path = os.path.join(data_dir, "id_word")
	answer_vocab_path = os.path.join(data_dir, "answer_id_word")
	docid_path = os.path.join(data_dir, "id_docId")
	read_weight = False

	if config:
		test_dir = os.path.join(config.data_dir, config.test_folder)
		vocab_path = os.path.join(config.data_dir, "id_word")
		answer_vocab_path = os.path.join(config.data_dir, "answer_id_word")
		docid_path = os.path.join(config.data_dir, "id_docId")
		if split:
			test_path_dict = {}
			for path in split:
				base_name = os.path.splitext(os.path.basename(path))[0]
				test_path_dict[base_name] = path
		else:
			test_path_tmp = glob.glob(os.path.join(test_dir, config.test_post))
			if len(test_path_tmp)>0:
				test_path_dict = {}
				for path in test_path_tmp:
					base_name = os.path.splitext(os.path.basename(path))[0]
					test_path_dict[base_name] = path
		read_weight = config.weightRNN_flag

	word_to_id = _build_vocab(vocab_path, key=1, value=0, value_func=lambda x:int(x))
	answer_word_to_id = _build_vocab(answer_vocab_path, key=1, value=0, value_func=lambda x:int(x))

	data = {}	
	for key, path in test_path_dict.items():
		data[key] =  _file_to_ids(path, word_to_id, answer_word_to_id, config, read_weight)
	
	return data, len(word_to_id)






if __name__ == "__main__":
	pass
