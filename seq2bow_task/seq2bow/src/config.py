
"""
This file is to define the configuration of the project.

"""
import sys, os
from model_seq2bow import Seq2bow_RNN

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
utils_dir = os.path.join(base_dir, "utils")
sys.path.append(utils_dir)
from config_basic import BasicConfiguration

class Configuration(BasicConfiguration):
	def __init__(self, model_name=None):
		self.DEBUG = False
		##### Classification Models #####
		self.model_func = Seq2bow_RNN

		self.base_dir = os.path.dirname(os.path.realpath(__file__))

		super(self.__class__, self).__init__()
		
		self.model_save_by_best_valid = True
		self.atn_hidden_size = 600
		self.weightRNN_flag = False
		self.decay_epoch = 0

		self.answer_vocab_file= "answer_id_word"
		self.answer_vocab_path = os.path.join(self.data_dir, self.answer_vocab_file)
		self.answer_num_classes = len(open(self.answer_vocab_path, 'r').readlines())

		'''
		GPU configuration
		'''
		self.gup_per_fraction = 0.333
		self.gpu_chosen = [0]


