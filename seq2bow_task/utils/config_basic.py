"""
This file is to define the configuration of the project.

"""
import os

class BasicConfiguration(object):
	def __init__(self, model_name=None):
		'''
		GPU configuration
		'''
		self.gup_per_fraction = 0.333
		self.gpu_chosen = [0]

		'''
		network parameters
		'''
		self.init_scale = 0.05
		self.hidden_size = 600
		self.num_layers = 2
		self.num_steps = 35
		self.dropout = 0.35
		self.batch_size = 20
		self.wordvec_size = 300
		self.num_k = 10
		self.embedding_random_flag = False

		'''
		training parameters
		'''
		self.learning_rate = 0.5
		self.decay_epoch = 6
		self.lr_decay = 0.9
		self.num_epoch = 39
		self.early_stop_epoch = 10
		self.max_grad_norm = 5
		# model saved flag when meet best valid score
		self.model_save_by_best_valid = True
		# model saved period when model_save_by_best_valid==False
		self.model_save_period = 5

		"""
		data parameters
		"""
		self.shuffle_data = True

		'''
		Trainig and test files path
		'''
		# data_dir is the data folder path, here it is equivalent to "../data/"
		self.data_dir = os.path.join(os.path.dirname(self.base_dir), "data/")
		self.model_dir = os.path.join(os.path.dirname(self.base_dir), "models")
		self.log_dir = os.path.join(os.path.dirname(self.base_dir), "logs")

		self.pb_as_text = False

		# The followings are only the file names (under data_dir folder), not the path.
		# used in training section
		self.train_file = "train.txt"
		self.valid_file = "valid.txt"
		self.test_file = "valid.txt"

		# These two variables used to evaluating section, for example, the test files would
		# be all ".txt" file in "test" folder under the parent 'data_dir'.
		self.test_folder = "test"
		self.test_post = "*.txt"


		# recommend no modification
		self.vocab_file = "id_word"
		self.docid_file = "id_docId"
		self.word_vec_file = "word_vector"

		'''
		Parameters dependens on external files. Generally no need to modify
		'''
		# file path definition
		self.train_path = os.path.join(self.data_dir, self.train_file)
		self.valid_path = os.path.join(self.data_dir, self.valid_file)
		self.test_path = os.path.join(self.data_dir, self.test_file)
		self.vocab_path = os.path.join(self.data_dir, self.vocab_file)
		self.docid_path = os.path.join(self.data_dir, self.docid_file)
		self.word_vec_path = os.path.join(self.data_dir, self.word_vec_file)

		# model path definition
		if model_name:
			self.model_name = model_name
		else:
			self.model_name = self.model_func.__name__
		self.model_path = os.path.join(self.model_dir, self.model_name)

		# log path definition
		self.log_name = "all." + self.model_name + ".log"
		self.log_path = os.path.join(self.log_dir, self.log_name)
		self.log_train_dir = os.path.join(self.log_dir, "train")
		self.log_train_loss_path = os.path.join(self.log_train_dir, self.model_name+".train.loss.log")
		self.log_train_acc_path = os.path.join(self.log_train_dir, self.model_name+".train.acc.log")
		self.log_valid_acc_path = os.path.join(self.log_train_dir, self.model_name+".valid.acc.log")
		self.log_train_fig_path = os.path.join(self.log_train_dir, self.model_name+".png")
		# get numbers from extenal files
		self.num_classes = len(open(self.docid_path, 'r').readlines())
		self.vocab_size = len(open(self.vocab_path, 'r').readlines())
		self.train_size = len(open(self.train_path, 'r').readlines())


		if self.DEBUG:
			self.num_epoch = 1
			self.embedding_random_flag = True
