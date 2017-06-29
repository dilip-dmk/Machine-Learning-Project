
import tensorflow as tf

from config import *

def main(_):
	config = Configuration()

	if len(config.gpu_chosen)>0: 
		gpu_options = tf.GPUOptions(
			per_process_gpu_memory_fraction=config.gup_per_fraction, 
			visible_device_list=",".join(map(str, config.gpu_chosen))
		)
	else:
		gpu_options = tf.GPUOptions(
			per_process_gpu_memory_fraction=config.gup_per_fraction
		)

	with tf.Graph().as_default(), tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:

		initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)

		# with tf.variable_scope("model", reuse=None):
		model = config.model_func(config)
			
		_all = set(tf.global_variables())
		saver = tf.train.Saver(list(_all))
		saver.restore(session, config.model_path)

		with open(config.model_path+".txt", "w") as fp:
			for ele in _all:
				print ele.name 
				shape = ele.get_shape()
				value = session.run(ele)
				if (len(shape)) == 0:
					print "0 0"
				elif (len(shape)) == 1:
					print "1 "+str(shape[0])
					print " ".join([str(x) for x in value])
				elif (len(shape)) == 2:
					print str(shape[0])+" "+str(shape[1])
					for i in range(shape[0]):
						print " ".join([str(x) for x in value[i]])
				else:
					line = ""
					for i in range(len(shape)):
						line += str(shape[i])+" "
					print line


if __name__ == "__main__":
	tf.app.run()
