
import sys


if len(sys.argv)>0:
	model_name = sys.argv[1]
else:
	raise "Usage: python split_word_vec_model.py [model_name]"

id_word_name = model_name+".id_word"
word_vec_name = model_name+".word_vec"

with open(model_name, "r") as model_fp, \
		open(id_word_name, "w") as id_fp, \
		open(word_vec_name, "w") as vec_fp:
	lines = model_fp.readlines()
	
	
	for i, line in enumerate(lines):
		if i==0:
			word = "<eos>"
			vec = ["0"]*300
		else:
			elems = line.strip().split(" ")
			word = elems[0]
			vec = elems[1:]
		
		id_word_line = str(i) + '\t' + word + '\n'
		word_vec_line = " ".join(vec) + '\n'

		id_fp.write(id_word_line)
		vec_fp.write(word_vec_line)


