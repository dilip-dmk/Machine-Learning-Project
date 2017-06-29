
import sys


if len(sys.argv)>0:
    file_name = sys.argv[1]
else:
    raise "Usage: python get_bow_word_table.py [file_name]"

id_word_name = file_name+".id_word"
min_count = 5

stop_word_name = "bow_stop_words.new.txt"

stop_words = []
with open(stop_word_name, "r") as file_fp:
    stop_lines = file_fp.readlines()
    for line in stop_lines:
        stop_words.append(line.strip())



remove_words_dict = {}

with open(file_name, "r") as file_fp, \
        open(id_word_name, "w") as id_fp:


    lines = file_fp.readlines()
    word_dict = {}
    
    for line in lines:
        words = line.strip().split(" ")
        for word in words:

            if "." in word:
                word = word.replace(".", "")
            if "-" in word:
                word = word.replace("-", "")
            if word.isalnum():
                continue
            
            if word in stop_words:
                continue

            if "\\" in word:
                if word not in remove_words_dict:
                    remove_words_dict[word] = 1
                continue

            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    
    #for word, _ in remove_words_dict.items():
    #    print word
    
    word_id = 0
    word_dict = sorted(word_dict.items(), key=lambda x: x[1])
    for word, count in word_dict:
        if count >= min_count:
            id_word_line = str(word_id) + '\t' + word + '\t' + str(count) + '\n'
            
            id_fp.write(id_word_line)
            word_id += 1


