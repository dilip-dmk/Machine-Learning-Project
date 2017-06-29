import logging
import os.path
import sys
import multiprocessing

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

# Usage: .py query_path model_path

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    # check and process input arguments
    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)
    inp,  outp2 = sys.argv[1:]

    model = Word2Vec(LineSentence(inp), size=300, window=5, min_count=2, sg=1,
            workers=multiprocessing.cpu_count())

    # trim unneeded model memory = use(much) less RAM
    #model.init_sims(replace=True)
    #model.save(outp1)
    model.wv.save_word2vec_format(outp2, binary=False)
