from gensim.models import word2vec
import logging

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.Text8Corpus('./wikiextractor/wiki.txt')

model = word2vec.Word2Vec(sentences, vector_size=200, min_count=20, window=15)
model.wv.save_word2vec_format("./wiki.vec.pt", binary=True)


# target = ['ADV', 'VERB', 'NOUN', 'ADJ', 'PROPN', 'SYM']
# non_target = ['is', 'am', 'are']

# dic={'CC':'X',
# 'CD':'NUM',
# 'DT':'X',
# 'EX':'X', 
# 'FW':'X', 
# 'IN':'X', 
# 'JJ':'ADJ',
# 'JJR':'ADJ',
# 'JJS':'ADJ',
# 'LS':'X',
# 'MD':'X',
# 'NN':'NOUN',
# 'NNS':'NOUN',
# 'NNP':'PROPN',
# 'NNPS':'PROPN',
# 'PDT':'X',
# 'POS':'X',
# 'PRP':'X',
# 'PRP$':'X',
# 'RB':'ADV',
# 'RBR':'ADV',
# 'RBS':'ADV',
# 'RP':'X',
# 'SYM':'SYM',
# 'TO':'X',
# 'UH':'INTJ',
# 'VB':'VERB',
# 'VBD':'VERB',
# 'VBG':'VERB',
# 'VBN':'VERB',
# 'VBP':'VERB',
# 'VBZ':'VERB',
# 'WDT':'X',
# 'WP':'X',
# 'WP$':'X',
# 'WRB':'X'}

