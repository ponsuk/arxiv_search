import nltk
import time
import yaml
import datetime
import warnings
from dataclasses import dataclass
import arxiv
import pandas as pd
import numpy as np
from tqdm import tqdm
import gensim.models
warnings.filterwarnings('ignore')


@dataclass
class Result:
    url: str
    title: str
    abstract: str
    words: list
    score: float = 0.0

def calc_score(title: str, keywords:list, keywords_weight, word_vectors) -> (float):
    target_part_of_speech = ['ADV', 'VERB', 'NOUN', 'ADJ', 'PROPN']
    non_target_verb = ['is', 'am', 'are']
    dic_part_of_speech ={'CC':'X','CD':'NUM','DT':'X','EX':'X','FW':'X','IN':'X', 'JJ':'ADJ','JJR':'ADJ','JJS':'ADJ','LS':'X','MD':'X','NN':'NOUN','NNS':'NOUN','NNP':'PROPN','NNPS':'PROPN','PDT':'X','POS':'X','PRP':'X','PRP$':'X','RB':'ADV','RBR':'ADV','RBS':'ADV','RP':'X','SYM':'SYM','TO':'X','UH':'INTJ','VB':'VERB','VBD':'VERB','VBG':'VERB','VBN':'VERB','VBP':'VERB','VBZ':'VERB','WDT':'X','WP':'X','WP$':'X','WRB':'X'}
 
    morph = nltk.word_tokenize(title.lower())
    pos = nltk.pos_tag(morph)

    sum_vec = np.zeros(200)
    word_count = 0

    for p in pos:
        if p[1] not in dic_part_of_speech:
            continue
        else:
            if (dic_part_of_speech[p[1]] in target_part_of_speech) & (p[0] not in non_target_verb):
                try:
                    sum_vec += word_vectors[p[0]]
                    word_count += 1
                except KeyError:
                    continue
    v1 = sum_vec / word_count

    scores = []
    for keyword in keywords:
        v2 = word_vectors[keyword]
        scores.append(cos_sim(v1, v2))

    score = np.dot(scores, keywords_weight)

    if np.isnan(score):
        return -1
    else:
        return score

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# def search_keyword(
#         articles: list, keywords: dict, score_threshold: float
#         ) -> list:
#     results = []

#     for article in articles:
#         url = article['arxiv_url']
#         title = article['title']
#         print(title)
#         abstract = article['summary']
#         score, hit_keywords = calc_score(abstract, keywords)
#         if (score != 0) and (score >= score_threshold):
#             tr = Translator()
#             time.sleep(0.1)
#             title_trans = tr.translate(title, dest="ja").text
#             keywords_join = ', '.join(hit_keywords)
#             result = [title, title_trans, score, keywords_join, url, abstract]
#             results.append(result)
#     return results

def get_config() -> dict:
    config_path = './config.yaml'
    with open(config_path, 'r') as yml:
        config = yaml.load(yml)
    return config

def main():
    config = get_config()
    subject = config['subject']
    keywords_dic = config['keywords']
    keywords = list(keywords_dic.keys())
    keywords_weight = np.array(list(keywords_dic.values()))
    keywords_weight = keywords_weight / keywords_weight.sum()

    num_papers = int(config['num_papers'])
    go_back_date = int(config['go_back_date'])

    model_path = "./word2vec_test/wiki.vec.pt"
    word_vectors = gensim.models.KeyedVectors.load_word2vec_format(model_path, binary=True)


    output = []
    for i in tqdm(range(2, go_back_date+2), ncols=50):
        day_target = datetime.datetime.today() - datetime.timedelta(days=i)
        day_target_str = day_target.strftime('%Y%m%d')
        arxiv_query = f'({subject}) AND ' \
                    f'submittedDate:' \
                    f'[{day_target_str}000000 TO {day_target_str}235959]'
        articles = arxiv.query(query=arxiv_query,
                            max_results=1000,
                            sort_by='submittedDate',
                            iterative=False)
        for article in articles:
            score = calc_score(article['title'], keywords, keywords_weight, word_vectors)
            output.append([article['title'], article['arxiv_url'], score])

    df_output = pd.DataFrame(np.array(output), columns=['title', 'url', 'score'])
    df_output = df_output.sort_values('score', ascending=False).reset_index(drop=True)
    df_output = df_output.loc[:num_papers,:]
    df_output.to_csv("./result.csv")


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()
    print('elapsed time : {}[sec]'.format(round((end - start),2)))
