import arxiv
import pandas as pd
import numpy as np
import datetime as dt
import yaml
from sentence_transformers import SentenceTransformer

def get_config() -> dict:
    config_path = './config.yaml'
    with open(config_path, 'r') as yml:
        config = yaml.load(yml, Loader=yaml.Loader)
    return config

def calc_score(model, title, keywords_embeddings):
    title_embeddings = model.encode(title)
    score = np.dot(title_embeddings, keywords_embeddings)

    if np.isnan(score):
        return -1
    else:
        return score

def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def main(model):
    config = get_config()
    subject = config['subject']
    keywords_dic = config['keywords']
    keywords = list(keywords_dic.keys())
    keywords_embeddings_tmp = model.encode(keywords)

    keywords_weight = np.array(list(keywords_dic.values()))
    keywords_weight = keywords_weight / keywords_weight.sum()
    
    keywords_embeddings = [0]*keywords_embeddings_tmp.shape[1]
    for i in range(keywords_embeddings_tmp.shape[0]):
        keywords_embeddings += keywords_embeddings_tmp[i,:]*keywords_weight[i]

    num_papers = int(config['num_papers'])
    go_back_date = int(config['go_back_date'])

    output = []
    df_output = pd.DataFrame()
    for i in range(2, go_back_date+2):
        day_target = dt.datetime.today() - dt.timedelta(days=i)
        day_target_str = day_target.strftime('%Y%m%d')
        arxiv_query = f'({subject}) AND ' \
                    f'submittedDate:' \
                    f'[{day_target_str}000000 TO {day_target_str}235959]'
        articles = arxiv.query(query=arxiv_query,
                            max_results=1000,
                            sort_by='submittedDate',
                            iterative=False)
        print('--')
        print(day_target_str)
        cnt = 0
        for article in articles:
            score = calc_score(model, article['title'], keywords_embeddings)
            output.append([article['title'], day_target.strftime('%Y-%m-%d'), article['arxiv_url'], article['journal_reference'], score])
            cnt += 1
        print(f'{cnt}件ヒット')

        if i % 10 == 0:
            df_output_tmp = pd.DataFrame(output, columns=['title', 'published_date', 'url', 'journal_reference', 'score'])
            df_output = pd.concat([df_output, df_output_tmp])
            df_output = df_output.sort_values(by='score', ascending=False).reset_index(drop=True)
            df_output = df_output.iloc[:num_papers]
            output = []
            print('----')
            print('reset')
            print('----')
        elif i == go_back_date+1:
            df_output_tmp = pd.DataFrame(output, columns=['title', 'published_date', 'url', 'journal_reference', 'score'])
            df_output = pd.concat([df_output, df_output_tmp])
    df_output = df_output.sort_values(by='score', ascending=False).reset_index(drop=True)
    df_output.iloc[:num_papers].to_csv('result.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')
main(model)

