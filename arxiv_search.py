from googletrans import Translator
import os
import time
import yaml
import datetime
import warnings
from dataclasses import dataclass
import arxiv
import pandas as pd
import numpy as np
from tqdm import tqdm
# setting
warnings.filterwarnings('ignore')


@dataclass
class Result:
    url: str
    title: str
    abstract: str
    words: list
    score: float = 0.0


def calc_score(abst: str, keywords: dict) -> (float, list):
    sum_score = 0.0
    hit_kwd_list = []

    for word in keywords.keys():
        score = keywords[word]
        if word.lower() in abst.lower():
            sum_score += score
            hit_kwd_list.append(word)
    return sum_score, hit_kwd_list


def search_keyword(
        articles: list, keywords: dict, score_threshold: float
        ) -> list:
    results = []

    for article in articles:
        url = article['arxiv_url']
        title = article['title']
        abstract = article['summary']
        score, hit_keywords = calc_score(abstract, keywords)
        if (score != 0) and (score >= score_threshold):
            tr = Translator()
            time.sleep(0.1)
            title_trans = tr.translate(title, dest="ja").text
            keywords_join = ', '.join(hit_keywords)
            result = [title, title_trans, score, keywords_join, url, abstract]
            results.append(result)
    return results

def get_config() -> dict:
    # file_abs_path = os.path.abspath(__file__)
    # file_dir = os.path.dirname(file_abs_path)
    # config_path = f'{file_dir}/../config.yaml'
    config_path = './config.yaml'
    with open(config_path, 'r') as yml:
        config = yaml.load(yml)
    return config

def main():
    config = get_config()
    subject = config['subject']
    keywords = config['keywords']
    score_threshold = float(config['score_threshold'])
    max_date = int(config['max_date'])

    results = np.array([['title', 'title_ja', 'score', 'keywords', 'url', 'abstract']])
    for i in tqdm(range(2, max_date), ncols=50):
        day_before_yesterday = datetime.datetime.today() - datetime.timedelta(days=i)
        day_before_yesterday_str = day_before_yesterday.strftime('%Y%m%d')
        arxiv_query = f'({subject}) AND ' \
                    f'submittedDate:' \
                    f'[{day_before_yesterday_str}000000 TO {day_before_yesterday_str}235959]'
        articles = arxiv.query(query=arxiv_query,
                            max_results=1000,
                            sort_by='submittedDate',
                            iterative=False)
        results_tmp = search_keyword(articles, keywords, score_threshold)
        if len(results_tmp)==0:
            continue
        results = np.vstack([results, np.array(results_tmp)])

    df_output = pd.DataFrame(results[1:], columns=['title', 'title_ja', 'score', 'keywords', 'url', 'abstract'])
    print('{}件がヒットしました'.format(len(df_output)))
    df_output.to_csv("result.csv", encoding="shift_jis")

if __name__ == "__main__":
    main()