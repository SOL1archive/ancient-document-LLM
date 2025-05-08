from typing import Union
from pathlib import Path

import requests
import yaml
import re
from pprint import pprint

from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

from crawler import Crawler

class VRJD_Crawler(Crawler):
    def __init__(self, config):
        super().__init__(config)
        self.target_df = None

    def crawl(self):
        self.get_target_df()
        
        total_data = []
        for i, row in tqdm(self.target_df.iterrows()):
            king = row['name']
            data = self.get_given_king(king)
            data['name'] = king
            data['order'] = row['order']
            data['start_year'] = row['start_year']
            total_data.append(pd.DataFrame(data))
            self.move_to_top_url()

    def get_target_df(self) -> pd.DataFrame:
        self.move_to_top_url()
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        target_tags = soup.find(id='m_cont_list').find_all('a')
        result_lt = []
        for tag in target_tags:
            href = tag['href']
            text = tag.text
            order = re.findall(r'[0-9]+', text)
            if len(order) == 0:
                order = result_lt[-1]['order']
            else:
                order = int(order[0])
            if '대 ' in text:
                name = text[text.find('대 ') + 2 : text.find('(')]
            else:
                name = text[: text.find('(')]
            start_year = text[text.find('(') + 1 : text.find('~')]
            result_lt.append({
                'href': href,
                'order': order,
                'name': name,
                'start_year': start_year,
            })
        self.target_df = pd.DataFrame(result_lt)
        return self.target_df
    
    def get_given_king(self, king_name: str) -> dict:
        if self.target_df is None:
            self.get_target_df()
        self.move_to_top_url()
        js_code = self.target_df.loc[self.target_df['name'] == king_name, 'href'].iloc[0]
        self.execute_script_and_wait(js_code)
        
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        result = dict()
        # 총서 파싱

        collections_js_code = soup.find('ul', 'king_year1 clear2').find('span')['onclick']
        self.execute_script_and_wait(collections_js_code)

        soup = BeautifulSoup(self.driver.page_source, 'html.parser')
        result['collection'] = self.parse_collection(soup)

        # 본문 파싱
        main_js_code = soup.find('ul')
        # 부록 파싱

        self.driver.back()
        return result
    
    def preprocess_text(self, text: str) -> str:
        return (text.strip()
                    .replace('\t', ' ')
        )

    def parse_collection(self, page_soup) -> dict:
        if page_soup.find('div','ins_list'):
            return self.parse_list(page_soup)
        else:
            return self.parse_content(page_soup)

    def parse_list(self, page_soup) -> dict:
        original_texts = []
        translated_texts = []

        session = requests.Session()

        for li in page_soup.select('li'):
            a = li.find('a')
            if not a or 'searchView' not in a.get('href', ''):
                continue

            content_id = a['href'].split("'")[1]
            content_url = f"https://sillok.history.go.kr/id/{content_id}"
            resp = session.get(content_url, timeout=10)
            resp.raise_for_status()

            content_soup = BeautifulSoup(resp.text, 'html.parser')
            parsed = self.parse_content(content_soup)

            original_texts.append(parsed['original_text'])
            translated_texts.append(parsed['translated_text'])

        return {
            'original_text': '\n'.join(original_texts),
            'translated_text': '\n'.join(translated_texts),
        }

    def parse_content(self, page_soup) -> dict:
        translated_text = (page_soup.find('div', 'ins_view_in ins_left_in')
                         .find_all('p', 'paragraph')
                         )
        translated_text = '\n'.join([self.preprocess_text(tag.text) for tag in translated_text])
        original_text = (page_soup.find('div', 'ins_view_in ins_right_in')
                           .find_all('p', 'paragraph')
                           )
        original_text = '\n'.join([self.preprocess_text(tag.text) for tag in original_text])
        return {
            'original_text': original_text,
            'translated_text': translated_text,
        }

def main():
    main_path = Path(__file__).parent
    with open(main_path / 'sillok-crawler-config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    crawler = VRJD_Crawler(config)
    crawler.get_target_df()
    print(crawler.target_df)
    print('=' * 60)
    pprint(crawler.get_given_king('세조'))

if __name__ == '__main__':
    main()
