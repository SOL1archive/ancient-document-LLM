from typing import union
from pathlib import Path
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
        self.get_target_lt()
        for king in tqdm(self.target_df['name']):
            self.get_given_king(king)
            self.move_to_top_url()

    def get_target_lt(self) -> pd.DataFrame:
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
            self.get_target_lt()
        self.move_to_top_url()
        result = dict()
        js_code = self.target_df.loc[self.target_df['name'] == king_name, 'href'].iloc[0]
        self.execute_script_and_wait(js_code)
        
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        collections_js_code = soup.find('ul', 'king_year1 clear2').find('span')['onclick']
        self.execute_script_and_wait(collections_js_code)
        result['collection'] = self.parse_collection(self.driver.page_source)
        self.driver.back()

        return result
    
    def preprocess_text(self, text: str) -> str:
        return (text.strip()
                    .replace('\t', ' ')
        )

    def parse_collection(self, collection_html) -> dict:
        collection_soup = BeautifulSoup(collection_html, 'html.parser')
        original_text = (collection_soup.find('div', 'ins_view_in ins_left_in')
                                        .find_all('p', 'paragraph')
        )
        original_text = '\n'.join([tag.text for tag in original_text])
        translated_text = (collection_soup.find('div', 'ins_view_in ins_right_in')
                                          .find_all('p', 'paragraph')
        )
        translated_text = '\n'.join([tag.text for tag in translated_text])
        return {
            'original_text': original_text,
            'translated_text': translated_text.strip(),
        }

def main():
    main_path = Path(__file__).parent
    with open(main_path / 'sillok-crawler-config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    crawler = VRJD_Crawler(config)
    crawler.get_target_lt()
    print(crawler.target_df)
    print('=' * 60)
    pprint(crawler.get_given_king('순종'))

if __name__ == '__main__':
    main()
