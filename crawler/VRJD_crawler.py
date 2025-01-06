import random
import yaml
import re

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

    def get_target_lt(self):
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
    
    def get_given_king(self, king_name):
        if self.target_df is None:
            self.get_target_lt()
        result = dict()
        js_code = self.target_df.loc[self.target_df['name'] == king_name, 'href'].iloc[0]
        self.driver.execute(js_code)
        html = self.driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        
        collections_js_code = soup.find(attrs='king_year1 clear2').find('span')['onclick']
        self.driver.execute(collections_js_code)
        result['collection'] = self.parse_collection()
        self.driver.back()

        return result

    def parse_collection(self):
        collection_html = self.driver.page_source
        collection_soup = BeautifulSoup(collection_html, 'html.parser')
        original_text = (collection_soup.find('div', 'ins_view_in ins_right_in')
                                        .find('p', 'paragraph')
                                        .text
        )
        translated_text = (collection_soup.find('div', 'ins_view_in ins_right_in')
                                          .find('p', 'paragraph')
                                          .text
        )
        return {
            'original_text': original_text,
            'translated_text': translated_text,
        }

def main():
    with open('./VRJD-crawler-config.yaml', 'r') as f:
        config = yaml.load(f)
    crawler = VRJD_Crawler(config)

if __name__ == '__main__':
    main()
