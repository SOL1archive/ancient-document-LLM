import random
import yaml
import re

import numpy as np
import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

from .crawler import Crawler

class VRJD_Crawler(Crawler):
    def __init__(self, config):
        super().__init__(config)
        self.target_df = None

    def crawl(self):
        self.get_target_lt()
        for king in self.target_df['name']:
            self.get_given_king(king)
            self.move_to_top_url()

    def get_target_lt(self):
        self.move_to_top_url()
        result_html = self.driver.page_source
        soup = BeautifulSoup(result_html, 'html.parser')
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
                'start_year': start_year
            })
        self.target_df = pd.DataFrame(target_tags)
        return self.target_df
    
    def get_given_king(self, king_name):
        if self.target_df is None:
            self.get_target_lt()
        
        js_code = self.target_df.loc[self.target_df['name'] == king_name, 'href']
        self.driver.execute(js_code)

def main():
    with open('./VRJD-crawler-config.yaml', 'r') as f:
        config = yaml.load(f)
    crawler = VRJD_Crawler(config)
