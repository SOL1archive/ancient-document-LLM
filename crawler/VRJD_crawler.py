import random
import yaml

import numpy as np
import pandas as pd
import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup

from .crawler import Crawler

class VRJD_Crawler(Crawler):
    def crawl(self):
        pass

    def get_target_lt(self):
        self.driver.get(self.config['top_url'])
        self.implicitly_wait()
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
        self.target_tags = target_tags
        return self.target_tags

def main():
    with open('./VRJD-crawler-config.yaml', 'r') as f:
        config = yaml.load(f)
    crawler = VRJD_Crawler(config)
