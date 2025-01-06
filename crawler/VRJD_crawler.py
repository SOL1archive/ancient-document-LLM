import random
import yaml

import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

from .crawler import Crawler

class VRJD_Crawler(Crawler):
    def crawl(self):
        pass

    def get_target_lt(self):
        self.driver.get(self.config['top_url'])
        self.driver.implicitly_wait(10)

def main():
    with open('./VRJD-crawler-config.yaml', 'r') as f:
        config = yaml.load(f)
    crawler = VRJD_Crawler(config)
