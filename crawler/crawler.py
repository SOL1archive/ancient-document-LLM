from typing import Union
import time
import random
import yaml

import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options

class Crawler:
    def __init__(self, config: dict):
        self.config = config
        self.chrome_options = Options()
        if self.config['headless']:
            self.chrome_options.add_argument("--headless")
        self.driver = webdriver.Chrome(options=self.chrome_options)

    def wait_randomly(self, min: float=0.3, max: float=2):
        time.sleep(random.uniform(min, max))

    def implicitly_wait(self, implicitly_wait_time: float=5., min: float=0.3, max: float=2):
        self.driver.implicitly_wait(implicitly_wait_time)
        self.wait_randomly(min, max)

    def move_to_top_url(self):
        self.driver.get(self.config['top_url'])
        self.implicitly_wait()

    def execute_script_and_wait(self, js_code: str):
        self.driver.execute_script(js_code)
        self.implicitly_wait()
