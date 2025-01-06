import time
import random
import yaml

import selenium
from selenium import webdriver
from selenium.webdriver.common.keys import Keys

class Crawler:
    def __init__(self, config):
        self.config = config
        self.driver = webdriver.Chrome()

    def wait_randomly(self, min=0.3, max=2):
        time.sleep(random.uniform(min, max))

    def implicitly_wait(self, implicitly_wait_time=5, min=0.3, max=2):
        self.driver.implicitly_wait(implicitly_wait_time)
        self.wait_randomly(min, max)
