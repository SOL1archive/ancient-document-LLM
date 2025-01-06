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
