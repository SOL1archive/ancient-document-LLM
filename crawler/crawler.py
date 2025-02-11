from typing import Union, Literal
import time
import random
import yaml
from pathlib import Path
import pickle

import json
import numpy as np
import pandas as pd

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

    def store(
            self,
            data: Union[list, dict, pd.DataFrame, pd.Series],
            name: str,
            path: Union[Path, str, None]=None, 
            format: Literal['json', 'jsonl', 'csv', 'pickle']='json'):
        if path is None:
            path = self.config['target_path']
        path = Path(path) / (name + '.' + format)
        if isinstance(data, pd.DataFrame) or isinstance(data, pd.Series):
            if format == 'json':
                data.to_json(path)
            elif format == 'jsonl':
                data.to_json(path, lines=True)
            elif format == 'csv':
                data.to_csv(path)
            elif format == 'pickle':
                data.to_pickle(path)
        else:
            if format == 'json' or format == 'jsonl':
                with open(path, 'w') as f:
                    json.dump(data, f)
            elif format == 'csv':
                pd.DataFrame(data).to_csv(path)
            elif format == 'pickle':
                pickle.dump(data, f)
