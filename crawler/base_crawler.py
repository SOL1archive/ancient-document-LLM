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
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

class BaseCrawler:
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

    def execute_script_and_wait(
            self,
            js_code: str,
            timeout: float = 8.0,  # 필요시 조정
    ):
        """JS 실행 후 ① 페이지 리로드 또는 ② ins_list/ins_view 등장까지 대기"""
        # (1) 현재 <html> 태그 핸들 → 페이지가 새로 그려지면 stale 이 됨
        old_root = self.driver.find_element(By.TAG_NAME, "html")

        # JS 실행
        self.driver.execute_script(js_code)

        # (2) ① 페이지가 reload 되어 old_root 가 stale 되거나,
        #    ② ins_list 또는 ins_view_in 이 등장할 때까지 기다림
        cond_any = lambda d: (
                EC.staleness_of(old_root)(d) or
                d.find_elements(By.CSS_SELECTOR,
                                "div.ins_list, div.ins_view_in.ins_left_in")
        )

        try:
            WebDriverWait(self.driver, timeout).until(cond_any)
        except TimeoutException:
            # 그래도 안 뜨면 한 번 더 0.5~1s 랜덤 슬립 후 진행
            self.wait_randomly(1.0, 2.0)

    def store(
            self,
            data: Union[list, dict, pd.DataFrame, pd.Series],
            name: str,
            path: Union[Path, str, None]=None, 
            format: Literal['json', 'jsonl', 'csv', 'pickle', 'parquet']='jsonl'):
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
            elif format == 'parquet':
                data.to_parquet(path)
        else:
            if format == 'json' or format == 'jsonl':
                with open(path, 'w') as f:
                    json.dump(data, f)
            elif format == 'csv':
                pd.DataFrame(data).to_csv(path)
            elif format == 'pickle':
                pickle.dump(data, f)
            elif format == 'parquet':
                raise NotImplementedError
