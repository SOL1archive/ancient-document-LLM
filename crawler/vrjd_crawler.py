from __future__ import annotations

from typing import Union, Callable, Type
from pathlib import Path
import functools
import time
import re
import yaml
import requests
import numpy as np
import pandas as pd
from pprint import pprint
from bs4 import BeautifulSoup
from tqdm.auto import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import selenium

from base_crawler import BaseCrawler  # 기존 파일 그대로 사용

def retry(
    exceptions: Union[Type[BaseException], tuple[Type[BaseException], ...]],
    tries: int = 5,
    delay: float = 1.0,
    backoff: float = 2.0,
) -> Callable[[Callable], Callable]:
    """
    지정한 예외가 발생하면 최대 `tries` 번까지 재시도한다.
    delay(초) → delay*backoff → … 로 지수적으로 증가.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            _tries, _delay = tries, delay
            while _tries > 0:
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    _tries -= 1
                    if _tries == 0:
                        raise          # 마지막 시도도 실패하면 예외 재전파
                    print(f"[retry] {fn.__name__}: {e} — 남은 재시도 {_tries}회")
                    time.sleep(_delay)
                    _delay *= backoff
        return wrapper
    return decorator

class VRJD_Crawler(BaseCrawler):
    def __init__(self, config):
        super().__init__(config)
        self.target_df: pd.DataFrame | None = None

    def _iter_year_month_blocks(self, king_page_soup):
        """
        king_year2 블록(연도) → 하위 li(월)까지 순회하며
        (연도, 상대연도, 월표기, onclick JS) 튜플 yield
        """
        for year_li in king_page_soup.select("ul.king_year2.clear2 > li"):
            header_div = year_li.find("div")
            if not header_div:   # 방어 코드
                continue
            header_txt = header_div.get_text(" ", strip=True)
            relative_year = int(re.search(r"(\d+)년", header_txt)[1])
            year = int(re.search(r"(\d{4})년", header_txt)[1])

            for month_a in year_li.select("ul.clear2 > li > a"):
                month = month_a.text.strip()
                js_code = month_a["href"][len("javascript:") :]
                yield year, relative_year, month, js_code

    def crawl(self):
        """
        왕별 크롤링을 수행한다.
        - get_given_king() 실패 → 해당 왕 스킵, 다음 왕으로 진행
        - 최소 1명이라도 성공하면 결과 DataFrame 반환
        - 전부 실패하면 빈 DataFrame 반환
        """
        self.get_target_df()      # 임금 목록
        total_data: list[pd.DataFrame] = []

        for _, row in tqdm(self.target_df.iterrows(), total=len(self.target_df)):
            king = row["name"]
            try:
                king_df = self.get_given_king(king)           # ⇐ 재시도 포함
            except Exception as e:
                print(f"[skip] {king}: 모든 재시도 실패 — {e} — 건너뜀")
                # 혹시 남은 탐색을 위해 페이지를 원상복구
                try:
                    self.move_to_top_url()
                except Exception:
                    pass
                continue

            king_df["name"] = king
            king_df["order"] = row["order"]
            king_df["start_year"] = row["start_year"]

            total_data.append(king_df)
            self.move_to_top_url()  # 다음 왕을 위해 초기 화면 복귀

        if total_data:
            return pd.concat(total_data, ignore_index=True)
        # 전부 실패했을 때는 빈 DataFrame 반환
        return pd.DataFrame(
            columns=["king", "year", "relative_year", "month",
                     "original_text", "translated_text",
                     "name", "order", "start_year","entry_order"]
        )

    def get_target_df(self) -> pd.DataFrame:
        if self.target_df is not None:
            return self.target_df

        self.move_to_top_url()
        soup = BeautifulSoup(self.driver.page_source, "html.parser")
        target_tags = soup.find(id="m_cont_list").find_all("a")

        rows = []
        for tag in target_tags:
            href, text = tag["href"], tag.text
            order_match = re.findall(r"\d+", text)
            order = int(order_match[0]) if order_match else rows[-1]["order"]

            name = (
                text[text.find("대 ") + 2 : text.find("(")]
                if "대 " in text
                else text[: text.find("(")]
            )
            start_year = text[text.find("(") + 1 : text.find("~")]

            rows.append(
                {"href": href, "order": order, "name": name, "start_year": start_year}
            )

        self.target_df = pd.DataFrame(rows)
        return self.target_df

    @retry(
        (AttributeError, selenium.common.exceptions.WebDriverException),
        tries=5,
        delay=1,
        backoff=2,
    )
    def get_given_king(self, king_name: str) -> pd.DataFrame:
        # 왕별 첫 화면
        if self.target_df is None:
            self.get_target_df()
        self.move_to_top_url()

        top_js = self.target_df.loc[self.target_df.name == king_name, "href"].iloc[0]
        self.execute_script_and_wait(top_js)

        king_page = BeautifulSoup(self.driver.page_source, "html.parser")

        # 총서
        collection_span = king_page.select_one("ul.king_year1 span")
        rows = []
        if collection_span:
            collection_js = collection_span["onclick"]
            self.execute_script_and_wait(collection_js)
            parsed = self.parse_collection_with_retry(
                BeautifulSoup(self.driver.page_source, "html.parser")
            )
            for p in parsed:
                rows.append(
                    {
                        "king": king_name,
                        "year": None,
                        "relative_year": None,
                        "month": "총서",
                        **p,
                    }
                )
            self.driver.back()  # 왕 페이지로 복귀

        # 연, 월 본문
        king_page = BeautifulSoup(self.driver.page_source, "html.parser")
        for year, rel_year, month, js in self._iter_year_month_blocks(king_page):
            print(f"{king_name} {year} {month}")
            self.execute_script_and_wait(js)

            parsed = self.parse_collection_with_retry(
                BeautifulSoup(self.driver.page_source, "html.parser")
            )
            for p in parsed:
                rows.append(
                    {
                        "king": king_name,
                        "year": year,
                        "relative_year": rel_year,
                        "month": month,
                        **p
                    }
                )
            self.driver.back()

        return pd.DataFrame(rows)

    def preprocess_text(self, text: str) -> str:
        return text.strip().replace("\t", " ")

    def parse_collection_with_retry(self, soup: BeautifulSoup) -> list[dict]:
        """
        목록/본문 자동 판별 + 재시도
        파싱 과정에서 AttributeError 가 나면 driver.refresh() 뒤 다시 시도
        """
        for attempt in range(5):
            try:
                return self.parse_collection(soup)
            except AttributeError as e:
                if attempt == 4:
                    raise
                print(f"[retry] parse_collection: {e} — 재시도 {attempt+1}/4")
                self.driver.refresh()
                time.sleep(1.5)
                soup = BeautifulSoup(self.driver.page_source, "html.parser")

    def parse_collection(self, soup: BeautifulSoup) -> list[dict]:
        if soup.find("div", "ins_list"):
            return self.parse_list(soup)
        return [self.parse_content(soup)]


    @retry(
        (AttributeError, requests.exceptions.RequestException),
        tries=5,
        delay=1,
        backoff=2,
    )
    def parse_list(self, page_soup: BeautifulSoup) -> list[dict]:
        session = requests.Session()
        rows = []

        cnt = 1
        for idx, li in enumerate(page_soup.select("li"), start=1):
            a = li.find("a")
            if not a or "searchView" not in a.get("href", ""):
                continue

            content_id = a["href"].split("'")[1]
            content_url = f"https://sillok.history.go.kr/id/{content_id}"

            resp = session.get(content_url, timeout=10)
            resp.raise_for_status()

            parsed = self.parse_content(BeautifulSoup(resp.text, "html.parser"))
            parsed["entry_order"] = cnt
            cnt += 1
            rows.append(parsed)

        return rows

    @retry(AttributeError, tries=5, delay=1, backoff=2)
    def parse_content(self, page_soup: BeautifulSoup) -> dict:
        def clean(text: str) -> str:
            return " ".join(text.split())

        # 각주 수집
        footnotes = {}
        for li in page_soup.select("ul.ins_footnote li.clear2"):
            sup = li.find("a", class_="idx_annotation04_foot")
            if not sup:
                continue
            num = int(sup.text.strip("[]註 )"))
            desc_div = li.find("div", class_=re.compile(r"type_chr\d+"))
            desc = desc_div.get_text(strip=True) if desc_div else ""
            footnotes[num] = desc

        def inline_footnotes(container) -> str:
            parts = []
            for el in container.children:
                if el.name == "a":
                    href = el.get("href", "")
                    if href.startswith("#footnote_"):
                        num = int(href.lstrip("#footnote_"))
                        parts.append(f"({footnotes.get(num, el.get_text()).strip()})")
                    else:
                        parts.append(el.get_text())
                else:
                    parts.append(el.get_text())
            return "".join(parts)

        left_div = page_soup.find("div", "ins_view_in ins_left_in")
        right_div = page_soup.find("div", "ins_view_in ins_right_in")
        if not (left_div and right_div):
            raise AttributeError("본문 div를 찾을 수 없음")

        translated_lines = [
            clean(inline_footnotes(p)) for p in left_div.find_all("p", "paragraph")
        ]
        original_lines = [
            clean(inline_footnotes(p)) for p in right_div.find_all("p", "paragraph")
        ]

        return {
            "original_text": "\n".join(original_lines),
            "translated_text": "\n".join(translated_lines),
            "entry_order": None,
        }