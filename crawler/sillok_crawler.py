#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import yaml
import pandas as pd
from tqdm.auto import tqdm

from vrjd_crawler import VRJD_Crawler


def build_dataset(cfg_path: Path, out_path: Path, flush_every: int = 1000) -> None:
    """
    실록 사이트를 순회하며 번역쌍을 수집한 뒤 Parquet 파일로 직렬화

    Parameters
    ----------
    cfg_path : Path
        `sillok-crawler-config.yaml` 경로
    out_path : Path
        최종 Parquet 파일 출력 경로
    flush_every : int
        **누적 레코드(row) 수가 flush_every 에 도달할 때마다** 중간 결과를 저장
    """
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    crawler = VRJD_Crawler(config)

    target_df = crawler.get_target_df()  # 전체 임금

    if target_df.empty:
        print("⚠️  임금 목록을 불러오지 못했습니다. URL·HTML 구조를 다시 확인하세요.")
        sys.exit(1)

    records: list[dict] = []
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(
        target_df.iterrows(),
        total=len(target_df),
        desc="👑 Crawling Sillok",
        unit="king",
    ):
        king_name   = row["name"]
        order       = int(row["order"])
        start_year  = row["start_year"]

        try:
            king_df = crawler.get_given_king(king_name)
        except Exception as e:
            print(f"[WARN] {king_name} 수집 실패 → {e}")
            continue

        # king_df 의 각 행 → 하나의 레코드
        for _, r in king_df.iterrows():
            records.append(
                {
                    "king"          : king_name,
                    "king_order"         : order,
                    # "start_year"    : start_year,
                    "year"          : int(r["year"])           if pd.notna(r["year"]) else None,
                    "relative_year" : int(r["relative_year"])  if pd.notna(r["relative_year"]) else None,
                    "month"         : r["month"],
                    "entry_order":r["entry_order"] if pd.notna(r["entry_order"]) else None,
                    "original_text"        : r["original_text"],
                    "translated_text"    : r["translated_text"],
                }
            )

            if flush_every and len(records) >= flush_every:
                flush_to_parquet(records, out_path)
                records.clear()

    if records:
        flush_to_parquet(records, out_path)


def flush_to_parquet(batch: list[dict], path: Path) -> None:
    if not batch:
        return

    df_new = pd.DataFrame(batch)

    if path.exists():
        df_old = pd.read_parquet(path, engine="pyarrow")
        df_combined = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_combined = df_new

    df_combined.to_parquet(
        path,
        engine="pyarrow",
        compression="brotli",
        index=False,
    )

    # CSV로도 저장
    csv_path = path.with_suffix(".csv")
    df_combined.to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"✅ Saved {len(df_combined):,} rows → {path} & {csv_path}")




def parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="조선왕조실록 번역 데이터셋(Parquet) 생성 스크립트"
    )
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("sillok-crawler-config.yaml"),
        help="VRJD_Crawler 설정 YAML 경로",
    )
    p.add_argument(
        "--out",
        "-o",
        type=Path,
        default=Path("output/sillok_finetune.parquet"),
        help="저장할 Parquet 파일 경로",
    )
    p.add_argument(
        "--flush",
        "-f",
        type=int,
        default=1000,
        help="N명 임금당 Parquet flush 주기(메모리 관리용)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_cli()
    build_dataset(args.config, args.out, args.flush)
    print("데이터셋 생성이 완료되었습니다.")


if __name__ == "__main__":
    main()
