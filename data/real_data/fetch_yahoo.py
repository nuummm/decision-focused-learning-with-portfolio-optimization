from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd

# Yahoo Finance API ラッパー（環境によっては未導入のため optional）
try:
    import yfinance as yf  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yf = None


# ダウンロード済み CSV を再利用するローカルキャッシュ
DEFAULT_CACHE = Path(__file__).resolve().parent / "cache"


# 取得条件やファイルパスなどを記録するためのメタ情報
@dataclass
class FetchResult:
    tickers: List[str]
    start: str
    end: str
    interval: str
    auto_adjust: bool
    cache_path: Path
    n_rows: int
    n_cols: int

    def to_json(self) -> str:
        payload = asdict(self)
        payload["cache_path"] = str(self.cache_path)
        return json.dumps(payload, ensure_ascii=False, indent=2)


class YahooFinanceUnavailable(RuntimeError):
    pass


def _ensure_datetime(value: str | datetime | pd.Timestamp) -> datetime:
    """文字列/Timestamp を datetime に統一する小物関数。"""
    if isinstance(value, datetime):
        return value
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    return pd.Timestamp(value).to_pydatetime()


def fetch_yahoo_prices(
    tickers: Iterable[str],
    start: str | datetime,
    end: str | datetime,
    *,
    interval: str = "1d",
    auto_adjust: bool = True,
    cache_dir: Path | None = None,
    force_refresh: bool = False,
    debug: bool = True,
) -> Tuple[pd.DataFrame, FetchResult]:
    """ティッカー群の OHLCV を取得し、フラットな DataFrame とメタ情報を返す。"""
    if yf is None:
        raise YahooFinanceUnavailable(
            "yfinance がインストールされていないためデータ取得に失敗しました。"
        )

    ticker_list = [t.strip().upper() for t in tickers if t.strip()]
    if not ticker_list:
        raise ValueError("少なくとも1つのティッカーを指定してください。")

    start_dt = _ensure_datetime(start)
    end_dt = _ensure_datetime(end)
    if end_dt <= start_dt:
        raise ValueError("end は start より後の日付を指定してください。")

    cache_root = cache_dir or DEFAULT_CACHE
    cache_root.mkdir(parents=True, exist_ok=True)

    slug = "_".join(ticker_list)
    cache_name = f"{slug}_{start_dt:%Y%m%d}_{end_dt:%Y%m%d}_{interval}.csv"
    cache_path = cache_root / cache_name

    # 既存キャッシュがあれば読込、なければ Yahoo から取得
    if cache_path.exists() and not force_refresh:
        if debug:
            print(f"[real-data] cache hit: {cache_path}")
        df = pd.read_csv(cache_path, parse_dates=["date"], index_col="date")
    else:
        if debug:
            print(
                "[real-data] fetching from Yahoo Finance",
                {
                    "tickers": ticker_list,
                    "start": start_dt.isoformat(),
                    "end": end_dt.isoformat(),
                    "interval": interval,
                    "auto_adjust": auto_adjust,
                },
            )
        data = yf.download(
            tickers=ticker_list,
            start=start_dt,
            end=end_dt,
            interval=interval,
            auto_adjust=auto_adjust,
            group_by="ticker",
            progress=False,
            threads=True,
        )
        if data.empty:
            raise RuntimeError("Yahoo Finance からデータを取得できませんでした。")

        # yfinance の MultiIndex 列をフラットな {ティッカー}_{列名} に統一
        if isinstance(data.columns, pd.MultiIndex):
            data = data.stack(0).rename_axis(["date", "ticker"]).reset_index()
            df = (data.pivot(index="date", columns="ticker")
                    .sort_index()
                    .swaplevel(axis=1))
        else:
            df = data.copy()

        df.columns = ["_".join(map(str, col)).strip("_") for col in df.columns]
        df.index.name = "date"
        df = df.sort_index()
        df.to_csv(cache_path)
        if debug:
            print(f"[real-data] cached fresh download -> {cache_path}")

    # 取得メタ情報を JSON 化しやすい dataclass にまとめて返す
    result = FetchResult(
        tickers=ticker_list,
        start=start_dt.isoformat(),
        end=end_dt.isoformat(),
        interval=interval,
        auto_adjust=auto_adjust,
        cache_path=cache_path,
        n_rows=df.shape[0],
        n_cols=df.shape[1],
    )

    if debug:
        print("[real-data] fetch summary:")
        print(result.to_json())
        print(df.head())
        print(df.tail())

    return df, result


__all__ = [
    "FetchResult",
    "YahooFinanceUnavailable",
    "fetch_yahoo_prices",
]
