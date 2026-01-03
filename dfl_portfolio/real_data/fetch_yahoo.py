from __future__ import annotations

import json
import os
import time
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


def _normalize_tickers(tickers: Iterable[str]) -> List[str]:
    cleaned = [t.strip().upper() for t in tickers if t and t.strip()]
    # キャッシュ共有のため、同じ集合なら同じキーになるよう順序を正規化する
    return sorted(set(cleaned))


def _download_yahoo(
    ticker_list: List[str],
    start_dt: datetime,
    end_dt: datetime,
    *,
    interval: str,
    auto_adjust: bool,
    debug: bool,
    allow_empty: bool = False,
) -> pd.DataFrame:
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
        if allow_empty:
            return pd.DataFrame()
        raise RuntimeError("Yahoo Finance からデータを取得できませんでした。")

    # yfinance の MultiIndex 列をフラットな {ティッカー}_{列名} に統一
    if isinstance(data.columns, pd.MultiIndex):
        stacked = data.stack(0).rename_axis(["date", "ticker"]).reset_index()
        df = (
            stacked.pivot(index="date", columns="ticker")
            .sort_index()
            .swaplevel(axis=1)
        )
    else:
        df = data.copy()

    df.columns = ["_".join(map(str, col)).strip("_") for col in df.columns]
    df.index.name = "date"
    return df.sort_index()


def _cache_name(
    ticker_list: List[str],
    start_dt: datetime,
    *,
    interval: str,
    auto_adjust: bool,
) -> str:
    # 重要: end をキャッシュキーに含めない（末尾延長で過去が変わる問題を避ける）
    slug = "_".join(ticker_list)
    adj = "adj1" if auto_adjust else "adj0"
    return f"{slug}_{start_dt:%Y%m%d}_{interval}_{adj}.csv"


def _slice_df(df: pd.DataFrame, start_dt: datetime, end_dt: datetime) -> pd.DataFrame:
    start_ts = pd.Timestamp(start_dt)
    end_ts = pd.Timestamp(end_dt)
    # end はユーザーの直感に合わせて inclusive で切り出す
    return df.loc[(df.index >= start_ts) & (df.index <= end_ts)].copy()


def _append_only_merge(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    if existing.empty:
        merged = new.copy()
    elif new.empty:
        merged = existing.copy()
    else:
        merged = pd.concat([existing, new], axis=0).sort_index()
        # “過去更新禁止”のため、重複日があれば既存行（先に来る方）を優先
        merged = merged[~merged.index.duplicated(keep="first")]
    merged.index.name = "date"
    return merged


def _atomic_to_csv(df: pd.DataFrame, path: Path) -> None:
    """Write CSV atomically to avoid corrupt/empty reads under multi-process runs."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp_{os.getpid()}")
    df.to_csv(tmp)
    os.replace(tmp, path)


def _safe_read_cached_csv(path: Path, *, debug: bool, retries: int = 8, sleep_sec: float = 0.25) -> pd.DataFrame:
    """Read cached CSV with retries to tolerate concurrent writers."""
    last_exc: Exception | None = None
    for i in range(max(1, int(retries))):
        try:
            # A concurrent writer may have created/truncated the file; wait until it has content.
            try:
                size = path.stat().st_size
            except FileNotFoundError:
                size = 0
            if size <= 0:
                raise pd.errors.EmptyDataError("cache file is empty (likely being written)")
            return pd.read_csv(path, parse_dates=["date"], index_col="date").sort_index()
        except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
            last_exc = exc
            if debug:
                print(f"[real-data] cache read retry {i+1}/{retries} failed: {path} ({exc})")
            time.sleep(sleep_sec * (1.5**i))
    assert last_exc is not None
    raise last_exc


def fetch_yahoo_prices(
    tickers: Iterable[str],
    start: str | datetime,
    end: str | datetime,
    *,
    interval: str = "1d",
    auto_adjust: bool = True,
    cache_dir: Path | None = None,
    cache_readonly: bool = False,
    force_refresh: bool = False,
    allow_empty: bool = False,
    debug: bool = True,
) -> Tuple[pd.DataFrame, FetchResult]:
    """ティッカー群の OHLCV を取得し、フラットな DataFrame とメタ情報を返す。"""
    if yf is None:
        raise YahooFinanceUnavailable(
            "yfinance がインストールされていないためデータ取得に失敗しました。"
        )

    ticker_list = _normalize_tickers(tickers)
    if not ticker_list:
        raise ValueError("少なくとも1つのティッカーを指定してください。")

    start_dt = _ensure_datetime(start)
    end_dt = _ensure_datetime(end)
    if end_dt <= start_dt:
        raise ValueError("end は start より後の日付を指定してください。")

    cache_root = cache_dir or DEFAULT_CACHE
    if cache_readonly:
        if not cache_root.exists():
            raise FileNotFoundError(
                f"cache-dir '{cache_root}' does not exist (cache is read-only). "
                "Disable --cache-readonly or provide an existing --cache-dir."
            )
    else:
        cache_root.mkdir(parents=True, exist_ok=True)

    cache_name = _cache_name(ticker_list, start_dt, interval=interval, auto_adjust=auto_adjust)
    cache_path = cache_root / cache_name

    # 既存キャッシュがあれば読込。
    # end はキーに含めないため、まずは “スナップショット” を読み、その後必要なら末尾だけ追記する。
    effective_start_dt = start_dt
    effective_end_dt = end_dt

    if cache_path.exists() and not force_refresh:
        if debug:
            print(f"[real-data] cache hit (append-only): {cache_path}")
        cached = _safe_read_cached_csv(cache_path, debug=debug)
        cache_min = cached.index.min()
        cache_max = cached.index.max()

        if cache_readonly:
            # Read-only mode: never touch the cache. If the requested range exceeds the
            # cached span, clamp to the available span instead of failing (common when
            # the requested start/end lands on a non-trading day or in the future).
            effective_start_dt = max(pd.Timestamp(start_dt), cache_min).to_pydatetime()
            effective_end_dt = min(pd.Timestamp(end_dt), cache_max).to_pydatetime()
            if effective_end_dt <= effective_start_dt:
                raise ValueError(
                    "cache is read-only but has no overlap with the requested date range. "
                    f"requested=[{pd.Timestamp(start_dt).date()}..{pd.Timestamp(end_dt).date()}], "
                    f"cache=[{cache_min.date()}..{cache_max.date()}]. "
                    "Disable --cache-readonly, change --cache-dir, or use --force-refresh."
                )
            if debug and (effective_start_dt != start_dt or effective_end_dt != end_dt):
                print(
                    "[real-data] cache-readonly: clamped requested range to cache span",
                    {
                        "requested": [pd.Timestamp(start_dt).date().isoformat(), pd.Timestamp(end_dt).date().isoformat()],
                        "effective": [pd.Timestamp(effective_start_dt).date().isoformat(), pd.Timestamp(effective_end_dt).date().isoformat()],
                        "cache": [cache_min.date().isoformat(), cache_max.date().isoformat()],
                    },
                )
            df = _slice_df(cached, effective_start_dt, effective_end_dt)
        else:
            cache_modified = False
            # 末尾延長: キャッシュの最終日以降のみ追記（過去は更新しない）
            if pd.Timestamp(end_dt) > cache_max:
                fetch_start = (cache_max + pd.Timedelta(days=1)).to_pydatetime()
                fetched_tail = _download_yahoo(
                    ticker_list,
                    fetch_start,
                    end_dt,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    debug=debug,
                    allow_empty=True,
                )
                cached = _append_only_merge(cached, fetched_tail)
                cache_modified = True

            # 先頭拡張（必要な場合のみ）：キャッシュより前の期間が要求された場合、先頭側を追加する。
            if pd.Timestamp(start_dt) < cache_min:
                fetch_end = (cache_min - pd.Timedelta(days=1)).to_pydatetime()
                fetched_head = _download_yahoo(
                    ticker_list,
                    start_dt,
                    fetch_end,
                    interval=interval,
                    auto_adjust=auto_adjust,
                    debug=debug,
                    allow_empty=True,
                )
                cached = _append_only_merge(fetched_head, cached)
                cache_modified = True

            # キャッシュ内容が変わった場合のみスナップショットを保存し直す
            if cache_modified:
                _atomic_to_csv(cached, cache_path)
            df = _slice_df(cached, start_dt, end_dt)
    else:
        if cache_readonly:
            raise FileNotFoundError(
                f"cache file '{cache_path}' is missing (cache is read-only). "
                "Disable --cache-readonly, change --cache-dir, or use --force-refresh."
            )
        # force_refresh または初回: 全期間を新規取得して保存
        df_full = _download_yahoo(
            ticker_list,
            start_dt,
            end_dt,
            interval=interval,
            auto_adjust=auto_adjust,
            debug=debug,
            allow_empty=allow_empty,
        )
        _atomic_to_csv(df_full, cache_path)
        if debug:
            print(f"[real-data] cached fresh download -> {cache_path}")
        df = _slice_df(df_full, start_dt, end_dt)

    # 取得メタ情報を JSON 化しやすい dataclass にまとめて返す
    result = FetchResult(
        tickers=ticker_list,
        start=effective_start_dt.isoformat(),
        end=effective_end_dt.isoformat(),
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
