import datetime as dt
import os
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from pandas_datareader import data as pdr


# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Global Markets Terminal",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .stApp { background: #0b0e13; }
        .block-container {
            max-width: 100%;
            padding-top: 1.2rem;
            padding-left: 2rem;
            padding-right: 2rem;
            padding-bottom: 3rem;
        }
        h1, h2, h3 { letter-spacing: -0.03em; }
        h1 { font-size: 2.05rem !important; margin-bottom: 0.15rem !important; }
        h2 { font-size: 1.35rem !important; margin-top: 1.1rem !important; }
        h3 { font-size: 1.05rem !important; }
        [data-testid="stMetric"] {
            background: #11161f;
            border: 1px solid #202735;
            border-radius: 8px;
            padding: 0.75rem 0.95rem;
        }
        [data-testid="stDataFrame"] {
            border: 1px solid #202735;
            border-radius: 8px;
            overflow: hidden;
        }
        [data-testid="stSidebar"] {
            background: #0f131b;
            border-right: 1px solid #202735;
        }
        .muted {
            color: #8e98a8;
            font-size: 0.86rem;
        }
        .eyebrow {
            color: #8e98a8;
            font-size: 0.70rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-top: 0.35rem;
            margin-bottom: -0.3rem;
        }
        .section-gap { height: 0.6rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================
REQUIRED_COLUMNS = [
    "Ticker", "Name", "Country", "Sector",
    "Industry", "Asset Class", "Theme", "Currency"
]

MAG7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]

OVERVIEW_ASSETS = {
    "S&P 500": "^GSPC",
    "Nasdaq": "^IXIC",
    "S&P 500 Futures": "ES=F",
    "Nasdaq 100 Futures": "NQ=F",
    "Dow Jones": "^DJI",
    "Nikkei 225": "^N225",
    "KOSPI": "^KS11",
    "STI": "^STI"
    "Hang Seng": "^HSI",
    "VIX": "^VIX",
    "Gold": "GC=F",
    "WTI": "CL=F",
    "USD/JPY": "JPY=X",
    "EUR/USD": "EURUSD=X",
    "Bitcoin": "BTC-USD",
}

FX_ASSETS = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "JPY=X",
    "USD/CNY": "CNY=X",
    "USD/SGD": "SGD=X",
    "AUD/USD": "AUDUSD=X",
}

COMMODITY_ASSETS = {
    "Gold": "GC=F",
    "Silver": "SI=F",
    "WTI": "CL=F",
    "Brent": "BZ=F",
    "Copper": "HG=F",
    "Natural Gas": "NG=F",
}

PERIOD_STARTS = {
    "1D": lambda x: x - pd.DateOffset(days=1),
    "5D": lambda x: x - pd.DateOffset(days=5),
    "1M": lambda x: x - pd.DateOffset(months=1),
    "3M": lambda x: x - pd.DateOffset(months=3),
    "6M": lambda x: x - pd.DateOffset(months=6),
    "YTD": lambda x: pd.Timestamp(year=x.year, month=1, day=1),
    "1Y": lambda x: x - pd.DateOffset(years=1),
    "3Y": lambda x: x - pd.DateOffset(years=3),
    "5Y": lambda x: x - pd.DateOffset(years=5),
}

TENORS = ["2 Year", "5 Year", "10 Year", "30 Year"]


# ============================================================
# DATA HELPERS
# ============================================================
@st.cache_data
def load_watchlist(path: str = "Watchlist.csv") -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = [str(c).strip() for c in frame.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in frame.columns]
    if missing:
        raise ValueError(
            "Watchlist.csv is missing columns: " + ", ".join(missing)
        )

    for c in REQUIRED_COLUMNS:
        frame[c] = frame[c].astype(str).str.strip()

    frame = frame.replace({"nan": "", "None": ""})
    return (
        frame[frame["Ticker"] != ""]
        .drop_duplicates("Ticker")
        .reset_index(drop=True)
    )


@st.cache_data(ttl=1800, show_spinner=False)
def download_history(
    tickers: tuple[str, ...],
    period: str = "6y",
) -> pd.DataFrame:
    if not tickers:
        return pd.DataFrame()

    return yf.download(
        list(tickers),
        period=period,
        interval="1d",
        auto_adjust=False,
        group_by="column",
        threads=True,
        progress=False,
    )


@st.cache_data(ttl=900, show_spinner=False)
def download_chart_history(ticker: str, chart_period: str) -> pd.DataFrame:
    """Download intraday or daily data appropriate for the selected chart period."""
    settings = {
        "1D": ("1d", "5m"),
        "5D": ("10d", "15m"),
        "1M": ("1mo", "1h"),
        "3M": ("3mo", "1d"),
        "6M": ("6mo", "1d"),
        "YTD": ("ytd", "1d"),
        "1Y": ("1y", "1d"),
        "3Y": ("3y", "1d"),
        "5Y": ("5y", "1d"),
    }

    period, interval = settings.get(chart_period, ("1y", "1d"))

    return yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )


@st.cache_data(ttl=21600, show_spinner=False)
def get_market_cap(ticker: str):
    try:
        obj = yf.Ticker(ticker)
        try:
            value = obj.fast_info.get("market_cap")
        except Exception:
            value = None
        if value is None:
            try:
                value = obj.info.get("marketCap")
            except Exception:
                value = None
        return float(value) if value is not None else np.nan
    except Exception:
        return np.nan


@st.cache_data(ttl=30, show_spinner=False)
def get_quote_snapshot(ticker: str) -> dict:
    """
    Return one internally consistent Yahoo regular-market quote.

    Priority:
    1. Yahoo quote metadata: regularMarketPrice paired with
       regularMarketPreviousClose.
    2. Yahoo fast_info pair.
    3. Unadjusted daily history fallback.
    """
    price = np.nan
    previous_close = np.nan
    source = "Unavailable"

    try:
        obj = yf.Ticker(ticker)

        # Priority 1: keep both values from the same quote-metadata source.
        try:
            info = obj.get_info()

            info_price = pd.to_numeric(
                info.get("regularMarketPrice"),
                errors="coerce",
            )
            info_previous_close = pd.to_numeric(
                info.get("regularMarketPreviousClose"),
                errors="coerce",
            )

            if (
                not pd.isna(info_price)
                and not pd.isna(info_previous_close)
                and info_previous_close != 0
            ):
                price = float(info_price)
                previous_close = float(info_previous_close)
                source = "Yahoo regular market"
        except Exception:
            pass

        # Priority 2: use fast_info only as a complete pair.
        if pd.isna(price) or pd.isna(previous_close):
            try:
                fast = obj.fast_info

                fast_price = pd.to_numeric(
                    fast.get("lastPrice", fast.get("last_price")),
                    errors="coerce",
                )
                fast_previous_close = pd.to_numeric(
                    fast.get(
                        "previousClose",
                        fast.get("previous_close"),
                    ),
                    errors="coerce",
                )

                if (
                    not pd.isna(fast_price)
                    and not pd.isna(fast_previous_close)
                    and fast_previous_close != 0
                ):
                    price = float(fast_price)
                    previous_close = float(fast_previous_close)
                    source = "Yahoo fast info"
            except Exception:
                pass

        # Priority 3: use only unadjusted daily closes as a pair.
        if pd.isna(price) or pd.isna(previous_close):
            try:
                history = obj.history(
                    period="10d",
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                )

                closes = pd.to_numeric(
                    history.get("Close", pd.Series(dtype=float)),
                    errors="coerce",
                ).dropna()

                if len(closes) >= 2:
                    price = float(closes.iloc[-1])
                    previous_close = float(closes.iloc[-2])
                    source = "Yahoo daily history"
            except Exception:
                pass

    except Exception:
        pass

    change_1d = (
        (price / previous_close - 1) * 100
        if (
            not pd.isna(price)
            and not pd.isna(previous_close)
            and previous_close != 0
        )
        else np.nan
    )

    return {
        "price": price,
        "previous_close": previous_close,
        "change_1d": change_1d,
        "source": source,
    }


@st.cache_data(ttl=60, show_spinner=False)
def get_regular_market_price(ticker: str):
    """Compatibility wrapper for callers that only need the latest price."""
    return get_quote_snapshot(ticker)["price"]


@st.cache_data(ttl=3600, show_spinner=False)
def search_yahoo(query: str) -> pd.DataFrame:
    query = query.strip()
    if not query:
        return pd.DataFrame()

    try:
        quotes = yf.Search(
            query,
            max_results=12,
            news_count=0,
            lists_count=0,
            include_research=False,
        ).quotes

        if not quotes:
            return pd.DataFrame()

        raw = pd.DataFrame(quotes)
        mapping = {
            "symbol": "Ticker",
            "shortname": "Name",
            "longname": "Long Name",
            "exchange": "Exchange",
            "quoteType": "Type",
            "currency": "Currency",
        }

        available = [c for c in mapping if c in raw.columns]
        return (
            raw[available]
            .rename(columns=mapping)
            .drop_duplicates("Ticker")
            .reset_index(drop=True)
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=1800, show_spinner=False)
def get_ticker_news(ticker: str) -> list[dict]:
    try:
        return yf.Ticker(ticker).news or []
    except Exception:
        return []


@st.cache_data(ttl=21600, show_spinner=False)
def get_earnings_date(ticker: str):
    try:
        calendar = yf.Ticker(ticker).calendar
        if calendar is None or len(calendar) == 0:
            return None

        if isinstance(calendar, dict):
            value = calendar.get("Earnings Date")
            if isinstance(value, list) and value:
                return pd.to_datetime(value[0])
            if value is not None:
                return pd.to_datetime(value)

        if isinstance(calendar, pd.DataFrame):
            for key in ["Earnings Date", "EarningsDate"]:
                if key in calendar.index:
                    value = calendar.loc[key].iloc[0]
                    return pd.to_datetime(value)
                if key in calendar.columns:
                    value = calendar[key].iloc[0]
                    return pd.to_datetime(value)
    except Exception:
        return None
    return None


def get_close_series(history: pd.DataFrame, ticker: str) -> pd.Series:
    if history.empty:
        return pd.Series(dtype=float)

    try:
        if isinstance(history.columns, pd.MultiIndex):
            l0 = history.columns.get_level_values(0)
            l1 = history.columns.get_level_values(1)

            if "Close" in l0 and ticker in l1:
                series = history["Close"][ticker]
            elif ticker in l0 and "Close" in l1:
                series = history[ticker]["Close"]
            else:
                return pd.Series(dtype=float)
        else:
            if "Close" not in history.columns:
                return pd.Series(dtype=float)
            series = history["Close"]

        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]

        series = pd.to_numeric(series, errors="coerce").dropna()
        series.index = pd.to_datetime(series.index)
        return series.sort_index()
    except Exception:
        return pd.Series(dtype=float)


def obs_return(prices: pd.Series, n: int):
    if len(prices) <= n:
        return np.nan
    return (prices.iloc[-1] / prices.iloc[-(n + 1)] - 1) * 100


def _normalise_daily_prices(prices: pd.Series) -> pd.Series:
    """Return a clean, date-indexed series of regular-session closes."""
    prices = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    if prices.empty:
        return prices

    index = pd.to_datetime(prices.index)
    if getattr(index, "tz", None) is not None:
        index = index.tz_localize(None)

    # Daily Yahoo data can occasionally contain duplicate timestamps.
    prices.index = index.normalize()
    return prices[~prices.index.duplicated(keep="last")].sort_index()


def _reference_close_on_or_before(
    prices: pd.Series,
    target: pd.Timestamp,
):
    """Final exchange close on or before a calendar boundary."""
    prices = _normalise_daily_prices(prices)
    if prices.empty:
        return np.nan

    target = pd.Timestamp(target)
    if target.tzinfo is not None:
        target = target.tz_localize(None)
    target = target.normalize()

    eligible = prices.loc[prices.index <= target]
    if eligible.empty:
        return np.nan
    return float(eligible.iloc[-1])


def _reference_close_on_or_after(
    prices: pd.Series,
    target: pd.Timestamp,
):
    """First exchange close on or after a calendar boundary."""
    prices = _normalise_daily_prices(prices)
    if prices.empty:
        return np.nan

    target = pd.Timestamp(target)
    if target.tzinfo is not None:
        target = target.tz_localize(None)
    target = target.normalize()

    eligible = prices.loc[prices.index >= target]
    if eligible.empty:
        return np.nan
    return float(eligible.iloc[0])


def _safe_return(end_price, start_price):
    if pd.isna(end_price) or pd.isna(start_price) or start_price == 0:
        return np.nan
    return (float(end_price) / float(start_price) - 1) * 100


def ytd_return(prices: pd.Series, end_price: float | None = None):
    """
    Chart-style YTD return.

    Google/Yahoo chart percentages generally use the first visible close in
    the selected year as the chart base. Each ticker therefore follows its
    own exchange calendar and holidays.
    """
    prices = _normalise_daily_prices(prices)
    if prices.empty:
        return np.nan

    latest_price = float(prices.iloc[-1]) if end_price is None else float(end_price)
    year_start = pd.Timestamp(year=prices.index[-1].year, month=1, day=1)
    reference = _reference_close_on_or_after(prices, year_start)
    return _safe_return(latest_price, reference)


def calculate_period_return(prices: pd.Series):
    """Percentage change from the first to last visible chart point."""
    prices = pd.to_numeric(prices, errors="coerce").dropna().sort_index()
    if len(prices) < 2:
        return np.nan
    return _safe_return(prices.iloc[-1], prices.iloc[0])


def standard_period_return(
    daily_prices: pd.Series,
    period: str,
    end_price: float | None = None,
    previous_close: float | None = None,
):
    """
    Unified dynamic chart-return engine for equities, ETFs and indices.

    Conventions:
    - 1D/3D/5D: trading-session observations.
    - YTD: first exchange close on or after 1 January.
    - 1M/3M/6M/1Y/3Y: first exchange close on or after the calendar boundary.
    - 5Y: final exchange close on or before the exact five-year boundary,
      matching the observed Google/Yahoo convention more closely.

    The series itself supplies the relevant exchange calendar, so US, Japan,
    Hong Kong and other local holidays are handled without hard-coding dates.
    """
    prices = _normalise_daily_prices(daily_prices)
    if len(prices) < 2:
        return np.nan

    latest_price = (
        float(prices.iloc[-1])
        if end_price is None or pd.isna(end_price)
        else float(end_price)
    )

    # Yahoo/Google 1D percentage is current regular-market price versus
    # Yahoo's official regularMarketPreviousClose. Do not infer this from
    # the daily series because that series may already contain today's bar.
    if period == "1D":
        if previous_close is not None and not pd.isna(previous_close):
            return _safe_return(latest_price, previous_close)
        if len(prices) < 2:
            return np.nan
        return _safe_return(prices.iloc[-1], prices.iloc[-2])

    session_map = {"3D": 3, "5D": 5}
    if period in session_map:
        sessions = session_map[period]
        if len(prices) <= sessions:
            return np.nan
        reference = float(prices.iloc[-(sessions + 1)])
        return _safe_return(latest_price, reference)

    if period == "YTD":
        return ytd_return(prices, end_price=latest_price)

    if period not in PERIOD_STARTS:
        return np.nan

    anchor = PERIOD_STARTS[period](prices.index[-1])
    if period == "5Y":
        reference = _reference_close_on_or_before(prices, anchor)
    else:
        reference = _reference_close_on_or_after(prices, anchor)

    return _safe_return(latest_price, reference)


def build_market_table(
    universe: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for _, item in universe.iterrows():
        ticker = item["Ticker"]
        prices = _normalise_daily_prices(get_close_series(history, ticker))
        quote = get_quote_snapshot(ticker)
        live_price = quote["price"]
        previous_close = quote["previous_close"]

        if pd.isna(live_price) and not prices.empty:
            live_price = float(prices.iloc[-1])

        if prices.empty or pd.isna(live_price):
            values = {
                "Price": np.nan,
                "1D %": np.nan,
                "3D %": np.nan,
                "1M %": np.nan,
                "6M %": np.nan,
                "YTD %": np.nan,
                "3Y %": np.nan,
                "5Y %": np.nan,
            }
        else:
            values = {
                "Price": float(live_price),
                "1D %": standard_period_return(
                    prices,
                    "1D",
                    end_price=live_price,
                    previous_close=previous_close,
                ),
                "3D %": standard_period_return(prices, "3D", live_price),
                "1M %": standard_period_return(prices, "1M", live_price),  
                "6M %": standard_period_return(prices, "6M", live_price),
                "YTD %": standard_period_return(prices, "YTD", live_price),
                "3Y %": standard_period_return(prices, "3Y", live_price),
                "5Y %": standard_period_return(prices, "5Y", live_price),
            }

        market_cap = (
            get_market_cap(ticker)
            if item["Asset Class"].casefold() == "equity"
            else np.nan
        )

        rows.append(
            {
                "Ticker": ticker,
                "Name": item["Name"],
                "Country": item["Country"],
                "Sector": item["Sector"],
                "Industry": item["Industry"],
                "Theme": item["Theme"],
                "Currency": item["Currency"],
                "Asset Class": item["Asset Class"],
                "Market Cap": market_cap,
                **values,
            }
        )

    return pd.DataFrame(rows)


def format_market_cap(value):
    if pd.isna(value):
        return "—"
    if abs(value) >= 1e12:
        return f"{value / 1e12:.2f}T"
    if abs(value) >= 1e9:
        return f"{value / 1e9:.1f}B"
    if abs(value) >= 1e6:
        return f"{value / 1e6:.1f}M"
    return f"{value:,.0f}"


def color_return(value):
    if pd.isna(value):
        return "color: #7d8795"
    if value > 0:
        return "color: #33c481; font-weight: 600"
    if value < 0:
        return "color: #ff5c5c; font-weight: 600"
    return "color: #aab2bf"


def display_market_table(title: str, table: pd.DataFrame):
    if table.empty:
        return

    st.markdown(f'<div class="eyebrow">{title}</div>', unsafe_allow_html=True)

    cols = [
        "Ticker", "Name", "Country", "Sector", "Industry", "Currency",
        "Price", "1D %", "3D %", "1M %", "6M %", "YTD %", "3Y %", "5Y %",
        "Market Cap"
    ]

    display = table[cols].copy()
    display["Market Cap"] = display["Market Cap"].map(format_market_cap)

    styled = (
        display.style
        .map(
            color_return,
            subset=["1D %", "3D %", "1M %", "6M %", "YTD %", "3Y %", "5Y %"],
        )
        .format(
            {
                "Price": lambda x: "—" if pd.isna(x) else f"{x:,.2f}",
                "1D %": lambda x: "—" if pd.isna(x) else f"{x:+.2f}%",
                "3D %": lambda x: "—" if pd.isna(x) else f"{x:+.2f}%",
                "1M %": lambda x: "—" if pd.isna(x) else f"{x:+.2f}%",
                "6M %": lambda x: "—" if pd.isna(x) else f"{x:+.2f}%",
                "YTD %": lambda x: "—" if pd.isna(x) else f"{x:+.2f}%",
                "3Y %": lambda x: "—" if pd.isna(x) else f"{x:+.2f}%",
                "5Y %": lambda x: "—" if pd.isna(x) else f"{x:+.2f}%",
            }
        )
    )

    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=min(680, 42 + len(display) * 35),
    )


def make_price_chart(
    prices: pd.Series,
    title: str,
    display_return=None,
):
    period_return = (
        calculate_period_return(prices)
        if display_return is None
        else display_return
    )

    if pd.isna(period_return):
        chart_title = title
    else:
        chart_title = f"{title}   {period_return:+.2f}%"

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode="lines",
            name=title,
            line=dict(width=2),
            hovertemplate=(
                "%{x|%d %b %Y %H:%M}"
                "<br>Price: %{y:,.2f}"
                "<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(
            text=chart_title,
            x=0,
            xanchor="left",
            font=dict(size=18),
        ),
        height=430,
        margin=dict(l=0, r=0, t=55, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hovermode="x unified",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, color="#8e98a8"),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(142,152,168,0.13)",
            zeroline=False,
            color="#8e98a8",
        ),
    )
    return fig


# ============================================================
# MARKET SNAPSHOT
# ============================================================
@st.cache_data(ttl=60, show_spinner=False)
def market_snapshot() -> pd.DataFrame:
    rows = []

    for name, ticker in OVERVIEW_ASSETS.items():
        price = np.nan
        previous_close = np.nan

        try:
            obj = yf.Ticker(ticker)

            try:
                fast = obj.fast_info
                price = fast.get("lastPrice") or fast.get("last_price")
                previous_close = fast.get("previousClose") or fast.get("previous_close")
            except Exception:
                pass

            if price is None or pd.isna(price):
                info = obj.info
                price = info.get("regularMarketPrice", np.nan)
                previous_close = info.get("regularMarketPreviousClose", np.nan)

            if pd.isna(price):
                hist = obj.history(period="2d")
                if not hist.empty:
                    price = float(hist["Close"].iloc[-1])
                    if len(hist) >= 2:
                        previous_close = float(hist["Close"].iloc[-2])

        except Exception:
            pass

        daily_change = (
            (price / previous_close - 1) * 100
            if not pd.isna(price)
            and not pd.isna(previous_close)
            and previous_close != 0
            else np.nan
        )

        rows.append(
            {
                "Name": name,
                "Ticker": ticker,
                "Value": price,
                "1D %": daily_change,
            }
        )

    return pd.DataFrame(rows)

def display_snapshot_cards():
    frame = market_snapshot()

    for start in range(0, len(frame), 4):
        columns = st.columns(4)
        batch = frame.iloc[start : start + 4]

        for col, (_, row) in zip(columns, batch.iterrows()):
            value = "—" if pd.isna(row["Value"]) else f"{row['Value']:,.2f}"
            delta = None if pd.isna(row["1D %"]) else f"{row['1D %']:+.2f}%"
            col.metric(row["Name"], value, delta)


# ============================================================
# YIELDS
# ============================================================
@st.cache_data(ttl=3600, show_spinner=False)
def get_us_yields() -> pd.DataFrame:
    mapping = {
        "2 Year": "DGS2",
        "5 Year": "DGS5",
        "10 Year": "DGS10",
        "30 Year": "DGS30",
    }

    end = dt.datetime.today()
    start = end - dt.timedelta(days=35)
    raw = pdr.DataReader(list(mapping.values()), "fred", start, end)
    raw = raw.ffill().dropna()

    if len(raw) < 2:
        return pd.DataFrame()

    latest = raw.iloc[-1]
    previous = raw.iloc[-2]

    return pd.DataFrame(
        [
            {
                "Country": "United States",
                "Tenor": tenor,
                "Yield": float(latest[code]),
                "Change": (float(latest[code]) - float(previous[code])) * 100,
                "Source": "FRED",
            }
            for tenor, code in mapping.items()
        ]
    )


@st.cache_data
def load_manual_yields(path: str = "BondYields.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    raw = pd.read_csv(path)
    needed = {"Country", "Tenor", "Yield", "Change"}
    if not needed.issubset(raw.columns):
        return pd.DataFrame()

    raw = raw[["Country", "Tenor", "Yield", "Change"]].copy()
    raw["Yield"] = pd.to_numeric(raw["Yield"], errors="coerce")
    raw["Change"] = pd.to_numeric(raw["Change"], errors="coerce")
    raw["Source"] = "BondYields.csv"
    return raw.dropna(subset=["Country", "Tenor", "Yield"])


def get_all_yields() -> pd.DataFrame:
    frames = []

    try:
        us = get_us_yields()
        if not us.empty:
            frames.append(us)
    except Exception:
        pass

    manual = load_manual_yields()
    if not manual.empty:
        frames.append(manual)

    if not frames:
        return pd.DataFrame()

    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates(["Country", "Tenor"], keep="first")
    )


def yield_cell(level, change):
    if pd.isna(level):
        return "—"
    if pd.isna(change):
        return f"{level:.2f}%"
    return f"{level:.2f}%\n{change:+.1f} bps"


def color_yield(value):
    if not isinstance(value, str) or "bps" not in value:
        return "color: #d6dce6"
    try:
        change = float(value.splitlines()[-1].replace("bps", "").strip())
    except Exception:
        return "color: #d6dce6"
    if change > 0:
        return "color: #33c481; font-weight: 600"
    if change < 0:
        return "color: #ff5c5c; font-weight: 600"
    return "color: #aab2bf"


def display_yield_table():
    data = get_all_yields()
    if data.empty:
        st.warning("No government-yield source is currently available.")
        return

    order = ["United States", "Japan", "United Kingdom", "China"]
    available = data["Country"].dropna().unique().tolist()
    countries = [x for x in order if x in available]
    countries += sorted(x for x in available if x not in order)

    display = pd.DataFrame({"Country": countries})

    for tenor in TENORS:
        values = []
        for country in countries:
            match = data[
                (data["Country"] == country)
                & (data["Tenor"] == tenor)
            ]
            if match.empty:
                values.append("—")
            else:
                row = match.iloc[0]
                values.append(yield_cell(row["Yield"], row["Change"]))
        display[tenor] = values

    styled = (
        display.style
        .map(color_yield, subset=TENORS)
        .set_properties(
            subset=["Country"],
            **{"font-weight": "700", "color": "#f3f5f8"},
        )
    )

    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        height=min(350, 42 + len(display) * 48),
    )

    sources = ", ".join(sorted(data["Source"].dropna().unique()))
    st.caption(
        f"Latest yield with daily move underneath. "
        f"Green = yield higher; red = yield lower. Sources: {sources}."
    )


# ============================================================
# CORRELATION
# ============================================================
def build_normalised_frame(
    history: pd.DataFrame,
    tickers: Iterable[str],
    start_date: pd.Timestamp,
) -> pd.DataFrame:
    output = {}

    for ticker in tickers:
        prices = get_close_series(history, ticker)
        prices = prices.loc[prices.index >= start_date]
        if not prices.empty:
            output[ticker] = prices / prices.iloc[0] * 100

    return pd.DataFrame(output)


# ============================================================
# LOAD WATCHLIST AND FILTERS
# ============================================================
try:
    watchlist = load_watchlist()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.sidebar.markdown("## Filters")

selected_asset_classes = st.sidebar.multiselect(
    "Asset class",
    sorted(watchlist["Asset Class"].dropna().unique()),
)

selected_countries = st.sidebar.multiselect(
    "Country",
    sorted(watchlist["Country"].dropna().unique()),
)

selected_sectors = st.sidebar.multiselect(
    "Sector",
    sorted(watchlist["Sector"].dropna().unique()),
)

selected_themes = st.sidebar.multiselect(
    "Theme",
    sorted(watchlist["Theme"].dropna().unique()),
)

filtered_watchlist = watchlist.copy()

if selected_asset_classes:
    filtered_watchlist = filtered_watchlist[
        filtered_watchlist["Asset Class"].isin(selected_asset_classes)
    ]

if selected_countries:
    filtered_watchlist = filtered_watchlist[
        filtered_watchlist["Country"].isin(selected_countries)
    ]

if selected_sectors:
    filtered_watchlist = filtered_watchlist[
        filtered_watchlist["Sector"].isin(selected_sectors)
    ]

if selected_themes:
    filtered_watchlist = filtered_watchlist[
        filtered_watchlist["Theme"].isin(selected_themes)
    ]

st.sidebar.caption(f"{len(filtered_watchlist)} securities selected")


# ============================================================
# HEADER + TABS
# ============================================================
st.title("Global Markets Terminal")
st.markdown(
    '<div class="muted">Cross-asset monitoring for equities, rates, FX and commodities</div>',
    unsafe_allow_html=True,
)

tabs = st.tabs(
    [
        "Overview",
        "Equities",
        "Rates",
        "FX & Commodities",
        "Search",
        "Charts",
        "Correlation",
        "News",
        "Calendar",
    ]
)


# ============================================================
# OVERVIEW
# ============================================================
with tabs[0]:
    st.markdown('<div class="eyebrow">Cross-asset snapshot</div>', unsafe_allow_html=True)
    st.subheader("Market Overview")
    display_snapshot_cards()

    a, b, c, d = st.columns(4)
    a.metric("Universe", len(watchlist))
    b.metric("Filtered", len(filtered_watchlist))
    c.metric("Countries", filtered_watchlist["Country"].nunique())
    d.metric("Updated", dt.datetime.now().strftime("%H:%M"))

    st.markdown('<div class="eyebrow">Rates</div>', unsafe_allow_html=True)
    st.subheader("Government Bond Yields")
    display_yield_table()


# ============================================================
# EQUITIES
# ============================================================
with tabs[1]:
    tickers = tuple(filtered_watchlist["Ticker"].dropna().unique())
    with st.spinner("Updating market data..."):
        history = download_history(tickers)
        market_table = build_market_table(filtered_watchlist, history)

    st.markdown('<div class="eyebrow">Coverage</div>', unsafe_allow_html=True)
    st.subheader("Market Monitor")

    display_market_table(
        "MAG 7",
        market_table[market_table["Ticker"].isin(MAG7)].copy(),
    )

    remaining = market_table[~market_table["Ticker"].isin(MAG7)].copy()

    for asset_class in ["Index", "ETF", "Equity"]:
        section = remaining[
            remaining["Asset Class"].str.casefold() == asset_class.casefold()
        ].copy()
        display_market_table(asset_class.upper(), section)

    other_classes = [
        x for x in remaining["Asset Class"].dropna().unique()
        if x.casefold() not in {"index", "etf", "equity"}
    ]

    for asset_class in sorted(other_classes):
        display_market_table(
            asset_class.upper(),
            remaining[
                remaining["Asset Class"].str.casefold()
                == asset_class.casefold()
            ].copy(),
        )


# ============================================================
# RATES
# ============================================================
with tabs[2]:
    st.markdown('<div class="eyebrow">Sovereign curves</div>', unsafe_allow_html=True)
    st.subheader("Government Bond Yields")
    display_yield_table()

    with st.expander("Add Japan, UK and China yields"):
        st.markdown(
            """
            Put a file named `BondYields.csv` beside `app.py`.

            ```csv
            Country,Tenor,Yield,Change
            Japan,2 Year,0.75,-1.2
            Japan,5 Year,1.10,2.3
            Japan,10 Year,1.85,3.0
            Japan,30 Year,3.10,-0.5
            United Kingdom,2 Year,4.10,-2.0
            China,10 Year,1.90,1.5
            ```

            `Yield` is in percent. `Change` is the daily move in basis points.
            """
        )


# ============================================================
# FX & COMMODITIES
# ============================================================
with tabs[3]:
    st.markdown('<div class="eyebrow">Macro markets</div>', unsafe_allow_html=True)
    st.subheader("FX")

    fx_history = download_history(tuple(FX_ASSETS.values()), "1y")
    fx_rows = []

    for name, ticker in FX_ASSETS.items():
        prices = get_close_series(fx_history, ticker)
        fx_rows.append(
            {
                "Pair": name,
                "Price": np.nan if prices.empty else prices.iloc[-1],
                "1D %": np.nan if len(prices) < 2 else obs_return(prices, 1),
                "YTD %": np.nan if prices.empty else ytd_return(prices),
            }
        )

    fx = pd.DataFrame(fx_rows)
    st.dataframe(
        fx.style
        .map(color_return, subset=["1D %", "YTD %"])
        .format(
            {
                "Price": lambda x: "—" if pd.isna(x) else f"{x:,.4f}",
                "1D %": lambda x: "—" if pd.isna(x) else f"{x:+.2f}%",
                "YTD %": lambda x: "—" if pd.isna(x) else f"{x:+.2f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )

    st.markdown('<div class="eyebrow">Real assets</div>', unsafe_allow_html=True)
    st.subheader("Commodities")

    commodity_history = download_history(
        tuple(COMMODITY_ASSETS.values()), "1y"
    )
    commodity_rows = []

    for name, ticker in COMMODITY_ASSETS.items():
        prices = get_close_series(commodity_history, ticker)
        commodity_rows.append(
            {
                "Commodity": name,
                "Price": np.nan if prices.empty else prices.iloc[-1],
                "1D %": np.nan if len(prices) < 2 else obs_return(prices, 1),
                "YTD %": np.nan if prices.empty else ytd_return(prices),
            }
        )

    commodities = pd.DataFrame(commodity_rows)
    st.dataframe(
        commodities.style
        .map(color_return, subset=["1D %", "YTD %"])
        .format(
            {
                "Price": lambda x: "—" if pd.isna(x) else f"{x:,.2f}",
                "1D %": lambda x: "—" if pd.isna(x) else f"{x:+.2f}%",
                "YTD %": lambda x: "—" if pd.isna(x) else f"{x:+.2f}%",
            }
        ),
        width="stretch",
        hide_index=True,
    )


# ============================================================
# SEARCH
# ============================================================
with tabs[4]:
    st.markdown('<div class="eyebrow">Discovery</div>', unsafe_allow_html=True)
    st.subheader("Security Search")

    query = st.text_input(
        "Search any listed security",
        placeholder="Microsoft, DBS, Toyota, Samsung, KWEB...",
        label_visibility="collapsed",
    )

    if query.strip():
        results = search_yahoo(query)

        if results.empty:
            st.info("No matching securities were returned.")
        else:
            st.dataframe(results, width="stretch", hide_index=True)

            choices = results["Ticker"].dropna().tolist()
            if choices:
                selected = st.selectbox("Open result", choices)
                period = st.radio(
                    "Period",
                    ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"],
                    horizontal=True,
                    key="search_period",
                )

                search_history = download_chart_history(selected, period)
                prices = get_close_series(search_history, selected)

                if prices.empty:
                    st.warning(f"No price history returned for {selected}.")
                else:
                    daily_history = download_history((selected,), "6y")
                    daily_prices = get_close_series(daily_history, selected)
                    quote = get_quote_snapshot(selected)
                    live_price = quote["price"]
                    previous_close = quote["previous_close"]
                    if pd.isna(live_price):
                        live_price = float(daily_prices.iloc[-1])

                    selected_return = standard_period_return(
                        daily_prices,
                        period,
                        end_price=live_price,
                        previous_close=previous_close,
                    )
                    one_day = standard_period_return(
                        daily_prices,
                        "1D",
                        end_price=live_price,
                        previous_close=previous_close,
                    )
                    five_day = standard_period_return(
                        daily_prices, "5D", end_price=live_price
                    )
                    ytd_perf = standard_period_return(
                        daily_prices, "YTD", end_price=live_price
                    )

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Price", f"{live_price:,.2f}")
                    m2.metric(
                        f"{period} Return",
                        "—" if pd.isna(selected_return)
                        else f"{selected_return:+.2f}%",
                    )
                    m3.metric(
                        "1 Day",
                        "—" if pd.isna(one_day) else f"{one_day:+.2f}%",
                    )
                    m4.metric(
                        "5 Day",
                        "—" if pd.isna(five_day) else f"{five_day:+.2f}%",
                    )
                    m5.metric(
                        "YTD",
                        "—" if pd.isna(ytd_perf) else f"{ytd_perf:+.2f}%",
                    )

                    st.plotly_chart(
                        make_price_chart(
                            prices,
                            selected,
                            selected_return,
                        ),
                        width="stretch",
                    )


# ============================================================
# CHARTS
# ============================================================
with tabs[5]:
    st.markdown('<div class="eyebrow">Technical view</div>', unsafe_allow_html=True)
    st.subheader("Price History")

    if filtered_watchlist.empty:
        st.info("No securities match the current filters.")
    else:
        left, right = st.columns([2, 3])

        with left:
            ticker = st.selectbox(
                "Security",
                filtered_watchlist["Ticker"].tolist(),
                format_func=lambda x: (
                    f"{x} — "
                    f"{filtered_watchlist.loc[filtered_watchlist['Ticker'] == x, 'Name'].iloc[0]}"
                ),
            )

        with right:
            period = st.radio(
                "Period",
                ["1D", "5D", "1M", "3M", "6M", "YTD", "1Y", "3Y", "5Y"],
                horizontal=True,
                key="main_period",
            )

        chart_data = download_chart_history(ticker, period)
        prices = get_close_series(chart_data, ticker)

        if prices.empty:
            st.warning(f"No price history returned for {ticker}.")
        else:
            daily_history = download_history((ticker,), "6y")
            daily_prices = get_close_series(daily_history, ticker)
            quote = get_quote_snapshot(ticker)
            live_price = quote["price"]
            previous_close = quote["previous_close"]
            if pd.isna(live_price):
                live_price = float(daily_prices.iloc[-1])

            selected_return = standard_period_return(
                daily_prices,
                period,
                end_price=live_price,
                previous_close=previous_close,
            )
            one_day = standard_period_return(
                daily_prices,
                "1D",
                end_price=live_price,
                previous_close=previous_close,
            )
            five_day = standard_period_return(
                daily_prices, "5D", end_price=live_price
            )
            ytd_perf = standard_period_return(
                daily_prices, "YTD", end_price=live_price
            )

            metric_1, metric_2, metric_3, metric_4, metric_5 = st.columns(5)

            metric_1.metric(
                "Price",
                f"{live_price:,.2f}",
            )

            metric_2.metric(
                f"{period} Return",
                "—" if pd.isna(selected_return)
                else f"{selected_return:+.2f}%",
            )

            metric_3.metric(
                "1 Day",
                "—" if pd.isna(one_day) else f"{one_day:+.2f}%",
            )

            metric_4.metric(
                "5 Day",
                "—" if pd.isna(five_day) else f"{five_day:+.2f}%",
            )

            metric_5.metric(
                "YTD",
                "—" if pd.isna(ytd_perf) else f"{ytd_perf:+.2f}%",
            )

            st.plotly_chart(
                make_price_chart(
                    prices,
                    ticker,
                    selected_return,
                ),
                width="stretch",
            )


# ============================================================
# CORRELATION
# ============================================================
with tabs[6]:
    st.markdown('<div class="eyebrow">Risk analytics</div>', unsafe_allow_html=True)
    st.subheader("Correlation Matrix")

    default_corr = [
        x for x in ["^GSPC", "^IXIC", "^N225", "^HSI", "GC=F", "CL=F", "BTC-USD"]
        if x in list(OVERVIEW_ASSETS.values()) or True
    ]

    corr_choices = sorted(
        set(filtered_watchlist["Ticker"].tolist())
        | set(OVERVIEW_ASSETS.values())
    )

    selected_corr = st.multiselect(
        "Select 2 to 10 assets",
        corr_choices,
        default=default_corr[:6],
        max_selections=10,
    )

    if len(selected_corr) >= 2:
        corr_history = download_history(tuple(selected_corr), "1y")
        returns = {}

        for ticker in selected_corr:
            prices = get_close_series(corr_history, ticker)
            if not prices.empty:
                returns[ticker] = prices.pct_change()

        return_frame = pd.DataFrame(returns).dropna(how="all")

        if return_frame.shape[1] >= 2:
            corr = return_frame.corr()

            fig = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
                zmin=-1,
                zmax=1,
            )
            fig.update_layout(
                height=520,
                margin=dict(l=0, r=0, t=20, b=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Not enough overlapping data for a correlation matrix.")
    else:
        st.info("Choose at least two assets.")


# ============================================================
# NEWS
# ============================================================
with tabs[7]:
    st.markdown('<div class="eyebrow">Headlines</div>', unsafe_allow_html=True)
    st.subheader("Company News")

    news_ticker = st.selectbox(
        "Security",
        filtered_watchlist["Ticker"].tolist()
        if not filtered_watchlist.empty
        else watchlist["Ticker"].tolist(),
        key="news_ticker",
    )

    news_items = get_ticker_news(news_ticker)

    if not news_items:
        st.info("No news was returned for this security.")
    else:
        for item in news_items[:10]:
            content = item.get("content", item)
            title = content.get("title", "Untitled")
            provider = content.get("provider", {})
            provider_name = (
                provider.get("displayName", "")
                if isinstance(provider, dict)
                else str(provider)
            )
            summary = content.get("summary", "")
            canonical = content.get("canonicalUrl", {})
            url = canonical.get("url") if isinstance(canonical, dict) else None

            if url:
                st.markdown(f"**[{title}]({url})**")
            else:
                st.markdown(f"**{title}**")

            if provider_name:
                st.caption(provider_name)
            if summary:
                st.write(summary)
            st.divider()


# ============================================================
# CALENDAR
# ============================================================
with tabs[8]:
    st.markdown('<div class="eyebrow">Events</div>', unsafe_allow_html=True)
    st.subheader("Earnings Calendar")

    equities = watchlist[
        watchlist["Asset Class"].str.casefold() == "equity"
    ].copy()

    selected_calendar_names = st.multiselect(
        "Choose up to 15 companies",
        equities["Ticker"].tolist(),
        default=[x for x in MAG7 if x in equities["Ticker"].tolist()],
        max_selections=15,
    )

    rows = []

    for ticker in selected_calendar_names:
        rows.append(
            {
                "Ticker": ticker,
                "Name": equities.loc[
                    equities["Ticker"] == ticker, "Name"
                ].iloc[0],
                "Next Earnings": get_earnings_date(ticker),
            }
        )

    calendar = pd.DataFrame(rows)

    if calendar.empty:
        st.info("Choose one or more equities.")
    else:
        calendar = calendar.sort_values(
            "Next Earnings",
            na_position="last",
        )
        st.dataframe(
            calendar,
            width="stretch",
            hide_index=True,
            column_config={
                "Next Earnings": st.column_config.DatetimeColumn(
                    "Next Earnings",
                    format="DD MMM YYYY",
                )
            },
        )


st.divider()
st.caption(
    "Market prices, search, news, market capitalisation and earnings data are "
    "retrieved through yfinance. US Treasury yields are retrieved from FRED. "
    "International government yields use the optional BondYields.csv file. "
    "Equity and ETF performance figures are price returns based on regular-session "
    "Yahoo closes and do not reinvest dividends. Data may be delayed and is for research use."
)
