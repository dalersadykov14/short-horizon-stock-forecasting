import pandas as pd
import yfinance as yf

def load_prices(ticker: str, years: int = 5) -> pd.Series:
    """
    Returns daily Close prices as a tz-naive pandas Series indexed by date.
    Raises ValueError if data is missing/empty.
    """
    period = f"{years}y"
    df = yf.download(ticker, period=period, interval="1d", auto_adjust=False, progress=False)

    if df is None or df.empty:
        raise ValueError(f"No data returned for {ticker} from yfinance.")

    # yfinance sometimes returns multi-index columns; normalize
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    if "Close" not in df.columns:
        raise ValueError(f"Missing Close column for {ticker}. Columns: {list(df.columns)}")

    close = df["Close"].dropna()

    if close.empty:
        raise ValueError(f"Close series empty after dropna for {ticker}.")

    # ensure datetime index is tz-naive
    close.index = pd.to_datetime(close.index).tz_localize(None)

    close.name = ticker
    return close
