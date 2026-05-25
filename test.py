import pandas as pd
import yfinance as yf
import gspread
import requests
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestRegressor


# ================== GOOGLE SHEETS ==================

import json
import os

scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

# ✅ Use GitHub secrets if available
if os.getenv("GOOGLE_CREDS"):
    creds_dict = json.loads(os.getenv("GOOGLE_CREDS"))
    creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_dict, scope)

# ✅ Otherwise use local file
else:
    creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)

client = gspread.authorize(creds)
sheet = client.open("NSE500 Tracker").sheet1



# ================== TELEGRAM ==================
BOT_TOKEN = os.getenv("8595041350:AAHNzPFfWgIlQ-EvWM2kWh-GJ5md4D8dKyw")
CHAT_ID = os.getenv("637317120")

def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
    except:
        pass


# ================== STOCK LIST ==================
def get_nse():
    df = pd.read_csv("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
    symbols = [s for s in df["SYMBOL"].tolist() if "DUMMY" not in s]
    return [s + ".NS" for s in symbols[:200]]


# ================== RSI ==================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).iloc[-1]


# ================== FETCH DATA ==================
def fetch(stock):
    try:
        t = yf.Ticker(stock)
        info = t.info

        price = info.get("currentPrice")
        eps = info.get("trailingEps")
        feps = info.get("forwardEps")
        pe = info.get("trailingPE")
        fpe = info.get("forwardPE")
        mcap = info.get("marketCap")
        sector = info.get("sector")
        roe = info.get("returnOnEquity")
        debt = info.get("debtToEquity")

        if price is None:
            return None

        hist = t.history(period="1y")
        if hist.empty:
            return None

        hist_price = hist["Close"].mean()
        momentum = hist["Close"].iloc[-1] - hist["Close"].iloc[0]
        rsi = calculate_rsi(hist["Close"])
        hist_pe = hist_price / eps if eps else None

        return {
            "Stock": stock,
            "Price": price,
            "EPS": eps,
            "Forward EPS": feps,
            "PE": pe,
            "Forward PE": fpe,
            "MarketCap": mcap,
            "Sector": sector,
            "Historical Price": hist_price,
            "Historical PE": hist_pe,
            "RSI": rsi,
            "Momentum": momentum,
            "ROE": roe,
            "Debt": debt
        }

    except:
        return None


# ================== LOAD DATA ==================
stocks = get_nse()
data = []

with ThreadPoolExecutor(max_workers=15) as ex:
    futures = [ex.submit(fetch, s) for s in stocks]
    for f in as_completed(futures):
        result = f.result()
        if result:
            data.append(result)

df = pd.DataFrame(data)

# ================== CLEAN ==================
df = df.replace([float("inf"), -float("inf")], "")
df = df.fillna("")


# ================== ADV NORMALIZATION ==================
def normalize_features(df, cols):
    df = df.copy()

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

        # ✅ use median (robust)
        df[col] = df[col].fillna(df[col].median())

        min_v = df[col].min()
        max_v = df[col].max()

        if pd.isna(min_v) or pd.isna(max_v) or min_v == max_v:
            df[col] = 0
        else:
            df[col] = (df[col] - min_v) / (max_v - min_v)

        df[col] = df[col].clip(0, 1)

    return df


# ================== AI MODEL ==================
def run_ai(df):
    try:
        df2 = df.replace("", None).dropna()

        features = [
            "PE","Forward PE","EPS","Forward EPS",
            "RSI","Momentum","ROE","Debt","Historical PE"
        ]

        df2[features] = df2[features].astype(float)
        df2["Price"] = df2["Price"].astype(float)

        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=10,
            random_state=42
        )

        X = df2[features]
        y = df2["Price"]

        model.fit(X, y)

        pred_input = df[features].replace("", 0).fillna(0).astype(float)

        df["Predicted Price"] = model.predict(pred_input)
        df["Expected Return"] = ((df["Predicted Price"] - df["Price"]) / df["Price"]) * 100

    except:
        df["Predicted Price"] = ""
        df["Expected Return"] = ""

    return df


# ================== SCORING ==================
def advanced_score(df):
    cols = [
        "PE","Forward PE","EPS","Forward EPS",
        "RSI","Momentum","ROE","Debt","Historical PE"
    ]

    df = normalize_features(df, cols)

    df["Score"] = (
        (1 - df["PE"]) * 0.15 +
        (1 - df["Forward PE"]) * 0.15 +
        df["EPS"] * 0.10 +
        df["Forward EPS"] * 0.10 +
        df["ROE"] * 0.10 +
        (1 - df["Debt"]) * 0.05 +
        df["Momentum"] * 0.15 +
        (1 - abs(df["RSI"] - 0.5)) * 0.10 +
        (1 - df["Historical PE"]) * 0.10
    )

    df["Score"] = (df["Score"] * 100).round(2)
    return df


# ================== SECTOR ROTATION ==================
def sector_rotation(df):
    df["Momentum"] = pd.to_numeric(df["Momentum"], errors="coerce")

    sector_strength = (
        df.groupby("Sector")["Momentum"].mean() * 0.7 +
        df.groupby("Sector")["ROE"].mean() * 0.3
    ).reset_index()

    sector_strength.columns = ["Sector", "Strength"]

    top_sectors = sector_strength.sort_values(
        by="Strength", ascending=False
    ).head(3)["Sector"]

    df["Top Sector"] = df["Sector"].isin(top_sectors)

    print("🔥 Top Sectors:", list(top_sectors))
    return df


# ================== BACKTEST ==================
def backtest(df):
    try:
        selected = df[df["Score"] > 60]

        avg = selected["Expected Return"].mean()
        win = (selected["Expected Return"] > 0).mean() * 100

        print(f"✅ Avg Return: {round(avg,2)}%")
        print(f"✅ Win Rate: {round(win,2)}%")

    except:
        pass

    return df

# ✅ PORTFOLIO ALLOCATION
def portfolio_allocation(df, capital=100000):
    try:
        df = df.copy()

        # ✅ Convert numeric safely
        df["Score"] = pd.to_numeric(df["Score"], errors="coerce")
        df["Expected Return"] = pd.to_numeric(df["Expected Return"], errors="coerce")
        df["Price"] = pd.to_numeric(df["Price"], errors="coerce")

        df = df.dropna()

        # ✅ Combine strength score
        df["Weight Score"] = df["Score"] * 0.6 + df["Expected Return"] * 0.4

        # ✅ Normalize weights
        total_weight = df["Weight Score"].sum()
        df["Allocation %"] = (df["Weight Score"] / total_weight) * 100
        
        # ✅ RISK CONTROL (LIMIT MAX 20%)
        df["Allocation %"] = df["Allocation %"].clip(upper=20)

        # ✅ Allocate capital
        df["Allocated Amount"] = (df["Allocation %"] / 100) * capital

        # ✅ Number of shares
        df["Shares"] = (df["Allocated Amount"] / df["Price"]).astype(int)

        # ✅ Final invested amount
        df["Invested"] = df["Shares"] * df["Price"]
        
       # ✅ Re-normalize weights
        total = df["Allocation %"].sum()
        df["Allocation %"] = (df["Allocation %"] / total) * 100


        print("✅ Portfolio created with total capital:", capital)

    except Exception as e:
        print("Portfolio error:", e)

    return df



# ================== PIPELINE ==================
df = run_ai(df)
df = advanced_score(df)
df = sector_rotation(df)
df = backtest(df)


# ================== TOP STOCKS ==================
top10 = df[
    (df["Score"] > 60) &
    (df["Expected Return"] > 5) &
    (df["Top Sector"] == True)
].sort_values(
    by=["Score","Expected Return"], ascending=False
).head(10)

# ✅ PORTFOLIO ALLOCATION
portfolio = portfolio_allocation(top10, capital=100000)


# ================== FINAL CLEAN ==================
df = df.replace([float("inf"), -float("inf")], "")
df = df.fillna("")
df = df.astype(str)

# ================== UPLOAD ==================
sheet.clear()
sheet.update([portfolio.columns.tolist()] + portfolio.astype(str).values.tolist())



# ================== TELEGRAM ==================

msg = "📊 PORTFOLIO ALLOCATION\n\n"

for _, r in portfolio.iterrows():
    msg += (
        f"{r['Stock']} | ₹{round(r['Invested'],0)} "
        f"| {round(r['Allocation %'],1)}%\n"
    )


send_telegram(msg)

print("✅ SYSTEM COMPLETE")
