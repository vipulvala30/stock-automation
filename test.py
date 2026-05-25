import pandas as pd
import yfinance as yf
import gspread
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


# ✅ GOOGLE SHEETS CONNECTION
scope = ["https://spreadsheets.google.com/feeds",
         "https://www.googleapis.com/auth/drive"]

creds = ServiceAccountCredentials.from_json_keyfile_name("credentials.json", scope)
client = gspread.authorize(creds)
sheet = client.open("NSE500 Tracker").sheet1


# ✅ TELEGRAM (REPLACE TOKEN AFTER TESTING)
import os

BOT_TOKEN = os.getenv("BOT_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")


# ✅ NSE 500 LIST
def get_nse_500():
    df = pd.read_csv("https://archives.nseindia.com/content/equities/EQUITY_L.csv")
    symbols = df["SYMBOL"].tolist()
    symbols = [s for s in symbols if "DUMMY" not in s]
    return [s + ".NS" for s in symbols[:300]]


# ✅ RSI
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = -delta.clip(upper=0).rolling(period).mean()
    rs = gain / loss
    return (100 - (100 / (1 + rs))).iloc[-1]


# ✅ TELEGRAM
def send_telegram(msg):
    try:
        url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": msg})
    except:
        pass


# ✅ FETCH DATA
def fetch_stock(stock):
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

        hist_pe = hist_price / eps if hist_price and eps else None

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


# ✅ MULTITHREAD FETCH
stocks = get_nse_500()
data = []

with ThreadPoolExecutor(max_workers=15) as ex:
    futures = [ex.submit(fetch_stock, s) for s in stocks]

    for f in as_completed(futures):
        r = f.result()
        if r:
            data.append(r)


df = pd.DataFrame(data)

# ✅ CLEAN DATA
df = df.replace([float("inf"), -float("inf")], "")
df = df.fillna("")


# ✅ SECTOR ANALYSIS
df["PE"] = pd.to_numeric(df["PE"], errors="coerce")
sector_pe = df.groupby("Sector")["PE"].mean().to_dict()
df["Sector PE"] = df["Sector"].map(sector_pe)


# ✅ ADVANCED SCORING
def score(row):
    s = 0
    try:
        pe = float(row["PE"])
        fpe = float(row["Forward PE"])
        eps = float(row["EPS"])
        feps = float(row["Forward EPS"])
        hist_pe = float(row["Historical PE"])
        rsi = float(row["RSI"])
        momentum = float(row["Momentum"])
        price = float(row["Price"])
        hist_price = float(row["Historical Price"])
        mcap = float(row["MarketCap"])
        roe = float(row["ROE"]) if row["ROE"] else 0
        debt = float(row["Debt"]) if row["Debt"] else 0
        sector_pe = float(row["Sector PE"])

        # VALUE
        if pe < hist_pe:
            s += 10
        if fpe < pe:
            s += 10
        if pe < sector_pe:
            s += 5

        # GROWTH
        if feps > eps:
            s += 10

        # MOMENTUM
        if 35 <= rsi <= 55:
            s += 10
        if momentum > 0:
            s += 5

        # QUALITY
        if mcap > 1e11:
            s += 10
        if roe > 0.15:
            s += 10
        if debt < 1:
            s += 5

        # VALUE ENTRY
        if price < hist_price:
            s += 5

    except:
        pass

    return s


df["Score"] = df.apply(score, axis=1)


# ✅ AI MODEL
def run_ai(df):
    try:
        df2 = df.replace("", None).dropna()

        features = [
            "PE", "Forward PE", "EPS", "Forward EPS",
            "RSI", "Momentum", "ROE", "Debt", "Historical PE"
        ]

        df2[features] = df2[features].astype(float)
        df2["Price"] = df2["Price"].astype(float)

        X = df2[features]
        y = df2["Price"]

        reg = RandomForestRegressor(n_estimators=50)
        reg.fit(X, y)

        # prediction safe
        pred_input = df[features].replace("", 0)
        pred_input = pred_input.fillna(0).astype(float)

        df["Predicted Price"] = reg.predict(pred_input)

        # classification
        future = y.shift(-1)
        df2["target"] = (future > y).astype(int)

        clf = RandomForestClassifier(n_estimators=50)
        clf.fit(X, df2["target"])

        probs = clf.predict_proba(pred_input)[:,1]
        df["Buy Probability"] = (probs * 100).round(2)

    except Exception as e:
        print("AI error:", e)
        df["Predicted Price"] = ""
        df["Buy Probability"] = ""

    return df


df = run_ai(df)


# ✅ FINAL SORT
df = df.sort_values(["Score", "Buy Probability"], ascending=False)


# ✅ TOP 10 FILTER (SMART)
top10 = df[
    (df["Score"] > 40) &
    (df["Buy Probability"] > 60)
].head(10)


# ✅ FINAL CLEAN BEFORE UPLOAD
df = df.replace([float("inf"), -float("inf")], "")
df = df.fillna("")
df = df.astype(str)


# ✅ UPLOAD TO GOOGLE SHEETS
try:
    sheet.clear()
    sheet.update([df.columns.tolist()] + df.values.tolist())
    print("✅ SHEET UPDATED")
except Exception as e:
    print("UPLOAD ERROR:", e)


# ✅ TELEGRAM ALERT
msg = "📊 TOP STOCK PICKS\n\n"

for _, r in top10.iterrows():
    msg += f"{r['Stock']} | Score:{r['Score']} | Prob:{r['Buy Probability']}%\n"

send_telegram(msg)

print("✅ SYSTEM COMPLETE")
