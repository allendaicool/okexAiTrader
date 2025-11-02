import os, json
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from okx.Account import AccountAPI
from okx.MarketData import MarketAPI
from okx.Trade import TradeAPI
import ta

# ----------------------------
# Load ENV
# ----------------------------
load_dotenv("/Users/yihongdai/Desktop/project/config.env")

client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
okx_account = AccountAPI(os.getenv("OKX_API_KEY"), os.getenv("OKX_API_SECRET"), os.getenv("OKX_PASS"))
okx_market = MarketAPI()
okx_trade = TradeAPI(os.getenv("OKX_API_KEY"), os.getenv("OKX_API_SECRET"), os.getenv("OKX_PASS"))

STATE_FILE = "/Users/yihongdai/Desktop/project/okex/account_state.json"

# ----------------------------
# Load or init trading state
# ----------------------------
if not os.path.exists(STATE_FILE):
    initial_cash = float(okx_account.get_account_balance("USDT")["data"][0]["details"][0]["availEq"])

    init = {
        "start_time": datetime.now().isoformat(),
        "invocations": 0,
        "initial_equity": initial_cash,
    }
    json.dump(init, open(STATE_FILE, "w"))

state = json.load(open(STATE_FILE, "r"))


# ----------------------------
# Helpers
# ----------------------------
def get_klines(inst="BTC-USDT-SWAP", bar="3m", limit=200):
    raw = okx_market.get_candlesticks(instId=inst, bar=bar, limit=str(limit))
    df = pd.DataFrame(raw["data"], columns=["ts","o","h","l","c","vol","volCcy","volCcyQuote","confirm"])
    df = df[["ts","o","h","l","c"]].astype(float)
    df["t"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("t")
    return df

def add_indicators(df):
    df["ema20"] = ta.trend.ema_indicator(df["c"], 20)
    df["macd"] = ta.trend.macd_diff(df["c"])
    df["rsi7"] = ta.momentum.rsi(df["c"], 7)
    df["rsi14"] = ta.momentum.rsi(df["c"], 14)
    return df

def get_funding_oi(symbol="BTC-USDT-SWAP"):
    fr = okx_market.get_funding_rate(symbol)["data"][0]
    oi = okx_market.get_open_interest(symbol)["data"][0]
    return float(fr["fundingRate"]), float(oi["openInterest"])

def get_account():
    bal = okx_account.get_account_balance("USDT")["data"][0]["details"][0]
    return float(bal["availEq"]), float(bal["eq"]), float(bal["upl"])

def get_positions():
    return okx_account.get_positions()["data"]

def order(inst, side, size, lev=5):
    okx_trade.set_leverage(instId=inst, lever=str(lev), mgnMode="cross")
    print(f"📈 Order: {side} {inst} x{size} lev={lev}")
    return okx_trade.place_order(instId=inst, tdMode="cross", side=side, ordType="market", sz=str(size))


# ----------------------------
# Build Prompt
# ----------------------------
def build_prompt(df, funding, oi, cash, equity, pnl, positions):
    state["invocations"] += 1

    elapsed = (datetime.now() - datetime.fromisoformat(state["start_time"])).total_seconds()/60
    latest = df.iloc[-1]

    ret_pct = (equity - state["initial_equity"]) / state["initial_equity"] * 100

    return f"""
It has been {elapsed:.0f} minutes since you started trading.
Current time: {datetime.now()}
Invoked: {state['invocations']}

=== BTC Signals ===
price={latest.c}, ema20={latest.ema20}, macd={latest.macd}, rsi7={latest.rsi7}
Funding={funding}, OI={oi}

Recent prices: {df['c'][-10:].tolist()}
EMA20: {df['ema20'][-10:].tolist()}
MACD: {df['macd'][-10:].tolist()}
RSI7: {df['rsi7'][-10:].tolist()}
RSI14: {df['rsi14'][-10:].tolist()}

=== Account ===
Total Return: {ret_pct:.2f}%
Cash: {cash}
Equity: {equity}
PnL: {pnl}
Positions: {positions}

Return ONLY JSON:
{{
"action": "BUY/SELL/HOLD",
"symbol": "BTC-USDT-SWAP",
"size": 0.01,
"leverage": 3,
"reason": "text"
}}
"""


# ----------------------------
# AI decision
# ----------------------------
def ask_ai(p):
    r = client.chat.completions.create(
        model="deepseek-reasoner-v3.1",
        messages=[
            {"role":"system","content":"You are a disciplined crypto trading AI."},
            {"role":"user","content":p}
        ]
    )
    return r.choices[0].message.content


# ----------------------------
# Main Execution (single run for Cron)
# ----------------------------
df = add_indicators(get_klines())
funding, oi = get_funding_oi()
cash, equity, pnl = get_account()
positions = get_positions()

prompt = build_prompt(df, funding, oi, cash, equity, pnl, positions)
resp = ask_ai(prompt)
print("🤖 AI:", resp)

try:
    d = json.loads(resp)
    if d["action"] == "BUY":
        order(d["symbol"], "buy", d["size"], d["leverage"])
    elif d["action"] == "SELL":
        order(d["symbol"], "sell", d["size"], d["leverage"])
    else:
        print("✅ AI chose HOLD")
except Exception as e:
    print("⚠️ JSON error:", e)

json.dump(state, open(STATE_FILE, "w"))
