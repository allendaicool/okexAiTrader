import os, json, math, traceback
from datetime import datetime
import httpx

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from okx.Account import AccountAPI
from okx.MarketData import MarketAPI
from okx.Trade import TradeAPI
import ta
from ta.volatility import AverageTrueRange


# ================================
# Config
# ================================
ENV_PATH = "/Users/yihongdai/Desktop/project/config.env"
STATE_FILE = "/Users/yihongdai/Desktop/project/okex/account_state.json"

TRADING_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]

MAX_COIN_EXPOSURE = 0.30    # 单币最大仓位 = 30% 账户
MAX_RISK_PER_TRADE = 0.04   # 单笔风险 = 2% 账户
ATR_STOP_MULTIPLIER = 1.5
VOL_TARGET_ANN = 0.50
KELLY_FRACTION = 0.25
VOL_SCALE_CLIP = (0.5, 2.0)

BAR = "3m"
LIMIT = 300

# ================================
# Init
# ================================
load_dotenv(ENV_PATH)
print('----')
print(os.getenv("DEEPSEEK_API_KEY"))
print('----')

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(
        trust_env=False,  # 不读取系统代理环境变量
        timeout=120
    )  # 🚫 无代理
)
is_simulated=True
acct = AccountAPI(os.getenv("OKX_API_KEY"), os.getenv("OKX_API_SECRET"), os.getenv("OKX_PASS"), use_server_time=False,
    flag='1')   # 模拟盘必须设为 '1'
mkt = MarketAPI()
trade = TradeAPI(os.getenv("OKX_API_KEY"), os.getenv("OKX_API_SECRET"), os.getenv("OKX_PASS"),False, flag='1')

# ================================
# State
# ================================
def load_state():
    if not os.path.exists(STATE_FILE):
        bal = acct.get_account_balance("USDT")["data"][0]["details"][0]
        init = {
            "start_time": datetime.now().isoformat(),
            "invocations": 0,
            "initial_equity": float(bal["eq"]),
            "trade_history": []
        }
        json.dump(init, open(STATE_FILE, "w"), indent=2)
        return init
    return json.load(open(STATE_FILE))

def save_state(s):
    json.dump(s, open(STATE_FILE, "w"), indent=2)

state = load_state()

# ================================
# Helpers
# ================================
def get_klines(sym):
    r = mkt.get_candlesticks(instId=sym, bar=BAR, limit=str(LIMIT))
    df = pd.DataFrame(r["data"], columns=["ts","o","h","l","c","vol","volCcy","volCcyQuote","confirm"])
    df = df[["ts","o","h","l","c"]].astype(float)
    df["t"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("t")
    df["ret"] = df["c"].pct_change()
    indicator = AverageTrueRange(high=df["h"], low=df["l"], close=df["c"], window=14)
    df["atr14"] = indicator.average_true_range()
    return df

def get_balances():
    data = acct.get_account_balance()["data"][0]["details"]
    bal = {d["ccy"]: float(d["eq"]) for d in data}
    usdt = bal.get("USDT", 0.0)
    equity = usdt
    return bal, usdt, equity

def atr_size(equity, atr, px):
    if atr <= 0: return 0
    risk = equity * MAX_RISK_PER_TRADE
    per_unit = ATR_STOP_MULTIPLIER * atr
    qty = risk / per_unit / px * px
    return max(qty, 0)

def vol_scale(df):
    r = df["ret"].dropna()
    if len(r) < 20: return 1
    rv = r.std() * math.sqrt(480*365)
    if rv <= 0: return 1
    raw = VOL_TARGET_ANN / rv
    return max(VOL_SCALE_CLIP[0], min(VOL_SCALE_CLIP[1], raw))

def kelly(conf):
    p = max(0,min(1,conf))
    k = (p*2 - 1)
    return max(0, k*0.25)

def record(symbol, side, size, reason, okxresp, success):
    log = {
        "time": datetime.now().isoformat(),
        "symbol": symbol,
        "side": side,
        "size": size,
        "reason": reason,
        "success": success,
        "price": None
    }
    try: log["price"] = okxresp["data"][0].get("fillPx")
    except: pass
    st = load_state()
    st["trade_history"].append(log)
    st["trade_history"] = st["trade_history"][-2000:]
    save_state(st)

# ================================
# AI
# ================================
def build_prompt(sig, bal, equity, positions):
    state["invocations"] += 1
    save_state(state)
    ret = (equity-state["initial_equity"])/state["initial_equity"]*100

    return f"""
You are a SPOT crypto AI trader. No leverage. Only long.

Market:
{json.dumps(sig,indent=2)}

Account:
equity={equity}
return={ret:.2f}%
positions={positions}
balances={bal}

Output JSON only:
{{
 "action":"BUY/SELL/HOLD",
 "symbol":"BTC-USDT",
 "confidence":0.6,
 "reason":""
}}
"""

def ask_ai(p):
    r = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role":"system","content":"Spot trader. No leverage. Risk aware."},
                  {"role":"user","content":p}]
    )
    reasoning_content = r.choices[0].message.reasoning_content
    print("reasoning content:", reasoning_content)
    return r.choices[0].message.content

# ================================
# Main Loop
# ================================
def main():
    try:
        bal, usdt, equity = get_balances()

        sig = {}
        frames = {}
        for s in TRADING_SYMBOLS:
            df = get_klines(s)
            frames[s] = df
            latest = df.iloc[-1]
            sig[s] = {
                "price": latest.c,
                "atr": latest.atr14,
                "rsi7": ta.momentum.rsi(df["c"],7).iloc[-1],
                "ema20": ta.trend.ema_indicator(df["c"],20).iloc[-1],
                "macd": ta.trend.macd_diff(df["c"]).iloc[-1]
            }

        positions = {k:v for k,v in bal.items() if k in ["BTC","ETH","SOL","BNB"]}

        prompt = build_prompt(sig, bal, equity, positions)
        ai = ask_ai(prompt)
        print("🤖", ai)

        try: d = json.loads(ai)
        except:
            print("❌ AI非JSON")
            return

        action = d.get("action","HOLD").upper()
        symbol = d.get("symbol","BTC-USDT")
        conf = float(d.get("confidence",0.6))
        reason = d.get("reason","")

        if symbol not in TRADING_SYMBOLS:
            print("⚠️ invalid symbol")
            return

        df = frames[symbol]
        px = float(df["c"].iloc[-1])
        atr = float(df["atr14"].iloc[-1])

        coin = symbol.split("-")[0]

        # 当前持币数量
        holding = bal.get(coin,0.0)

        # 头寸 sizing
        size = atr_size(equity, atr, px)
        size *= vol_scale(df)
        size *= (kelly(conf) if conf>0 else 0.2)

        # 单币上限
        if holding*px > equity * MAX_COIN_EXPOSURE:
            print("⚠️ 超单币上限，禁止加仓")
            return

        if size <= 0: return

        if action=="BUY":
            cost = size*px
            print(f"Buy Cost {cost}")
            if cost > usdt:
                print("💡 USDT不足，按余额调整")
                size = usdt/px*0.98

            print(f"📈 BUY {symbol} {size}")
            res = trade.place_order(instId=symbol, tdMode="cash", side="buy", ordType="market", tgtCcy='base_ccy', sz=str(size))
            record(symbol,"BUY",size,reason,res,True)

        elif action=="SELL":
            if holding <= 0:
                print("⛔ 无币可卖")
                return

            size = min(size, holding)
            print(f"📉 SELL {symbol} {size}")
            res = trade.place_order(instId=symbol, tdMode="cash", side="sell", ordType="market", sz=str(size))
            record(symbol,"SELL",size,reason,res,True)

        else:
            print("🟡 HOLD")
            record(symbol,"HOLD",0,reason,{},True)

    except Exception as e:
        print("⚠️ ERR:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
    #res = trade.place_order(instId='BNB-USDT', tdMode="cash", side="buy", ordType="market", tgtCcy='base_ccy', sz=str(1))
    # print(res)
