import os, json, math, traceback, time
from datetime import datetime, timedelta
import httpx, pandas as pd, numpy as np, ta
from dotenv import load_dotenv
from openai import OpenAI
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import adx

# ================================
# Config
# ================================
ENV_PATH = "/Users/yihongdai/Desktop/project/config.env"
STATE_FILE = "/Users/yihongdai/Desktop/project/okex/account_state.json"

TRADING_SYMBOLS = ["BTC-USDT", "ETH-USDT", "SOL-USDT", "BNB-USDT"]

MAX_COIN_EXPOSURE = 0.30
MAX_TOTAL_RISK = 0.10
MAX_RISK_PER_TRADE = 0.03
ATR_STOP_MULTIPLIER = 1.5
VOL_TARGET_ANN = 0.30
KELLY_BASE = 0.25
VOL_SCALE_CLIP = (0.5, 2.0)
COOLDOWN_MINUTES = 15
STOPLOSS_MULT = 1.5
TRAIL_MULT = 1.0
MAX_RETRY = 3

# ================================
# Init
# ================================
load_dotenv(ENV_PATH)
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
    http_client=httpx.Client(trust_env=False, timeout=120)
)

from okx.Account import AccountAPI
from okx.MarketData import MarketAPI
from okx.Trade import TradeAPI
acct = AccountAPI(os.getenv("OKX_API_KEY"), os.getenv("OKX_API_SECRET"), os.getenv("OKX_PASS"), use_server_time=False, flag='1')
mkt = MarketAPI()
trade = TradeAPI(os.getenv("OKX_API_KEY"), os.getenv("OKX_API_SECRET"), os.getenv("OKX_PASS"), False, flag='1')

# ================================
# State
# ================================
def load_state():
    if not os.path.exists(STATE_FILE):
        init = {
            "start_time": datetime.now().isoformat(),
            "invocations": 0,
            "initial_equity": 0,
            "trade_history": [],
            "last_trade_time": {},
            "recent_results": [],
            "open_positions": {}
        }
        json.dump(init, open(STATE_FILE, "w"), indent=2)
        return init
    return json.load(open(STATE_FILE))

def save_state(s): json.dump(s, open(STATE_FILE, "w"), indent=2)
state = load_state()

# ================================
# Helpers
# ================================
def get_klines(sym, bar="3m", limit=300):
    r = mkt.get_candlesticks(instId=sym, bar=bar, limit=str(limit))
    df = pd.DataFrame(r["data"], columns=["ts","o","h","l","c","vol","volCcy","volCcyQuote","confirm"])
    df = df[["ts","o","h","l","c"]].astype(float)
    df["t"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("t")
    df["ret"] = df["c"].pct_change()
    df["atr14"] = AverageTrueRange(df["h"], df["l"], df["c"], window=14).average_true_range()
    df["rsi"] = ta.momentum.rsi(df["c"],14)
    bb = BollingerBands(df["c"], window=20)
    df["bb_high"], df["bb_low"] = bb.bollinger_hband(), bb.bollinger_lband()
    df["adx"] = adx(df["h"], df["l"], df["c"], window=14)
    return df

def get_balances():
    data = acct.get_account_balance()["data"][0]["details"]
    bal = {d["ccy"]: float(d["eq"]) for d in data}
    usdt = bal.get("USDT", 0.0)
    equity = sum(float(d["eqUsd"]) for d in data)
    return bal, usdt, equity

def vol_scale(df):
    r = df["ret"].dropna()
    if len(r) < 20: return 1
    rv = r.std() * math.sqrt(480*365)
    if rv <= 0: return 1
    raw = VOL_TARGET_ANN / rv
    return max(VOL_SCALE_CLIP[0], min(VOL_SCALE_CLIP[1], raw))

def dynamic_kelly(winrate):
    k = (winrate - (1 - winrate))
    return max(0, KELLY_BASE * k)

def record(symbol, side, size, reason, res, success, pnl=None):
    log = {
        "time": datetime.now().isoformat(),
        "symbol": symbol, "side": side, "size": size,
        "reason": reason, "success": success,
        "pnl": pnl, "res_code": res.get("code") if res else None
    }
    state["trade_history"].append(log)
    state["trade_history"] = state["trade_history"][-2000:]
    save_state(state)

def compute_stats():
    results = state.get("recent_results", [])
    if len(results) < 2: return 0.5, 0, 0
    arr = np.array(results)
    winrate = np.mean(arr > 0)
    avg_pnl = np.mean(arr)
    sharpe = np.mean(arr) / (np.std(arr)+1e-9) * math.sqrt(252)
    return winrate, avg_pnl, sharpe

# ================================
# Safe Order Execution
# ================================
def safe_place_order(side, symbol, size, reason, max_retry=MAX_RETRY):
    attempt = 0
    while attempt < max_retry:
        try:
            res = trade.place_order(
                instId=symbol, tdMode="cash", side=side,
                ordType="market", tgtCcy='base_ccy', sz=str(size)
            )
            attempt += 1
            if not res or "code" not in res:
                raise Exception("No response from OKX API")

            code, msg = str(res.get("code","")), res.get("msg","")
            if code == "0":
                data = res.get("data", [{}])[0]
                fill_px = data.get("fillPx")
                print(f"✅ Order success: {side} {symbol} {size}, fill={fill_px}")
                record(symbol, side, size, reason, res, True)
                return res
            elif code in ["50001","58001","58004"]:
                print(f"⚠️ System busy [{code}] {msg}, attempt={attempt}/{max_retry}")
                time.sleep(3)
                continue
            else:
                print(f"❌ Order failed [{code}] {msg}")
                record(symbol, side, size, reason, res, False)
                return None
        except Exception as e:
            print(f"⚠️ Exception: {e}")
            traceback.print_exc()
            time.sleep(2)
            attempt += 1
    print(f"🚨 Aborted after {max_retry} retries. Skipping this cycle.")
    record(symbol, side, size, reason, {}, False)
    return None

# ================================
# AI Section
# ================================
def build_prompt(sig, bal, equity, positions, market_state, winrate):
    state["invocations"] += 1
    save_state(state)
    ret = (equity - state.get("initial_equity", equity)) / max(state.get("initial_equity", equity),1)*100
    return f"""
You are an AI SPOT crypto trader. No leverage. Market regime: {market_state}

If market_state == 'trending': use trend-following strategy (EMA + MACD)
If market_state == 'ranging': use mean-reversion strategy (RSI + Bollinger)

Signals:
{json.dumps(sig, indent=2)}

Account:
equity={equity:.2f}USDT, return={ret:.2f}%, recent_winrate={winrate:.2f}

Output JSON only:
{{"action":"BUY/SELL/HOLD","symbol":"BTC-USDT","confidence":0.6,"reason":""}}
"""

def ask_ai(p):
    r = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=[{"role":"system","content":"Crypto spot trader, risk managed."},
                  {"role":"user","content":p}]
    )
    print(f"[{datetime.now()}] 🧠 AI reasoning:", r.choices[0].message.reasoning_content)
    return r.choices[0].message.content

# ================================
# Main
# ================================
def main():
    try:
        bal, usdt, equity = get_balances()
        if state["initial_equity"] == 0: state["initial_equity"] = equity
        winrate, avgpnl, sharpe = compute_stats()
        kelly_adj = dynamic_kelly(winrate)
        sig, frames = {}, {}

        for s in TRADING_SYMBOLS:
            df = get_klines(s)
            latest = df.iloc[-1]
            market_state = "ranging" if df["adx"].iloc[-1] < 20 else "trending"
            sig[s] = {
                "price": latest.c, "atr": latest.atr14, "rsi": latest.rsi,
                "ema20": ta.trend.ema_indicator(df["c"],20).iloc[-1],
                "macd": ta.trend.macd_diff(df["c"]).iloc[-1],
                "bb_high": latest.bb_high, "bb_low": latest.bb_low,
                "adx": latest.adx, "market_state": market_state
            }
            frames[s] = df

        overall_state = "ranging" if np.mean([v["adx"] for v in sig.values()]) < 20 else "trending"
        positions = {k:v for k,v in bal.items() if k in ["BTC","ETH","SOL","BNB"]}
        prompt = build_prompt(sig, bal, equity, positions, overall_state, winrate)
        ai = ask_ai(prompt)
        print("🤖 AI raw:", ai)
        try: d = json.loads(ai)
        except: print("❌ AI非JSON输出"); return

        action = d.get("action","HOLD").upper()
        symbol = d.get("symbol","BTC-USDT")
        conf = float(d.get("confidence",0.6))
        reason = d.get("reason","")
        if symbol not in TRADING_SYMBOLS or conf < 0.55: print("🟡 HOLD"); return

        now = datetime.now()
        last_trade_t = state["last_trade_time"].get(symbol)
        if last_trade_t and (now - datetime.fromisoformat(last_trade_t)).seconds < COOLDOWN_MINUTES*60:
            print("🕒 Cooldown active"); return

        df = frames[symbol]
        px, atr = float(df["c"].iloc[-1]), float(df["atr14"].iloc[-1])
        coin = symbol.split("-")[0]
        holding = bal.get(coin,0.0)

        # === Dynamic Stoploss / Trailing ===
        if coin in state["open_positions"]:
            entry = state["open_positions"][coin]["entry"]
            stop = entry - STOPLOSS_MULT*atr
            trail = entry + TRAIL_MULT*atr
            if px < stop:
                print(f"⛔ STOP LOSS {coin}")
                res = safe_place_order("sell", symbol, holding, "StopLoss")
                pnl = (px-entry)/entry
                state["recent_results"].append(pnl)
                state["recent_results"]=state["recent_results"][-200:]
                state["open_positions"].pop(coin, None)
                save_state(state)
                return
            elif px > trail:
                print(f"💰 TAKE PROFIT {coin}")
                res = safe_place_order("sell", symbol, holding, "TrailingTP")
                pnl = (px-entry)/entry
                state["recent_results"].append(pnl)
                state["recent_results"]=state["recent_results"][-200:]
                state["open_positions"].pop(coin, None)
                save_state(state)
                return

        risk_amt = equity * MAX_RISK_PER_TRADE
        per_unit = ATR_STOP_MULTIPLIER * atr
        size = (risk_amt / per_unit) * vol_scale(df) * kelly_adj
        if holding*px > equity * MAX_COIN_EXPOSURE:
            print("⚠️ Position cap hit"); return
        if size <= 0: return

        if action=="BUY":
            cost = size*px
            if cost > usdt: size = usdt/px*0.98
            print(f"📈 BUY {symbol} size={size:.4f} conf={conf} ({overall_state})")
            res = safe_place_order("buy", symbol, size, reason)
            if res:
                state["last_trade_time"][symbol] = now.isoformat()
                state["open_positions"][coin] = {"entry": px, "time": now.isoformat()}
                save_state(state)

        elif action=="SELL":
            if holding <= 0: print("⛔ No position to sell"); return
            size = min(size, holding)
            print(f"📉 SELL {symbol} size={size:.4f} ({overall_state})")
            res = safe_place_order("sell", symbol, size, reason)
            if res:
                entry = state["open_positions"].get(coin,{"entry":px})["entry"]
                pnl = (px-entry)/entry
                state["recent_results"].append(pnl)
                state["recent_results"]=state["recent_results"][-200:]
                state["open_positions"].pop(coin, None)
                state["last_trade_time"][symbol]=now.isoformat()
                save_state(state)

        winrate, avgpnl, sharpe = compute_stats()
        print(f"📊 WinRate={winrate:.2f} | AvgPnL={avgpnl:.4f} | Sharpe={sharpe:.2f}")

    except Exception as e:
        print("⚠️ ERR:", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
