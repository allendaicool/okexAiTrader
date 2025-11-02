import os, json, math, traceback
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from okx.Account import AccountAPI
from okx.MarketData import MarketAPI
from okx.Trade import TradeAPI
import ta

# =========================
# ====== Configuration =====
# =========================
# 路径：改成你的绝对路径（Cron 下需要绝对路径）
ENV_PATH = "/Users/yihongdai/Desktop/project/config.env"
STATE_FILE = "/Users/yihongdai/Desktop/project/okex/account_state.json"

# 交易币种（线性 USDT 本位永续）
TRADING_SYMBOLS = ["BTC-USDT-SWAP", "ETH-USDT-SWAP", "SOL-USDT-SWAP", "BNB-USDT-SWAP"]

# 风控参数（可按需调整）
MAX_LEVERAGE = 3.0             # 组合最大杠杆倍数（总名义敞口 / 权益）
MAX_COIN_EXPOSURE = 0.25       # 单币最大名义敞口占比（占账户权益）
MAX_RISK_PER_TRADE = 0.02      # 单笔最大风险 2% 账户权益
ATR_STOP_MULTIPLIER = 1.5      # 停损距离 ~ 1.5 * ATR
VOL_TARGET_ANN = 0.60          # 组合目标年化波动（简单缩放器）
VOL_SCALE_CLIP = (0.5, 2.0)    # 波动缩放上下限
KELLY_FRACTION = 0.25          # 分数凯利（避免过度冒险）
CORR_CAP = 0.85                # 相关性阈值（高度同向）
CLUSTER_EXPO_CAP = 0.50        # 高度相关簇的总敞口上限（占权益）

# K线参数
BAR = "3m"
LIMIT = 300   # 多币相关性计算需要更长窗口

# =========================
# ====== Init Clients =====
# =========================
load_dotenv(ENV_PATH)
client = OpenAI(api_key=os.getenv("DEEPSEEK_API_KEY"), base_url="https://api.deepseek.com")
okx_account = AccountAPI(os.getenv("OKX_API_KEY"), os.getenv("OKX_API_SECRET"), os.getenv("OKX_PASS"))
okx_market  = MarketAPI()
okx_trade   = TradeAPI(os.getenv("OKX_API_KEY"), os.getenv("OKX_API_SECRET"), os.getenv("OKX_PASS"))

# =========================
# ====== State Load =======
# =========================
def load_state() -> dict:
    if not os.path.exists(STATE_FILE):
        # 初始化 initial_equity：包含未实现盈亏
        bal = okx_account.get_account_balance("USDT")["data"][0]["details"][0]
        initial_equity = float(bal["eq"])
        state = {
            "start_time": datetime.now().isoformat(),
            "invocations": 0,
            "initial_equity": initial_equity,
            "trade_history": []   # 交易日志
        }
        json.dump(state, open(STATE_FILE, "w"), indent=2)
        return state
    return json.load(open(STATE_FILE, "r"))

def save_state(s: dict):
    json.dump(s, open(STATE_FILE, "w"), indent=2)

state = load_state()

# =========================
# ====== OKX Helpers ======
# =========================
def get_klines(inst: str, bar: str = BAR, limit: int = LIMIT) -> pd.DataFrame:
    raw = okx_market.get_candlesticks(instId=inst, bar=bar, limit=str(limit))
    df = pd.DataFrame(raw["data"], columns=["ts","o","h","l","c","vol","volCcy","volCcyQuote","confirm"])
    df = df[["ts","o","h","l","c"]].astype(float)
    df["t"] = pd.to_datetime(df["ts"], unit="ms")
    df = df.sort_values("t")
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 技术指标（EMA、MACD、RSI、ATR）
    df["ema20"] = ta.trend.ema_indicator(df["c"], 20)
    df["macd"] = ta.trend.macd_diff(df["c"])
    df["rsi7"] = ta.momentum.rsi(df["c"], 7)
    df["rsi14"] = ta.momentum.rsi(df["c"], 14)
    df["atr14"] = ta.volatility.average_true_range(high=df["h"], low=df["l"], close=df["c"], window=14)
    # 实现波动（用于 vol targeting）
    df["ret"] = df["c"].pct_change()
    return df

def get_funding_oi(inst: str) -> Tuple[float, float]:
    fr = okx_market.get_funding_rate(inst)["data"][0]
    oi = okx_market.get_open_interest(inst)["data"][0]
    return float(fr["fundingRate"]), float(oi["openInterest"])

def get_account_equity() -> Tuple[float, float, float]:
    bal = okx_account.get_account_balance("USDT")["data"][0]["details"][0]
    return float(bal["availEq"]), float(bal["eq"]), float(bal["upl"])

def get_positions_map() -> Dict[str, float]:
    """返回 {instId: qty_signed}，多头为正，空头为负"""
    pos_list = okx_account.get_positions()["data"]
    pos_map = {}
    for p in pos_list:
        sym = p["instId"]
        qty = float(p["pos"])          # 绝对数量
        side = p.get("posSide", "net") # "long"/"short" 或 "net"
        if side == "short":
            qty = -abs(qty)
        elif side == "long":
            qty = abs(qty)
        else:
            # 有些账户是净头寸模式：用 posCcy/posSide 不一定可用，这里用 mgnMode=net 的 sign
            if p.get("posSide","") == "net" and p.get("avgPx"):
                # 仅作为兜底：正负未知时当作正（谨慎）
                qty = abs(qty)
        pos_map[sym] = qty
    return pos_map

def last_price(df: pd.DataFrame) -> float:
    return float(df["c"].iloc[-1])

def place_order(inst: str, side: str, size: float, lev: int):
    okx_trade.set_leverage(instId=inst, lever=str(lev), mgnMode="cross")
    return okx_trade.place_order(instId=inst, tdMode="cross", side=side, ordType="market", sz=str(size))

# =========================
# ===== Risk Engine =======
# =========================
def realized_vol_annualized(df: pd.DataFrame, bars_per_day: int = 480, days_per_year: int = 365) -> float:
    """
    3m 频率：每天 480 根
    年化波动 ~ std(ret) * sqrt(bars_per_day * days_per_year)
    """
    r = df["ret"].dropna()
    if len(r) < 20:
        return 0.0
    vol = r.std() * math.sqrt(bars_per_day * days_per_year)
    return float(vol)

def kelly_fraction_from_confidence(conf: float, b: float = 1.0) -> float:
    """
    p = 模型信心（胜率近似），b = 赔率（简单近似为 1）
    凯利 f* = (p*(b+1)-1)/b；使用分数凯利
    """
    p = max(0.0, min(1.0, conf))
    k = (p*(b+1.0)-1.0)/b
    return max(0.0, k)

def portfolio_exposure_usd(pos_map: Dict[str, float], price_map: Dict[str, float]) -> float:
    return sum(abs(pos_map.get(sym,0.0))*price_map.get(sym,0.0) for sym in TRADING_SYMBOLS)

def symbol_exposure_usd(qty: float, px: float) -> float:
    return abs(qty) * px

def allowed_by_portfolio_limits(equity: float,
                                proposed_expo: float,
                                pos_map: Dict[str, float],
                                price_map: Dict[str, float]) -> Tuple[bool, str]:
    """组合杠杆限制检查"""
    current = portfolio_exposure_usd(pos_map, price_map)
    if current + proposed_expo > equity * MAX_LEVERAGE:
        return False, "Portfolio leverage limit"
    return True, "OK"

def allowed_by_single_coin(equity: float, symbol: str, proposed_expo: float,
                           pos_map: Dict[str, float], price_map: Dict[str, float]) -> Tuple[bool, str]:
    """单币最大名义敞口限制"""
    cur_qty = pos_map.get(symbol, 0.0)
    cur_expo = symbol_exposure_usd(cur_qty, price_map[symbol])
    if cur_expo + proposed_expo > equity * MAX_COIN_EXPOSURE:
        return False, "Single asset exposure cap"
    return True, "OK"

def correlation_cluster_scale(symbol: str,
                              price_map: Dict[str, float],
                              pos_map: Dict[str, float],
                              rets_df: pd.DataFrame,
                              equity: float,
                              proposed_expo: float) -> float:
    """
    针对与目标 symbol 高度相关(>CORR_CAP)的一组资产，控制簇总敞口
    返回一个 0~1 的缩放系数，若簇超限则按比例缩小。
    """
    if rets_df is None or rets_df.empty or symbol not in rets_df.columns:
        return 1.0

    corr = rets_df.corr()
    if symbol not in corr.index:
        return 1.0

    high_corr_syms = [s for s in TRADING_SYMBOLS if s in corr.columns and corr.loc[symbol, s] >= CORR_CAP]
    # 计算该簇的当前敞口
    cluster_expo = 0.0
    for s in high_corr_syms:
        cluster_expo += symbol_exposure_usd(pos_map.get(s,0.0), price_map.get(s,0.0))

    limit = equity * CLUSTER_EXPO_CAP
    if cluster_expo + proposed_expo <= limit:
        return 1.0
    else:
        # 需要按比例缩放使 cluster_expo + scale*proposed_expo = limit
        remaining = max(0.0, limit - cluster_expo)
        scale = remaining / max(1e-9, proposed_expo)
        return max(0.0, min(1.0, scale))

def atr_based_size(equity: float, atr: float, price: float,
                   risk_pct: float, k_atr: float = ATR_STOP_MULTIPLIER) -> float:
    """
    ATR 头寸 sizing（合约以币计）：risk_dollars / (k_atr * atr)
    为防止过大，再配合单币最大名义敞口限制。
    """
    if atr <= 0:
        return 0.0
    risk_dollars = equity * risk_pct
    per_unit_risk = k_atr * atr
    size = risk_dollars / max(1e-9, per_unit_risk)
    return max(0.0, size)

def vol_target_scale(df: pd.DataFrame, target_ann_vol: float = VOL_TARGET_ANN) -> float:
    rv = realized_vol_annualized(df)
    if rv <= 0:
        return 1.0
    raw = target_ann_vol / rv
    return float(max(VOL_SCALE_CLIP[0], min(VOL_SCALE_CLIP[1], raw)))

# =========================
# ===== Prompt & AI =======
# =========================
def build_market_state() -> Tuple[dict, dict, dict, pd.DataFrame]:
    """
    返回:
    - market_state: 每币技术面/资金面摘要（给模型）
    - price_map: {sym: last_price}
    - frames: {sym: df_with_indicators}
    - rets_df: 对齐后的收益序列 DataFrame（用于相关性/集群敞口）
    """
    frames = {}
    price_map = {}
    market_state = {}
    for sym in TRADING_SYMBOLS:
        df = add_indicators(get_klines(sym))
        frames[sym] = df
        price_map[sym] = last_price(df)
        funding, oi = get_funding_oi(sym)
        latest = df.iloc[-1]
        market_state[sym] = {
            "price": float(latest.c),
            "ema20": float(latest.ema20),
            "macd": float(latest.macd),
            "rsi7": float(latest.rsi7),
            "atr14": float(latest.atr14),
            "funding": funding,
            "oi": oi,
            "recent_prices": df["c"].tail(10).round(6).tolist(),
            "ema20_list": df["ema20"].tail(10).round(6).tolist(),
            "macd_list": df["macd"].tail(10).round(6).tolist(),
            "rsi7_list": df["rsi7"].tail(10).round(6).tolist(),
            "rsi14_list": df["rsi14"].tail(10).round(6).tolist()
        }

    # 对齐收益率（用于相关性）
    rets_df = None
    for sym, df in frames.items():
        sub = df[["t","ret"]].dropna().rename(columns={"ret": sym})
        rets_df = sub if rets_df is None else pd.merge(rets_df, sub, on="t", how="outer")
    if rets_df is not None:
        rets_df = rets_df.sort_values("t").set_index("t").ffill().bfill()

    return market_state, price_map, frames, rets_df

def build_prompt(market_state: dict, cash: float, equity: float, pnl: float,
                 pos_map: Dict[str, float]) -> str:
    state["invocations"] += 1
    elapsed = (datetime.now() - datetime.fromisoformat(state["start_time"])).total_seconds()/60
    ret_pct = (equity - state["initial_equity"]) / max(1e-9, state["initial_equity"]) * 100

    return f"""
It has been {elapsed:.0f} minutes since you started trading.
Current time: {datetime.now().isoformat()}
Invoked: {state['invocations']}

=== Market State (BTC/ETH/SOL/BNB) ===
{json.dumps(market_state, indent=2)}

=== Account ===
Total Return: {ret_pct:.2f}%
Cash: {cash}
Equity: {equity}
PnL: {pnl}
Positions: {pos_map}

TASK:
Return ONLY one JSON:
{{
  "action": "BUY/SELL/HOLD",
  "symbol": "BTC-USDT-SWAP|ETH-USDT-SWAP|SOL-USDT-SWAP|BNB-USDT-SWAP",
  "size_hint": 0.01,
  "leverage": 3,
  "confidence": 0.65,
  "reason": "short text"
}}
"""

def ask_ai(prompt: str) -> str:
    r = client.chat.completions.create(
        model="deepseek-reasoner-v3.1",
        messages=[
            {"role":"system","content":"You are a disciplined multi-asset crypto trading AI. Prefer high risk-adjusted return, respect risk constraints."},
            {"role":"user","content":prompt}
        ]
    )
    return r.choices[0].message.content

# =========================
# === Trade & Logging =====
# =========================
def record_trade(symbol: str, side: str, size: float, lev: int, reason: str, okx_resp: dict, success: bool):
    log = {
        "timestamp": datetime.now().isoformat(),
        "symbol": symbol,
        "side": side,
        "size": float(size),
        "leverage": int(lev),
        "reason": reason,
        "price": None,
        "success": bool(success),
        "order_id": None
    }
    if okx_resp:
        try:
            log["order_id"] = okx_resp["data"][0]["ordId"]
        except Exception:
            pass
        try:
            log["price"] = okx_resp["data"][0].get("fillPx")
        except Exception:
            pass

    trades = state.get("trade_history", [])
    trades.append(log)
    state["trade_history"] = trades[-2000:]  # 防止无限增长
    save_state(state)

# =========================
# ===== Main (Single) =====
# =========================
def main():
    try:
        # 账户信息
        cash, equity, pnl = get_account_equity()
        pos_map = get_positions_map()

        # 行情与技术面
        market_state, price_map, frames, rets_df = build_market_state()

        # 构造 Prompt & AI 决策
        prompt = build_prompt(market_state, cash, equity, pnl, pos_map)
        ai_resp = ask_ai(prompt)
        print("🤖 AI:", ai_resp)

        try:
            dec = json.loads(ai_resp)
        except Exception:
            print("⚠️ AI 返回非 JSON，忽略本次。")
            return

        action = str(dec.get("action","HOLD")).upper()
        symbol = str(dec.get("symbol","BTC-USDT-SWAP"))
        lev    = int(dec.get("leverage", 3))
        conf   = float(dec.get("confidence", 0.55))
        reason = str(dec.get("reason",""))

        if symbol not in TRADING_SYMBOLS:
            print(f"⚠️ 非允许标的：{symbol}")
            return

        # === 读取该标的的行情指标 ===
        df = frames[symbol]
        px = price_map[symbol]
        atr = float(df["atr14"].iloc[-1]) if not math.isnan(df["atr14"].iloc[-1]) else 0.0

        # === 计算基础尺寸：ATR-based sizing + Kelly + Vol targeting ===
        # 1) 基于模型信心得到分数凯利风控上限
        kelly_raw = kelly_fraction_from_confidence(conf)   # 0~1，p=confidence 假设赔率=1
        risk_pct_kelly = min(MAX_RISK_PER_TRADE, KELLY_FRACTION * kelly_raw) if kelly_raw > 0 else (MAX_RISK_PER_TRADE * 0.5)

        # 2) ATR sizing（以“币”为单位）
        base_size = atr_based_size(equity, atr, px, risk_pct=risk_pct_kelly)

        # 3) Vol targeting 缩放
        vol_scale = vol_target_scale(df, VOL_TARGET_ANN)
        sized = base_size * vol_scale

        # 4) 如果 AI 提供 size_hint，可轻微调节（只当作提示，仍受风控约束）
        size_hint = float(dec.get("size_hint", 0.0))
        if size_hint > 0:
            sized = (sized * 0.8) + (size_hint * 0.2)

        # 5) 名义敞口（不含杠杆）→ 这里以“合约币数 * 价格”近似
        proposed_expo = sized * px

        # === 风控：组合 & 单币限额 ===
        allow, why = allowed_by_portfolio_limits(equity, proposed_expo * lev, pos_map, price_map)
        if not allow:
            print(f"❌ 拒绝：{why}")
            record_trade(symbol, f"BLOCK_{action}_PORT", sized, lev, f"{reason} | {why}", {}, False)
            return

        allow, why = allowed_by_single_coin(equity, symbol, proposed_expo * lev, pos_map, price_map)
        if not allow:
            print(f"❌ 拒绝：{why}")
            record_trade(symbol, f"BLOCK_{action}_COIN", sized, lev, f"{reason} | {why}", {}, False)
            return

        # === 相关性簇风控：对同向高度相关的资产总敞口做上限 ===
        cluster_scale = correlation_cluster_scale(symbol, price_map, pos_map, rets_df, equity, proposed_expo * lev)
        if cluster_scale < 1.0:
            sized *= cluster_scale
            proposed_expo = sized * px
            print(f"⚠️ 相关性簇缩放：scale={cluster_scale:.2f}")

        # === 当前是否已有仓位（避免重复同向加仓；若相反则先对冲平掉） ===
        existing_qty = pos_map.get(symbol, 0.0)
        side = None

        if action == "BUY":
            if existing_qty > 0:
                print(f"⚠️ 已有多头 {symbol}，跳过加仓。")
                record_trade(symbol, "SKIP_BUY_EXISTS", 0.0, lev, reason, {}, True)
                return
            elif existing_qty < 0:
                # 先平空
                print(f"🔄 平空 {symbol} 数量={abs(existing_qty)}")
                res_close = place_order(symbol, "buy", abs(existing_qty), lev)
                record_trade(symbol, "CLOSE_SHORT", abs(existing_qty), lev, "Auto close before long", res_close, True)
                # 再开多
                side = "buy"

            else:
                side = "buy"

        elif action == "SELL":
            if existing_qty < 0:
                print(f"⚠️ 已有空头 {symbol}，跳过加仓。")
                record_trade(symbol, "SKIP_SELL_EXISTS", 0.0, lev, reason, {}, True)
                return
            elif existing_qty > 0:
                # 先平多
                print(f"🔄 平多 {symbol} 数量={abs(existing_qty)}")
                res_close = place_order(symbol, "sell", abs(existing_qty), lev)
                record_trade(symbol, "CLOSE_LONG", abs(existing_qty), lev, "Auto close before short", res_close, True)
                # 再开空
                side = "sell"
            else:
                side = "sell"

        else:
            print("✅ HOLD")
            record_trade(symbol, "HOLD", 0.0, 0, reason, {}, True)
            save_state(state)
            return

        # === 执行下单 ===
        if sized <= 0:
            print("⚠️ sized=0，放弃下单。")
            record_trade(symbol, f"SKIP_{action}_ZERO", 0.0, lev, reason, {}, True)
            return

        print(f"📈 下单 {action} {symbol} size={sized:.6f} lev={lev}")
        res = place_order(symbol, side, sized, lev)
        record_trade(symbol, action, sized, lev, reason, res, True)

        save_state(state)

    except Exception as e:
        print("❌ 运行异常：", e)
        traceback.print_exc()

if __name__ == "__main__":
    main()
