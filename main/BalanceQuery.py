import okx.Account as Account
import pandas as pd
import json

# API 初始化
apikey = "ab76cd78-0311-44f5-90f8-4bab0e2b0e1c"
secretkey = "7E6EA8C8E2873E40C6296014EEA83E17"
passphrase = "9617Ios@"

flag = "1"  # 实盘:0 , 模拟盘:1

accountAPI = Account.AccountAPI(apikey, secretkey, passphrase, False, flag)

# 查看账户余额
result = accountAPI.get_account_balance()
json_str = json.dumps(result)  # '{"name": "Alice", "age": 30}' ✅ 合法 JSON
print(json_str)

data = result["data"][0]
total = float(data["totalEq"])
details = data["details"]
    
    # 整理为 DataFrame
df = pd.DataFrame(details)[["ccy", "eq", "availEq", "frozenBal"]]
df.columns = ["币种", "总资产", "可用", "冻结"]
    
print("💰 OKX 账户资产汇总")
print(df.to_string(index=False))
print(f"\n📊 总资产折合 (USDT): {total:.4f}")
print(result)