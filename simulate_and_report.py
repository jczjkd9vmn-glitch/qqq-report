import json
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from pathlib import Path

OUT_DIR = Path("docs")
OUT_DIR.mkdir(exist_ok=True)

# ===== 你的固定設定 =====
START = "2015-01-01"
END = dt.date.today().isoformat()
MONTHLY_TWD = 10000.0
TRADE_MODE = "month_start"  # 月初：每月第一個交易日

# 手續費模式（二選一）
# - etf_normal：一般 ETF 每筆 3 USD（較保守）
# - dca：定期定額買進每筆 0.1 USD
COMMISSION_MODE = "etf_normal"

# 匯差模式（二選一）
# - digital：數位通路單邊約 0.10%
# - spot：一般即期單邊約 0.19%
FX_SPREAD_MODE = "digital"
# =======================


def pick_trade_dates(price_index, start, end, mode="month_start"):
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    dates = price_index[(price_index >= start_dt) & (price_index <= end_dt)]
    df = pd.DataFrame(index=dates)
    df["y"] = df.index.year
    df["m"] = df.index.month
    if mode == "month_start":
        return df.groupby(["y", "m"]).head(1).index
    elif mode == "month_end":
        return df.groupby(["y", "m"]).tail(1).index
    else:
        raise ValueError("mode must be month_start or month_end")


def cathay_commission_usd(mode: str) -> float:
    if mode == "etf_normal":
        return 3.0
    if mode == "dca":
        return 0.1
    raise ValueError("COMMISSION_MODE must be 'etf_normal' or 'dca'")


def cathay_fx_oneway_spread(mode: str) -> float:
    if mode == "digital":
        return 0.0010  # 0.10%
    if mode == "spot":
        return 0.0019  # 0.19%
    raise ValueError("FX_SPREAD_MODE must be 'digital' or 'spot'")


def simulate(start, end, monthly_twd, trade_mode):
    qqq = yf.download("QQQ", start=start, end=end, auto_adjust=True, progress=False)
    fx = yf.download("TWD=X", start=start, end=end, auto_adjust=True, progress=False)

    if qqq.empty:
        raise RuntimeError("QQQ 價格資料抓不到（Yahoo Finance）。")
    if fx.empty:
        raise RuntimeError("TWD=X 匯率資料抓不到（Yahoo Finance）。")

    qqq_price = qqq["Close"].dropna()
    fx_rate = fx["Close"].dropna().reindex(qqq_price.index).ffill()

    trade_dates = pick_trade_dates(qqq_price.index, start, end, trade_mode)

    commission = cathay_commission_usd(COMMISSION_MODE)
    fx_spread = cathay_fx_oneway_spread(FX_SPREAD_MODE)

    total_shares = 0.0
    rows = []

    for d in trade_dates:
        # 用 iloc[0] 避免 FutureWarning（yfinance 有時回 Series）
        rate = float(fx_rate.loc[d].iloc[0])       # 近似中間價（TWD per USD）
        price = float(qqq_price.loc[d].iloc[0])    # QQQ USD price

        # 基本 sanity check：正常應該在 20~50 左右
        if rate < 1:
            raise RuntimeError("匯率看起來反了（fx 太小）。請檢查 TWD=X 的定義。")

        # TWD -> USD：扣單邊匯差
        usd_gross = monthly_twd / rate
        usd_net_fx = usd_gross * (1 - fx_spread)

        # 扣固定手續費（USD）
        usd_after_fee = max(usd_net_fx - commission, 0.0)

        # 允許小數股（更接近「理論 DCA」）
        shares = usd_after_fee / price
        total_shares += shares

        port_usd = total_shares * price
        port_twd = port_usd * rate

        rows.append({
            "date": d.date().isoformat(),
            "fx_twd_per_usd_mid": rate,
            "qqq_price_usd": price,
            "twd_contribution": monthly_twd,
            "fx_oneway_spread": fx_spread,
            "commission_usd": commission,
            "usd_after_fx_and_fee": usd_after_fee,
            "shares_bought": shares,
            "total_shares": total_shares,
            "portfolio_value_twd": port_twd,
        })

    return pd.DataFrame(rows)


def build_report(df: pd.DataFrame):
    months = len(df)
    total_in = MONTHLY_TWD * months
    final_value = float(df["portfolio_value_twd"].iloc[-1]) if months else 0.0
    profit = final_value - total_in
    roi = (profit / total_in) if total_in > 0 else 0.0

    fig = px.line(
        df,
        x="date",
        y="portfolio_value_twd",
        title="QQQ 每月投入 10,000 TWD（國泰成本假設）資產曲線"
    )
    fig.update_layout(xaxis_title="日期", yaxis_title="資產（TWD）")

    summary = f"""
    <h2>摘要</h2>
    <ul>
      <li>月份數：{months}</li>
      <li>總投入：{total_in:,.0f} TWD</li>
      <li>期末資產：{final_value:,.0f} TWD</li>
      <li>損益：{profit:,.0f} TWD</li>
      <li>ROI：{roi*100:.2f}%</li>
    </ul>
    """

    params = f"""
    <h2>參數</h2>
    <ul>
      <li>買入規則：月初（每月第一個交易日）</li>
      <li>每月投入：{MONTHLY_TWD:,.0f} TWD</li>
      <li>手續費模式：{COMMISSION_MODE}（每筆 {cathay_commission_usd(COMMISSION_MODE):.2f} USD）</li>
      <li>匯差模式：{FX_SPREAD_MODE}（單邊 {cathay_fx_oneway_spread(FX_SPREAD_MODE)*100:.2f}%）</li>
      <li>匯率資料：Yahoo Finance 的 TWD=X（近似中間價）</li>
    </ul>
    """

    # ===== 實際持倉快照（你用手機截圖抄數字更新）=====
    snapshot_html = ""
    snap_path = Path("actual_snapshot.json")
    if snap_path.exists():
        snap = json.loads(snap_path.read_text(encoding="utf-8"))

        total_cost_twd = float(snap.get("total_cost_twd", 0))
        market_value_twd = float(snap.get("market_value_twd", 0))
        pnl_twd = market_value_twd - total_cost_twd
        pnl_pct = (pnl_twd / total_cost_twd) if total_cost_twd > 0 else 0.0

        snapshot_html = f"""
        <h2>實際持倉快照（手動更新）</h2>
        <ul>
          <li>日期：{snap.get("as_of","")}</li>
          <li>券商：{snap.get("broker","")}</li>
          <li>標的：{snap.get("symbol","")}</li>
          <li>股數：{snap.get("shares","")}</li>
          <li>成交均價（USD）：{snap.get("avg_cost_usd","")}</li>
          <li>總成本（TWD）：{total_cost_twd:,.0f}</li>
          <li>總預估現值（TWD）：{market_value_twd:,.0f}</li>
          <li>預估損益（TWD）：{pnl_twd:,.0f}（{pnl_pct*100:.2f}%）</li>
        </ul>
        """
    else:
        snapshot_html = """
        <h2>實際持倉快照（手動更新）</h2>
        <p style="color:#b00;">找不到 actual_snapshot.json（請放在 repo 根目錄）。</p>
        """
    # ===============================================

    table_html = df.tail(24).to_html(index=False)

    html = f"""
    <html><head><meta charset="utf-8"><title>QQQ 每月投入回測（國泰成本）</title></head>
    <body style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding:16px;">
      <h1>QQQ 每月投入 10,000 TWD 回測（國泰成本假設）</h1>
      {summary}
      {snapshot_html}
      {params}
      <h2>資產曲線</h2>
      {fig.to_html(full_html=False, include_plotlyjs="cdn")}
      <h2>最近 24 筆</h2>
      {table_html}
      <p style="color:#666;">更新時間：{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </body></html>
    """

    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    df.to_csv(OUT_DIR / "data.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    df = simulate(START, END, MONTHLY_TWD, TRADE_MODE)
    build_report(df)
