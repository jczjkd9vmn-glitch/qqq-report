import json
import datetime as dt
import pandas as pd
import yfinance as yf
import plotly.express as px
from pathlib import Path

OUT_DIR = Path("docs")
OUT_DIR.mkdir(exist_ok=True)

# ===== 你要的起始日（只看 2026/01/01 以後）=====
START = "2026-01-01"

# yfinance 的 end 是「不包含」當天，所以用「明天」避免抓不到今天/最近交易日
END = (dt.date.today() + dt.timedelta(days=1)).isoformat()

MONTHLY_TWD = 10000.0
TRADE_MODE = "month_start"  # 月初：每月第一個交易日

# 國泰成本（你可自行切換）
COMMISSION_MODE = "etf_normal"  # "etf_normal"=3USD/筆, "dca"=0.1USD/筆
FX_SPREAD_MODE = "digital"      # "digital"=0.10%, "spot"=0.19%
# ===============================================


def pick_trade_dates(price_index, start, end, mode="month_start"):
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    dates = price_index[(price_index >= start_dt) & (price_index <= end_dt)]
    if dates.empty:
        return dates
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
    # 下載時往前多抓一段，避免遇到假日/資料空窗就整個空掉
    dl_start = (pd.to_datetime(start) - pd.Timedelta(days=45)).date().isoformat()
    dl_end = (pd.to_datetime(end) + pd.Timedelta(days=1)).date().isoformat()

    qqq = yf.download("QQQ", start=dl_start, end=dl_end, auto_adjust=True, progress=False)
    fx = yf.download("TWD=X", start=dl_start, end=dl_end, auto_adjust=True, progress=False)

    # 回傳固定欄位，避免 df 空造成 plotly 爆炸
    cols = [
        "date", "fx_twd_per_usd_mid", "qqq_price_usd", "twd_contribution",
        "fx_oneway_spread", "commission_usd", "usd_after_fx_and_fee",
        "shares_bought", "total_shares", "portfolio_value_twd"
    ]

    if qqq.empty or "Close" not in qqq:
        return pd.DataFrame(columns=cols)
    if fx.empty or "Close" not in fx:
        return pd.DataFrame(columns=cols)

    qqq_price = qqq["Close"].dropna()
    fx_rate = fx["Close"].dropna().reindex(qqq_price.index).ffill()

    # 只保留 START 以後
    start_dt = pd.to_datetime(start)
    qqq_price = qqq_price[qqq_price.index >= start_dt]
    fx_rate = fx_rate.reindex(qqq_price.index).ffill()

    if qqq_price.empty or fx_rate.empty:
        return pd.DataFrame(columns=cols)

    trade_dates = pick_trade_dates(qqq_price.index, start, end, trade_mode)
    if trade_dates.empty:
        return pd.DataFrame(columns=cols)

    commission = cathay_commission_usd(COMMISSION_MODE)
    fx_spread = cathay_fx_oneway_spread(FX_SPREAD_MODE)

    total_shares = 0.0
    rows = []

    for d in trade_dates:
        rate = float(fx_rate.loc[d])     # 單值
        price = float(qqq_price.loc[d])  # 單值

        if rate < 1:
            return pd.DataFrame(columns=cols)

        usd_gross = monthly_twd / rate
        usd_net_fx = usd_gross * (1 - fx_spread)
        usd_after_fee = max(usd_net_fx - commission, 0.0)

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

    return pd.DataFrame(rows, columns=cols)


def fmt_int(n: float) -> str:
    return f"{n:,.0f}"


def build_cathay_style_snapshot():
    snap_path = Path("actual_snapshot.json")
    if not snap_path.exists():
        return ("<h2>實際持倉</h2><p style='color:#b00;'>找不到 actual_snapshot.json（請放 repo 根目錄）</p>", "")

    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    as_of = snap.get("as_of", "")
    broker = snap.get("broker", "")
    currency = snap.get("currency", "TWD")
    positions = snap.get("positions", [])

    if not isinstance(positions, list) or len(positions) == 0:
        return ("<h2>實際持倉</h2><p style='color:#b00;'>positions 格式錯誤</p>", "")

    total_mv = float(sum(p.get("market_value_twd", 0) for p in positions))
    total_cost = float(sum(p.get("total_cost_twd", 0) for p in positions))
    pnl = total_mv - total_cost
    pnl_pct = (pnl / total_cost) if total_cost > 0 else 0.0

    # 台灣券商慣例：正數紅、負數綠
    pnl_class = "pos" if pnl >= 0 else "neg"
    pnl_sign = "+" if pnl >= 0 else "-"

    card_html = f"""
    <h2>實際持倉（{broker}）</h2>
    <div class="card">
      <div class="row">
        <div class="label">總預估現值（{currency}）</div>
        <div class="big">{fmt_int(total_mv)}</div>
      </div>
      <div class="split">
        <div class="cell">
          <div class="label">總成本</div>
          <div class="num">{fmt_int(total_cost)}</div>
        </div>
        <div class="cell">
          <div class="label">總預估損益</div>
          <div class="num {pnl_class}">{pnl_sign}{fmt_int(abs(pnl))}（{pnl_sign}{abs(pnl_pct)*100:.2f}%）</div>
        </div>
      </div>
      <div class="sub">更新日期：{as_of}</div>
    </div>
    """

    # 明細表
    rows_html = ""
    for p in positions:
        mv = float(p.get("market_value_twd", 0))
        cost = float(p.get("total_cost_twd", 0))
        pp = mv - cost
        pp_pct = (pp / cost) if cost > 0 else 0.0
        pp_class = "pos" if pp >= 0 else "neg"
        pp_sign = "+" if pp >= 0 else "-"

        rows_html += f"""
        <tr>
          <td><b>{p.get("symbol","")}</b><br><span class="muted">{p.get("name","")}</span></td>
          <td style="text-align:right;">{p.get("shares","")}</td>
          <td style="text-align:right;">{p.get("avg_cost_usd","")}</td>
          <td style="text-align:right;">{fmt_int(cost)}</td>
          <td style="text-align:right;">{fmt_int(mv)}</td>
          <td style="text-align:right;" class="{pp_class}">{pp_sign}{fmt_int(abs(pp))}<br>{pp_sign}{abs(pp_pct)*100:.2f}%</td>
        </tr>
        """

    holdings_html = f"""
    <h2>持倉明細</h2>
    <div class="card">
      <table class="holdings">
        <thead>
          <tr>
            <th style="text-align:left;">標的</th>
            <th style="text-align:right;">股數</th>
            <th style="text-align:right;">成交均價(USD)</th>
            <th style="text-align:right;">總成本(TWD)</th>
            <th style="text-align:right;">現值(TWD)</th>
            <th style="text-align:right;">損益</th>
          </tr>
        </thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
    """

    # 分配圖
    alloc_df = pd.DataFrame([{"symbol": p.get("symbol",""), "value": float(p.get("market_value_twd", 0))} for p in positions])
    alloc_plot_html = ""
    if alloc_df["value"].sum() > 0:
        fig_alloc = px.pie(alloc_df, names="symbol", values="value", title="資產分配（依現值 TWD）", hole=0.55)
        fig_alloc.update_traces(textposition="inside", textinfo="percent+label")
        alloc_plot_html = fig_alloc.to_html(full_html=False, include_plotlyjs="cdn")

    return (card_html + holdings_html, alloc_plot_html)


def build_report(df: pd.DataFrame):
    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding:16px; }
      h1 { margin: 8px 0 12px; }
      h2 { margin: 18px 0 10px; }
      .card { border:1px solid #e6e6e6; border-radius:14px; padding:14px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,0.05); }
      .row { display:flex; flex-direction:column; gap:6px; }
      .label { color:#666; font-size:13px; }
      .big { font-size:34px; font-weight:800; }
      .split { display:flex; gap:12px; margin-top:10px; }
      .cell { flex:1; border-top:1px solid #eee; padding-top:10px; }
      .num { font-size:18px; font-weight:700; }
      .sub { margin-top:10px; color:#777; font-size:12px; }
      .pos { color:#d10b2f; } /* 正數紅 */
      .neg { color:#0a8f45; } /* 負數綠 */
      .muted { color:#888; font-size:12px; }
      table.holdings { width:100%; border-collapse:collapse; }
      table.holdings th, table.holdings td { padding:10px 6px; border-bottom:1px solid #eee; vertical-align:top; font-size:13px; }
      table.holdings th { color:#555; font-weight:700; }
    </style>
    """

    snapshot_html, alloc_plot_html = build_cathay_style_snapshot()

    # 模擬區塊：如果 df 空，就顯示提示，不畫圖（避免你現在那個 plotly error）
    if df.empty:
        sim_block = f"""
        <h2>資產曲線（模擬）</h2>
        <div class="card">
          <p class="muted">目前 {START} 之後尚無可用交易日資料（可能遇到假日/資料延遲）。稍後再手動 Run workflow，或等下一個交易日。</p>
        </div>
        """
        recent_block = ""
        summary_block = ""
        params_block = f"""
        <h2>參數</h2>
        <div class="card">
          <ul>
            <li>起始日：{START}</li>
            <li>每月投入：{fmt_int(MONTHLY_TWD)} TWD</li>
            <li>手續費模式：{COMMISSION_MODE}（每筆 {cathay_commission_usd(COMMISSION_MODE):.2f} USD）</li>
            <li>匯差模式：{FX_SPREAD_MODE}（單邊 {cathay_fx_oneway_spread(FX_SPREAD_MODE)*100:.2f}%）</li>
          </ul>
        </div>
        """
    else:
        months = len(df)
        total_in = MONTHLY_TWD * months
        final_value = float(df["portfolio_value_twd"].iloc[-1])
        profit = final_value - total_in
        roi = (profit / total_in) if total_in > 0 else 0.0

        fig_curve = px.line(df, x="date", y="portfolio_value_twd", title=f"QQQ 模擬資產曲線（從 {START} 起）")
        fig_curve.update_layout(xaxis_title="日期", yaxis_title="資產（TWD）")

        summary_block = f"""
        <h2>模擬摘要（不是你的實際帳戶）</h2>
        <div class="card">
          <ul>
            <li>月份數：{months}</li>
            <li>總投入：{fmt_int(total_in)} TWD</li>
            <li>期末資產：{fmt_int(final_value)} TWD</li>
            <li>損益：{fmt_int(profit)} TWD</li>
            <li>ROI：{roi*100:.2f}%</li>
          </ul>
        </div>
        """

        params_block = f"""
        <h2>參數</h2>
        <div class="card">
          <ul>
            <li>起始日：{START}</li>
            <li>買入規則：月初（每月第一個交易日）</li>
            <li>每月投入：{fmt_int(MONTHLY_TWD)} TWD</li>
            <li>手續費模式：{COMMISSION_MODE}（每筆 {cathay_commission_usd(COMMISSION_MODE):.2f} USD）</li>
            <li>匯差模式：{FX_SPREAD_MODE}（單邊 {cathay_fx_oneway_spread(FX_SPREAD_MODE)*100:.2f}%）</li>
            <li>匯率資料：Yahoo Finance 的 TWD=X（近似中間價）</li>
          </ul>
        </div>
        """

        sim_block = f"""
        <h2>資產曲線（模擬）</h2>
        <div class="card">
          {fig_curve.to_html(full_html=False, include_plotlyjs="cdn")}
        </div>
        """

        table_html = df.tail(24).to_html(index=False)
        recent_block = f"""
        <h2>最近 24 筆（模擬）</h2>
        <div class="card">{table_html}</div>
        """

    html = f"""
    <html><head><meta charset="utf-8"><title>QQQ 報告</title>{css}</head>
    <body>
      <h1>QQQ 報告</h1>

      {snapshot_html}

      <h2>資產分配</h2>
      <div class="card">
        {alloc_plot_html if alloc_plot_html else "<p class='muted'>目前只有單一標的或現值為 0，分配圖不顯示。</p>"}
      </div>

      {summary_block}
      {params_block}
      {sim_block}
      {recent_block}

      <p class="muted">更新時間：{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </body></html>
    """

    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    # df 可能空，但也寫出 csv（方便你 debug）
    df.to_csv(OUT_DIR / "data.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    df = simulate(START, END, MONTHLY_TWD, TRADE_MODE)
    build_report(df)