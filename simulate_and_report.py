import json
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from pathlib import Path

OUT_DIR = Path("docs")
OUT_DIR.mkdir(exist_ok=True)

# ===== 固定設定 =====
START = "2026-01-01"
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
# ===================


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
        # yfinance 有時回 Series，統一用 iloc[0]
        rate = float(fx_rate.loc[d].iloc[0])       # TWD per USD（近似中間價）
        price = float(qqq_price.loc[d].iloc[0])    # QQQ price in USD

        if rate < 1:
            raise RuntimeError("匯率看起來反了（fx 太小）。請檢查 TWD=X 的定義。")

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

    return pd.DataFrame(rows)


def fmt_int(n: float) -> str:
    return f"{n:,.0f}"


def build_cathay_style_snapshot():
    """
    讀取 actual_snapshot.json（新版：positions[]），產出：
    - 國泰風格總覽卡片（總現值/總成本/總損益/報酬率）
    - 資產分配圓餅圖（positions>1 時很有感；只有 QQQ 也照畫）
    - 持倉明細表（像 App 那樣列出）
    """
    snap_path = Path("actual_snapshot.json")
    if not snap_path.exists():
        return (
            """<h2>實際持倉快照</h2>
               <p style="color:#b00;">找不到 actual_snapshot.json（請放在 repo 根目錄）。</p>""",
            ""  # plotly html
        )

    snap = json.loads(snap_path.read_text(encoding="utf-8"))
    as_of = snap.get("as_of", "")
    broker = snap.get("broker", "")
    currency = snap.get("currency", "TWD")

    positions = snap.get("positions", [])
    if not isinstance(positions, list) or len(positions) == 0:
        return (
            """<h2>實際持倉快照</h2>
               <p style="color:#b00;">actual_snapshot.json 的 positions 格式不正確。</p>""",
            ""
        )

    # 彙總（像國泰那張卡）
    total_mv = float(sum(p.get("market_value_twd", 0) for p in positions))
    total_cost = float(sum(p.get("total_cost_twd", 0) for p in positions))
    pnl = total_mv - total_cost
    pnl_pct = (pnl / total_cost) if total_cost > 0 else 0.0

    pnl_class = "pos" if pnl >= 0 else "neg"
    pnl_sign = "+" if pnl >= 0 else "-"

    # 卡片 HTML（仿國泰：現值大字、損益紅綠）
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

    # 明細表（像 App：標的、股數、均價、成本、現值、損益）
    rows = []
    for p in positions:
        mv = float(p.get("market_value_twd", 0))
        cost = float(p.get("total_cost_twd", 0))
        pp = mv - cost
        pp_pct = (pp / cost) if cost > 0 else 0.0
        pp_class = "pos" if pp >= 0 else "neg"
        pp_sign = "+" if pp >= 0 else "-"
        rows.append({
            "symbol": p.get("symbol", ""),
            "name": p.get("name", ""),
            "shares": p.get("shares", ""),
            "avg_cost_usd": p.get("avg_cost_usd", ""),
            "cost_twd": cost,
            "mv_twd": mv,
            "pnl_twd": pp,
            "pnl_pct": pp_pct,
            "pnl_class": pp_class,
            "pnl_sign": pp_sign,
        })

    # 產明細表 html（自做，方便套顏色）
    table_rows_html = ""
    for r in rows:
        table_rows_html += f"""
        <tr>
          <td><b>{r["symbol"]}</b><br><span class="muted">{r["name"]}</span></td>
          <td style="text-align:right;">{r["shares"]}</td>
          <td style="text-align:right;">{r["avg_cost_usd"]}</td>
          <td style="text-align:right;">{fmt_int(r["cost_twd"])}</td>
          <td style="text-align:right;">{fmt_int(r["mv_twd"])}</td>
          <td style="text-align:right;" class="{r["pnl_class"]}">{r["pnl_sign"]}{fmt_int(abs(r["pnl_twd"]))}<br>
              <span class="{r["pnl_class"]}">{r["pnl_sign"]}{abs(r["pnl_pct"])*100:.2f}%</span>
          </td>
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
        <tbody>
          {table_rows_html}
        </tbody>
      </table>
    </div>
    """

    # 分配圖（圓餅）
    alloc_df = pd.DataFrame([{
        "symbol": p.get("symbol", ""),
        "value": float(p.get("market_value_twd", 0))
    } for p in positions])

    # 避免 value 全 0 造成 plotly 爆炸
    if alloc_df["value"].sum() <= 0:
        alloc_plot_html = ""
    else:
        fig_alloc = px.pie(alloc_df, names="symbol", values="value", title="資產分配（依現值 TWD）", hole=0.55)
        fig_alloc.update_traces(textposition="inside", textinfo="percent+label")
        alloc_plot_html = fig_alloc.to_html(full_html=False, include_plotlyjs="cdn")

    snapshot_section = card_html + holdings_html
    return snapshot_section, alloc_plot_html


def build_report(df: pd.DataFrame):
    months = len(df)
    total_in = MONTHLY_TWD * months
    final_value = float(df["portfolio_value_twd"].iloc[-1]) if months else 0.0
    profit = final_value - total_in
    roi = (profit / total_in) if total_in > 0 else 0.0

    fig_curve = px.line(
        df,
        x="date",
        y="portfolio_value_twd",
        title="QQQ 每月投入 10,000 TWD（國泰成本假設）資產曲線"
    )
    fig_curve.update_layout(xaxis_title="日期", yaxis_title="資產（TWD）")

    summary = f"""
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

    params = f"""
    <h2>參數</h2>
    <div class="card">
      <ul>
        <li>買入規則：月初（每月第一個交易日）</li>
        <li>每月投入：{fmt_int(MONTHLY_TWD)} TWD</li>
        <li>手續費模式：{COMMISSION_MODE}（每筆 {cathay_commission_usd(COMMISSION_MODE):.2f} USD）</li>
        <li>匯差模式：{FX_SPREAD_MODE}（單邊 {cathay_fx_oneway_spread(FX_SPREAD_MODE)*100:.2f}%）</li>
        <li>匯率資料：Yahoo Finance 的 TWD=X（近似中間價）</li>
      </ul>
    </div>
    """

    snapshot_html, alloc_plot_html = build_cathay_style_snapshot()

    table_html = df.tail(24).to_html(index=False)

    css = """
    <style>
      body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; padding:16px; }
      h1 { margin: 8px 0 12px; }
      h2 { margin: 18px 0 10px; }
      .card { border:1px solid #e6e6e6; border-radius:14px; padding:14px; background:#fff; box-shadow:0 1px 2px rgba(0,0,0,0.05); }
      .row { display:flex; flex-direction:column; gap:6px; }
      .label { color:#666; font-size:13px; }
      .big { font-size:34px; font-weight:800; letter-spacing:0.3px; }
      .split { display:flex; gap:12px; margin-top:10px; }
      .cell { flex:1; border-top:1px solid #eee; padding-top:10px; }
      .num { font-size:18px; font-weight:700; }
      .sub { margin-top:10px; color:#777; font-size:12px; }
      .pos { color:#d10b2f; }  /* 國泰常見：獲利偏紅（台股習慣） */
      .neg { color:#0a8f45; }
      .muted { color:#888; font-size:12px; }
      table.holdings { width:100%; border-collapse:collapse; }
      table.holdings th, table.holdings td { padding:10px 6px; border-bottom:1px solid #eee; vertical-align:top; font-size:13px; }
      table.holdings th { color:#555; font-weight:700; }
    </style>
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

      {summary}
      {params}

      <h2>資產曲線（模擬）</h2>
      <div class="card">
        {fig_curve.to_html(full_html=False, include_plotlyjs="cdn")}
      </div>

      <h2>最近 24 筆（模擬）</h2>
      <div class="card">
        {table_html}
      </div>

      <p class="muted">更新時間：{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </body></html>
    """

    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")
    df.to_csv(OUT_DIR / "data.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    df = simulate(START, END, MONTHLY_TWD, TRADE_MODE)
    build_report(df)
