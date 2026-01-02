import json
import datetime as dt
import time
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.express as px
from pathlib import Path

OUT_DIR = Path("docs")
OUT_DIR.mkdir(exist_ok=True)

# ===== 固定設定（你要的：2026/01/01 以後開始）=====
START = "2026-01-01"

# yfinance 的 end 常常是「不含 end 當天」，所以用「明天」確保含今天
END = (dt.date.today() + dt.timedelta(days=1)).isoformat()

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

# 匯率備援：抓不到 Yahoo 的 TWD=X 時，就用這個（你也可改成自己常用匯率）
FX_FALLBACK_TWD_PER_USD = 32.0
# ===================================================


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


def safe_download(ticker: str, start: str, end: str, tries: int = 3, sleep_sec: float = 1.0) -> pd.DataFrame:
    last_err = None
    for _ in range(tries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False, threads=False)
            if isinstance(df, pd.DataFrame) and not df.empty:
                return df
        except Exception as e:
            last_err = e
        time.sleep(sleep_sec)
    # 失敗就回空表（由上層決定怎麼處理）
    return pd.DataFrame()


def read_fx_fallback_from_snapshot(default_fx: float) -> float:
    """
    如果你未來想用「國泰 App 當下匯率」：可以在 actual_snapshot.json 裡加 fx_twd_per_usd。
    """
    snap_path = Path("actual_snapshot.json")
    if not snap_path.exists():
        return default_fx
    try:
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
        fx = snap.get("fx_twd_per_usd", None)
        if fx is None:
            return default_fx
        fx = float(fx)
        return fx if fx > 1 else default_fx
    except Exception:
        return default_fx


def simulate(start, end, monthly_twd, trade_mode):
    """
    回傳：
      df: 模擬結果（就算沒資料也會有固定欄位）
      warn: 警告訊息（例如匯率用備援）
    """
    warn = []

    # 1) QQQ
    qqq = safe_download("QQQ", start=start, end=end)
    if qqq.empty:
        # 直接回「空但有欄位」的 df，避免後面 plotly 爆炸
        cols = [
            "date", "fx_twd_per_usd_mid", "qqq_price_usd", "twd_contribution",
            "fx_oneway_spread", "commission_usd", "usd_after_fx_and_fee",
            "shares_bought", "total_shares", "portfolio_value_twd"
        ]
        warn.append("QQQ 價格資料抓不到（Yahoo Finance）。本次僅更新持倉快照區塊。")
        return pd.DataFrame(columns=cols), warn

    qqq_price = qqq["Close"].dropna()
    if qqq_price.empty:
        cols = [
            "date", "fx_twd_per_usd_mid", "qqq_price_usd", "twd_contribution",
            "fx_oneway_spread", "commission_usd", "usd_after_fx_and_fee",
            "shares_bought", "total_shares", "portfolio_value_twd"
        ]
        warn.append("QQQ Close 欄位為空。本次僅更新持倉快照區塊。")
        return pd.DataFrame(columns=cols), warn

    # 2) 匯率（TWD=X）
    fx = safe_download("TWD=X", start=start, end=end)
    fx_mid = None

    if not fx.empty and "Close" in fx.columns:
        fx_mid = fx["Close"].dropna()
    else:
        fx_mid = pd.Series(dtype=float)

    # 對齊到 QQQ 的交易日
    if fx_mid is not None and len(fx_mid) > 0:
        fx_rate = fx_mid.reindex(qqq_price.index).ffill()
    else:
        # 用備援匯率：先看 snapshot 有沒有 fx_twd_per_usd，沒有就用常數
        fallback = read_fx_fallback_from_snapshot(FX_FALLBACK_TWD_PER_USD)
        fx_rate = pd.Series([fallback] * len(qqq_price.index), index=qqq_price.index, dtype=float)
        warn.append(f"TWD=X 匯率資料抓不到（Yahoo Finance），改用備援匯率 {fallback:.4f} TWD/USD。")

    trade_dates = pick_trade_dates(qqq_price.index, start, end, trade_mode)
    if len(trade_dates) == 0:
        cols = [
            "date", "fx_twd_per_usd_mid", "qqq_price_usd", "twd_contribution",
            "fx_oneway_spread", "commission_usd", "usd_after_fx_and_fee",
            "shares_bought", "total_shares", "portfolio_value_twd"
        ]
        warn.append("本區間沒有可用交易日（可能是剛好假日或資料未更新）。")
        return pd.DataFrame(columns=cols), warn

    commission = cathay_commission_usd(COMMISSION_MODE)
    fx_spread = cathay_fx_oneway_spread(FX_SPREAD_MODE)

    total_shares = 0.0
    rows = []

    for d in trade_dates:
        # 用 iloc[0] 避免 FutureWarning
        rate_val = fx_rate.loc[d]
        price_val = qqq_price.loc[d]

        rate = float(rate_val.iloc[0]) if isinstance(rate_val, pd.Series) else float(rate_val)
        price = float(price_val.iloc[0]) if isinstance(price_val, pd.Series) else float(price_val)

        if rate < 1:
            # 不直接 raise，避免整個報表死掉
            warn.append("匯率數值異常（<1），本次以備援匯率取代。")
            rate = read_fx_fallback_from_snapshot(FX_FALLBACK_TWD_PER_USD)

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

    df = pd.DataFrame(rows)
    return df, warn


def fmt_int(n: float) -> str:
    try:
        return f"{float(n):,.0f}"
    except Exception:
        return str(n)


def build_cathay_style_snapshot():
    """
    讀取 actual_snapshot.json（positions[]）：
    - 國泰風格總覽卡片
    - 持倉明細
    - 分配圓餅圖
    """
    snap_path = Path("actual_snapshot.json")
    if not snap_path.exists():
        return (
            """<h2>實際持倉（未提供）</h2>
               <p class="muted">找不到 actual_snapshot.json（請放在 repo 根目錄）。</p>""",
            ""
        )

    try:
        snap = json.loads(snap_path.read_text(encoding="utf-8"))
    except Exception:
        return (
            """<h2>實際持倉（讀取失敗）</h2>
               <p class="muted">actual_snapshot.json 格式錯誤（不是合法 JSON）。</p>""",
            ""
        )

    as_of = snap.get("as_of", "")
    broker = snap.get("broker", "")
    currency = snap.get("currency", "TWD")
    positions = snap.get("positions", [])

    if not isinstance(positions, list) or len(positions) == 0:
        return (
            """<h2>實際持倉（空）</h2>
               <p class="muted">positions[] 為空，請填入至少一筆持倉。</p>""",
            ""
        )

    total_mv = float(sum(p.get("market_value_twd", 0) for p in positions))
    total_cost = float(sum(p.get("total_cost_twd", 0) for p in positions))
    pnl = total_mv - total_cost
    pnl_pct = (pnl / total_cost) if total_cost > 0 else 0.0

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

    # 明細
    table_rows_html = ""
    for p in positions:
        mv = float(p.get("market_value_twd", 0))
        cost = float(p.get("total_cost_twd", 0))
        pp = mv - cost
        pp_pct = (pp / cost) if cost > 0 else 0.0
        pp_class = "pos" if pp >= 0 else "neg"
        pp_sign = "+" if pp >= 0 else "-"

        table_rows_html += f"""
        <tr>
          <td><b>{p.get("symbol","")}</b><br><span class="muted">{p.get("name","")}</span></td>
          <td style="text-align:right;">{p.get("shares","")}</td>
          <td style="text-align:right;">{p.get("avg_cost_usd","")}</td>
          <td style="text-align:right;">{fmt_int(cost)}</td>
          <td style="text-align:right;">{fmt_int(mv)}</td>
          <td style="text-align:right;" class="{pp_class}">{pp_sign}{fmt_int(abs(pp))}<br>
              <span class="{pp_class}">{pp_sign}{abs(pp_pct)*100:.2f}%</span>
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

    # 分配圓餅
    alloc_df = pd.DataFrame([{
        "symbol": p.get("symbol", ""),
        "value": float(p.get("market_value_twd", 0))
    } for p in positions])

    alloc_plot_html = ""
    if alloc_df["value"].sum() > 0:
        fig_alloc = px.pie(alloc_df, names="symbol", values="value", title="資產分配（依現值 TWD）", hole=0.55)
        fig_alloc.update_traces(textposition="inside", textinfo="percent+label")
        alloc_plot_html = fig_alloc.to_html(full_html=False, include_plotlyjs="cdn")

    return (card_html + holdings_html), alloc_plot_html


def build_report(df: pd.DataFrame, warn_list):
    # 模擬摘要（df 可能為空）
    if df is None:
        df = pd.DataFrame()

    warn_html = ""
    if warn_list:
        items = "".join([f"<li>{w}</li>" for w in warn_list])
        warn_html = f"""
        <h2>系統提示</h2>
        <div class="card">
          <ul>{items}</ul>
        </div>
        """

    # 參數卡
    params = f"""
    <h2>參數</h2>
    <div class="card">
      <ul>
        <li>模擬起點：{START}（你指定 2026/01/01 以後）</li>
        <li>買入規則：月初（每月第一個交易日）</li>
        <li>每月投入：{fmt_int(MONTHLY_TWD)} TWD</li>
        <li>手續費模式：{COMMISSION_MODE}（每筆 {cathay_commission_usd(COMMISSION_MODE):.2f} USD）</li>
        <li>匯差模式：{FX_SPREAD_MODE}（單邊 {cathay_fx_oneway_spread(FX_SPREAD_MODE)*100:.2f}%）</li>
        <li>匯率資料：Yahoo Finance 的 TWD=X（抓不到會用備援）</li>
      </ul>
    </div>
    """

    # 實際持倉區（國泰風格）
    snapshot_html, alloc_plot_html = build_cathay_style_snapshot()

    # 模擬圖（df 可能為空 -> 不畫）
    curve_html = ""
    summary_html = ""
    table_html = ""

    if isinstance(df, pd.DataFrame) and (not df.empty) and ("date" in df.columns) and ("portfolio_value_twd" in df.columns):
        months = len(df)
        total_in = MONTHLY_TWD * months
        final_value = float(df["portfolio_value_twd"].iloc[-1]) if months else 0.0
        profit = final_value - total_in
        roi = (profit / total_in) if total_in > 0 else 0.0

        summary_html = f"""
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

        fig_curve = px.line(
            df,
            x="date",
            y="portfolio_value_twd",
            title="QQQ 每月投入 10,000 TWD（國泰成本假設）資產曲線"
        )
        fig_curve.update_layout(xaxis_title="日期", yaxis_title="資產（TWD）")
        curve_html = f"""
        <h2>資產曲線（模擬）</h2>
        <div class="card">
          {fig_curve.to_html(full_html=False, include_plotlyjs="cdn")}
        </div>
        """

        table_html = f"""
        <h2>最近 24 筆（模擬）</h2>
        <div class="card">
          {df.tail(24).to_html(index=False)}
        </div>
        """
    else:
        summary_html = """
        <h2>模擬摘要</h2>
        <div class="card"><p class="muted">本次沒有產生可繪圖的模擬資料（可能是交易日資料未更新或抓取失敗）。</p></div>
        """

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
      /* 台灣券商習慣：紅=漲/獲利、綠=跌/虧損 */
      .pos { color:#d10b2f; }
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

      {warn_html}

      {snapshot_html}

      <h2>資產分配</h2>
      <div class="card">
        {alloc_plot_html if alloc_plot_html else "<p class='muted'>目前只有單一標的或現值為 0，分配圖不顯示。</p>"}
      </div>

      {summary_html}
      {params}
      {curve_html}
      {table_html}

      <p class="muted">更新時間：{dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </body></html>
    """

    (OUT_DIR / "index.html").write_text(html, encoding="utf-8")

    # df 可能空，仍輸出 data.csv（方便你日後下載）
    if isinstance(df, pd.DataFrame) and not df.empty:
        df.to_csv(OUT_DIR / "data.csv", index=False, encoding="utf-8-sig")
    else:
        pd.DataFrame().to_csv(OUT_DIR / "data.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    df, warn = simulate(START, END, MONTHLY_TWD, TRADE_MODE)
    build_report(df, warn)
