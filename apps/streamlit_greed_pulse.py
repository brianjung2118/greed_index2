# streamlit_greed_pulse.py
# ---------------------------------------------------------------------------
# Single-stock page: everything on the page is about one chosen stock.
# Greed index and greed data are shown in the context of that stock's page.
#
# Run: streamlit run greed_index2/apps/streamlit_greed_pulse.py
# Data: greed_index2/pipeline_output_attention/attention_greed_panel_daily_focus10_kcbert_10k.csv (or _kcbert.csv)
#       greed_index2/prices/{code}.csv, greed_index2/fundamentals/{code}.csv
# ---------------------------------------------------------------------------

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st
import altair as alt

BASE_DIR = Path(__file__).resolve().parent.parent
PRICES_DIR = BASE_DIR / "prices"
FUNDAMENTALS_DIR = BASE_DIR / "fundamentals"
PANEL_10K = BASE_DIR / "pipeline_output_attention" / "attention_greed_panel_daily_focus10_kcbert_10k.csv"
PANEL_FALLBACK = BASE_DIR / "pipeline_output_attention" / "attention_greed_panel_daily_focus10_kcbert.csv"

FOCUS_STOCKS = [
    "005930", "000660", "005380", "105560", "373220",
    "080220", "033100", "190510", "064850", "272110",
]
STOCK_NAMES = {
    "005930": "삼성전자", "000660": "SK하이닉스", "005380": "현대차", "105560": "KB금융",
    "373220": "LG에너지솔루션", "080220": "한솔케미칼", "033100": "청담러닝",
    "190510": "나노엔텍", "064850": "이노션", "272110": "한화솔루션",
}
WINDOW_DAYS = 7
CHART_DAYS = 365


@st.cache_data(show_spinner=False)
def load_panel() -> pd.DataFrame:
    path = PANEL_10K if PANEL_10K.exists() else PANEL_FALLBACK
    if not path.exists():
        raise FileNotFoundError(f"Panel not found: {path}")
    df = pd.read_csv(path, parse_dates=["dt"], low_memory=False)
    df["company_code"] = df["company_code"].astype(str).str.zfill(6)
    df["mean_greed_score"] = pd.to_numeric(df["mean_greed_score"], errors="coerce")
    df["greed"] = (df["mean_greed_score"] / 4.0) * 100.0
    return df


@st.cache_data(show_spinner=False)
def load_price(code: str) -> pd.DataFrame | None:
    path = PRICES_DIR / f"{code}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    df["price"] = pd.to_numeric(df["close"].astype(str).str.replace(",", "", regex=False), errors="coerce")
    return df[["date", "price"]].dropna(subset=["date"]).sort_values("date")


@st.cache_data(show_spinner=False)
def load_fundamentals(code: str) -> pd.DataFrame | None:
    path = FUNDAMENTALS_DIR / f"{code}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y", errors="coerce")
    for col in ("operating_income", "net_income", "net_profit"):
        if col in df.columns:
            df["income"] = pd.to_numeric(df[col], errors="coerce")
            return df[["date", "income"]].dropna(subset=["date"]).sort_values("date")
    return None


def rolling_greed(panel: pd.DataFrame, code: str, window: int) -> pd.DataFrame:
    sub = panel.loc[panel["company_code"] == code, ["dt", "greed"]].copy()
    sub = sub.sort_values("dt").dropna(subset=["greed"])
    sub["greed_roll"] = sub["greed"].rolling(window=window, min_periods=1).mean()
    return sub.rename(columns={"dt": "date"})


def latest_greed(panel: pd.DataFrame, code: str, window: int) -> float | None:
    s = rolling_greed(panel, code, window)
    if s.empty:
        return None
    return float(s["greed_roll"].iloc[-1])


def greed_bounds(panel: pd.DataFrame, code: str, window: int) -> tuple[float, float]:
    s = rolling_greed(panel, code, window)
    if s.empty or s["greed_roll"].isna().all():
        return 0.0, 100.0
    lo, hi = float(s["greed_roll"].min()), float(s["greed_roll"].max())
    return (lo, hi + 1.0) if hi <= lo else (lo, hi)


def greed_display(panel: pd.DataFrame, code: str, window: int, use_relative: bool) -> float | None:
    raw = latest_greed(panel, code, window)
    if raw is None:
        return None
    if not use_relative:
        return raw
    lo, hi = greed_bounds(panel, code, window)
    return max(0, min(100, (raw - lo) / (hi - lo) * 100)) if hi > lo else 50.0


def mood(g: float) -> tuple[str, str, str]:
    if g >= 65:
        return "Greedy", "🔥", "#c0392b"
    if g >= 35:
        return "Neutral", "😐", "#7f8c8d"
    return "Fearful", "🥶", "#2980b9"


def main():
    st.set_page_config(
        page_title="Greed | Stock",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    try:
        panel = load_panel()
    except FileNotFoundError as e:
        st.error(str(e))
        return

    # ----- Stock selector (primary: this page is for one stock) -----
    st.sidebar.markdown("### Pick a stock")
    options = [f"{STOCK_NAMES.get(c, c)} ({c})" for c in FOCUS_STOCKS]
    sel = st.sidebar.selectbox("Stock", options, index=0, label_visibility="collapsed")
    code = FOCUS_STOCKS[options.index(sel)]
    name = STOCK_NAMES.get(code, code)

    window = st.sidebar.selectbox(
        "Smoothing",
        [1, 7, 14, 30],
        index=1,
        format_func=lambda x: {1: "Daily", 7: "7 days", 14: "14 days", 30: "30 days"}[x],
    )
    use_relative = st.sidebar.checkbox(
        "Scale vs this stock's history",
        value=True,
        help="0 = lowest greed for this stock, 100 = highest. Makes the number easier to read.",
    )
    with st.sidebar.expander("What is this?"):
        st.caption(
            "**Greed** = how greedy or fearful the crowd is about this stock, from discussion titles (Naver Finance). "
            "0 = fear, 100 = greed. All charts on this page are for the selected stock only."
        )

    # ----- Page title: stock name (like a stock detail page) -----
    st.title(f"{name}")
    st.caption(f"Stock code {code} · Crowd greed and price for this stock")

    g = greed_display(panel, code, window, use_relative)
    if g is None:
        st.warning(f"No greed data for {name}. This stock may not be in the focus list or data is missing.")
        st.stop()

    # ----- Hero: one number for this stock -----
    label, emoji, color = mood(g)
    st.markdown(
        f"""
        <div style="text-align: center; padding: 1.5rem; border-radius: 12px; background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%); color: #eee; margin: 1rem 0;">
            <div style="font-size: 0.95rem; opacity: 0.9;">Crowd greed for this stock</div>
            <div style="font-size: 3.5rem; font-weight: 800;">{int(g)} <span style="font-size: 2rem;">{emoji}</span></div>
            <div style="font-size: 1.1rem; margin-top: 0.25rem;">{label}</div>
            <div style="margin-top: 0.75rem; height: 6px; background: #333; border-radius: 3px; overflow: hidden;">
                <div style="height: 100%; width: {min(100, max(0, g))}%; background: {color}; border-radius: 3px;"></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ----- Greed over time (this stock only) -----
    st.subheader("Greed over time")
    st.caption(f"Rolling greed (0–100) for **{name}** over the last {CHART_DAYS} days.")
    s = rolling_greed(panel, code, window)
    s = s[s["date"] >= (s["date"].max() - pd.Timedelta(days=CHART_DAYS))]
    if not s.empty:
        if use_relative:
            lo, hi = greed_bounds(panel, code, window)
            s["display"] = s["greed_roll"].apply(lambda x: max(0, min(100, (x - lo) / (hi - lo) * 100)) if hi > lo else 50)
        else:
            s["display"] = s["greed_roll"]
        s["date_str"] = s["date"].dt.strftime("%Y-%m-%d")
        chart = alt.Chart(s).mark_line(stroke=color, strokeWidth=2).encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y("display:Q", title="Greed (0–100)", scale=alt.Scale(domain=[0, 100])),
            tooltip=["date_str:N", "display:Q"],
        ).properties(height=280)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("Not enough history for this stock.")

    # ----- Price & greed together (this stock only) -----
    st.subheader("Price vs greed")
    st.caption(f"**{name}**: price and greed (both normalized to 0–1 over this period) so you can see if they move together.")
    price_df = load_price(code)
    if price_df is not None and not s.empty:
        s_date = s[["date", "greed_roll"]].copy()
        if use_relative:
            lo, hi = greed_bounds(panel, code, window)
            s_date["greed_norm"] = s_date["greed_roll"].apply(lambda x: max(0, min(1, (x - lo) / (hi - lo))) if hi > lo else 0.5)
        else:
            gmin, gmax = s_date["greed_roll"].min(), s_date["greed_roll"].max()
            r = (gmax - gmin) or 1
            s_date["greed_norm"] = (s_date["greed_roll"] - gmin) / r
        merged = price_df.merge(s_date[["date", "greed_norm"]], on="date", how="inner")
        merged = merged.dropna(subset=["price", "greed_norm"]).sort_values("date")
        merged = merged[merged["date"] >= (merged["date"].max() - pd.Timedelta(days=CHART_DAYS))]
        if len(merged) >= 2:
            pmin, pmax = merged["price"].min(), merged["price"].max()
            pr = (pmax - pmin) or 1
            merged["price_norm"] = (merged["price"] - pmin) / pr
            merged["date_str"] = merged["date"].dt.strftime("%Y-%m-%d")
            line_p = alt.Chart(merged).mark_line(stroke="steelblue", strokeWidth=2).encode(
                x=alt.X("date:T"), y=alt.Y("price_norm:Q", title="Normalized (0–1)", scale=alt.Scale(domain=[0, 1])),
                tooltip=["date_str:N", "price:Q", "greed_norm:Q"],
            )
            line_g = alt.Chart(merged).mark_line(stroke="green", strokeWidth=2, strokeDash=[4, 2]).encode(
                x=alt.X("date:T"), y=alt.Y("greed_norm:Q", scale=alt.Scale(domain=[0, 1])),
                tooltip=["date_str:N", "price:Q", "greed_norm:Q"],
            )
            st.altair_chart((line_p + line_g).properties(height=260), use_container_width=True)
        else:
            st.info("Not enough overlapping dates for this stock.")
    else:
        st.info("No price data for this stock.")

    # ----- Price, greed & fundamentals (this stock only) -----
    st.subheader("Price, greed & fundamentals")
    st.caption(f"**{name}**: price, greed, and income (quarterly, forward-filled). All normalized to 0–1.")
    fund_df = load_fundamentals(code)
    if fund_df is None or fund_df.empty:
        st.info("No fundamentals file for this stock. Add data to see this chart.")
    elif price_df is not None and not s.empty:
        greed_series = panel.loc[panel["company_code"] == code, ["dt", "greed_ratio"]].copy()
        greed_series = greed_series.rename(columns={"dt": "date"})
        greed_series["greed_ratio"] = pd.to_numeric(greed_series["greed_ratio"], errors="coerce")
        full = price_df.merge(greed_series, on="date", how="outer").sort_values("date")
        full = full.merge(fund_df[["date", "income"]], on="date", how="left")
        full["income"] = full["income"].ffill()
        full = full.dropna(subset=["date", "price", "greed_ratio", "income"]).reset_index(drop=True)
        if full.empty:
            st.info("No overlapping dates with price, greed, and income.")
        else:
            roll = st.selectbox("Rolling window for price & greed", [1, 7, 30, 90], index=1, format_func=lambda x: {1: "Daily", 7: "7 days", 30: "30 days", 90: "90 days"}[x], key="triple_roll")
            full["price_r"] = full["price"].rolling(roll, min_periods=1).mean()
            full["greed_r"] = full["greed_ratio"].rolling(roll, min_periods=1).mean()
            full = full[full["date"] >= (full["date"].max() - pd.Timedelta(days=CHART_DAYS))]
            pmin, pmax = full["price_r"].min(), full["price_r"].max()
            gmin, gmax = full["greed_r"].min(), full["greed_r"].max()
            imin, imax = full["income"].min(), full["income"].max()
            full["pn"] = (full["price_r"] - pmin) / ((pmax - pmin) or 1)
            full["gn"] = (full["greed_r"] - gmin) / ((gmax - gmin) or 1)
            full["in"] = (full["income"] - imin) / ((imax - imin) or 1)
            full["date_str"] = full["date"].dt.strftime("%Y-%m-%d")
            lp = alt.Chart(full).mark_line(stroke="steelblue", strokeWidth=2).encode(x=alt.X("date:T"), y=alt.Y("pn:Q", scale=alt.Scale(domain=[0, 1])), tooltip=["date_str:N", "price_r:Q", "greed_r:Q", "income:Q"])
            lg = alt.Chart(full).mark_line(stroke="green", strokeWidth=2, strokeDash=[4, 2]).encode(x=alt.X("date:T"), y=alt.Y("gn:Q", scale=alt.Scale(domain=[0, 1])), tooltip=["date_str:N", "price_r:Q", "greed_r:Q", "income:Q"])
            li = alt.Chart(full).mark_line(stroke="orange", strokeWidth=2, strokeDash=[2, 2]).encode(x=alt.X("date:T"), y=alt.Y("in:Q", scale=alt.Scale(domain=[0, 1])), tooltip=["date_str:N", "price_r:Q", "greed_r:Q", "income:Q"])
            st.altair_chart((lp + lg + li).properties(height=260, title="Price (blue) · Greed (green) · Income (orange)"), use_container_width=True)
    else:
        st.info("Need price and greed data for this stock first.")

    st.markdown("---")
    st.caption("Greed from Naver Finance discussion titles (AI-classified). This page shows data for the selected stock only.")


if __name__ == "__main__":
    main()

