import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from io import StringIO
import plotly.graph_objects as go

st.set_page_config(page_title="S&P 500 Stock Recommender", layout="wide")

@st.cache_data(ttl=86400)  # 24 jam
def load_universe():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
    df = pd.read_html(StringIO(html))[0]
    df["Symbol"] = df["Symbol"].str.replace(".", "-", regex=False)
    return df[["Symbol", "Security", "GICS Sector"]].rename(
        columns={"Symbol": "Ticker", "Security": "Nama", "GICS Sector": "Sektor"})

@st.cache_data(ttl=900, show_spinner="Mengunduh harga ~500 saham...")  # 15 menit
def load_prices(tickers, period="1y"):
    df = yf.download(list(tickers) + ["SPY"], period=period, auto_adjust=True,
                     progress=False, threads=True)["Close"]
    return df.dropna(how="all")

@st.cache_data(ttl=900)
def load_ohlc(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def rsi(p, n=14):
    d = p.diff()
    gain = d.clip(lower=0).rolling(n).mean()
    loss = (-d.clip(upper=0)).rolling(n).mean()
    return 100 - 100 / (1 + gain / loss)

def hitung_skor(prices, w_tech, w_mom):
    spy = prices["SPY"].dropna()
    spy_ret = spy.iloc[-1] / spy.iloc[-126] - 1

    p = prices.drop(columns="SPY")
    p = p.loc[:, p.count() >= 200]          # buang saham dengan data kurang

    last = p.ffill().iloc[-1]
    ma50 = p.rolling(50).mean().iloc[-1]
    ma200 = p.rolling(200).mean().iloc[-1]
    r = rsi(p).iloc[-1]
    ret6m = p.ffill().iloc[-1] / p.ffill().iloc[-126] - 1

    out = pd.DataFrame({
        "Harga": last, "RSI": r, "Return 6M": ret6m, "vs SPY": ret6m - spy_ret,
        "Teknikal": 40*(last > ma50) + 40*(ma50 > ma200) + 20*((r > 30) & (r < 70)),
        "Momentum": ret6m.rank(pct=True) * 100,
    }).dropna()
    out["Skor"] = (w_tech*out["Teknikal"] + w_mom*out["Momentum"]) / (w_tech + w_mom)
    out["Sinyal"] = pd.cut(out["Skor"], [-1, 40, 70, 101], labels=["Jual", "Tahan", "Beli"])
    out.index.name = "Ticker"
    return out.reset_index()

# ---------- Sidebar ----------
uni = load_universe()
st.sidebar.header("Filter")
sektor = st.sidebar.multiselect("Sektor", sorted(uni["Sektor"].unique()),
                                default=sorted(uni["Sektor"].unique()))
w_tech = st.sidebar.slider("Bobot Teknikal", 0, 100, 50)
w_mom = st.sidebar.slider("Bobot Momentum", 0, 100, 50)
min_skor = st.sidebar.slider("Skor minimum", 0, 100, 0)
sinyal = st.sidebar.multiselect("Sinyal", ["Beli", "Tahan", "Jual"],
                                default=["Beli", "Tahan", "Jual"])
top_n = st.sidebar.number_input("Tampilkan top N", 10, 500, 50, step=10)
periode = st.sidebar.select_slider("Periode grafik", ["3mo", "6mo", "1y", "2y"], value="1y")
if st.sidebar.button("Refresh data"):
    st.cache_data.clear()
    st.rerun()

# ---------- Halaman utama ----------
st.title("Dashboard Rekomendasi Saham S&P 500")
st.caption("Benchmark: SPY | Data yfinance, tertunda ±15 menit")

if (w_tech + w_mom) == 0:
    st.warning("Atur minimal satu bobot > 0."); st.stop()

prices = load_prices(tuple(uni["Ticker"]))          # tuple supaya bisa di-cache
hasil = hitung_skor(prices, w_tech, w_mom).merge(uni, on="Ticker")
hasil = hasil[hasil["Sektor"].isin(sektor) & hasil["Sinyal"].isin(sinyal)
              & (hasil["Skor"] >= min_skor)]
hasil = hasil.sort_values("Skor", ascending=False).head(int(top_n)).reset_index(drop=True)

c1, c2, c3 = st.columns(3)
c1.metric("Saham lolos filter", len(hasil))
c2.metric("Sinyal Beli", int((hasil["Sinyal"] == "Beli").sum()))
c3.metric("Rata-rata skor", f"{hasil['Skor'].mean():.0f}" if len(hasil) else "-")

tab1, tab2 = st.tabs(["Ranking", "Detail Saham"])

with tab1:
    st.dataframe(
        hasil[["Ticker", "Nama", "Sektor", "Harga", "RSI", "Return 6M", "vs SPY",
               "Teknikal", "Momentum", "Skor", "Sinyal"]].style.format({
            "Harga": "${:.2f}", "RSI": "{:.0f}", "Return 6M": "{:.1%}",
            "vs SPY": "{:+.1%}", "Teknikal": "{:.0f}", "Momentum": "{:.0f}",
            "Skor": "{:.0f}"}),
        use_container_width=True, hide_index=True)

with tab2:
    if hasil.empty:
        st.info("Tidak ada saham yang lolos filter.")
    else:
        t = st.selectbox("Pilih saham", hasil["Ticker"])
        d = load_ohlc(t, periode)
        fig = go.Figure(go.Candlestick(x=d.index, open=d["Open"], high=d["High"],
                                       low=d["Low"], close=d["Close"], name=t))
        fig.add_trace(go.Scatter(x=d.index, y=d["Close"].rolling(50).mean(), name="MA50"))
        fig.add_trace(go.Scatter(x=d.index, y=d["Close"].rolling(200).mean(), name="MA200"))
        fig.update_layout(xaxis_rangeslider_visible=False, height=550)
        st.plotly_chart(fig, use_container_width=True)

st.caption("⚠️ Alat bantu analisis, bukan nasihat investasi.")
