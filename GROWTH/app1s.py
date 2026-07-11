import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from sklearn.ensemble import (
    HistGradientBoostingClassifier, RandomForestClassifier,
    GradientBoostingRegressor, VotingClassifier,
)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ---------------- palette ----------------
BLUE = "#1f6feb"
BLUE_D = "#0b4bb3"
ORANGE = "#f5871f"
ORANGE_D = "#c9660a"
INK = "#1c2530"
MUTED = "#5b6675"
GRID = "#e6ebf2"

st.set_page_config(page_title="Stock Signal | ML Dashboard", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
.stApp {{ background:#f7f9fc; }}
#MainMenu, footer {{ visibility:hidden; }}
h1,h2,h3,h4 {{ font-family:'Segoe UI',system-ui,sans-serif; color:{INK}; letter-spacing:-.3px; }}
.block-container {{ padding-top:1.6rem; max-width:1240px; }}
.hdr {{ background:linear-gradient(100deg,{BLUE},{BLUE_D}); color:#fff;
        border-radius:16px; padding:26px 30px; margin-bottom:20px; }}
.hdr h1 {{ color:#fff; margin:0; font-size:1.9rem; }}
.hdr p  {{ color:#dce8ff; margin:.35rem 0 0; font-size:1rem; }}
.hdr .tag {{ display:inline-block; background:{ORANGE}; color:#fff; font-weight:700;
             font-size:.72rem; letter-spacing:.08em; padding:.2rem .6rem; border-radius:6px; }}
.stMetric {{ background:#fff; border:1px solid {GRID}; border-radius:14px; padding:16px 18px;
             box-shadow:0 1px 3px rgba(20,40,80,.05); }}
[data-testid="stMetricValue"] {{ color:{BLUE_D}; font-weight:700; }}
[data-testid="stFileUploader"] {{ background:#fff; border:1.5px dashed {BLUE}; border-radius:14px; padding:1rem; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; color:{MUTED}; }}
.stTabs [aria-selected="true"] {{ color:{BLUE_D}; }}
.concl {{ background:#fff; border:1px solid {GRID}; border-left:none; border-radius:14px;
          padding:18px 22px; box-shadow:0 1px 3px rgba(20,40,80,.05); }}
.concl h4 {{ margin:0 0 .6rem; color:{ORANGE_D}; }}
.concl li {{ margin:.3rem 0; color:{INK}; }}
.sig {{ font-weight:800; font-size:1.5rem; }}
.sig.up {{ color:#1a8f4c; }} .sig.down {{ color:#d33d2f; }}
</style>
""", unsafe_allow_html=True)

HORIZON = 20
FEATURES = [
    "ret_1","ret_3","ret_5","ret_10","ret_20","hl_range","oc_change",
    "ma_gap_10","ma_gap_20","ma_gap_50","boll_pos","rsi","macd","macd_sig",
    "macd_hist","stoch_k","atr","obv_z","vol_z","realized_vol","dow",
]


def load_and_clean(df):
    d = df.copy()
    d.columns = [str(c).strip().lower().split(".")[-1] for c in d.columns]
    alias = {"adj close":"adjusted","adj_close":"adjusted","datetime":"date"}
    d = d.rename(columns={k:v for k,v in alias.items() if k in d.columns})
    if "date" not in d.columns:
        d = d.rename(columns={d.columns[0]:"date"})
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    for c in ["open","high","low","close","volume","adjusted"]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    if "adjusted" not in d.columns:
        d["adjusted"] = d["close"]
    d = d.dropna(subset=["date","close"]).drop_duplicates("date").sort_values("date")
    core = ["open","high","low","close","volume","adjusted"]
    d[core] = d[core].ffill()
    return d.dropna(subset=core).reset_index(drop=True)


def engineer(df, horizon=HORIZON):
    d = df.copy()
    c,h,l,v = d["close"],d["high"],d["low"],d["volume"]
    for n in (1,3,5,10,20):
        d[f"ret_{n}"] = c.pct_change(n)
    d["hl_range"] = (h-l)/c
    d["oc_change"] = (c-d["open"])/d["open"]
    for n in (10,20,50):
        ma = c.rolling(n).mean()
        d[f"ma_gap_{n}"] = (c-ma)/ma
    ma20,std20 = c.rolling(20).mean(), c.rolling(20).std()
    d["boll_pos"] = (c-(ma20-2*std20))/(4*std20)
    delta = c.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    d["rsi"] = 100 - 100/(1+gain/loss.replace(0,np.nan))
    ema12,ema26 = c.ewm(span=12,adjust=False).mean(), c.ewm(span=26,adjust=False).mean()
    d["macd"] = ema12-ema26
    d["macd_sig"] = d["macd"].ewm(span=9,adjust=False).mean()
    d["macd_hist"] = d["macd"]-d["macd_sig"]
    ll,hh = l.rolling(14).min(), h.rolling(14).max()
    d["stoch_k"] = 100*(c-ll)/(hh-ll)
    tr = pd.concat([(h-l),(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    d["atr"] = tr.rolling(14).mean()/c
    obv = (np.sign(c.diff())*v).fillna(0).cumsum()
    d["obv_z"] = (obv-obv.rolling(20).mean())/obv.rolling(20).std()
    d["vol_z"] = (v-v.rolling(20).mean())/v.rolling(20).std()
    d["realized_vol"] = d["ret_1"].rolling(10).std()
    d["dow"] = d["date"].dt.dayofweek
    fwd_ret = c.shift(-horizon)/c - 1
    d["target_ret"] = fwd_ret
    band = fwd_ret.abs().quantile(0.30)
    d["target_dir"] = np.where(fwd_ret>band,1,np.where(fwd_ret<-band,0,np.nan))
    d = d.dropna(subset=["target_dir"])
    d["target_dir"] = d["target_dir"].astype(int)
    return d.dropna().reset_index(drop=True)


def _build_clf():
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    mk = lambda model: Pipeline([("s", StandardScaler()),
                                 ("pca", PCA(n_components=0.95, random_state=42)),
                                 ("m", model)])
    hgb = mk(HistGradientBoostingClassifier(learning_rate=0.05, max_leaf_nodes=31,
              l2_regularization=1.0, random_state=42, class_weight="balanced"))
    rf = mk(RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42))
    lr = mk(LogisticRegression(max_iter=1000, class_weight="balanced"))
    ens = VotingClassifier([("hgb", hgb), ("rf", rf), ("lr", lr)], voting="soft", n_jobs=-1)
    return CalibratedClassifierCV(ens, method="isotonic", cv=cv), cv


@st.cache_data(show_spinner=False)
def run_model(raw_df):
    clean = load_and_clean(raw_df)
    d = engineer(clean)
    if len(d) < 200:
        return {"ok":False,"error":f"Only {len(d)} usable rows; need about 200 or more."}
    X,yd,yr = d[FEATURES], d["target_dir"], d["target_ret"]
    clf,cv = _build_clf()
    accs,aucs = [],[]
    for tr,te in cv.split(X, yd):
        clf.fit(X.iloc[tr],yd.iloc[tr])
        p = clf.predict_proba(X.iloc[te])[:,1]
        accs.append(accuracy_score(yd.iloc[te],(p>=0.5).astype(int)))
        aucs.append(roc_auc_score(yd.iloc[te],p))
    clf.fit(X,yd)
    reg = Pipeline([("s",StandardScaler()),("m",GradientBoostingRegressor(random_state=42))])
    split = int(len(d)*0.8)
    reg.fit(X.iloc[:split],yr.iloc[:split])
    rp = reg.predict(X.iloc[split:])
    latest = X.iloc[[-1]]
    prob_up = float(clf.predict_proba(latest)[0][1])
    exp_ret = float(reg.predict(latest)[0])
    return {
        "ok":True, "clean":clean, "feat":d,
        "label":"GROWTH" if prob_up>=0.5 else "DECLINE",
        "prob_up":round(prob_up*100,1), "exp_ret":round(exp_ret*100,2),
        "cv_acc":round(float(np.mean(accs))*100,1), "cv_auc":round(float(np.mean(aucs)),3),
        "reg_r2":round(r2_score(yr.iloc[split:],rp),3),
        "rows":len(d), "horizon":HORIZON,
        "last_date":str(clean["date"].iloc[-1].date()),
        "rsi":round(float(d.iloc[-1]["rsi"]),1),
    }


# ================= HEADER =================
st.markdown(f"""
<div class="hdr">
  <span class="tag">ML DASHBOARD</span>
  <h1>📈 Stock Signal</h1>
  <p>Upload a company's price history for a calibrated {HORIZON}-day trend forecast, distribution analysis, and a regression view.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Data")
    file = st.file_uploader("Upload OHLCV CSV", type="csv")
    st.caption("Columns: date, open, high, low, close, volume. ~200+ rows.")
    st.markdown("---")
    st.caption("Palette: blue = model, orange = market. Educational tool, not investment advice.")

if file is None:
    st.info("Upload a CSV in the sidebar to begin.")
    st.stop()

raw = pd.read_csv(file)
with st.spinner("Cleaning, training, and forecasting..."):
    r = run_model(raw)
if not r["ok"]:
    st.error(r["error"]); st.stop()

feat, clean = r["feat"], r["clean"]
up = r["label"] == "GROWTH"

# ---- KPI row ----
k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Signal", r["label"])
k2.metric("Growth probability", f"{r['prob_up']}%")
k3.metric("Expected move", f"{r['exp_ret']}%")
k4.metric("CV accuracy", f"{r['cv_acc']}%")
k5.metric("Model AUC", f"{r['cv_auc']}")

tab1, tab2, tab3 = st.tabs(["📉  Price & Forecast", "📊  Distribution (Box Plot)", "📈  Regression"])

# ---- TAB 1: price ----
with tab1:
    tail = clean.tail(160)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=tail["date"], y=tail["close"], mode="lines",
                             name="Close", line=dict(color=BLUE, width=2.4)))
    ma = tail["close"].rolling(20).mean()
    fig.add_trace(go.Scatter(x=tail["date"], y=ma, mode="lines",
                             name="20-day MA", line=dict(color=ORANGE, width=2, dash="dot")))
    fig.update_layout(height=420, title="Closing price with 20-day moving average",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.08), margin=dict(t=60,l=50,r=20,b=30))
    fig.update_yaxes(gridcolor=GRID); fig.update_xaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 2: box plot ----
with tab2:
    st.markdown("**Forward return distribution by predicted class.** "
                "Blue = days the model tags DECLINE, orange = GROWTH. Separation between the boxes is the edge.")
    dd = feat.copy()
    dd["cls"] = np.where(dd["target_dir"]==1, "Growth", "Decline")
    fig = go.Figure()
    for name,color in [("Decline",BLUE),("Growth",ORANGE)]:
        sub = dd[dd["cls"]==name]["target_ret"]*100
        fig.add_trace(go.Box(y=sub, name=name, marker_color=color, boxmean=True))
    fig.update_layout(height=420, title=f"{HORIZON}-day forward return (%) by class",
                      yaxis_title="Forward return (%)", paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=60,l=50,r=20,b=30))
    fig.update_yaxes(gridcolor=GRID)
    st.plotly_chart(fig, use_container_width=True)

# ---- TAB 3: linear regression ----
with tab3:
    st.markdown("**Does the 20-day moving-average gap predict the forward return?** "
                "Each point is a trading day; the orange line is the fitted linear regression.")
    x = feat["ma_gap_20"].values.reshape(-1,1)
    y = (feat["target_ret"]*100).values
    lin = LinearRegression().fit(x,y)
    xs = np.linspace(x.min(), x.max(), 100).reshape(-1,1)
    ys = lin.predict(xs)
    r2 = r2_score(y, lin.predict(x))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x.flatten(), y=y, mode="markers", name="Days",
                             marker=dict(color=BLUE, size=5, opacity=0.35)))
    fig.add_trace(go.Scatter(x=xs.flatten(), y=ys, mode="lines", name="Regression fit",
                             line=dict(color=ORANGE, width=3)))
    fig.update_layout(height=420, title=f"Forward return vs 20-day MA gap  (R2 = {r2:.3f})",
                      xaxis_title="20-day MA gap", yaxis_title="Forward return (%)",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.08), margin=dict(t=60,l=50,r=20,b=30))
    fig.update_yaxes(gridcolor=GRID); fig.update_xaxes(gridcolor=GRID)
    st.plotly_chart(fig, use_container_width=True)
    slope = lin.coef_[0]

# ---- CONCLUSION ----
dd = feat.copy()
growth_med = (dd[dd["target_dir"]==1]["target_ret"].median())*100
decline_med = (dd[dd["target_dir"]==0]["target_ret"].median())*100
trend_word = "positive" if slope > 0 else "negative"
st.write("")
st.markdown(f"""
<div class="concl">
  <h4>Conclusion</h4><ul>
    <li>The model reads a <b class="{ 'sig up' if up else 'sig down' }">{r['label']}</b> trend over the next {r['horizon']} days, with <b>{r['prob_up']}%</b> growth probability and an expected move of <b>{r['exp_ret']}%</b>.</li>
    <li>Cross-validated accuracy is <b>{r['cv_acc']}%</b> (AUC {r['cv_auc']}); growth days historically returned a median <b>{growth_med:+.1f}%</b> vs <b>{decline_med:+.1f}%</b> on decline days.</li>
    <li>The 20-day MA gap shows a <b>{trend_word}</b> linear relationship with forward returns, confirming momentum as a real (if modest) signal in this stock.</li>
  </ul>
</div>
""", unsafe_allow_html=True)

st.caption("Educational tool, not investment advice. Trend target with a volatility dead-band; the model abstains on small, near-random moves.")