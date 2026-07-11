import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ---------- palette ----------
BLUE, BLUE_D = "#1f6feb", "#0b4bb3"
ORANGE, ORANGE_D = "#f5871f", "#c9660a"
INK, MUTED, GRID = "#1c2530", "#5b6675", "#e6ebf2"

TARGET = "Target_Growth_Rate"
LEAK_COLS = ["Company_ID", "Date", "Target_Anomaly_Class", "Fraud_Flag",
             "Market_Shock_Flag", "Policy_Change_Flag", "Audit_Flag"]
MODEL_PATH, FEATURES_PATH = "growth_model.pkl", "model_features.pkl"

st.set_page_config(page_title="Growth Predictor | ML", page_icon="📈",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown(f"""
<style>
.stApp {{ background:#f7f9fc; }}
#MainMenu, footer {{ visibility:hidden; }}
h1,h2,h3,h4 {{ font-family:'Segoe UI',system-ui,sans-serif; color:{INK}; letter-spacing:-.3px; }}
.block-container {{ padding-top:1.4rem; max-width:1240px; }}
.hdr {{ background:linear-gradient(100deg,{BLUE},{BLUE_D}); color:#fff;
        border-radius:16px; padding:24px 30px; margin-bottom:18px; }}
.hdr h1 {{ color:#fff; margin:0; font-size:1.85rem; }}
.hdr p {{ color:#dce8ff; margin:.35rem 0 0; }}
.hdr .tag {{ display:inline-block; background:{ORANGE}; color:#fff; font-weight:700;
             font-size:.72rem; letter-spacing:.08em; padding:.2rem .6rem; border-radius:6px; }}
.stMetric {{ background:#fff; border:1px solid {GRID}; border-radius:14px; padding:16px 18px;
             box-shadow:0 1px 3px rgba(20,40,80,.05); }}
[data-testid="stMetricValue"] {{ color:{BLUE_D}; font-weight:700; }}
[data-testid="stFileUploader"] {{ background:#fff; border:1.5px dashed {BLUE}; border-radius:14px; padding:1rem; }}
.stTabs [data-baseweb="tab"] {{ font-weight:600; color:{MUTED}; }}
.stTabs [aria-selected="true"] {{ color:{BLUE_D}; }}
div.stButton > button {{ background:{ORANGE}; color:#fff; border:none; border-radius:10px;
                         font-weight:600; padding:.5rem 1.4rem; }}
div.stButton > button:hover {{ background:{ORANGE_D}; color:#fff; }}
.note {{ background:#fff; border:1px solid {GRID}; border-radius:12px; padding:14px 18px; color:{MUTED}; }}
</style>
""", unsafe_allow_html=True)


def make_features(df):
    drop = [c for c in ([TARGET] + LEAK_COLS) if c in df.columns]
    X = df.drop(columns=drop)
    return X.select_dtypes(include=[np.number])


def train_model(df):
    X, y = make_features(df), df[TARGET]
    cols = X.columns.tolist()
    split = int(len(X) * 0.8)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(n_estimators=600, learning_rate=0.05, max_depth=5,
                               subsample=0.8, colsample_bytree=0.8,
                               random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X.iloc[:split], y.iloc[:split])
    pred = pipe.predict(X.iloc[split:])
    yte = y.iloc[split:]
    metrics = {
        "mae": mean_absolute_error(yte, pred),
        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
        "r2": r2_score(yte, pred),
        "n": len(X),
    }
    joblib.dump(pipe, MODEL_PATH)
    joblib.dump(cols, FEATURES_PATH)
    importances = pd.Series(
        pipe.named_steps["model"].feature_importances_, index=cols
    ).sort_values(ascending=False)
    return metrics, importances, yte.values, pred


def load_model():
    if os.path.exists(MODEL_PATH) and os.path.exists(FEATURES_PATH):
        return joblib.load(MODEL_PATH), joblib.load(FEATURES_PATH)
    return None, None


# ---------- header ----------
st.markdown(f"""
<div class="hdr">
  <span class="tag">XGBOOST REGRESSION</span>
  <h1>📈 Company Growth Predictor</h1>
  <p>Train on your featured dataset, then upload any company file to forecast its growth rate.</p>
</div>
""", unsafe_allow_html=True)

tab_train, tab_pred = st.tabs(["🎓  Train model", "🔮  Predict"])

# ================= TRAIN =================
with tab_train:
    st.subheader("Train on featured dataset")
    st.markdown(f"<div class='note'>Upload a CSV containing the <b>{TARGET}</b> column. "
                "Leaky/ID columns are dropped automatically.</div>", unsafe_allow_html=True)
    st.write("")
    up = st.file_uploader("Training CSV", type="csv", key="train")
    if up is not None:
        df = pd.read_csv(up)
        if TARGET not in df.columns:
            st.error(f"This file has no '{TARGET}' column. Use the featured dataset for training.")
        else:
            with st.expander("Preview"):
                st.dataframe(df.head(8), use_container_width=True)
            if st.button("Train model", type="primary"):
                with st.spinner("Training XGBoost..."):
                    m, imp, ytrue, ypred = train_model(df)
                st.success("Model trained and saved. Switch to the Predict tab to use it.")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("R2 score", f"{m['r2']:.3f}")
                c2.metric("MAE", f"{m['mae']:.4f}")
                c3.metric("RMSE", f"{m['rmse']:.4f}")
                c4.metric("Rows", f"{m['n']}")

                left, right = st.columns(2)
                with left:
                    top = imp.head(12)[::-1]
                    fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                           marker_color=BLUE))
                    fig.update_layout(title="Top features driving growth", height=420,
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      margin=dict(t=50,l=10,r=10,b=30))
                    fig.update_xaxes(gridcolor=GRID)
                    st.plotly_chart(fig, use_container_width=True)
                with right:
                    fig = go.Figure(go.Scatter(x=ytrue, y=ypred, mode="markers",
                                               marker=dict(color=ORANGE, size=6, opacity=.5)))
                    lo, hi = float(min(ytrue.min(), ypred.min())), float(max(ytrue.max(), ypred.max()))
                    fig.add_trace(go.Scatter(x=[lo,hi], y=[lo,hi], mode="lines",
                                             line=dict(color=BLUE, dash="dash"), name="perfect"))
                    fig.update_layout(title="Predicted vs actual (test set)", height=420,
                                      xaxis_title="Actual", yaxis_title="Predicted",
                                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                                      showlegend=False, margin=dict(t=50,l=40,r=10,b=40))
                    fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
                    st.plotly_chart(fig, use_container_width=True)

# ================= PREDICT =================
with tab_pred:
    st.subheader("Predict growth for a new file")
    pipe, cols = load_model()
    if pipe is None:
        st.warning("No trained model yet. Train one in the Train tab first.")
    else:
        st.markdown("<div class='note'>Upload a CSV with the same feature columns. "
                    "The target column is not required.</div>", unsafe_allow_html=True)
        st.write("")
        up2 = st.file_uploader("Data to score", type="csv", key="pred")
        if up2 is not None:
            df = pd.read_csv(up2)
            X = make_features(df)
            for c in cols:
                if c not in X.columns:
                    X[c] = 0
            X = X[cols]
            with st.spinner("Scoring..."):
                preds = pipe.predict(X)
            out = df.copy()
            out["Predicted_Growth_Rate"] = preds

            c1, c2, c3 = st.columns(3)
            c1.metric("Rows scored", f"{len(out)}")
            c2.metric("Mean predicted growth", f"{preds.mean():.3f}")
            c3.metric("Max predicted growth", f"{preds.max():.3f}")

            fig = go.Figure(go.Histogram(x=preds, marker_color=ORANGE, nbinsx=30))
            fig.update_layout(title="Distribution of predicted growth", height=340,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              margin=dict(t=50,l=40,r=10,b=30))
            fig.update_xaxes(gridcolor=GRID); fig.update_yaxes(gridcolor=GRID)
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(out.head(20), use_container_width=True)
            st.download_button("Download predictions CSV",
                               out.to_csv(index=False).encode(),
                               "predictions.csv", "text/csv")

st.caption("Educational tool. Model persists to disk (growth_model.pkl) between runs.")
