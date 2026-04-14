"""
LoanNet — Loan Approval Prediction Using Neural Networks
Python 3.12 | TensorFlow 2.17+ | Streamlit
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time, os, io

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="LoanNet — Loan Approval Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono&display=swap');
    .stApp { background: #ffffff; font-family: 'DM Sans', sans-serif; }
    section[data-testid="stSidebar"] { background: #f8fafc; border-right: 1px solid #e2e8f0; }
    .block-container { padding-top: 1.5rem; }
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #f0f7ff 0%, #f8fafc 100%);
        border: 1px solid #bfdbfe;
        border-radius: 12px;
        padding: 14px 18px;
    }
    div[data-testid="stMetricValue"] { color: #1d4ed8; font-weight: 600; }
    div[data-testid="stMetricLabel"] { color: #374151; }
    div[data-testid="stMetricDelta"] { color: #047857; }
    .stTabs [data-baseweb="tab"] { color: #6b7280; font-family: 'DM Sans'; }
    .stTabs [aria-selected="true"] { color: #1d4ed8 !important; }
    .stTabs [data-baseweb="tab-list"] { background: #f8fafc; border-bottom: 1px solid #e2e8f0; }
    h1,h2,h3 { color: #111827 !important; font-family: 'DM Sans' !important; }
    p, label, .stCaption { color: #374151 !important; }
    .stDataFrame { border: 1px solid #e2e8f0; border-radius: 8px; }
    .approved-box {
        background: linear-gradient(135deg, #ecfdf5, #d1fae5);
        border: 2px solid #10b981; border-radius: 16px;
        padding: 32px; text-align: center;
    }
    .rejected-box {
        background: linear-gradient(135deg, #fff1f2, #ffe4e6);
        border: 2px solid #ef4444; border-radius: 16px;
        padding: 32px; text-align: center;
    }
    .review-box {
        background: linear-gradient(135deg, #fffbeb, #fef3c7);
        border: 2px solid #f59e0b; border-radius: 16px;
        padding: 32px; text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────
for k in ["df","model","scaler","encoder","trained","features","history"]:
    if k not in st.session_state:
        st.session_state[k] = None
if st.session_state.trained is None:
    st.session_state.trained = False

# ── Plotly dark theme ─────────────────────────────────────────
PLOT = dict(
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f8fafc",
    font_color="#374151",
    margin=dict(t=30, b=10, l=10, r=10),
)
GRID = dict(gridcolor="rgba(0,0,0,0.07)")

# ── Colour palette ────────────────────────────────────────────
C = dict(
    blue="#60a5fa", purple="#a78bfa", green="#10b981",
    amber="#f59e0b", red="#ef4444",  teal="#14b8a6",
    text="#111827",  muted="#6b7280", bg2="#f8fafc",
)

# ══════════════════════════════════════════════════════════════
#  DATA GENERATION
# ══════════════════════════════════════════════════════════════
@st.cache_data
def generate_dataset(n=5000, seed=42):
    """Generate a realistic synthetic loan dataset."""
    np.random.seed(seed)

    age               = np.random.randint(21, 70, n)
    income            = np.random.lognormal(10.8, 0.55, n).astype(int).clip(15000, 500000)
    loan_amount       = np.random.lognormal(11.5, 0.7, n).astype(int).clip(5000, 1000000)
    loan_term         = np.random.choice([12, 24, 36, 48, 60, 84, 120, 180, 240, 360], n)
    credit_score      = np.random.normal(680, 75, n).astype(int).clip(300, 850)
    employment_years  = np.random.exponential(6, n).clip(0, 40).round(1)
    debt_to_income    = np.random.beta(2, 5, n).round(3)
    num_credit_lines  = np.random.randint(0, 20, n)
    num_delinquencies = np.random.choice([0,0,0,0,1,1,2,3,5], n)
    property_value    = np.random.lognormal(12.5, 0.6, n).astype(int).clip(50000, 2000000)
    education         = np.random.choice(["High School","Bachelor","Master","PhD","Associate"], n,
                                          p=[0.25,0.40,0.20,0.05,0.10])
    employment_type   = np.random.choice(["Salaried","Self-Employed","Business","Contract"], n,
                                          p=[0.55,0.20,0.15,0.10])
    loan_purpose      = np.random.choice(["Home","Car","Education","Personal","Business"], n,
                                          p=[0.30,0.20,0.15,0.25,0.10])

    # Derived
    loan_to_income    = (loan_amount / income).round(3)
    loan_to_value     = (loan_amount / property_value).round(3)
    monthly_payment   = (loan_amount * 0.005).round(2)
    savings_ratio     = np.random.beta(3, 7, n).round(3)

    # Approval logic (realistic scoring)
    score = (
        (credit_score - 300) / 550 * 0.35 +
        (1 - debt_to_income) * 0.20 +
        np.minimum(income / 200000, 1.0) * 0.15 +
        (1 - np.minimum(loan_to_income / 10, 1.0)) * 0.12 +
        (1 - num_delinquencies / 5).clip(0,1) * 0.10 +
        np.minimum(employment_years / 20, 1.0) * 0.08
    )
    edu_bonus = {"PhD":0.05,"Master":0.04,"Bachelor":0.02,"Associate":0.01,"High School":0.0}
    score += np.array([edu_bonus[e] for e in education])
    noise = np.random.normal(0, 0.04, n)
    prob  = 1 / (1 + np.exp(-8 * (score + noise - 0.55)))
    approved = (prob > 0.5).astype(int)

    df = pd.DataFrame({
        "age": age, "income": income, "loan_amount": loan_amount,
        "loan_term": loan_term, "credit_score": credit_score,
        "employment_years": employment_years, "debt_to_income": debt_to_income,
        "num_credit_lines": num_credit_lines, "num_delinquencies": num_delinquencies,
        "property_value": property_value, "loan_to_income": loan_to_income,
        "loan_to_value": loan_to_value, "monthly_payment": monthly_payment,
        "savings_ratio": savings_ratio,
        "education": education, "employment_type": employment_type,
        "loan_purpose": loan_purpose,
        "approved": approved
    })
    return df

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 LoanNet")
    st.caption("Neural Network Loan Approval · Python 3.11")
    st.divider()

    st.markdown("### 📂 Dataset")
    data_src = st.radio("Data source", ["Generate synthetic dataset", "Upload CSV"],
                        label_visibility="collapsed")

    if data_src == "Generate synthetic dataset":
        n_rows = st.slider("Number of records", 1000, 20000, 5000, 500)
        if st.button("⚙️ Generate Dataset", use_container_width=True):
            st.session_state.df = generate_dataset(n_rows)
            st.success(f"✅ {n_rows:,} records generated")
    else:
        uploaded = st.file_uploader("Upload CSV", type=["csv"])
        if uploaded:
            try:
                st.session_state.df = pd.read_csv(uploaded)
                st.success(f"✅ {len(st.session_state.df):,} rows loaded")
            except Exception as e:
                st.error(f"Error: {e}")

    st.divider()
    st.markdown("### 🔮 Live Applicant")
    age_s    = st.slider("Age",            21, 70, 35)
    inc_s    = st.slider("Annual Income ($)", 15000, 300000, 65000, 1000)
    loan_s   = st.slider("Loan Amount ($)",   5000, 500000, 150000, 1000)
    term_s   = st.selectbox("Loan Term (months)", [12,24,36,48,60,84,120,180,240,360], index=4)
    cred_s   = st.slider("Credit Score",  300, 850, 700)
    emp_s    = st.slider("Employment Years", 0.0, 30.0, 5.0, 0.5)
    dti_s    = st.slider("Debt-to-Income Ratio", 0.0, 0.8, 0.3, 0.01)
    del_s    = st.selectbox("Past Delinquencies", [0,1,2,3,5], index=0)
    edu_s    = st.selectbox("Education", ["High School","Associate","Bachelor","Master","PhD"], index=2)
    emp_type_s = st.selectbox("Employment Type", ["Salaried","Self-Employed","Business","Contract"])
    purpose_s  = st.selectbox("Loan Purpose", ["Home","Car","Education","Personal","Business"])

# ══════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════
st.title("🏦 LoanNet — Loan Approval Neural Network")
st.caption("Upload data or generate synthetic records → Explore → Train → Predict  |  Python 3.12 · TensorFlow 2.17+")

tabs = st.tabs(["📂 Dataset","📊 EDA","🧠 Model","⚡ Train","🔮 Predict","💻 Code"])

# ════════════════ TAB 1: DATASET ═════════════════════════════
with tabs[0]:
    if st.session_state.df is None:
        st.info("👆 Generate or upload a dataset from the sidebar to begin.")

        st.markdown("### 📋 Required CSV Schema")
        schema = pd.DataFrame({
            "Column":["age","income","loan_amount","loan_term","credit_score",
                      "employment_years","debt_to_income","num_credit_lines",
                      "num_delinquencies","property_value","education",
                      "employment_type","loan_purpose","approved"],
            "Type":["int","int","int","int","int","float","float","int","int",
                    "int","str","str","str","int (0/1)"],
            "Example":["35","65000","150000","60","720","5.0","0.32","4",
                       "0","350000","Bachelor","Salaried","Home","1"],
            "Description":["Applicant age","Annual income","Requested loan","Months","FICO score",
                            "Years at job","Debt payments / income","Open credit lines",
                            "Missed payments","Home value","Highest education",
                            "Employment category","Loan purpose","1=Approved, 0=Rejected"],
        })
        st.dataframe(schema, use_container_width=True, hide_index=True)
    else:
        df = st.session_state.df
        c1,c2,c3,c4,c5 = st.columns(5)
        c1.metric("Total Records",  f"{len(df):,}")
        c2.metric("Features",       f"{len(df.columns)-1}")
        c3.metric("Approval Rate",  f"{df['approved'].mean()*100:.1f}%")
        c4.metric("Missing Values", f"{df.isnull().sum().sum():,}")
        c5.metric("Duplicates",     f"{df.duplicated().sum():,}")

        st.markdown("### 🔍 Sample Records")
        st.dataframe(df.head(15), use_container_width=True)

        st.markdown("### 📊 Numerical Statistics")
        st.dataframe(df.select_dtypes("number").describe().round(2), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("### 🏷️ Approval Distribution")
            counts = df["approved"].value_counts()
            fig = go.Figure(go.Pie(
                labels=["Approved","Rejected"],
                values=[counts.get(1,0), counts.get(0,0)],
                marker_colors=[C["green"], C["red"]],
                hole=0.5, textinfo="percent+label",
            ))
            fig.update_layout(**PLOT, height=280)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            st.markdown("### 💳 Credit Score by Outcome")
            fig2 = go.Figure()
            for label, val, color in [("Approved",1,C["green"]),("Rejected",0,C["red"])]:
                fig2.add_trace(go.Histogram(
                    x=df[df["approved"]==val]["credit_score"],
                    name=label, marker_color=color, opacity=0.7, nbinsx=30
                ))
            fig2.update_layout(**PLOT, height=280, barmode="overlay",
                xaxis=dict(title="Credit Score",**GRID), yaxis=dict(title="Count",**GRID))
            st.plotly_chart(fig2, use_container_width=True)

        # Download
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Dataset CSV", csv_bytes,
                           "loan_dataset.csv", "text/csv", use_container_width=False)

# ════════════════ TAB 2: EDA ══════════════════════════════════
with tabs[1]:
    if st.session_state.df is None:
        st.info("Generate or upload a dataset first.")
    else:
        df = st.session_state.df
        num_cols = df.select_dtypes("number").columns.tolist()

        st.markdown("### 📈 Feature Distribution")
        col_sel = st.selectbox("Select feature:", [c for c in num_cols if c != "approved"])
        ca, cb = st.columns(2)
        with ca:
            fig = go.Figure()
            for label, val, color in [("Approved",1,C["green"]),("Rejected",0,C["red"])]:
                fig.add_trace(go.Histogram(
                    x=df[df["approved"]==val][col_sel], name=label,
                    marker_color=color, opacity=0.7, nbinsx=35
                ))
            fig.update_layout(**PLOT, height=270, barmode="overlay",
                xaxis=dict(title=col_sel,**GRID), yaxis=dict(title="Count",**GRID),
                title=dict(text=f"{col_sel} by Approval Status", font=dict(color=C["text"],size=13)))
            st.plotly_chart(fig, use_container_width=True)
        with cb:
            fig2 = px.box(df, y=col_sel, x=df["approved"].map({1:"Approved",0:"Rejected"}),
                          color=df["approved"].map({1:"Approved",0:"Rejected"}),
                          color_discrete_map={"Approved":C["green"],"Rejected":C["red"]})
            fig2.update_layout(**PLOT, height=270,
                xaxis=dict(**GRID), yaxis=dict(title=col_sel,**GRID),
                title=dict(text=f"{col_sel} Box Plot", font=dict(color=C["text"],size=13)),
                showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### 🔗 Correlation Heatmap")
        corr = df[num_cols].corr().round(2)
        fig3 = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                         text_auto=True, aspect="auto")
        fig3.update_layout(**PLOT, height=450)
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### 🏷️ Categorical Analysis")
        cat_cols = ["education","employment_type","loan_purpose"]
        cat_sel  = st.selectbox("Categorical column:", cat_cols)
        grp = df.groupby(cat_sel)["approved"].mean().reset_index()
        grp.columns = [cat_sel, "approval_rate"]
        grp["approval_rate"] = (grp["approval_rate"]*100).round(1)
        grp = grp.sort_values("approval_rate", ascending=True)
        fig4 = go.Figure(go.Bar(
            x=grp["approval_rate"], y=grp[cat_sel], orientation="h",
            marker_color=C["blue"], text=[f"{v}%" for v in grp["approval_rate"]],
            textposition="outside"
        ))
        fig4.update_layout(**PLOT, height=260,
            xaxis=dict(title="Approval Rate (%)",**GRID,range=[0,110]),
            yaxis=dict(**GRID),
            title=dict(text=f"Approval Rate by {cat_sel.title()}", font=dict(color=C["text"],size=13)))
        st.plotly_chart(fig4, use_container_width=True)

        st.markdown("### 💰 Income vs Loan Amount")
        sample = df.sample(min(1000, len(df)), random_state=42)
        fig5 = px.scatter(sample, x="income", y="loan_amount",
                          color=sample["approved"].map({1:"Approved",0:"Rejected"}),
                          color_discrete_map={"Approved":C["green"],"Rejected":C["red"]},
                          opacity=0.6, size_max=6)
        fig5.update_layout(**PLOT, height=320,
            xaxis=dict(title="Annual Income ($)",**GRID),
            yaxis=dict(title="Loan Amount ($)",**GRID),
            legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1))
        st.plotly_chart(fig5, use_container_width=True)

# ════════════════ TAB 3: MODEL ════════════════════════════════
with tabs[2]:
    st.markdown("### 🧠 Neural Network Architecture")

    col1, col2 = st.columns([1,1])
    with col1:
        st.markdown("**Layer Stack**")
        layers = [
            ("Input Layer",         "17 features (num + encoded cat)",  "#1d4ed8","#60a5fa"),
            ("Dense 256 + BN + Dropout(0.4)", "ReLU  |  L2 reg  |  4,864 params",  "#6d28d9","#a78bfa"),
            ("Dense 128 + BN + Dropout(0.3)", "ReLU  |  L2 reg  |  32,896 params", "#6d28d9","#a78bfa"),
            ("Dense 64  + BN",      "ReLU  |  8,256 params",             "#92400e","#f59e0b"),
            ("Dense 32",            "ReLU  |  2,080 params",             "#92400e","#f59e0b"),
            ("Output Dense 1",      "Sigmoid → loan approval probability","#064e3b","#10b981"),
        ]
        for name, detail, bg, border in layers:
            st.markdown(f"""
            <div style="border-left:3px solid {border};padding:10px 14px;margin-bottom:3px;
                 background:#f0f7ff;border:1px solid #bfdbfe;border-radius:0 8px 8px 0">
                <div style="color:#111827;font-weight:500;font-size:13px">{name}</div>
                <div style="color:#4b5563;font-size:11px;margin-top:2px">{detail}</div>
            </div>
            <div style="width:2px;height:8px;background:#bfdbfe;margin-left:12px"></div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("**Hyperparameters**")
        params = pd.DataFrame({
            "Parameter": ["Input features","Output","Loss","Optimiser","Learning rate",
                          "Batch size","Max epochs","Dropout layers","L2 lambda","Total params"],
            "Value":     ["17","1 (probability)","Binary Cross-Entropy","Adam","1×10⁻³",
                          "256","200","2 (0.4, 0.3)","1×10⁻⁴","~48,000"]
        })
        st.dataframe(params, use_container_width=True, hide_index=True)

        st.markdown("**Why Neural Network for Loan Approval?**")
        st.markdown("""
        <div style="background:#f0f7ff;border:1px solid #bfdbfe;border-radius:10px;padding:14px;font-size:13px;color:#374151">
        ✅ Captures non-linear interactions between credit score, income, and debt<br>
        ✅ Handles mixed data types (numerical + categorical)<br>
        ✅ Outperforms logistic regression on complex applicant profiles<br>
        ✅ Outputs calibrated probability scores for risk ranking<br>
        ✅ Scales to millions of applications with GPU acceleration
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("### 🔧 Build Code")
    st.code("""
import tensorflow as tf
from tensorflow import keras

def build_loannet(input_dim: int = 17, dropout1: float = 0.4,
                  dropout2: float = 0.3) -> keras.Model:
    reg = keras.regularizers.L2(1e-4)
    inp = keras.layers.Input(shape=(input_dim,), name='applicant_features')

    x = keras.layers.Dense(256, activation='relu',
            kernel_regularizer=reg, name='dense_1')(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout1)(x)

    x = keras.layers.Dense(128, activation='relu',
            kernel_regularizer=reg, name='dense_2')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout2)(x)

    x = keras.layers.Dense(64, activation='relu', name='dense_3')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(32, activation='relu', name='dense_4')(x)

    out = keras.layers.Dense(1, activation='sigmoid', name='approval_prob')(x)

    model = keras.Model(inputs=inp, outputs=out, name='LoanNet')
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model
    """, language="python")

# ════════════════ TAB 4: TRAIN ════════════════════════════════
with tabs[3]:
    if st.session_state.df is None:
        st.info("Generate or upload a dataset first.")
    else:
        st.markdown("### ⚙️ Training Configuration")
        c1,c2,c3 = st.columns(3)
        with c1:
            epochs   = st.slider("Max Epochs",   50, 300, 100, 10)
            batch_sz = st.select_slider("Batch Size",[32,64,128,256,512],256)
        with c2:
            dropout1 = st.slider("Dropout Layer 1", 0.0, 0.6, 0.4, 0.05)
            dropout2 = st.slider("Dropout Layer 2", 0.0, 0.6, 0.3, 0.05)
        with c3:
            lr       = st.select_slider("Learning Rate",[1e-4,5e-4,1e-3,5e-3],1e-3,
                                        format_func=lambda x: f"{x:.0e}")
            patience = st.slider("Early Stop Patience", 5, 30, 14)
            val_pct  = st.slider("Val / Test Split %", 10, 25, 15)

        st.divider()
        if st.button("🚀  Start Training", type="primary", use_container_width=True):
            try:
                import tensorflow as tf
                from tensorflow import keras
                from sklearn.preprocessing import StandardScaler, LabelEncoder
                from sklearn.metrics import (roc_auc_score, accuracy_score,
                    precision_score, recall_score, f1_score, confusion_matrix)
            except ImportError as e:
                st.error(f"Missing: {e} — run `pip install -r requirements.txt`")
                st.stop()

            df = st.session_state.df.copy()

            # ── Preprocessing ─────────────────────────────────
            cat_cols = ["education","employment_type","loan_purpose"]
            encoders = {}
            for col in cat_cols:
                if col in df.columns:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                    encoders[col] = le

            TARGET   = "approved"
            FEATURES = [c for c in df.columns if c != TARGET]
            X = df[FEATURES].values.astype(np.float32)
            y = df[TARGET].values.astype(np.float32)

            n  = len(X)
            i1 = int(n*(1 - 2*val_pct/100))
            i2 = int(n*(1 - val_pct/100))
            idx = np.random.permutation(n)
            X, y = X[idx], y[idx]

            X_tr, X_va, X_te = X[:i1], X[i1:i2], X[i2:]
            y_tr, y_va, y_te = y[:i1], y[i1:i2], y[i2:]

            sc = StandardScaler().fit(X_tr)
            X_tr_s,X_va_s,X_te_s = sc.transform(X_tr),sc.transform(X_va),sc.transform(X_te)

            # ── Build model ────────────────────────────────────
            reg = keras.regularizers.L2(1e-4)
            inp = keras.layers.Input(shape=(len(FEATURES),))
            x   = keras.layers.Dense(256,"relu",kernel_regularizer=reg)(inp)
            x   = keras.layers.BatchNormalization()(x)
            x   = keras.layers.Dropout(dropout1)(x)
            x   = keras.layers.Dense(128,"relu",kernel_regularizer=reg)(x)
            x   = keras.layers.BatchNormalization()(x)
            x   = keras.layers.Dropout(dropout2)(x)
            x   = keras.layers.Dense(64,"relu")(x)
            x   = keras.layers.BatchNormalization()(x)
            x   = keras.layers.Dense(32,"relu")(x)
            out = keras.layers.Dense(1,"sigmoid")(x)
            model = keras.Model(inp, out, name="LoanNet")
            model.compile(
                optimizer=keras.optimizers.Adam(lr),
                loss="binary_crossentropy",
                metrics=["accuracy", keras.metrics.AUC(name="auc")]
            )

            # ── Train ──────────────────────────────────────────
            st.markdown("### 📉 Live Training")
            prog     = st.progress(0,"Starting…")
            chart_ph = st.empty()
            tr_loss,va_loss,tr_acc,va_acc = [],[],[],[]

            class LiveCB(keras.callbacks.Callback):
                def on_epoch_end(self, ep, logs=None):
                    tr_loss.append(round(float(logs.get("loss",0)),4))
                    va_loss.append(round(float(logs.get("val_loss",0)),4))
                    tr_acc.append(round(float(logs.get("accuracy",0))*100,2))
                    va_acc.append(round(float(logs.get("val_accuracy",0))*100,2))
                    pct = min(int((ep+1)/self.params["epochs"]*100),100)
                    prog.progress(pct,
                        f"Epoch {ep+1}/{self.params['epochs']}  |  "
                        f"loss {tr_loss[-1]:.4f}  val_loss {va_loss[-1]:.4f}  "
                        f"acc {tr_acc[-1]:.1f}%  val_acc {va_acc[-1]:.1f}%")
                    ep_x = list(range(1,len(tr_loss)+1))
                    fig = make_subplots(rows=1,cols=2,
                        subplot_titles=["Loss (BCE)","Accuracy (%)"])
                    fig.add_trace(go.Scatter(x=ep_x,y=tr_loss,name="Train loss",
                        line=dict(color=C["blue"],width=2)),row=1,col=1)
                    fig.add_trace(go.Scatter(x=ep_x,y=va_loss,name="Val loss",
                        line=dict(color=C["purple"],width=2,dash="dash")),row=1,col=1)
                    fig.add_trace(go.Scatter(x=ep_x,y=tr_acc,name="Train acc",
                        line=dict(color=C["green"],width=2)),row=1,col=2)
                    fig.add_trace(go.Scatter(x=ep_x,y=va_acc,name="Val acc",
                        line=dict(color=C["amber"],width=2,dash="dash")),row=1,col=2)
                    fig.update_layout(**PLOT,height=260,
                        xaxis=dict(**GRID),yaxis=dict(**GRID),
                        xaxis2=dict(**GRID),yaxis2=dict(**GRID))
                    chart_ph.plotly_chart(fig,use_container_width=True)

            t0 = time.time()
            model.fit(X_tr_s,y_tr,validation_data=(X_va_s,y_va),
                      epochs=epochs,batch_size=batch_sz,verbose=0,
                      callbacks=[LiveCB(),
                          keras.callbacks.EarlyStopping(monitor="val_loss",
                              patience=patience,restore_best_weights=True),
                          keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                              factor=0.5,patience=max(3,patience//3),min_lr=1e-6)])
            elapsed = time.time()-t0
            prog.progress(100,f"✅ Done in {elapsed:.1f}s  ({len(tr_loss)} epochs)")

            # ══════════════════════════════════════════════════
            # ── EVALUATE on BOTH train + test sets ────────────
            # ══════════════════════════════════════════════════
            from sklearn.metrics import roc_curve

            # — Train set predictions —
            y_prob_tr = model.predict(X_tr_s, verbose=0).flatten()
            y_pred_tr = (y_prob_tr >= 0.5).astype(int)
            acc_tr    = accuracy_score(y_tr, y_pred_tr) * 100
            auc_tr    = roc_auc_score(y_tr, y_prob_tr)
            prec_tr   = precision_score(y_tr, y_pred_tr, zero_division=0) * 100
            rec_tr    = recall_score(y_tr, y_pred_tr, zero_division=0) * 100
            f1_tr     = f1_score(y_tr, y_pred_tr, zero_division=0) * 100
            cm_tr     = confusion_matrix(y_tr, y_pred_tr)
            tn_tr, fp_tr, fn_tr, tp_tr = cm_tr.ravel()
            spec_tr   = tn_tr / (tn_tr + fp_tr) * 100 if (tn_tr + fp_tr) > 0 else 0.0

            # — Test set predictions —
            y_prob_te = model.predict(X_te_s, verbose=0).flatten()
            y_pred_te = (y_prob_te >= 0.5).astype(int)
            acc_te    = accuracy_score(y_te, y_pred_te) * 100
            auc_te    = roc_auc_score(y_te, y_prob_te)
            prec_te   = precision_score(y_te, y_pred_te, zero_division=0) * 100
            rec_te    = recall_score(y_te, y_pred_te, zero_division=0) * 100
            f1_te     = f1_score(y_te, y_pred_te, zero_division=0) * 100
            cm_te     = confusion_matrix(y_te, y_pred_te)
            tn_te, fp_te, fn_te, tp_te = cm_te.ravel()
            spec_te   = tn_te / (tn_te + fp_te) * 100 if (tn_te + fp_te) > 0 else 0.0

            # ──────────────────────────────────────────────────
            # 📊 SECTION: TRAINING EVALUATION
            # ──────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("## 📊 Training Evaluation")
            m1,m2,m3,m4,m5,m6 = st.columns(6)
            m1.metric("Accuracy",    f"{acc_tr:.2f}%")
            m2.metric("AUC-ROC",     f"{auc_tr:.4f}")
            m3.metric("Precision",   f"{prec_tr:.2f}%")
            m4.metric("Recall",      f"{rec_tr:.2f}%")
            m5.metric("F1 Score",    f"{f1_tr:.2f}%")
            m6.metric("Specificity", f"{spec_tr:.2f}%")

            # ──────────────────────────────────────────────────
            # 📊 SECTION: TESTING EVALUATION
            # ──────────────────────────────────────────────────
            st.markdown("## 📊 Testing Evaluation")
            n1,n2,n3,n4,n5,n6 = st.columns(6)
            n1.metric("Accuracy",    f"{acc_te:.2f}%",  delta=f"{acc_te-acc_tr:.2f}%")
            n2.metric("AUC-ROC",     f"{auc_te:.4f}",   delta=f"{auc_te-auc_tr:.4f}")
            n3.metric("Precision",   f"{prec_te:.2f}%", delta=f"{prec_te-prec_tr:.2f}%")
            n4.metric("Recall",      f"{rec_te:.2f}%",  delta=f"{rec_te-rec_tr:.2f}%")
            n5.metric("F1 Score",    f"{f1_te:.2f}%",   delta=f"{f1_te-f1_tr:.2f}%")
            n6.metric("Specificity", f"{spec_te:.2f}%", delta=f"{spec_te-spec_tr:.2f}%")

            # ──────────────────────────────────────────────────
            # 📊 CONFUSION MATRICES — Training & Testing
            # ──────────────────────────────────────────────────
            st.markdown("### 📊 Confusion Matrices")
            cm_col1, cm_col2 = st.columns(2)

            # Helper: shared heatmap style
            _CM_SCALE = [[0, "#eff6ff"], [0.4, "#bfdbfe"], [0.7, "#3b82f6"], [1, "#1d4ed8"]]
            _CM_LABELS = dict(x="Predicted", y="Actual")
            _CM_AXES   = ["Rejected", "Approved"]

            with cm_col1:
                # ── Training Confusion Matrix ──────────────────
                st.markdown(
                    "<div style='text-align:center;color:#60a5fa;font-size:15px;"
                    "font-weight:600;margin-bottom:6px'>Training Confusion Matrix</div>",
                    unsafe_allow_html=True)
                annot_tr = [
                    [f"TN<br>{tn_tr:,}", f"FP<br>{fp_tr:,}"],
                    [f"FN<br>{fn_tr:,}", f"TP<br>{tp_tr:,}"],
                ]
                fig_cm_tr = go.Figure(go.Heatmap(
                    z=cm_tr,
                    x=_CM_AXES, y=_CM_AXES,
                    text=annot_tr,
                    texttemplate="%{text}",
                    colorscale=_CM_SCALE,
                    showscale=True,
                    colorbar=dict(tickfont=dict(color="#374151"), thickness=12),
                ))
                fig_cm_tr.update_layout(
                    **PLOT, height=340,
                    title=dict(text="Training Confusion Matrix",
                               font=dict(color="#111827", size=13), x=0.5),
                    xaxis=dict(title=dict(text="Predicted Label", font=dict(color="#374151")),
                               tickfont=dict(color="#374151")),
                    yaxis=dict(title=dict(text="Actual Label", font=dict(color="#374151")),
                               tickfont=dict(color="#374151")),
                    font=dict(color="#111827"),
                )
                st.plotly_chart(fig_cm_tr, use_container_width=True)
                st.markdown(
                    f"<div style='font-size:12px;color:#4b5563;text-align:center'>"
                    f"TN={tn_tr:,} · FP={fp_tr:,} · FN={fn_tr:,} · TP={tp_tr:,}</div>",
                    unsafe_allow_html=True)

            with cm_col2:
                # ── Testing Confusion Matrix ───────────────────
                st.markdown(
                    "<div style='text-align:center;color:#10b981;font-size:15px;"
                    "font-weight:600;margin-bottom:6px'>Testing Confusion Matrix</div>",
                    unsafe_allow_html=True)
                annot_te = [
                    [f"TN<br>{tn_te:,}", f"FP<br>{fp_te:,}"],
                    [f"FN<br>{fn_te:,}", f"TP<br>{tp_te:,}"],
                ]
                fig_cm_te = go.Figure(go.Heatmap(
                    z=cm_te,
                    x=_CM_AXES, y=_CM_AXES,
                    text=annot_te,
                    texttemplate="%{text}",
                    colorscale=[[0,"#0a1f12"],[0.4,"#064e3b"],[0.7,"#059669"],[1,"#10b981"]],
                    showscale=True,
                    colorbar=dict(tickfont=dict(color="#374151"), thickness=12),
                ))
                fig_cm_te.update_layout(
                    **PLOT, height=340,
                    title=dict(text="Testing Confusion Matrix",
                               font=dict(color="#111827", size=13), x=0.5),
                    xaxis=dict(title=dict(text="Predicted Label", font=dict(color="#374151")),
                               tickfont=dict(color="#374151")),
                    yaxis=dict(title=dict(text="Actual Label", font=dict(color="#374151")),
                               tickfont=dict(color="#374151")),
                    font=dict(color="#111827"),
                )
                st.plotly_chart(fig_cm_te, use_container_width=True)
                st.markdown(
                    f"<div style='font-size:12px;color:#4b5563;text-align:center'>"
                    f"TN={tn_te:,} · FP={fp_te:,} · FN={fn_te:,} · TP={tp_te:,}</div>",
                    unsafe_allow_html=True)

            # ──────────────────────────────────────────────────
            # 📈 SECTION: TRAINING vs VALIDATION PERFORMANCE
            # ──────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("## 📈 Training vs Validation Performance")

            ep_x = list(range(1, len(tr_loss)+1))

            # — Loss vs Epochs —
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=ep_x, y=tr_loss, name="Train Loss",
                mode="lines",
                line=dict(color=C["blue"], width=2.5),
                hovertemplate="Epoch %{x}<br>Train Loss: %{y:.4f}<extra></extra>",
            ))
            fig_loss.add_trace(go.Scatter(
                x=ep_x, y=va_loss, name="Validation Loss",
                mode="lines",
                line=dict(color=C["purple"], width=2.5, dash="dash"),
                hovertemplate="Epoch %{x}<br>Val Loss: %{y:.4f}<extra></extra>",
            ))
            fig_loss.update_layout(
                **PLOT, height=320,
                title=dict(text="Loss vs Epochs  (Train vs Validation)",
                           font=dict(color="#111827", size=14)),
                xaxis=dict(title="Epoch", **GRID),
                yaxis=dict(title="Binary Cross-Entropy Loss", **GRID),
                legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1, font=dict(color="#374151")),
                hovermode="x unified",
            )
            st.plotly_chart(fig_loss, use_container_width=True)

            # — Accuracy vs Epochs —
            fig_acc = go.Figure()
            fig_acc.add_trace(go.Scatter(
                x=ep_x, y=tr_acc, name="Train Accuracy",
                mode="lines",
                line=dict(color=C["green"], width=2.5),
                hovertemplate="Epoch %{x}<br>Train Acc: %{y:.2f}%<extra></extra>",
            ))
            fig_acc.add_trace(go.Scatter(
                x=ep_x, y=va_acc, name="Validation Accuracy",
                mode="lines",
                line=dict(color=C["amber"], width=2.5, dash="dash"),
                hovertemplate="Epoch %{x}<br>Val Acc: %{y:.2f}%<extra></extra>",
            ))
            fig_acc.update_layout(
                **PLOT, height=320,
                title=dict(text="Accuracy vs Epochs  (Train vs Validation)",
                           font=dict(color="#111827", size=14)),
                xaxis=dict(title="Epoch", **GRID),
                yaxis=dict(title="Accuracy (%)", **GRID),
                legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1, font=dict(color="#374151")),
                hovermode="x unified",
            )
            st.plotly_chart(fig_acc, use_container_width=True)

            # ──────────────────────────────────────────────────
            # 📊 METRICS COMPARISON — Grouped Bar Chart
            # ──────────────────────────────────────────────────
            st.markdown("---")
            st.markdown("## 📊 Train vs Test — Metrics Comparison")

            metric_names  = ["Accuracy", "Precision", "Recall", "F1 Score", "Specificity"]
            metric_train  = [acc_tr,  prec_tr, rec_tr, f1_tr, spec_tr]
            metric_test   = [acc_te,  prec_te, rec_te, f1_te, spec_te]

            fig_bar = go.Figure()
            fig_bar.add_trace(go.Bar(
                name="Train",
                x=metric_names,
                y=metric_train,
                marker_color=C["blue"],
                opacity=0.85,
                text=[f"{v:.1f}%" for v in metric_train],
                textposition="outside",
                textfont=dict(color=C["blue"], size=11),
            ))
            fig_bar.add_trace(go.Bar(
                name="Test",
                x=metric_names,
                y=metric_test,
                marker_color=C["green"],
                opacity=0.85,
                text=[f"{v:.1f}%" for v in metric_test],
                textposition="outside",
                textfont=dict(color=C["green"], size=11),
            ))
            fig_bar.update_layout(
                **PLOT, height=380,
                barmode="group",
                bargap=0.22,
                bargroupgap=0.06,
                title=dict(text="Train vs Test Metrics Comparison",
                           font=dict(color="#111827", size=14)),
                xaxis=dict(title="Metric", **GRID, tickfont=dict(color="#374151")),
                yaxis=dict(title="Score (%)", **GRID, range=[0, 115]),
                legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1, font=dict(color="#374151"),
                            orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

            # ──────────────────────────────────────────────────
            # ROC Curve + Probability Distribution (original)
            # ──────────────────────────────────────────────────
            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("#### ROC Curve — Test Set")
                fpr, tpr, _ = roc_curve(y_te, y_prob_te)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                    line=dict(color=C["blue"], width=2.5), name=f"AUC={auc_te:.3f}"))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                    line=dict(color=C["muted"], dash="dash", width=1), name="Random"))
                fig_roc.update_layout(**PLOT, height=300,
                    xaxis=dict(title="False Positive Rate", **GRID),
                    yaxis=dict(title="True Positive Rate", **GRID),
                    legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1, font=dict(color="#374151")))
                st.plotly_chart(fig_roc, use_container_width=True)

            with col_b:
                st.markdown("#### Predicted Probability Distribution")
                fig_dist = go.Figure()
                for label, val, col in [("Approved",1,C["green"]),("Rejected",0,C["red"])]:
                    mask = y_te == val
                    fig_dist.add_trace(go.Histogram(
                        x=y_prob_te[mask], name=label,
                        marker_color=col, opacity=0.7, nbinsx=40))
                fig_dist.update_layout(**PLOT, height=300, barmode="overlay",
                    xaxis=dict(title="Predicted Probability", **GRID),
                    yaxis=dict(title="Count", **GRID),
                    legend=dict(bgcolor="#ffffff", bordercolor="#e2e8f0", borderwidth=1, font=dict(color="#374151")))
                st.plotly_chart(fig_dist, use_container_width=True)

            # Save to session
            st.session_state.model    = model
            st.session_state.scaler   = sc
            st.session_state.encoder  = encoders
            st.session_state.trained  = True
            st.session_state.features = FEATURES
            st.success("✅ Model saved — go to **Predict** tab!")

        elif st.session_state.trained:
            st.info("✅ Model already trained. Adjust settings and retrain, or go to **Predict**.")

# ════════════════ TAB 5: PREDICT ══════════════════════════════
with tabs[4]:
    # Build input vector from sidebar
    def heuristic_predict(age,income,loan_amount,term,credit,emp_yrs,dti,delinq,edu,etype,purpose):
        score = (
            (credit-300)/550*0.35 +
            (1-dti)*0.20 +
            min(income/200000,1.0)*0.15 +
            (1-min(loan_amount/income/10,1.0))*0.12 +
            (1-delinq/5)*0.10 +
            min(emp_yrs/20,1.0)*0.08
        )
        edu_b = {"PhD":0.05,"Master":0.04,"Bachelor":0.02,"Associate":0.01,"High School":0.0}
        score += edu_b.get(edu,0)
        prob = 1/(1+np.exp(-8*(score-0.55)))
        return float(np.clip(prob,0,1))

    prob = heuristic_predict(age_s,inc_s,loan_s,term_s,cred_s,emp_s,dti_s,
                             del_s,edu_s,emp_type_s,purpose_s)
    src  = "💡 Heuristic estimator"

    if st.session_state.trained and st.session_state.model:
        try:
            from sklearn.preprocessing import LabelEncoder
            feats = st.session_state.features
            encs  = st.session_state.encoder or {}
            edu_enc  = encs.get("education",LabelEncoder().fit(["High School","Associate","Bachelor","Master","PhD"]))
            etype_enc= encs.get("employment_type",LabelEncoder().fit(["Salaried","Self-Employed","Business","Contract"]))
            purp_enc = encs.get("loan_purpose",LabelEncoder().fit(["Home","Car","Education","Personal","Business"]))

            def safe_enc(enc,val):
                try: return enc.transform([val])[0]
                except: return 0

            lti  = loan_s/(inc_s+1)
            ltv  = loan_s/350000
            mp   = loan_s*0.005
            sav  = 0.25
            ncl  = 4

            row_dict = {
                "age":age_s,"income":inc_s,"loan_amount":loan_s,"loan_term":term_s,
                "credit_score":cred_s,"employment_years":emp_s,"debt_to_income":dti_s,
                "num_credit_lines":ncl,"num_delinquencies":del_s,"property_value":350000,
                "loan_to_income":round(lti,3),"loan_to_value":round(ltv,3),
                "monthly_payment":mp,"savings_ratio":sav,
                "education":safe_enc(edu_enc,edu_s),
                "employment_type":safe_enc(etype_enc,emp_type_s),
                "loan_purpose":safe_enc(purp_enc,purpose_s),
            }
            x_in = np.array([[row_dict.get(f,0) for f in feats]],dtype=np.float32)
            x_sc = st.session_state.scaler.transform(x_in)
            prob = float(st.session_state.model.predict(x_sc,verbose=0)[0][0])
            src  = "🧠 Your trained LoanNet model"
        except Exception as e:
            src = f"⚠️ Heuristic (NN error: {e})"

    st.caption(src)
    pct  = int(prob*100)
    risk = 100-pct

    # Decision
    if prob >= 0.70:
        decision, box_cls, icon, advice = "APPROVED", "approved-box", "✅", \
            "Strong application profile. Recommend processing for final verification."
    elif prob >= 0.45:
        decision, box_cls, icon, advice = "MANUAL REVIEW", "review-box", "⚠️", \
            "Borderline profile. Recommend additional documentation and underwriter review."
    else:
        decision, box_cls, icon, advice = "REJECTED", "rejected-box", "❌", \
            "High risk profile. Application does not meet minimum approval thresholds."

    col_r, col_ref = st.columns([1,1])
    with col_r:
        st.subheader("Decision Result")
        st.markdown(f"""
        <div class="{box_cls}">
            <div style="font-size:48px;margin-bottom:8px">{icon}</div>
            <div style="font-size:32px;font-weight:700;color:#111827;letter-spacing:2px">{decision}</div>
            <div style="font-size:48px;font-weight:800;margin:12px 0;
                color:{'#10b981' if decision=='APPROVED' else '#ef4444' if decision=='REJECTED' else '#f59e0b'}">
                {pct}%
            </div>
            <div style="font-size:13px;color:#6b7280;margin-bottom:8px">Approval Probability</div>
            <div style="font-size:13px;color:#d1d5db;font-style:italic">{advice}</div>
        </div>
        """, unsafe_allow_html=True)

        # Gauge
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=pct,
            number={"suffix":"%","font":{"size":32,
                "color":"#10b981" if decision=="APPROVED" else "#ef4444" if decision=="REJECTED" else "#f59e0b"}},
            gauge={
                "axis":{"range":[0,100],"tickcolor":"#9ca3af","tickfont":{"color":"#374151"}},
                "bar":{"color":"#10b981" if pct>=70 else "#f59e0b" if pct>=45 else "#ef4444","thickness":0.22},
                "bgcolor":"#ffffff",
                "steps":[
                    {"range":[0,45], "color":"rgba(239,68,68,0.12)"},
                    {"range":[45,70],"color":"rgba(245,158,11,0.12)"},
                    {"range":[70,100],"color":"rgba(16,185,129,0.12)"},
                ],
                "threshold":{"line":{"color":"white","width":3},"thickness":0.75,"value":pct}
            }
        ))
        fig_g.update_layout(paper_bgcolor="#ffffff",font_color="#374151",
                            height=210,margin=dict(t=20,b=0,l=20,r=20))
        st.plotly_chart(fig_g,use_container_width=True)

    with col_ref:
        st.subheader("Applicant Risk Profile")
        # Risk breakdown bars
        factors = {
            "Credit Score":      min((cred_s-300)/550,1.0),
            "Debt-to-Income":    max(1-dti_s*2,0),
            "Income Level":      min(inc_s/200000,1.0),
            "Loan-to-Income":    max(1-loan_s/inc_s/8,0),
            "Delinquency":       max(1-del_s/3,0),
            "Employment":        min(emp_s/15,1.0),
        }
        for fname, score_f in factors.items():
            pct_f = int(score_f*100)
            col   = "#10b981" if pct_f>=70 else "#f59e0b" if pct_f>=45 else "#ef4444"
            st.markdown(f"""
            <div style="margin-bottom:10px">
                <div style="display:flex;justify-content:space-between;font-size:12px;
                     color:#374151;margin-bottom:4px">
                    <span>{fname}</span><span style="color:{col}">{pct_f}%</span>
                </div>
                <div style="background:#e2e8f0;border-radius:4px;height:6px">
                    <div style="width:{pct_f}%;height:100%;background:{col};
                         border-radius:4px;transition:width 0.5s"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Approval Thresholds")
        thresholds = pd.DataFrame({
            "Threshold":   ["Rejected","Manual Review","Approved"],
            "Probability": ["< 45%",   "45% – 69%",    "≥ 70%"],
            "Action":      ["Decline", "Underwriter",  "Auto-approve"],
        })
        st.dataframe(thresholds, use_container_width=True, hide_index=True)

        st.markdown("#### Input Summary")
        inp_df = pd.DataFrame({
            "Field":  ["Age","Income","Loan Amount","Term","Credit Score",
                       "Employment Yrs","DTI","Delinquencies","Education","Purpose"],
            "Value":  [age_s, f"${inc_s:,}", f"${loan_s:,}", f"{term_s}mo",
                       cred_s, f"{emp_s}y", f"{dti_s:.0%}", del_s, edu_s, purpose_s]
        })
        st.dataframe(inp_df, use_container_width=True, hide_index=True)

# ════════════════ TAB 6: CODE ══════════════════════════════════
with tabs[5]:
    st.markdown("### 📁 Project Structure")
    st.code("""
loan_approval_nn/
├── app.py                 ← Streamlit web application (this file)
├── requirements.txt       ← Python 3.11 dependencies
├── src/
│   ├── data_pipeline.py   ← Data loading, cleaning, feature engineering
│   ├── model.py           ← Neural network architecture
│   ├── train.py           ← Training loop with callbacks
│   └── predict.py         ← Inference for new applicants
├── models/
│   ├── loannet_best.h5    ← Saved Keras model
│   └── scaler.pkl         ← Fitted StandardScaler
└── data/
    └── loan_data.csv      ← Your dataset
    """, language="bash")

    st.markdown("### ▶️ Setup (Python 3.11)")
    st.code("""
# 1. Create virtual environment
python -m venv venv

# 2. Activate — Windows
venv\\Scripts\\activate

# 2. Activate — Mac/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run app
streamlit run app.py
    """, language="bash")

    with st.expander("📄 src/data_pipeline.py — Full code"):
        st.code(open("/dev/stdin").read() if False else """
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

CAT_COLS = ['education', 'employment_type', 'loan_purpose']
TARGET   = 'approved'

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna(subset=[TARGET])
    # Clip outliers on key numeric columns
    df['credit_score']   = df['credit_score'].clip(300, 850)
    df['debt_to_income'] = df['debt_to_income'].clip(0, 1)
    df[TARGET] = df[TARGET].astype(int)
    return df.reset_index(drop=True)

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    df['loan_to_income']  = (df['loan_amount'] / (df['income'] + 1)).round(3)
    df['loan_to_value']   = (df['loan_amount'] / (df['property_value'] + 1)).round(3)
    df['monthly_payment'] = (df['loan_amount'] * 0.005).round(2)
    df['credit_band']     = pd.cut(df['credit_score'],
        bins=[300,580,670,740,800,850], labels=[0,1,2,3,4]).astype(int)
    return df

def encode_and_split(df, val_pct=0.15, seed=42):
    encoders = {}
    for col in CAT_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le

    features = [c for c in df.columns if c != TARGET]
    X = df[features].values.astype(np.float32)
    y = df[TARGET].values.astype(np.float32)

    np.random.seed(seed)
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    n    = len(X)
    i1   = int(n * (1 - 2*val_pct))
    i2   = int(n * (1 - val_pct))

    X_tr, X_va, X_te = X[:i1], X[i1:i2], X[i2:]
    y_tr, y_va, y_te = y[:i1], y[i1:i2], y[i2:]

    sc = StandardScaler().fit(X_tr)
    joblib.dump(sc, 'models/scaler.pkl')
    joblib.dump(encoders, 'models/encoders.pkl')

    return (sc.transform(X_tr), sc.transform(X_va), sc.transform(X_te),
            y_tr, y_va, y_te, features)

def run_pipeline(path: str):
    df = load_data(path)
    df = clean(df)
    df = engineer(df)
    return encode_and_split(df)
        """, language="python")

    with st.expander("📄 src/model.py — Full code"):
        st.code("""
import tensorflow as tf
from tensorflow import keras

def build_loannet(input_dim: int, dropout1=0.4, dropout2=0.3) -> keras.Model:
    reg = keras.regularizers.L2(1e-4)
    inp = keras.layers.Input(shape=(input_dim,), name='applicant_features')

    x = keras.layers.Dense(256, 'relu', kernel_regularizer=reg)(inp)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout1)(x)

    x = keras.layers.Dense(128, 'relu', kernel_regularizer=reg)(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(dropout2)(x)

    x = keras.layers.Dense(64, 'relu')(x)
    x = keras.layers.BatchNormalization()(x)

    x = keras.layers.Dense(32, 'relu')(x)
    out = keras.layers.Dense(1, 'sigmoid', name='approval_prob')(x)

    model = keras.Model(inp, out, name='LoanNet')
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model
        """, language="python")

    with st.expander("📄 src/train.py — Full code"):
        st.code("""
from src.data_pipeline import run_pipeline
from src.model import build_loannet
from tensorflow import keras
import numpy as np

X_tr, X_va, X_te, y_tr, y_va, y_te, features = run_pipeline('data/loan_data.csv')

model = build_loannet(input_dim=len(features))
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=14,
        restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=7, min_lr=1e-6),
    keras.callbacks.ModelCheckpoint(
        'models/loannet_best.h5',
        save_best_only=True, verbose=0),
    keras.callbacks.TensorBoard(log_dir='logs/')
]

history = model.fit(
    X_tr, y_tr,
    validation_data=(X_va, y_va),
    epochs=200, batch_size=256,
    callbacks=callbacks, verbose=1
)

# Evaluate
from sklearn.metrics import classification_report, roc_auc_score
y_prob = model.predict(X_te).flatten()
y_pred = (y_prob >= 0.5).astype(int)
print(classification_report(y_te, y_pred, target_names=['Rejected','Approved']))
print(f"AUC-ROC: {roc_auc_score(y_te, y_prob):.4f}")
        """, language="python")

    with st.expander("📄 src/predict.py — Full code"):
        st.code("""
import numpy as np
import joblib
from tensorflow import keras

_model    = keras.models.load_model('models/loannet_best.h5')
_scaler   = joblib.load('models/scaler.pkl')
_encoders = joblib.load('models/encoders.pkl')

DECISIONS = [
    (0.70, 1.0,  'APPROVED',       'Strong application. Proceed to verification.'),
    (0.45, 0.70, 'MANUAL REVIEW',  'Borderline. Requires underwriter assessment.'),
    (0.00, 0.45, 'REJECTED',       'High risk. Does not meet approval thresholds.'),
]

def predict_applicant(applicant: dict) -> dict:
    '''
    applicant: dict with keys matching training feature names
    Returns: {decision, probability, confidence, advice}
    '''
    # Encode categoricals
    for col, enc in _encoders.items():
        if col in applicant:
            try:
                applicant[col] = enc.transform([applicant[col]])[0]
            except ValueError:
                applicant[col] = 0

    # Build feature vector (must match training order)
    FEATURES = list(applicant.keys())
    x = np.array([[applicant[f] for f in FEATURES]], dtype=np.float32)
    x_scaled = _scaler.transform(x)

    prob = float(_model.predict(x_scaled, verbose=0)[0][0])

    for lo, hi, decision, advice in DECISIONS:
        if lo <= prob < hi or (hi == 1.0 and prob >= lo):
            return {
                'decision':    decision,
                'probability': round(prob * 100, 1),
                'advice':      advice
            }

if __name__ == '__main__':
    sample = {
        'age': 35, 'income': 75000, 'loan_amount': 200000,
        'loan_term': 360, 'credit_score': 720, 'employment_years': 7,
        'debt_to_income': 0.28, 'num_credit_lines': 5,
        'num_delinquencies': 0, 'property_value': 350000,
        'loan_to_income': 2.67, 'loan_to_value': 0.57,
        'monthly_payment': 1000, 'savings_ratio': 0.3,
        'education': 'Bachelor', 'employment_type': 'Salaried',
        'loan_purpose': 'Home'
    }
    result = predict_applicant(sample)
    print(f"Decision: {result['decision']}  ({result['probability']}%)")
    print(f"Advice:   {result['advice']}")
        """, language="python")

        