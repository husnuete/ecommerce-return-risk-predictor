"""
Return Risk Scoring — Bilingual Streamlit Web Application
==========================================================
An ML-powered Proof of Concept that predicts the probability
of a customer returning their e-commerce order, using a
Random Forest Classifier trained on synthetic data.

Features:
  - Dynamic TR / EN language switcher in the sidebar
  - All UI text driven by a translations dictionary
  - Backend logic untouched by language selection

Run:  streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# ╔══════════════════════════════════════════════════════════════╗
# ║  TRANSLATIONS DICTIONARY                                    ║
# ╚══════════════════════════════════════════════════════════════╝
TR = {
    # Page & hero
    "page_title":        "İade Riski Skorlama",
    "hero_title":        "🛒 İade Riski Skorlama",
    "hero_subtitle":     "Makine Öğrenmesi ile Müşteri İade Tahmini — Random Forest Classifier",

    # Sidebar
    "lang_label":        "🌐 Dil / Language",
    "control_panel":     "🎛️ Kontrol Paneli",
    "control_desc":      "Müşteri bilgilerini girerek anlık iade riski tahmini alın.",
    "age_label":         "👤 Müşteri Yaşı",
    "age_help":          "Müşterinin yaşını seçin",
    "gender_label":      "⚧ Cinsiyet",
    "gender_help":       "Müşterinin cinsiyetini seçin",
    "gender_female":     "Kadın",
    "gender_male":       "Erkek",
    "cart_label":        "🛒 Sepetteki Ürün Sayısı",
    "cart_help":         "Sepetteki toplam ürün adedi",
    "prev_label":        "📦 Önceki Alışveriş Sayısı",
    "prev_help":         "Müşterinin geçmiş toplam alışveriş sayısı",
    "hour_label":        "🕐 İşlem Saati",
    "hour_help":         "Siparişin verildiği saat (0-23)",
    "about_title":       "📌 Bilgi",
    "about_text":        ("Bu uygulama, 1.000 satırlık sentetik veri ile eğitilmiş "
                          "bir **Random Forest** modeli kullanır. "
                          "Dengesiz veri problemi `class_weight='balanced'` ile çözülmüştür."),

    # Risk card
    "return_probability": "İade Riski Olasılığı",
    "low_risk":          "✅ DÜŞÜK RİSK",
    "medium_risk":       "⚠️ ORTA RİSK",
    "high_risk":         "🚨 YÜKSEK RİSK",
    "low_desc":          "Bu müşterinin iade yapma olasılığı düşüktür.",
    "medium_desc":       "Bu müşteri iade yapabilir, dikkatli olunmalı.",
    "high_desc":         "Bu müşterinin iade yapma olasılığı yüksek!",
    "prediction_summary": "Tahmin Özeti",
    "will_return":       "İade Var",
    "no_return":         "İade Yok",

    # Customer profile
    "customer_profile":  "👤 Girilen Müşteri Profili",
    "col_age":           "Yaş",
    "col_gender":        "Cinsiyet",
    "col_cart":          "Sepet",
    "col_history":       "Geçmiş",
    "col_hour":          "Saat",
    "cart_unit":         "ürün",
    "history_unit":      "alışveriş",

    # Tabs & charts
    "model_analysis":    "📊 Model Analizi",
    "tab_importance":    "🌲 Değişken Önem Düzeyleri",
    "tab_confusion":     "📋 Karmaşıklık Matrisi",
    "tab_performance":   "📈 Model Performansı",

    "chart_imp_title":   "Değişken Önem Düzeyleri — Random Forest",
    "chart_imp_xlabel":  "Önem Değeri (Importance)",
    "chart_imp_caption": "★ En etkili değişkenler: **Yaş** ve **Önceki Alışveriş Sayısı**",

    "chart_cm_title":    "Karmaşıklık Matrisi (Confusion Matrix)",
    "chart_cm_xlabel":   "Tahmin Edilen (Predicted)",
    "chart_cm_ylabel":   "Gerçek (Actual)",
    "cm_no_return":      "İade Yok (0)",
    "cm_return":         "İade Var (1)",

    "metric_accuracy":   "Doğruluk",
    "metric_recall":     "İade Var — Recall",
    "metric_precision":  "İade Var — Precision",
    "metric_f1":         "İade Var — F1",
    "class_report":      "📋 Detaylı Sınıflandırma Raporu",
    "comparison_title":  "🔄 Lojistik Regresyon vs Random Forest",
    "comp_metric":       "Metrik",
    "comp_lr":           "Lojistik Regresyon",
    "comp_rf":           "Random Forest",

    # Feature display names for charts
    "feat_age":          "Yaş",
    "feat_gender":       "Cinsiyet",
    "feat_cart":         "Sepetteki Ürün Sayısı",
    "feat_prev":         "Önceki Alışveriş Sayısı",
    "feat_hour":         "İşlem Saati",

    # Report target names
    "report_no_return":  "İade Yok",
    "report_return":     "İade Var",

    # Footer
    "footer":            ("<strong>İade Riski Skorlama</strong> — Proof of Concept<br>"
                          "Random Forest Classifier · 1.000 Sentetik Veri · Python + Streamlit<br>"
                          "📊 İstatistik Bölümü Projesi"),
}

EN = {
    # Page & hero
    "page_title":        "Return Risk Scoring",
    "hero_title":        "🛒 Return Risk Scoring",
    "hero_subtitle":     "ML-Powered Customer Return Prediction — Random Forest Classifier",

    # Sidebar
    "lang_label":        "🌐 Dil / Language",
    "control_panel":     "🎛️ Control Panel",
    "control_desc":      "Enter customer details to get an instant return risk prediction.",
    "age_label":         "👤 Customer Age",
    "age_help":          "Select the customer's age",
    "gender_label":      "⚧ Gender",
    "gender_help":       "Select the customer's gender",
    "gender_female":     "Female",
    "gender_male":       "Male",
    "cart_label":        "🛒 Cart Items",
    "cart_help":         "Number of items in the shopping cart",
    "prev_label":        "📦 Previous Purchases",
    "prev_help":         "Total number of past orders by this customer",
    "hour_label":        "🕐 Transaction Hour",
    "hour_help":         "Hour of the day the order was placed (0–23)",
    "about_title":       "📌 About",
    "about_text":        ("This app uses a **Random Forest Classifier** trained on "
                          "1,000 rows of synthetic e-commerce data. "
                          "Class imbalance is handled via `class_weight='balanced'`."),

    # Risk card
    "return_probability": "Return Probability",
    "low_risk":          "✅ LOW RISK",
    "medium_risk":       "⚠️ MEDIUM RISK",
    "high_risk":         "🚨 HIGH RISK",
    "low_desc":          "This customer is unlikely to return their order.",
    "medium_desc":       "This customer may return their order — monitor closely.",
    "high_desc":         "This customer has a high probability of returning their order!",
    "prediction_summary": "Prediction Summary",
    "will_return":       "Will Return",
    "no_return":         "No Return",

    # Customer profile
    "customer_profile":  "👤 Customer Profile",
    "col_age":           "Age",
    "col_gender":        "Gender",
    "col_cart":          "Cart",
    "col_history":       "History",
    "col_hour":          "Hour",
    "cart_unit":         "items",
    "history_unit":      "orders",

    # Tabs & charts
    "model_analysis":    "📊 Model Analysis",
    "tab_importance":    "🌲 Feature Importance",
    "tab_confusion":     "📋 Confusion Matrix",
    "tab_performance":   "📈 Model Performance",

    "chart_imp_title":   "Feature Importance — Random Forest",
    "chart_imp_xlabel":  "Importance Score",
    "chart_imp_caption": "★ Most influential features: **Age** and **Previous Purchases**",

    "chart_cm_title":    "Confusion Matrix",
    "chart_cm_xlabel":   "Predicted",
    "chart_cm_ylabel":   "Actual",
    "cm_no_return":      "No Return (0)",
    "cm_return":         "Return (1)",

    "metric_accuracy":   "Accuracy",
    "metric_recall":     "Return — Recall",
    "metric_precision":  "Return — Precision",
    "metric_f1":         "Return — F1",
    "class_report":      "📋 Classification Report",
    "comparison_title":  "🔄 Logistic Regression vs Random Forest",
    "comp_metric":       "Metric",
    "comp_lr":           "Logistic Regression",
    "comp_rf":           "Random Forest",

    # Feature display names for charts
    "feat_age":          "Age",
    "feat_gender":       "Gender",
    "feat_cart":         "Cart Items",
    "feat_prev":         "Previous Purchases",
    "feat_hour":         "Transaction Hour",

    # Report target names
    "report_no_return":  "No Return",
    "report_return":     "Return",

    # Footer
    "footer":            ("<strong>Return Risk Scoring</strong> — Proof of Concept<br>"
                          "Random Forest Classifier · 1,000 Synthetic Records · Python + Streamlit<br>"
                          "📊 Statistics Department Project"),
}

LANGS = {"Türkçe": TR, "English": EN}


# ─── Page Configuration ────────────────────────────────────────
st.set_page_config(
    page_title="Return Risk Scoring | İade Riski Skorlama",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS — Premium dark theme ──────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .hero-card {
        background: linear-gradient(135deg, #6C63FF 0%, #3B3795 100%);
        border-radius: 16px; padding: 28px 32px; margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(108, 99, 255, 0.25);
    }
    .hero-card h1 { color: white; font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.5px; }
    .hero-card p  { color: rgba(255,255,255,0.8); font-size: 1rem; margin: 6px 0 0 0; }

    .risk-card {
        border-radius: 20px; padding: 36px; text-align: center; margin: 16px 0;
        box-shadow: 0 8px 32px rgba(0,0,0,0.15); transition: transform 0.3s ease;
    }
    .risk-card:hover { transform: translateY(-2px); }
    .risk-low    { background: linear-gradient(135deg, #00B894 0%, #00796B 100%); box-shadow: 0 8px 32px rgba(0,184,148,0.3); }
    .risk-medium { background: linear-gradient(135deg, #FDCB6E 0%, #E17055 100%); box-shadow: 0 8px 32px rgba(253,203,110,0.3); }
    .risk-high   { background: linear-gradient(135deg, #FF6B6B 0%, #C0392B 100%); box-shadow: 0 8px 32px rgba(255,107,107,0.3); }

    .risk-score { font-size: 4.5rem; font-weight: 900; color: white; line-height: 1; text-shadow: 0 4px 12px rgba(0,0,0,0.2); }
    .risk-label { font-size: 1.3rem; font-weight: 600; color: rgba(255,255,255,0.9); margin-top: 8px; }
    .risk-tag   { display: inline-block; font-size: 0.9rem; font-weight: 700; color: white;
                  background: rgba(255,255,255,0.2); border-radius: 20px; padding: 6px 18px;
                  margin-top: 12px; backdrop-filter: blur(4px); }

    .info-card {
        background: linear-gradient(135deg, #1A1D27 0%, #22263A 100%);
        border: 1px solid rgba(108,99,255,0.2); border-radius: 14px; padding: 20px 24px; margin: 8px 0;
    }
    .info-card h3 { color: #6C63FF; font-size: 0.85rem; font-weight: 600;
                    text-transform: uppercase; letter-spacing: 1px; margin: 0 0 4px 0; }
    .info-card .value { color: #E8E8E8; font-size: 1.8rem; font-weight: 800; }

    section[data-testid="stSidebar"] { background: linear-gradient(180deg, #121420 0%, #1A1D27 100%); }
    section[data-testid="stSidebar"] .stMarkdown h2 { color: #6C63FF; }
    [data-testid="stMetricValue"] { font-weight: 700; }

    .divider { height: 2px; background: linear-gradient(90deg, transparent, #6C63FF, transparent);
               margin: 24px 0; border: none; }
</style>
""", unsafe_allow_html=True)


# ╔══════════════════════════════════════════════════════════════╗
# ║  TRAIN MODEL — cached, runs only once per session           ║
# ╚══════════════════════════════════════════════════════════════╝
@st.cache_resource
def train_model():
    """Load dataset, train Random Forest, return model + metrics."""
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "eticaret_iade_verisi.csv")
    df = pd.read_csv(data_path)

    # Encode gender numerically (Kadın=0, Erkek=1)
    df["cinsiyet_kod"] = df["cinsiyet"].map({"Kadın": 0, "Erkek": 1})

    feature_cols = ["yas", "cinsiyet_kod", "sepet_urun_sayisi",
                    "onceki_alisveris", "islem_saati"]

    X = df[feature_cols]
    y = df["iade_durumu"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200, class_weight="balanced",
        random_state=42, max_depth=8, min_samples_split=10, n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    return model, cm, feature_cols


model, cm, feature_cols = train_model()


# ╔══════════════════════════════════════════════════════════════╗
# ║  SESSION STATE — persist user inputs across language changes ║
# ╚══════════════════════════════════════════════════════════════╝
# Initialize defaults only once; subsequent reruns keep the values
_defaults = {"w_age": 30, "w_gender": 0, "w_cart": 3,
             "w_prev": 10, "w_hour": 14, "w_lang": 1}
for _k, _v in _defaults.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ╔══════════════════════════════════════════════════════════════╗
# ║  SIDEBAR — Language selector + control panel                ║
# ╚══════════════════════════════════════════════════════════════╝
with st.sidebar:
    # Language selector at the very top — stable key keeps its value
    lang_choice = st.radio("🌐 Dil / Language",
                           options=["Türkçe", "English"],
                           horizontal=True,
                           key="w_lang")
    t = LANGS[lang_choice]            # active translation dict

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f"## {t['control_panel']}")
    st.markdown(t["control_desc"])
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Every widget uses a fixed key so Streamlit preserves its value
    # when the page reruns due to a language switch.
    age = st.slider(t["age_label"], min_value=18, max_value=65,
                    help=t["age_help"], key="w_age")

    # Gender stored as an index (0=Female/Kadın, 1=Male/Erkek) so
    # changing the display language doesn't reset the selection.
    gender_options = [t["gender_female"], t["gender_male"]]
    gender_idx = st.selectbox(t["gender_label"], options=[0, 1],
                              format_func=lambda i: gender_options[i],
                              help=t["gender_help"], key="w_gender")

    cart_items = st.slider(t["cart_label"], min_value=1, max_value=15,
                           help=t["cart_help"], key="w_cart")

    prev_purchases = st.slider(t["prev_label"], min_value=0, max_value=50,
                               help=t["prev_help"], key="w_prev")

    txn_hour = st.slider(t["hour_label"], min_value=0, max_value=23,
                         help=t["hour_help"], key="w_hour")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown(f"##### {t['about_title']}")
    st.info(t["about_text"])


# ╔══════════════════════════════════════════════════════════════╗
# ║  PREDICT                                                    ║
# ╚══════════════════════════════════════════════════════════════╝
# gender_idx is already 0 or 1 (language-independent)
gender_code = gender_idx
# Display string for the Customer Profile section
gender_display = gender_options[gender_idx]

input_df = pd.DataFrame([{
    "yas":               age,
    "cinsiyet_kod":      gender_code,
    "sepet_urun_sayisi": cart_items,
    "onceki_alisveris":  prev_purchases,
    "islem_saati":       txn_hour,
}])

probs = model.predict_proba(input_df)[0]
return_prob   = probs[1] * 100
no_return_prob = probs[0] * 100


# ╔══════════════════════════════════════════════════════════════╗
# ║  MAIN SCREEN                                                ║
# ╚══════════════════════════════════════════════════════════════╝

# ── Hero ──
st.markdown(f"""
<div class="hero-card">
    <h1>{t['hero_title']}</h1>
    <p>{t['hero_subtitle']}</p>
</div>
""", unsafe_allow_html=True)

# ── Risk score ──
if return_prob < 30:
    risk_css   = "risk-low"
    risk_label = t["low_risk"]
    risk_emoji = "🟢"
    risk_desc  = t["low_desc"]
elif return_prob < 60:
    risk_css   = "risk-medium"
    risk_label = t["medium_risk"]
    risk_emoji = "🟡"
    risk_desc  = t["medium_desc"]
else:
    risk_css   = "risk-high"
    risk_label = t["high_risk"]
    risk_emoji = "🔴"
    risk_desc  = t["high_desc"]

col_risk, col_detail = st.columns([1.2, 1])

with col_risk:
    st.markdown(f"""
    <div class="risk-card {risk_css}">
        <div class="risk-score">{return_prob:.1f}%</div>
        <div class="risk-label">{t['return_probability']}</div>
        <div class="risk-tag">{risk_label}</div>
    </div>
    """, unsafe_allow_html=True)

with col_detail:
    # Strip emoji prefix for the summary card
    clean_label = risk_label
    for prefix in ("✅ ", "⚠️ ", "🚨 "):
        clean_label = clean_label.replace(prefix, "")

    st.markdown(f"""
    <div class="info-card">
        <h3>{t['prediction_summary']}</h3>
        <div class="value">{risk_emoji} {clean_label}</div>
    </div>
    """, unsafe_allow_html=True)

    d1, d2 = st.columns(2)
    with d1:
        st.markdown(f"""
        <div class="info-card">
            <h3>{t['will_return']}</h3>
            <div class="value">{return_prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    with d2:
        st.markdown(f"""
        <div class="info-card">
            <h3>{t['no_return']}</h3>
            <div class="value">{no_return_prob:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f"*{risk_desc}*")


# ── Customer profile ──
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(f"### {t['customer_profile']}")

p1, p2, p3, p4, p5 = st.columns(5)
p1.metric(t["col_age"], f"{age}")
p2.metric(t["col_gender"], gender_display)
p3.metric(t["col_cart"], f"{cart_items} {t['cart_unit']}")
p4.metric(t["col_history"], f"{prev_purchases} {t['history_unit']}")
p5.metric(t["col_hour"], f"{txn_hour}:00")


# ╔══════════════════════════════════════════════════════════════╗
# ║  CHARTS                                                     ║
# ╚══════════════════════════════════════════════════════════════╝
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(f"### {t['model_analysis']}")

tab1, tab2, tab3 = st.tabs([t["tab_importance"], t["tab_confusion"],
                             t["tab_performance"]])

# Dark chart palette
DARK_BG  = "#0F1117"
CARD_BG  = "#1A1D27"
ACCENT   = "#6C63FF"
ACCENT_2 = "#FF6B6B"
WHITE    = "#E8E8E8"
GRID_CLR = "#2A2D37"

plt.rcParams.update({
    "figure.facecolor": DARK_BG,  "axes.facecolor": CARD_BG,
    "axes.edgecolor":   GRID_CLR, "axes.labelcolor": WHITE,
    "text.color":       WHITE,    "xtick.color": WHITE,
    "ytick.color":      WHITE,    "font.size": 12,
    "font.family":      "sans-serif",
})

# Map internal column names ➜ display labels using active language
feat_display = {
    "yas":               t["feat_age"],
    "cinsiyet_kod":      t["feat_gender"],
    "sepet_urun_sayisi": t["feat_cart"],
    "onceki_alisveris":  t["feat_prev"],
    "islem_saati":       t["feat_hour"],
}


# ── TAB 1 — Feature Importance ──
with tab1:
    imp_df = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=True)

    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = [ACCENT_2 if v == imp_df["Importance"].max() else ACCENT
              for v in imp_df["Importance"]]
    labels = [feat_display.get(f, f) for f in imp_df["Feature"]]

    bars = ax1.barh(labels, imp_df["Importance"], color=colors,
                    edgecolor="none", height=0.55, zorder=3)
    for bar, val in zip(bars, imp_df["Importance"]):
        ax1.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f"{val:.3f}", va="center", ha="left",
                 fontsize=12, fontweight="bold", color=WHITE)

    ax1.set_xlabel(t["chart_imp_xlabel"], labelpad=10)
    ax1.set_title(t["chart_imp_title"], fontsize=14, fontweight="bold", pad=15)
    ax1.set_xlim(0, imp_df["Importance"].max() * 1.25)
    ax1.grid(axis="x", color=GRID_CLR, linestyle="--", alpha=0.5, zorder=0)
    plt.tight_layout()
    st.pyplot(fig1)
    plt.close(fig1)

    st.caption(t["chart_imp_caption"])


# ── TAB 2 — Confusion Matrix ──
with tab2:
    fig2, ax2 = plt.subplots(figsize=(7, 5.5))
    cmap = LinearSegmentedColormap.from_list("custom", [CARD_BG, ACCENT, "#A29BFE"])
    im = ax2.imshow(cm, interpolation="nearest", cmap=cmap, aspect="auto")

    class_labels = [t["cm_no_return"], t["cm_return"]]
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            pct = val / cm[i].sum() * 100
            ax2.text(j, i, f"{val}\n({pct:.0f}%)",
                     ha="center", va="center",
                     fontsize=18, fontweight="bold", color=WHITE)

    ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
    ax2.set_xticklabels(class_labels, fontsize=12)
    ax2.set_yticklabels(class_labels, fontsize=12)
    ax2.set_xlabel(t["chart_cm_xlabel"], labelpad=10)
    ax2.set_ylabel(t["chart_cm_ylabel"], labelpad=10)
    ax2.set_title(t["chart_cm_title"], fontsize=14, fontweight="bold", pad=15)
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=WHITE)
    cbar.outline.set_edgecolor(GRID_CLR)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close(fig2)


# ── TAB 3 — Model Performance ──
with tab3:
    # Re-split data and predict to generate a language-aware report
    _data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "eticaret_iade_verisi.csv")
    _df = pd.read_csv(_data_path)
    _df["cinsiyet_kod"] = _df["cinsiyet"].map({"Kadın": 0, "Erkek": 1})
    _X = _df[feature_cols]
    _y = _df["iade_durumu"]
    _, _X_test, _, _y_test = train_test_split(
        _X, _y, test_size=0.20, random_state=42, stratify=_y)
    _y_pred = model.predict(_X_test)
    report = classification_report(
        _y_test, _y_pred,
        target_names=[t["report_no_return"], t["report_return"]],
        output_dict=True,
    )

    ret_key = t["report_return"]   # "Return" or "İade Var"

    m1, m2, m3, m4 = st.columns(4)
    m1.metric(t["metric_accuracy"],  f"{report['accuracy']*100:.1f}%")
    m2.metric(t["metric_recall"],    f"{report[ret_key]['recall']*100:.1f}%")
    m3.metric(t["metric_precision"], f"{report[ret_key]['precision']*100:.1f}%")
    m4.metric(t["metric_f1"],        f"{report[ret_key]['f1-score']*100:.1f}%")

    st.markdown("---")
    st.markdown(f"#### {t['class_report']}")
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.drop(["accuracy"], errors="ignore").round(3)
    st.dataframe(report_df, width="stretch" if hasattr(st, "dataframe") else None)

    st.markdown("---")
    st.markdown(f"#### {t['comparison_title']}")
    comparison = pd.DataFrame({
        t["comp_metric"]: [t["metric_recall"], t["metric_f1"], t["metric_accuracy"]],
        t["comp_lr"]:     ["7.9%", "13.0%", "67.5%"],
        t["comp_rf"]:     [
            f"{report[ret_key]['recall']*100:.1f}%",
            f"{report[ret_key]['f1-score']*100:.1f}%",
            f"{report['accuracy']*100:.1f}%",
        ],
    })
    st.dataframe(comparison, width="stretch" if hasattr(st, "dataframe") else None,
                 hide_index=True)


# ── Footer ──
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div style="text-align: center; color: #666; font-size: 0.85rem; padding: 16px 0;">
    {t['footer']}
</div>
""", unsafe_allow_html=True)
