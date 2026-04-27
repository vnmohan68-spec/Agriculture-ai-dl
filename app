"""
🌿 Plant Disease Detection — Premium Streamlit App
Model: MobileNetV2 | 11 Classes | Dark Agriculture Theme
"""

import io, json, time
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from PIL import Image

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Smart Agriculture System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Theme ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #020817;
    color: #e2e8f0;
}

/* ── Main background ── */
.stApp { background: #020817; }

/* ── Hide default header ── */
header[data-testid="stHeader"] { display: none; }
.block-container { padding-top: 2rem !important; padding-bottom: 3rem !important; }

/* ── Glassmorphism card ── */
.glass-card {
    background: rgba(15, 23, 42, 0.75);
    border: 1px solid rgba(34, 197, 94, 0.18);
    border-radius: 18px;
    padding: 28px;
    backdrop-filter: blur(12px);
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}

/* ── Result cards ── */
.result-healthy {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(34,197,94,0.5);
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 0 40px rgba(34,197,94,0.12);
}
.result-disease {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(239,68,68,0.45);
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 0 40px rgba(239,68,68,0.1);
}
.result-moderate {
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(245,158,11,0.45);
    border-radius: 18px;
    padding: 28px;
    box-shadow: 0 0 40px rgba(245,158,11,0.1);
}

/* ── Metric box ── */
.metric-box {
    background: rgba(30,41,59,0.6);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(10,15,30,0.97) !important;
    border-right: 1px solid rgba(34,197,94,0.12) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #16a34a, #22c55e) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 14px 32px !important;
    font-weight: 700 !important;
    font-size: 16px !important;
    width: 100% !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(34,197,94,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 30px rgba(34,197,94,0.45) !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    background: rgba(30,41,59,0.5) !important;
    border: 1px solid rgba(34,197,94,0.15) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(15,23,42,0.7) !important;
    border: 2px dashed rgba(34,197,94,0.3) !important;
    border-radius: 16px !important;
    padding: 20px !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #16a34a, #22c55e) !important;
    border-radius: 999px !important;
}

/* ── Divider ── */
hr { border-color: rgba(34,197,94,0.12) !important; }

/* ── Info / Warning / Error ── */
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ── Config ────────────────────────────────────────────────────────────────────
API_URL = "http://localhost:8000"
USE_API = False  # Set True if FastAPI backend is running

MODEL_PATH  = Path("model/plant_disease_model.keras")
CLASS_MAP   = Path("model/plant_classes.json")
IMG_SIZE    = (224, 224)

DISEASE_INFO = {
    "pepper bell bacterial spot": {
        "status": "diseased", "severity": "Moderate",
        "treatment": "Apply copper-based bactericides. Remove infected leaves immediately.",
        "fertilizer": "Balanced NPK 10-10-10. Avoid excess nitrogen.",
        "prevention": "Use disease-free seeds. Rotate crops every 2 seasons. Maintain proper spacing.",
    },
    "pepper bell healthy": {
        "status": "healthy", "severity": "None",
        "treatment": "No treatment needed. Continue regular monitoring.",
        "fertilizer": "Balanced NPK 10-10-10 with micronutrients.",
        "prevention": "Maintain proper irrigation. Monitor for early signs of disease.",
    },
    "potato early blight": {
        "status": "diseased", "severity": "Moderate",
        "treatment": "Apply chlorothalonil or mancozeb fungicide. Remove affected foliage.",
        "fertilizer": "Potassium-rich fertilizer (K2O). Avoid excess nitrogen.",
        "prevention": "Plant certified disease-free tubers. Avoid overhead irrigation.",
    },
    "potato late blight": {
        "status": "diseased", "severity": "High",
        "treatment": "Apply metalaxyl or cymoxanil fungicide immediately. Destroy infected plants.",
        "fertilizer": "Calcium and potassium supplements to strengthen cell walls.",
        "prevention": "Use resistant varieties. Ensure proper drainage. Monitor humidity.",
    },
    "tomato bacterial spot": {
        "status": "diseased", "severity": "Moderate",
        "treatment": "Copper-based bactericides + mancozeb. Prune infected branches.",
        "fertilizer": "Reduce nitrogen. Add calcium nitrate to strengthen tissue.",
        "prevention": "Avoid wet foliage. Use drip irrigation. Crop rotation mandatory.",
    },
    "tomato early blight": {
        "status": "diseased", "severity": "Moderate",
        "treatment": "Apply azoxystrobin or chlorothalonil fungicide every 7–10 days.",
        "fertilizer": "Balanced fertilizer. Add magnesium sulfate for leaf strength.",
        "prevention": "Mulch soil. Stake plants for air circulation. Remove lower leaves.",
    },
    "tomato late blight": {
        "status": "diseased", "severity": "High",
        "treatment": "Apply mancozeb + cymoxanil immediately. Remove all infected tissue.",
        "fertilizer": "Potassium-rich supplement. Reduce nitrogen input.",
        "prevention": "Avoid evening irrigation. Use resistant varieties. Improve airflow.",
    },
    "tomato leaf mold": {
        "status": "diseased", "severity": "Moderate",
        "treatment": "Apply copper oxychloride or chlorothalonil. Increase ventilation.",
        "fertilizer": "Balanced nutrition. Avoid over-fertilizing with nitrogen.",
        "prevention": "Reduce humidity below 85%. Space plants properly. Remove debris.",
    },
    "tomato septoria leaf spot": {
        "status": "diseased", "severity": "Moderate",
        "treatment": "Apply chlorothalonil or mancozeb. Remove lower infected leaves.",
        "fertilizer": "Calcium and potassium supplement. Moderate nitrogen.",
        "prevention": "Avoid overhead watering. Rotate crops. Use mulch to prevent splash.",
    },
    "tomato target spot": {
        "status": "diseased", "severity": "Moderate",
        "treatment": "Apply azoxystrobin or tebuconazole fungicide. Destroy crop debris.",
        "fertilizer": "Balanced NPK. Add boron micronutrient.",
        "prevention": "Avoid plant stress. Ensure proper drainage. Monitor regularly.",
    },
    "tomato tomato mosaic virus": {
        "status": "diseased", "severity": "High",
        "treatment": "No chemical cure. Remove and destroy infected plants immediately.",
        "fertilizer": "Balanced nutrition to support plant immunity.",
        "prevention": "Use virus-free seeds. Control aphid vectors. Disinfect tools.",
    },
}

CLASS_NAMES = sorted(DISEASE_INFO.keys())

# ── Local inference ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        import tensorflow as tf
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as ppi
        m = tf.keras.models.load_model(str(MODEL_PATH))
        m.predict(np.zeros((1, 224, 224, 3)))  # warm-up
        return m, ppi
    except Exception as e:
        return None, None

def predict_local(img_bytes):
    model, ppi = load_model()
    if model is None:
        return None, "Model not found at model/plant_disease_model.keras"
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize(IMG_SIZE)
    arr = np.expand_dims(np.array(img, dtype=np.float32), 0)
    arr = ppi(arr)
    preds = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(preds))
    return {
        "predicted_class": CLASS_NAMES[idx],
        "confidence": round(float(preds[idx]) * 100, 2),
        "all_probabilities": {CLASS_NAMES[i]: round(float(preds[i]), 4) for i in range(len(CLASS_NAMES))},
    }, None

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 10px 0 30px;">
  <div style="display:inline-flex; align-items:center; gap:8px;
       background:rgba(34,197,94,0.1); border:1px solid rgba(34,197,94,0.3);
       border-radius:999px; padding:6px 18px; margin-bottom:20px;">
    <span style="color:#22c55e; font-size:12px; font-weight:700; letter-spacing:1.5px;">
      🌿 AI SMART AGRICULTURE SYSTEM
    </span>
  </div>
  <h1 style="font-size:46px; font-weight:800; margin:0 0 12px;
       background:linear-gradient(135deg,#ffffff 0%,#22c55e 55%,#3b82f6 100%);
       -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
    Detect. Diagnose.<br>Protect Your Crops.
  </h1>
  <p style="color:#64748b; font-size:17px; max-width:540px; margin:0 auto;">
    Upload a leaf image — MobileNetV2 identifies 11 crop diseases in under a second
    and delivers actionable treatment plans.
  </p>
</div>
""", unsafe_allow_html=True)

# ── Stats bar ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("⚡", "< 1s Inference", "#f59e0b"),
    ("🛡️", "11 Disease Classes", "#22c55e"),
    ("🧠", "MobileNetV2 Model", "#3b82f6"),
    ("📊", "78% PV Accuracy", "#a855f7"),
]
for col, (icon, label, color) in zip([c1, c2, c3, c4], stats):
    col.markdown(f"""
    <div class="metric-box">
      <div style="font-size:22px; margin-bottom:4px;">{icon}</div>
      <div style="color:{color}; font-size:12px; font-weight:700;">{label}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Main Layout ────────────────────────────────────────────────────────────────
left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("#### 📷 Upload Crop Leaf Image")
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")

    if uploaded:
        img_bytes = uploaded.read()
        img = Image.open(io.BytesIO(img_bytes))
        st.image(img, use_column_width=True, caption="Uploaded Image")

    analyze = st.button("🌿 Analyze Crop", disabled=uploaded is None)
    st.markdown('</div>', unsafe_allow_html=True)

    # ── Weather widget ──
    st.markdown("""
    <div class="glass-card">
      <p style="color:#64748b; font-size:11px; font-weight:700; letter-spacing:1.5px; margin-bottom:16px;">
        🌤 FIELD CONDITIONS (MOCK)
      </p>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px;">
        <div class="metric-box"><div style="color:#f59e0b;font-size:20px;font-weight:800;">28°C</div>
             <div style="color:#475569;font-size:11px;">Temperature</div></div>
        <div class="metric-box"><div style="color:#3b82f6;font-size:20px;font-weight:800;">72%</div>
             <div style="color:#475569;font-size:11px;">Humidity</div></div>
        <div class="metric-box"><div style="color:#8b5cf6;font-size:20px;font-weight:800;">14 km/h</div>
             <div style="color:#475569;font-size:11px;">Wind Speed</div></div>
        <div class="metric-box"><div style="color:#ef4444;font-size:20px;font-weight:800;">HIGH</div>
             <div style="color:#475569;font-size:11px;">UV Index</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

with right:
    if analyze and uploaded:
        with st.spinner("🧠 Running AI analysis..."):
            t0 = time.time()
            if USE_API:
                try:
                    r = requests.post(f"{API_URL}/predict",
                                      files={"file": (uploaded.name, img_bytes, uploaded.type)},
                                      timeout=30)
                    result = r.json()
                    err = None
                except Exception as e:
                    result, err = None, str(e)
            else:
                result, err = predict_local(img_bytes)
                if result:
                    info = DISEASE_INFO.get(result["predicted_class"], {})
                    result.update(info)
                    result["inference_ms"] = round((time.time() - t0) * 1000, 1)

        if err:
            st.error(f"Error: {err}")
        elif result:
            cls    = result.get("predicted_class", "Unknown")
            conf   = result.get("confidence", 0)
            status = result.get("status", "unknown")
            sev    = result.get("severity", "Unknown")

            if status == "healthy":
                card_class, badge_color, badge_text = "result-healthy", "#22c55e", "✅ HEALTHY"
            elif sev == "High":
                card_class, badge_color, badge_text = "result-disease", "#ef4444", "⚠️ HIGH RISK"
            else:
                card_class, badge_color, badge_text = "result-moderate", "#f59e0b", "⚠️ MODERATE RISK"

            st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
              <span style="background:{badge_color}20; border:1px solid {badge_color}60;
                    color:{badge_color}; font-size:12px; font-weight:700; padding:5px 14px; border-radius:999px;">
                {badge_text}
              </span>
              <span style="color:#475569; font-size:12px;">{result.get('inference_ms','—')}ms</span>
            </div>
            <h2 style="font-size:22px; font-weight:800; text-transform:capitalize;
                 color:#f1f5f9; margin-bottom:8px;">{cls}</h2>
            """, unsafe_allow_html=True)

            st.markdown("**Confidence**")
            st.progress(int(conf))
            st.markdown(f"<p style='color:{badge_color}; font-weight:700; margin-top:-12px;'>{conf}%</p>",
                        unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

            # Insights
            with st.expander("💊 Treatment Plan", expanded=True):
                st.markdown(f"<p style='color:#cbd5e1;'>{result.get('treatment','—')}</p>",
                            unsafe_allow_html=True)
            with st.expander("🌱 Fertilizer Recommendation"):
                st.markdown(f"<p style='color:#cbd5e1;'>{result.get('fertilizer','—')}</p>",
                            unsafe_allow_html=True)
            with st.expander("🛡️ Prevention Tips"):
                st.markdown(f"<p style='color:#cbd5e1;'>{result.get('prevention','—')}</p>",
                            unsafe_allow_html=True)

            # Top-3 probabilities
            if "all_probabilities" in result:
                st.markdown("<br>**Top Predictions**", unsafe_allow_html=True)
                probs = sorted(result["all_probabilities"].items(), key=lambda x: -x[1])[:3]
                for name, prob in probs:
                    pct = round(prob * 100, 1)
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                      <div style="display:flex; justify-content:space-between; margin-bottom:4px;">
                        <span style="color:#94a3b8; font-size:12px; text-transform:capitalize;">{name}</span>
                        <span style="color:#22c55e; font-size:12px; font-weight:700;">{pct}%</span>
                      </div>
                      <div style="height:4px; background:#1e293b; border-radius:999px;">
                        <div style="height:100%; width:{pct}%; background:linear-gradient(90deg,#16a34a,#22c55e);
                             border-radius:999px;"></div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(15,23,42,0.5); border:1px solid rgba(34,197,94,0.1);
             border-radius:18px; padding:60px; text-align:center;">
          <div style="font-size:52px; margin-bottom:16px;">🌿</div>
          <p style="color:#334155; font-size:15px;">Upload an image and click Analyze<br>to see AI predictions here</p>
        </div>
        """, unsafe_allow_html=True)

# ── Soil Dashboard (full width) ───────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="glass-card">
  <p style="color:#64748b; font-size:11px; font-weight:700; letter-spacing:1.5px; margin-bottom:20px;">
    🌱 SOIL HEALTH DASHBOARD (MOCK)
  </p>
</div>
""", unsafe_allow_html=True)

cols = st.columns(4)
soil = [
    ("Nitrogen", 68, "#22c55e"),
    ("Phosphorus", 45, "#3b82f6"),
    ("Potassium", 81, "#a855f7"),
    ("pH Level", 63, "#f59e0b"),
]
for col, (label, val, color) in zip(cols, soil):
    col.markdown(f"""
    <div class="metric-box">
      <div style="color:{color}; font-size:28px; font-weight:800;">{val}%</div>
      <div style="color:#475569; font-size:12px; margin-top:4px;">{label}</div>
    </div>
    """, unsafe_allow_html=True)
    col.progress(val)

st.markdown("<br><br>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:20px; border-top:1px solid rgba(34,197,94,0.1);">
  <p style="color:#334155; font-size:12px;">
    🌿 AI Smart Agriculture System · MobileNetV2 Plant Disease Detection · 11 Classes
  </p>
</div>
""", unsafe_allow_html=True)
