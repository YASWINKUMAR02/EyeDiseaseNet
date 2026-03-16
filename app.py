import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
import os

# Grad-CAM
try:
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.image import show_cam_on_image
    GRADCAM_AVAILABLE = True
except ImportError:
    GRADCAM_AVAILABLE = False

st.set_page_config(
    page_title="RetinaAI — DR Screening",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp { font-family: 'Inter', sans-serif !important; background: #080b14 !important; color: #e4e6f1; }
.block-container { padding: 2rem 2.5rem 4rem 2.5rem !important; max-width: 1400px; }
.stFileUploader section { background: transparent !important; border: 2px dashed #2a2d4a !important; border-radius: 16px !important; }
.stFileUploader section:hover { border-color: #4f9cf9 !important; }
[data-testid="stSidebar"] { background: #0c0f1d !important; border-right: 1px solid #1a1d30; }
.stProgress > div > div { background: linear-gradient(90deg, #4f9cf9, #a78bfa) !important; border-radius: 99px !important; }
.stProgress { background: #1a1d2e !important; border-radius: 99px !important; }
div[data-testid="stImage"] img { border-radius: 16px !important; }
hr { border-color: #1a1d30 !important; }
.stWarning { background: #241800 !important; border: 1px solid #f59e0b !important; border-radius: 12px !important; }
.stAlert { border-radius: 12px !important; }

/* Sidebar */
.sidebar-title { font-size: 1.1rem; font-weight: 700; color: #fff; margin-bottom: 0.3rem; }
.sidebar-sub   { font-size: 0.78rem; color: #6b7280; line-height: 1.6; }

/* Navbar */
.navbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 0.5rem 0 2rem 0; border-bottom: 1px solid #1a1d30; margin-bottom: 2rem;
}
.nav-brand { display: flex; align-items: center; gap: 0.75rem; }
.nav-logo {
  width: 40px; height: 40px; border-radius: 10px;
  background: linear-gradient(135deg, #4f9cf9, #a78bfa);
  display: grid; place-items: center; font-weight: 900; color: #fff; font-size: 1rem;
}
.nav-title { font-size: 1.25rem; font-weight: 700; color: #fff; }
.nav-sub   { font-size: 0.72rem; color: #6b7280; }
.nav-pill  {
  background: #0d1d38; border: 1px solid #1e3a5f; color: #60a5fa;
  border-radius: 99px; padding: 0.3rem 0.9rem; font-size: 0.75rem; font-weight: 600;
}

/* Hero */
.hero-title {
  font-size: 2.6rem; font-weight: 800; line-height: 1.15; letter-spacing: -0.5px;
  background: linear-gradient(135deg, #ffffff 30%, #a78bfa 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  margin-bottom: 0.5rem;
}
.hero-sub { color: #6b7280; font-size: 1rem; margin-bottom: 2rem; line-height: 1.6; max-width: 600px; }

/* Section label */
.sec-label {
  font-size: 0.7rem; font-weight: 700; color: #6b7280;
  letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.8rem;
}

/* Glass card */
.glass-card {
  background: rgba(17, 20, 37, 0.9); border: 1px solid #1e2240;
  border-radius: 20px; padding: 1.6rem 1.8rem; margin-bottom: 1.1rem;
  backdrop-filter: blur(24px);
}

/* Check rows */
.ck-row { display: flex; align-items: flex-start; gap: 0.6rem; padding: 0.5rem 0; border-bottom: 1px solid #111425; }
.ck-row:last-child { border-bottom: none; }
.ck-icon { font-size: 0.9rem; font-weight: 700; flex-shrink: 0; margin-top: 2px; }
.ck-ok .ck-icon  { color: #34d399; }
.ck-no .ck-icon  { color: #f87171; }
.ck-ok .ck-name  { font-size: 0.86rem; font-weight: 600; color: #34d399; }
.ck-no .ck-name  { font-size: 0.86rem; font-weight: 600; color: #f87171; }
.ck-msg { font-size: 0.76rem; color: #6b7280; margin-top: 2px; }

/* Grade badges */
.grade-pill {
  display: inline-flex; align-items: center; gap: 0.5rem;
  padding: 0.6rem 1.4rem; border-radius: 99px;
  font-size: 1.05rem; font-weight: 700; margin-bottom: 0.8rem;
}
.grade-0 { background: rgba(6,78,59,0.4);  color: #6ee7b7; border: 1.5px solid #059669; }
.grade-1 { background: rgba(30,58,95,0.4); color: #93c5fd; border: 1.5px solid #2563eb; }
.grade-2 { background: rgba(69,26,3,0.5);  color: #fcd34d; border: 1.5px solid #d97706; }
.grade-3 { background: rgba(76,5,25,0.5);  color: #fca5a5; border: 1.5px solid #dc2626; }
.grade-4 { background: rgba(59,7,100,0.5); color: #e879f9; border: 1.5px solid #a21caf; }

/* Stat chips */
.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; margin-top: 1.1rem; }
.stat-chip {
  background: #111425; border: 1px solid #1e2240; border-radius: 10px;
  padding: 0.5rem 0.9rem; display: flex; flex-direction: column; gap: 0.1rem;
}
.stat-val { font-size: 1rem; font-weight: 800; color: #e4e6f1; }
.stat-lbl { font-size: 0.63rem; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; }

/* Block screen */
.blk-card {
  background: rgba(45,16,16,0.85); border: 2px solid #ef4444;
  border-radius: 20px; padding: 2.2rem; text-align: center;
}
.blk-sym   { font-size: 2.8rem; font-weight: 900; color: #ef4444; margin-bottom: 0.8rem; line-height: 1; }
.blk-title { font-size: 1.4rem; font-weight: 700; color: #fca5a5; margin-bottom: 0.5rem; }
.blk-body  { color: #f87171; font-size: 0.9rem; line-height: 1.7; }
.blk-tip   {
  margin-top: 1.1rem; padding: 0.8rem 1rem;
  background: rgba(239,68,68,0.08); border: 1px solid rgba(239,68,68,0.2);
  border-radius: 10px; font-size: 0.8rem; color: #fca5a5; line-height: 1.6;
}

/* Conf rows */
.conf-row { display: flex; align-items: center; gap: 0.7rem; padding: 0.45rem 0; border-bottom: 1px solid #111425; }
.conf-row:last-child { border-bottom: none; }
.conf-sym  {
  width: 26px; height: 26px; border-radius: 7px;
  display: grid; place-items: center; font-size: 0.75rem; font-weight: 700; flex-shrink: 0;
}
.conf-name { font-size: 0.83rem; color: #9ca3af; flex: 1; }
.conf-top .conf-name { color: #e4e6f1; font-weight: 600; }
.conf-bar-t { flex: 2; height: 5px; background: #111425; border-radius: 99px; overflow: hidden; }
.conf-bar-f { height: 100%; border-radius: 99px; }
.conf-pct  { font-size: 0.8rem; font-weight: 600; color: #6b7280; width: 42px; text-align: right; }
.conf-top .conf-pct { color: #e4e6f1; }

/* Footer */
.footer { text-align: center; color: #374151; font-size: 0.76rem; padding: 2rem 0 0.5rem; border-top: 1px solid #111425; margin-top: 3rem; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH  = r"C:\DR_CP\efficientnet_retinal.pth"
CM_PATH     = r"C:\DR_CP\confusion_matrix.png"
NUM_CLASSES = 5
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DR_INFO = {
    0: ("No DR",            "grade-0", "#34d399", "rgba(6,78,59,0.5)",   "◉", "No signs of diabetic retinopathy. Annual routine screening recommended.",         "Routine"),
    1: ("Mild NPDR",        "grade-1", "#60a5fa", "rgba(30,58,95,0.5)",  "◈", "Microaneurysms present. Schedule follow-up every 6–9 months.",                    "Monitor"),
    2: ("Moderate NPDR",    "grade-2", "#fbbf24", "rgba(69,26,3,0.6)",   "◆", "Vessel blockage detected. Ophthalmologist referral recommended within 3 months.", "Refer Soon"),
    3: ("Severe NPDR",      "grade-3", "#f87171", "rgba(76,5,25,0.6)",   "▲", "Significant retinal damage. Urgent specialist referral required.",                "Urgent"),
    4: ("Proliferative DR", "grade-4", "#e879f9", "rgba(59,7,100,0.6)",  "■", "Advanced neovascularisation. Immediate ophthalmologist consultation required.",   "Immediate"),
}
CONF_FG  = ["#34d399","#60a5fa","#fbbf24","#f87171","#e879f9"]
URG_COLS = {0:("#064e3b","#34d399","#065f46"),1:("#1e3a5f","#60a5fa","#1e40af"),2:("#451a03","#fbbf24","#92400e"),3:("#450a0a","#f87171","#991b1b"),4:("#3b0764","#e879f9","#6b21a8")}

# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    m = models.efficientnet_b0(weights=None)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    m.to(DEVICE); m.eval(); return m

preprocess = transforms.Compose([
    transforms.Resize((224,224)), transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

def run_guardrails(pil_image):
    img  = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res  = []
    ok = w>=224 and h>=224
    res.append(("Resolution",       ok, f"{w}&#215;{h} px" if ok else f"{w}&#215;{h} px &mdash; min 224x224"))
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    ok = score>=15.0
    res.append(("Sharpness",        ok, f"Score {score:.1f}" if ok else f"Score {score:.1f} &mdash; too blurry"))
    brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2])
    ok = 40<=brightness<=220
    res.append(("Brightness",       ok, f"Value {brightness:.1f}" if ok else f"Value {brightness:.1f} &mdash; {'too dark' if brightness<40 else 'overexposed'}"))
    cb = float(np.mean([np.mean(c) for c in [gray[0:10,0:10],gray[0:10,w-10:w],gray[h-10:h,0:10],gray[h-10:h,w-10:w]]]))
    ok = cb<=50
    res.append(("Fundus Structure",  ok, f"Corner brightness {cb:.1f}" if ok else f"Corner brightness {cb:.1f} &mdash; no dark border"))
    return res

def predict(model, pil_image):
    t = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probs = torch.softmax(model(t),dim=1)[0].cpu().numpy()
    return int(np.argmax(probs)), probs


def generate_gradcam(model, pil_image, class_idx):
    """
    Generate a Grad-CAM heatmap overlay for the predicted class.
    Hooks into EfficientNet-B0's last convolutional block.
    Returns a PIL Image with heatmap blended onto the original.
    """
    if not GRADCAM_AVAILABLE:
        return None
    try:
        # Target: last convolutional block of EfficientNet-B0
        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)

        # Prepare input tensor
        input_tensor = preprocess(pil_image).unsqueeze(0).to(DEVICE)

        # Target class wrapper
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        targets = [ClassifierOutputTarget(class_idx)]

        # Generate CAM
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

        # Prepare original image as float array [0,1]
        img_resized = pil_image.resize((224, 224))
        rgb_img = np.array(img_resized, dtype=np.float32) / 255.0

        # Overlay heatmap on image
        visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        return Image.fromarray(visualization)
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# Sidebar
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("<div class='sidebar-title'>&#9670; About RetinaAI</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sidebar-sub'>EfficientNet-B0 trained on 115,000+ fundus images to classify Diabetic Retinopathy into 5 severity grades.<br/><br/>
    <strong>Disclaimer:</strong> This is an AI-assisted screening tool. It does not replace a qualified ophthalmologist's diagnosis.</div>""", unsafe_allow_html=True)
    st.divider()
    st.markdown("<div class='sec-label'>DR Severity Grades</div>", unsafe_allow_html=True)
    for g, (name, badge, fill, bg, sym, _, _) in DR_INFO.items():
        st.markdown(f"<span class='grade-pill {badge}'>{sym} {g} &mdash; {name}</span>", unsafe_allow_html=True)
    st.divider()
    st.markdown(f"<div class='sec-label'>System</div>", unsafe_allow_html=True)
    st.markdown(f"Device: `{DEVICE}`", unsafe_allow_html=False)
    if os.path.exists(MODEL_PATH):
        mb = os.path.getsize(MODEL_PATH)/(1024*1024)
        st.markdown(f"Model size: `{mb:.1f} MB`")

# ══════════════════════════════════════════════════════════════════════════════
# Navbar + Hero
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="navbar">
  <div class="nav-brand">
    <div class="nav-logo">R</div>
    <div><div class="nav-title">RetinaAI</div><div class="nav-sub">Diabetic Retinopathy Screening</div></div>
  </div>
  <div class="nav-pill">EfficientNet-B0 &middot; 115K Images &middot; Val Acc 81.3%</div>
</div>
<div class="hero-title">Retinal Disease Diagnosis</div>
<p class="hero-sub">Upload a fundus photograph to screen for Diabetic Retinopathy severity across all 5 grades using AI.</p>
""", unsafe_allow_html=True)

# ── Model check ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error("Model not found. Run train_model.py first.")
    st.stop()
model = load_model()

# ── Upload ─────────────────────────────────────────────────────────────────────
st.markdown("<div class='sec-label'>&#9632; Upload Fundus Image</div>", unsafe_allow_html=True)
uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

if uploaded is None:
    st.info("Upload a retinal fundus photograph above to begin screening.")
    if os.path.exists(CM_PATH):
        st.divider()
        st.markdown("<div class='sec-label'>&#9698; Model Performance &mdash; Confusion Matrix</div>", unsafe_allow_html=True)
        _, c2, _ = st.columns([1,2,1])
        with c2:
            st.image(CM_PATH, use_container_width=True, caption="14,201 test images | Overall Test Accuracy: 72%")
    st.stop()

# ── Process ────────────────────────────────────────────────────────────────────
pil_image    = Image.open(uploaded).convert("RGB")
checks       = run_guardrails(pil_image)
fundus_ok    = next((ok for n,ok,_ in checks if n=="Fundus Structure"), True)
quality_warn = [n for n,ok,_ in checks if not ok and n!="Fundus Structure"]

col_img, col_res = st.columns([1, 1.4], gap="large")

# ── LEFT: Image ────────────────────────────────────────────────────────────────
with col_img:
    st.markdown("<div class='sec-label'>&#9670; Uploaded Image</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.image(pil_image, use_container_width=True, caption=uploaded.name)
    st.markdown("</div>", unsafe_allow_html=True)

# ── RIGHT: Checks + Results ────────────────────────────────────────────────────
with col_res:
    # Quality checks
    st.markdown("<div class='sec-label'>&#9670; Image Quality Checks</div>", unsafe_allow_html=True)
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    for name, ok, msg in checks:
        sym = "&#10003;" if ok else "&#10007;"
        cls = "ck-ok" if ok else "ck-no"
        st.markdown(f"""
        <div class="ck-row {cls}">
          <div class="ck-icon">{sym}</div>
          <div>
            <div class="ck-name">{name}</div>
            <div class="ck-msg">{msg}</div>
          </div>
        </div>""", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()

    # Fundus block
    if not fundus_ok:
        st.markdown("""
        <div class="glass-card">
          <div class="blk-card">
            <div class="blk-sym">&#8856;</div>
            <div class="blk-title">Inappropriate Image</div>
            <div class="blk-body">
              This does not appear to be a valid <strong>retinal fundus photograph</strong>.<br/>
              Please upload a proper fundus image of the eye.
            </div>
            <div class="blk-tip">
              &#9432; Valid fundus images have a dark circular border around the retina.
              Photos of people, animals, or other objects cannot be processed.
            </div>
          </div>
        </div>""", unsafe_allow_html=True)
    else:
        if quality_warn:
            st.warning(f"[ ! ] Quality issues: {', '.join(quality_warn)}. Results may be less reliable.")

        # Prediction
        st.markdown("<div class='sec-label'>&#9670; Diagnosis Result</div>", unsafe_allow_html=True)
        with st.spinner("Running AI inference..."):
            grade, probs = predict(model, pil_image)

        name, badge_cls, fill, bg, sym, advice, urg_lbl = DR_INFO[grade]
        conf = float(probs[grade]) * 100
        urg_bg, urg_fg, urg_bdr = URG_COLS[grade]

        st.markdown(f"""
        <div class="glass-card">
          <span class="grade-pill {badge_cls}">{sym} Grade {grade} &mdash; {name}</span>
          <p style='font-size:.9rem;color:#9ca3af;line-height:1.65;margin-bottom:.5rem;'>{advice}</p>
          <span style='background:{urg_bg};border:1px solid {urg_bdr};color:{urg_fg};border-radius:8px;
                       padding:.28rem .75rem;font-size:.75rem;font-weight:700;'>{urg_lbl}</span>
          <div class="stat-row">
            <div class="stat-chip"><span class="stat-val" style="color:{fill}">{conf:.1f}%</span><span class="stat-lbl">Confidence</span></div>
            <div class="stat-chip"><span class="stat-val">{grade}/4</span><span class="stat-lbl">Grade</span></div>
            <div class="stat-chip"><span class="stat-val" style="font-size:.82rem">EfficientNet-B0</span><span class="stat-lbl">Model</span></div>
            <div class="stat-chip"><span class="stat-val" style="font-size:.85rem">{'GPU' if 'cuda' in str(DEVICE) else 'CPU'}</span><span class="stat-lbl">Device</span></div>
          </div>
        </div>""", unsafe_allow_html=True)

        # Confidence breakdown
        st.markdown("<div class='sec-label'>&#9698; Confidence Breakdown</div>", unsafe_allow_html=True)
        st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
        for i, (n2, _, fi, cbg, s2, _, _) in DR_INFO.items():
            pct = float(probs[i])*100
            top = "conf-top" if i==grade else ""
            st.markdown(f"""
            <div class="conf-row {top}">
              <div class="conf-sym" style="background:{cbg};color:{fi};">{s2}</div>
              <div class="conf-name">{n2}</div>
              <div class="conf-bar-t"><div class="conf-bar-f" style="width:{pct:.1f}%;background:{fi};"></div></div>
              <div class="conf-pct">{pct:.1f}%</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # Grad-CAM Section
        if GRADCAM_AVAILABLE:
            st.markdown("<div class='sec-label'>&#9670; Grad-CAM — Explainability Heatmap</div>", unsafe_allow_html=True)
            with st.spinner("Generating Grad-CAM heatmap..."):
                gradcam_img = generate_gradcam(model, pil_image, grade)
            if gradcam_img is not None:
                cam_col1, cam_col2 = st.columns(2, gap="medium")
                with cam_col1:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.image(pil_image.resize((224,224)), use_container_width=True, caption="Original Fundus Image")
                    st.markdown("</div>", unsafe_allow_html=True)
                with cam_col2:
                    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
                    st.image(gradcam_img, use_container_width=True, caption=f"Grad-CAM — Grade {grade} ({name}) activation regions")
                    st.markdown("</div>", unsafe_allow_html=True)
                st.markdown("""
                <div class='glass-card' style='font-size:.8rem;color:#6b7280;line-height:1.7;'>
                  <div class='sec-label'>&#9632; How to read the heatmap</div>
                  <span style='color:#ef4444;font-weight:700;'>&#9632; Red/Warm</span> = regions the AI focused on most for this prediction (highest activation)<br/>
                  <span style='color:#3b82f6;font-weight:700;'>&#9632; Blue/Cool</span> = regions that had little influence on the prediction<br/><br/>
                  In diabetic retinopathy, important regions typically include the <strong>optic disc</strong>, <strong>macula</strong>, and areas with <strong>microaneurysms or haemorrhages</strong>.
                </div>""", unsafe_allow_html=True)
            else:
                st.info("Grad-CAM visualization could not be generated for this image.")
        else:
            st.info("Install `grad-cam` library to enable Grad-CAM explainability: `pip install grad-cam`")

        # Clinical note
        st.markdown(f"""
        <div class="glass-card" style="font-size:.83rem;color:#6b7280;line-height:1.7;">
          <div class="sec-label">&#9632; Clinical Guidance</div>
          {advice}<br/><br/>
          <span style="color:#374151"><strong>&#9650; Disclaimer:</strong>
          RetinaAI is an AI-assisted screening tool intended to support &mdash; not replace &mdash;
          qualified ophthalmological diagnosis. Always consult a certified medical professional.</span>
        </div>""", unsafe_allow_html=True)

# ── Confusion Matrix ───────────────────────────────────────────────────────────
if os.path.exists(CM_PATH):
    st.divider()
    st.markdown("<div class='sec-label'>&#9698; Model Performance &mdash; Confusion Matrix</div>", unsafe_allow_html=True)
    _, c2, _ = st.columns([1,2,1])
    with c2:
        st.image(CM_PATH, use_container_width=True, caption="14,201 test images | Overall Test Accuracy: 72%")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  RetinaAI &middot; EfficientNet-B0 &middot; 115,241 training images &middot; Val Acc 81.3%<br/>
  <span style="color:#1f2937">For research and educational purposes only. Not a substitute for professional medical advice.</span>
</div>""", unsafe_allow_html=True)
