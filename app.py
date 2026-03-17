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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root {
  --bg-main: #f8fafc;
  --bg-card: #ffffff;
  --border-subtle: #e2e8f0;
  --border-strong: #cbd5e1;
  --text-main: #0f172a;
  --text-muted: #64748b;
  --primary-blue: #2563eb;
  --primary-hover: #1d4ed8;
  --status-ok: #059669;
  --status-warn: #d97706;
  --status-error: #dc2626;
  --shadow-sm: 0 1px 2px 0 rgba(0, 0, 0, 0.05);
  --shadow-md: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}

*, *::before, *::after { box-sizing: border-box; }
html, body, [class*="css"], .stApp {
  font-family: 'Inter', sans-serif !important;
  background: var(--bg-main) !important;
  color: var(--text-main);
}

.block-container {
  padding: 2rem 3rem 4rem 3rem !important;
  max-width: 1400px !important;
}

/* ─── Animations ─────────────────────────────────────────────────────── */
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
.animate-fade { animation: fadeIn 0.4s ease-out forwards; }

/* ─── Sidebar ─────────────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: #ffffff !important;
  border-right: 1px solid var(--border-subtle) !important;
}
.sb-logo { display: flex; align-items: center; gap: 0.8rem; margin-bottom: 2rem; padding-bottom: 1.5rem; border-bottom: 1px solid var(--border-subtle); }
.sb-logo-icon {
  width: 36px; height: 36px; border-radius: 8px;
  background: var(--primary-blue);
  display: grid; place-items: center; font-size: 1.1rem; font-weight: 800; color: #fff;
}
.sb-logo-text { font-size: 1.25rem; font-weight: 800; color: var(--text-main); letter-spacing: -0.02em; }
.sb-section { font-size: 0.65rem; font-weight: 700; color: var(--text-muted); letter-spacing: 0.08em; text-transform: uppercase; margin: 1.5rem 0 0.5rem; }
.sb-disclaimer {
  background: #f1f5f9; border: 1px solid var(--border-subtle);
  border-radius: 8px; padding: 0.8rem 1rem; font-size: 0.75rem; color: var(--text-muted);
  line-height: 1.5; margin-bottom: 2rem;
}
.sb-grade-row {
  display: flex; align-items: center; gap: 0.6rem;
  padding: 0.4rem 0; margin-bottom: 0.2rem;
}
.sb-grade-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.sb-grade-name { font-size: 0.8rem; color: var(--text-main); flex: 1; font-weight: 500; }
.sb-chip {
  font-size: 0.65rem; font-weight: 700; padding: 0.15rem 0.5rem;
  border-radius: 4px; letter-spacing: 0.02em;
}
.sb-sys { font-size: 0.75rem; color: var(--text-muted); padding: 0.25rem 0; }
.sb-sys span { color: var(--text-main); font-weight: 600; font-family: monospace; }

/* ─── Top Brand Bar ──────────────────────────────────────────────────── */
.topbar {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1rem 1.5rem; margin-bottom: 2rem;
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 12px;
  box-shadow: var(--shadow-sm);
}
.topbar-brand { display: flex; align-items: center; gap: 0.8rem; }
.topbar-icon {
  width: 40px; height: 40px; border-radius: 8px; display: grid; place-items: center;
  background: var(--primary-blue);
  font-size: 1.1rem; font-weight: 800; color: #fff;
}
.topbar-title { font-size: 1.2rem; font-weight: 800; color: var(--text-main); }
.topbar-sub   { font-size: 0.75rem; color: var(--text-muted); margin-top: 2px; }
.badge {
  font-size: 0.7rem; font-weight: 600; padding: 0.3rem 0.8rem;
  border-radius: 6px; border: 1px solid var(--border-subtle);
  background: #f8fafc; color: var(--text-main);
}

/* ─── Hero Section ────────────────────────────────────────────────────── */
.hero { margin-bottom: 2.5rem; }
.hero-title {
  font-size: 2.8rem; font-weight: 800; line-height: 1.15;
  color: var(--text-main);
  margin-bottom: 0.5rem; letter-spacing: -0.03em;
}
.hero-sub {
  font-size: 1.05rem; color: var(--text-muted); line-height: 1.6; max-width: 700px;
}

/* ─── Cards & Containers ─────────────────────────────────────────────── */
.card {
  background: var(--bg-card);
  border: 1px solid var(--border-subtle);
  border-radius: 12px; padding: 1.5rem; margin-bottom: 1.2rem;
  box-shadow: var(--shadow-sm);
}
.card-sm { padding: 1rem 1.2rem; border-radius: 8px; }

.sec-lbl {
  font-size: 0.75rem; font-weight: 700; color: var(--text-muted);
  letter-spacing: 0.05em; text-transform: uppercase; margin-bottom: 1rem;
  border-bottom: 1px solid var(--border-subtle); padding-bottom: 0.4rem;
}

/* ─── Grading & Pills ────────────────────────────────────────────────── */
.grade-pill {
  display: inline-flex; align-items: center; gap: 0.6rem;
  padding: 0.6rem 1.2rem; border-radius: 6px;
  font-size: 1.1rem; font-weight: 700; margin-bottom: 1rem;
  border: 1px solid transparent;
}
.grade-0 { background: #ecfdf5; color: #065f46; border-color: #a7f3d0; }
.grade-1 { background: #eff6ff; color: #1e40af; border-color: #bfdbfe; }
.grade-2 { background: #fffbeb; color: #92400e; border-color: #fde68a; }
.grade-3 { background: #fef2f2; color: #991b1b; border-color: #fecaca; }
.grade-4 { background: #faf5ff; color: #6b21a8; border-color: #e9d5ff; }

.advice-text { font-size: 0.9rem; color: var(--text-main); line-height: 1.5; margin-bottom: 1.2rem; }

/* ─── Diagnostic Metrics ──────────────────────────────────────────────── */
.match-row {
  display: flex; align-items: center; justify-content: space-between;
  padding: 1rem 1.2rem; background: #f8fafc;
  border-radius: 8px; border: 1px solid var(--border-subtle);
}
.match-val { font-size: 1.5rem; font-weight: 800; color: var(--text-main); }
.urg-badge {
  font-size: 0.72rem; font-weight: 700; padding: 0.3rem 0.8rem;
  border-radius: 4px; text-transform: uppercase;
}

.dme-card {
  display: flex; align-items: flex-start; gap: 1rem;
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: 8px; padding: 1.2rem; margin-bottom: 1.2rem;
  box-shadow: var(--shadow-sm);
}
.dme-dot { width: 12px; height: 12px; border-radius: 50%; margin-top: 4px; }
.dme-label { font-size: 1rem; font-weight: 700; color: var(--text-main); margin-bottom: 4px; }
.dme-note  { font-size: 0.85rem; color: var(--text-muted); line-height: 1.4; }

.sev-row { display: grid; grid-template-columns: 120px 1fr 45px; align-items: center; gap: 1rem; padding: 0.5rem 0; }
.sev-label { font-size: 0.8rem; color: var(--text-muted); font-weight: 500; }
.sev-label.active { color: var(--text-main); font-weight: 700; }
.sev-track { height: 8px; background: #e2e8f0; border-radius: 4px; overflow: hidden; }
.sev-fill { height: 100%; border-radius: 4px; transition: width 0.8s ease; }
.sev-pct { font-size: 0.8rem; color: var(--text-muted); text-align: right; font-weight: 600; }
.sev-pct.active { color: var(--text-main); font-weight: 800; }

/* ─── File Uploader Override ─────────────────────────────────────────── */
.stFileUploader section {
  background: #ffffff !important;
  border: 1px dashed var(--border-strong) !important;
  border-radius: 8px !important;
}
.stFileUploader section:hover { border-color: var(--primary-blue) !important; background: #f8fafc !important; }
.stFileUploader [data-testid="stFileUploaderDropzone"] button {
  background-color: var(--primary-blue) !important;
  color: #ffffff !important;
  border: none !important;
  padding: 0.5rem 1rem !important;
  border-radius: 6px !important;
  font-weight: 600 !important;
}
.stFileUploader [data-testid="stFileUploaderDropzone"] button:hover {
  background-color: var(--primary-hover) !important;
}

/* ─── Status Board ───────────────────────────────────────────────────── */
.int-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem; }
.int-cell {
  background: var(--bg-card); border: 1px solid var(--border-subtle);
  border-radius: 8px; padding: 1.2rem; box-shadow: var(--shadow-sm);
  border-top: 3px solid var(--border-subtle);
}
.int-cell.ok { border-top-color: var(--status-ok); }
.int-cell.fail { border-top-color: var(--status-error); }
.int-name { font-size: 0.7rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 0.5rem; }
.int-status { font-size: 1rem; font-weight: 800; margin-bottom: 0.3rem; color: var(--text-main); }
.int-val { font-size: 0.8rem; color: var(--text-muted); }

/* ─── Tabs ───────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] { background: transparent !important; gap: 0; border-bottom: 1px solid var(--border-subtle) !important; }
.stTabs [data-baseweb="tab"] {
  background: transparent !important; border: none !important;
  color: var(--text-muted) !important; font-size: 0.85rem; font-weight: 500;
  padding: 0.8rem 1.2rem !important; border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] { color: var(--primary-blue) !important; border-bottom: 2px solid var(--primary-blue) !important; font-weight: 600; }
div[data-testid="stImage"] img { border-radius: 8px !important; border: 1px solid var(--border-subtle); }

/* ─── Legend ─────────────────────────────────────────────────────────────── */
.legend-row { display: flex; flex-wrap: wrap; gap: 0.8rem 1.5rem; margin-top: 0.8rem; }
.legend-item { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; color: var(--text-main); }
.legend-dot { width: 10px; height: 10px; border-radius: 2px; }

/* ─── Footer ─────────────────────────────────────────────────────────── */
.footer {
  text-align: center; font-size: 0.8rem; color: var(--text-muted);
  padding: 3rem 0; border-top: 1px solid var(--border-subtle); margin-top: 4rem;
}
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
MULTI_MODEL_PATH = r"C:\DR_CP\efficientnet_multitask.pth"
UNET_MODEL_PATH  = r"C:\DR_CP\unet_segmentation.pth"
IDRID_MODEL_PATH = r"C:\DR_CP\efficientnet_retinal_idrid.pth"
BASE_MODEL_PATH  = r"C:\DR_CP\efficientnet_retinal.pth"

if os.path.exists(MULTI_MODEL_PATH):
    MODEL_PATH  = MULTI_MODEL_PATH
    MODEL_LABEL = "Multi-Task (DR+DME+Loc)"
    HAS_MULTI   = True
else:
    MODEL_PATH  = IDRID_MODEL_PATH if os.path.exists(IDRID_MODEL_PATH) else BASE_MODEL_PATH
    MODEL_LABEL = "MESSIDOR + IDRiD" if os.path.exists(IDRID_MODEL_PATH) else "MESSIDOR"
    HAS_MULTI   = False

HAS_UNET    = os.path.exists(UNET_MODEL_PATH)
# Use the large MESSIDOR model (trained on 100k+ images) for sharp, accurate DR grading.
# The IDRiD model only has ~54 training images and overfits badly on general fundus photos.
HAS_DR_ONLY = HAS_MULTI and os.path.exists(BASE_MODEL_PATH)
CM_PATH  = (r"C:\DR_CP\confusion_matrix_idrid.png"
            if os.path.exists(r"C:\DR_CP\confusion_matrix_idrid.png")
            else r"C:\DR_CP\confusion_matrix.png")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DR_INFO = {
    0: ("No DR",            "grade-0", "#059669", "rgba(5, 150, 105, 0.1)", "◉", "No visible retinal changes. Annual routine screening recommended.", "Routine",   "#d1fae5", "#065f46"),
    1: ("Mild NPDR",        "grade-1", "#2563eb", "rgba(37, 99, 235, 0.1)",  "◈", "Early microaneurysms present. Follow-up every 6–9 months.",           "Monitor",   "#dbeafe", "#1e3a8a"),
    2: ("Moderate NPDR",    "grade-2", "#d97706", "rgba(217, 119, 6, 0.1)",  "◆", "Vessel blockage detected. Ophthalmologist referral within 3 months.", "Refer Soon","#fef3c7", "#92400e"),
    3: ("Severe NPDR",      "grade-3", "#dc2626", "rgba(220, 38, 38, 0.1)",  "▲", "Significant retinal damage. Urgent specialist referral required.",    "Urgent",    "#fee2e2", "#991b1b"),
    4: ("Proliferative DR", "grade-4", "#7c3aed", "rgba(124, 58, 237, 0.1)", "■", "Advanced neovascularisation. Immediate consultation required.",       "Immediate", "#ede9fe", "#5b21b6"),
}

DME_INFO = {
    0: ("No DME Risk",       "#059669", "#d1fae5", "No clinical signs of macular edema detected."),
    1: ("Moderate DME Risk", "#d97706", "#fef3c7", "Possible macular thickening. Close monitoring recommended."),
    2: ("High DME Risk",     "#dc2626", "#fee2e2", "Significant macular edema risk. Specialist consultation required."),
}

# ── Model Architectures ────────────────────────────────────────────────────────
class MultiTaskEfficientNet(nn.Module):
    def __init__(self, num_dr=5, num_dme=3, num_loc=4):
        super().__init__()
        backbone = models.efficientnet_b0(weights=None)
        self.features = backbone.features
        self.avgpool  = backbone.avgpool
        self.dr_head  = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_dr))
        self.dme_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 128), nn.ReLU(), nn.Dropout(0.2), nn.Linear(128, num_dme))
        self.loc_head = nn.Sequential(nn.Dropout(0.3), nn.Linear(1280, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, num_loc), nn.Sigmoid())
    def forward(self, x):
        f = self.avgpool(self.features(x)).flatten(1)
        return self.dr_head(f), self.dme_head(f), self.loc_head(f)

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)

class LightUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=4):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 32); self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128);   self.enc4 = DoubleConv(128, 256)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(256, 512)
        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2); self.dec4 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2); self.dec3 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64,  2, stride=2); self.dec2 = DoubleConv(128, 64)
        self.up1 = nn.ConvTranspose2d(64,  32,  2, stride=2); self.dec1 = DoubleConv(64, 32)
        self.final = nn.Conv2d(32, out_ch, 1)
    def forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2)); e4 = self.enc4(self.pool(e3))
        b  = self.bottleneck(self.pool(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        return self.final(d1)

# ── Loading ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    # ── Multi-task model (DR + DME + Loc) ──
    if HAS_MULTI:
        m = MultiTaskEfficientNet().to(DEVICE)
    else:
        m = models.efficientnet_b0(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, 5)
        m = m.to(DEVICE)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    m.eval()

    # ── Single-task DR model: MESSIDOR (100k+ images → much better generalization) ──
    dr_model = None
    if HAS_DR_ONLY:
        dr_model = models.efficientnet_b0(weights=None)
        dr_model.classifier[1] = nn.Linear(dr_model.classifier[1].in_features, 5)
        dr_model = dr_model.to(DEVICE)
        dr_model.load_state_dict(torch.load(BASE_MODEL_PATH, map_location=DEVICE, weights_only=True))
        dr_model.eval()

    # ── U-Net segmentation model ──
    unet = None
    if HAS_UNET:
        unet = LightUNet().to(DEVICE)
        unet.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=DEVICE, weights_only=True))
        unet.eval()
    return m, dr_model, unet

preprocess = transforms.Compose([
    transforms.Resize((224, 224)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

seg_preprocess = transforms.Compose([
    transforms.Resize((256, 256)), transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ── Inference ──────────────────────────────────────────────────────────────────
def run_guardrails(pil_image):
    img  = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    results = []
    ok = w >= 224 and h >= 224
    results.append(("Resolution",      ok, f"{w}×{h} px" if ok else f"{w}×{h} px (min 224×224)"))
    sc = cv2.Laplacian(gray, cv2.CV_64F).var()
    ok = sc >= 15.0
    results.append(("Sharpness",       ok, f"Score {sc:.1f}" if ok else f"Score {sc:.1f} — too blurry"))
    bri = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2])
    ok  = 40 <= bri <= 220
    results.append(("Brightness",      ok, f"Value {bri:.1f}" if ok else f"Value {bri:.1f} — {'too dark' if bri < 40 else 'overexposed'}"))
    cb  = float(np.mean([np.mean(c) for c in [gray[0:10, 0:10], gray[0:10, w-10:w], gray[h-10:h, 0:10], gray[h-10:h, w-10:w]]]))
    ok  = cb <= 50
    results.append(("Fundus Structure", ok, f"Corner brightness {cb:.1f}" if ok else f"Corner brightness {cb:.1f} — no dark border"))
    return results

def predict(model, dr_model, unet, pil_image):
    t = preprocess(pil_image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        if HAS_MULTI:
            dr_out, dme_out, loc_out = model(t)
            dme_probs = torch.softmax(dme_out, dim=1)[0].cpu().numpy()
            coords    = loc_out[0].cpu().numpy()
            # Use single-task model for DR confidence if available (sharper predictions)
            if dr_model is not None:
                dr_probs = torch.softmax(dr_model(t), dim=1)[0].cpu().numpy()
            else:
                dr_probs = torch.softmax(dr_out, dim=1)[0].cpu().numpy()
        else:
            dr_probs  = torch.softmax(model(t), dim=1)[0].cpu().numpy()
            dme_probs, coords = None, None

        seg_mask = None
        if HAS_UNET and unet is not None:
            t_seg    = seg_preprocess(pil_image).unsqueeze(0).to(DEVICE)
            seg_out  = torch.sigmoid(unet(t_seg))[0].cpu().numpy()
            seg_mask = (seg_out > 0.5).astype(np.uint8)
    return dr_probs, dme_probs, coords, seg_mask

def draw_visuals(pil_img, coords, seg_mask):
    img = np.array(pil_img.resize((512, 512)))
    w, h = 512, 512
    if seg_mask is not None:
        seg_colors = [(255, 80, 80), (80, 80, 255), (255, 230, 50), (220, 80, 220)]
        for i in range(min(4, seg_mask.shape[0])):
            mask = cv2.resize(seg_mask[i], (w, h), interpolation=cv2.INTER_NEAREST)
            overlay = img.copy()
            overlay[mask > 0] = seg_colors[i]
            cv2.addWeighted(overlay, 0.38, img, 0.62, 0, img)
    return Image.fromarray(img)

def generate_gradcam(model, pil_image, class_idx):
    if not GRADCAM_AVAILABLE:
        return None
    try:
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
        target_layers = [model.features[-1]]
        cam = GradCAM(model=model, target_layers=target_layers)
        input_tensor  = preprocess(pil_image).unsqueeze(0).to(DEVICE)
        targets       = [ClassifierOutputTarget(class_idx)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        rgb_img       = np.array(pil_image.resize((224, 224)), dtype=np.float32) / 255.0
        return Image.fromarray(show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True))
    except Exception:
        return None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div class='sb-logo'>
      <div class='sb-logo-icon'>R</div>
      <div>
        <div class='sb-logo-text'>RetinaAI</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sb-section'>About</div>", unsafe_allow_html=True)
    st.markdown("""<div class='sb-disclaimer'>
    EfficientNet-B0 trained on 100k+ fundus images. Detects DR severity, DME risk, lesion segmentation, and anatomical landmarks.<br/><br/>
    <strong>⚠ Disclaimer:</strong> This AI tool does not replace clinical diagnosis by a qualified ophthalmologist.
    </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sb-section'>DR Severity Scale</div>", unsafe_allow_html=True)
    GRADE_DOTS = ["#34d399", "#60a5fa", "#fbbf24", "#f87171", "#e879f9"]
    GRADE_URGENCY = ["Routine", "Monitor", "Refer", "Urgent", "Immediate"]
    URGENCY_COLS  = ["#064e3b", "#1e3a8a", "#78350f", "#7f1d1d", "#581c87"]
    for g, (name, _, fill, _, _, _, _, _, _) in DR_INFO.items():
        st.markdown(f"""<div class='sb-grade-row'>
          <div class='sb-grade-dot' style='background:{fill}'></div>
          <div class='sb-grade-num'>{g}</div>
          <div class='sb-grade-name'>{name}</div>
          <div class='sb-chip' style='background:{URGENCY_COLS[g]};color:{fill}'>{GRADE_URGENCY[g]}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div class='sb-section'>System Info</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='sb-sys'>Device: <span>{DEVICE}</span></div>", unsafe_allow_html=True)
    if os.path.exists(MODEL_PATH):
        mb = os.path.getsize(MODEL_PATH) / (1024 * 1024)
        st.markdown(f"<div class='sb-sys'>Model: <span>{mb:.1f} MB</span></div>", unsafe_allow_html=True)
    unet_status = "Loaded" if HAS_UNET else "Not Found"
    unet_col    = "#34d399" if HAS_UNET else "#f87171"
    st.markdown(f"<div class='sb-sys'>U-Net: <span style='color:{unet_col}'>{unet_status}</span></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TOP BAR
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="topbar">
  <div class="topbar-brand">
    <div class="topbar-icon">R</div>
    <div>
      <div class="topbar-title">RetinaAI</div>
      <div class="topbar-sub">Multi-Task Retinal Analysis System</div>
    </div>
  </div>
  <div class="topbar-badges">
    <span class="badge badge-blue">📡 {MODEL_LABEL}</span>
    <span class="badge badge-purple">🧠 4-Task AI Core</span>
    <span class="badge badge-green">⦿ IDRiD Engine</span>
  </div>
</div>
<div class="hero">
  <div class="hero-title">Precision Retinal Analysis</div>
  <p class="hero-sub">Upload a fundus photograph to initiate an advanced multi-task screening for DR severity, DME risk, and anatomical localization with AI explainability.</p>
</div>
""", unsafe_allow_html=True)

# ── Model guard ────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    st.error("⚠ Model weights not found. Please run `train_multitask.py` first.")
    st.stop()

model, dr_model, unet = load_models()

# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD SECTION
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sec-lbl'>Upload Fundus Image</div>", unsafe_allow_html=True)
uploaded = st.file_uploader(
    "Drag and drop a retinal fundus photograph (JPG / PNG)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

if uploaded is None:
    st.info("📂 Upload a retinal fundus photograph to begin AI-powered screening.")
    if os.path.exists(CM_PATH):
        st.divider()
        st.markdown("<div class='sec-lbl'>Model Performance — Confusion Matrix</div>", unsafe_allow_html=True)
        _, c2, _ = st.columns([1, 2, 1])
        with c2:
            st.image(CM_PATH, use_container_width=True,
                     caption="Test accuracy — 14,201 images | Overall: ~72%")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ══════════════════════════════════════════════════════════════════════════════
pil_image = Image.open(uploaded).convert("RGB")
checks    = run_guardrails(pil_image)
fundus_ok = next((ok for n, ok, _ in checks if n == "Fundus Structure"), True)

with st.spinner("Running AI inference…"):
    dr_probs, dme_probs, coords, seg_mask = predict(model, dr_model, unet, pil_image)
    visual_img  = draw_visuals(pil_image, coords, seg_mask)
    grade       = int(np.argmax(dr_probs))
    gradcam_img = generate_gradcam(dr_model if dr_model else model, pil_image, grade)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN LAYOUT  — two columns
# ══════════════════════════════════════════════════════════════════════════════
col_left, col_right = st.columns([1.05, 1], gap="large")

# ─────────────────────────── LEFT COLUMN ─────────────────────────────────────
with col_left:

    # ── Image viewer ──
    st.markdown("<div class='sec-lbl'>Analysis Visualizations</div>", unsafe_allow_html=True)
    with st.container():
        tab1, tab2 = st.tabs(["🔬  Diagnostics (Seg + Loc)", "🔥  AI Explanation (Grad-CAM)"])
        with tab1:
            st.image(visual_img, use_container_width=True,
                     caption="Lesion Segmentation & Anatomical Landmark Localization")
        with tab2:
            if gradcam_img:
                st.image(gradcam_img, use_container_width=True,
                         caption="Gradient-weighted Class Activation Map — AI Attention Regions")
            else:
                st.info("Grad-CAM is not available for this model configuration.")

    # ── Overlay legend ──
    st.markdown("""
    <div class='card card-sm' style='margin-top:0.6rem;'>
      <div class='sec-lbl' style='margin-bottom:0.5rem;'>Lesion Overlay Legend</div>
      <div class='legend-row'>
        <div class='legend-item'><div class='legend-dot' style='background:#ff5050'></div>Microaneurysms</div>
        <div class='legend-item'><div class='legend-dot' style='background:#5050ff'></div>Haemorrhages</div>
        <div class='legend-item'><div class='legend-dot' style='background:#ffe632'></div>Hard Exudates</div>
        <div class='legend-item'><div class='legend-dot' style='background:#dc50dc'></div>Soft Exudates</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────── RIGHT COLUMN ────────────────────────────────────
with col_right:

    if not fundus_ok:
        st.markdown("""
        <div class='card' style='border-color:#7f1d1d;background:rgba(127,29,29,0.15);text-align:center;padding:2rem;'>
          <div style='font-size:2.4rem;margin-bottom:0.6rem;'>⚠</div>
          <div style='font-size:1.1rem;font-weight:800;color:#f87171;margin-bottom:0.5rem;'>Invalid Fundus Image</div>
          <div style='font-size:0.83rem;color:#fca5a5;line-height:1.65;'>
            The uploaded image does not appear to be a valid retinal fundus photograph.
            Please upload a properly captured fundus image with a clear dark border.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        name, badge_cls, fill, bg, sym, advice, urg_lbl, urg_bg, urg_fg = DR_INFO[grade]
        conf = dr_probs[grade] * 100

        # ── DR Grading card ──
        st.markdown("<div class='sec-lbl'>DR Severity Grading</div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class='card animate-fade'>
          <span class='grade-pill {badge_cls}'>{sym} Grade {grade} &mdash; {name}</span>
          <p class='advice-text'>{advice}</p>
          <div class='match-row'>
            <div>
              <div style='font-size:0.65rem;color:var(--text-muted);font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;'>AI Confidence</div>
              <div class='match-val'>{conf:.1f}%</div>
            </div>
            <div style='text-align:right;'>
              <div style='font-size:0.65rem;color:var(--text-muted);font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px;'>Action Priority</div>
              <div class='urg-badge' style='background:{urg_bg};color:{urg_fg};display:inline-block;'>{urg_lbl}</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── DME card ──
        if dme_probs is not None:
            dme_grade = int(np.argmax(dme_probs))
            dme_name, dme_fg, dme_bg, dme_note = DME_INFO[dme_grade]
            dme_conf = dme_probs[dme_grade] * 100
            st.markdown("<div class='sec-lbl'>Macular Edema (DME) Risk</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='dme-card' style='border-left:3px solid {dme_fg};'>
              <div class='dme-dot' style='background:{dme_fg};'></div>
              <div style='flex:1;'>
                <div class='dme-label' style='color:{dme_fg};'>{dme_name}</div>
                <div class='dme-note'>{dme_note}</div>
              </div>
              <div style='text-align:right;'>
                <div style='font-size:0.65rem;color:#4b5563;font-weight:700;letter-spacing:.08em;text-transform:uppercase;margin-bottom:2px;'>Confidence</div>
                <div style='font-size:0.92rem;font-weight:800;color:{dme_fg};'>{dme_conf:.1f}%</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Severity distribution ──
        st.markdown("<div class='sec-lbl'>Severity Distribution</div>", unsafe_allow_html=True)
        st.markdown("<div class='card card-sm' style='padding-top:1.4rem;'>", unsafe_allow_html=True)
        for i, (n2, _, fi, _, s2, _, _, _, _) in DR_INFO.items():
            p      = dr_probs[i] * 100
            active = "active" if i == grade else ""
            st.markdown(f"""
            <div class='sev-row'>
              <div class='sev-label {active}' title='Grade {i}: {n2}'>{s2} {n2}</div>
              <div class='sev-track'>
                <div class='sev-fill' style='width:{p:.1f}%;background:{fi};box-shadow: 0 0 8px {fi}88;'></div>
              </div>
              <div class='sev-pct {active}'>{p:.0f}%</div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# IMAGE INTEGRITY REPORT
# ══════════════════════════════════════════════════════════════════════════════
st.divider()
st.markdown("<div class='sec-lbl'>Image Integrity Report</div>", unsafe_allow_html=True)
int_cells = ""
for check_name, ok, msg in checks:
    css   = "ok" if ok else "fail"
    icon  = "✓ Pass" if ok else "✗ Fail"
    int_cells += f"""
    <div class='int-cell {css}'>
      <div class='int-name'>{check_name}</div>
      <div class='int-status {css}'>{icon}</div>
      <div class='int-val'>{msg}</div>
    </div>"""
st.markdown(f"<div class='int-grid animate-fade'>{int_cells}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class='footer'>
  RetinaAI v2.1 &middot; IDRiD Multi-Task Architecture &middot; DR + DME + Segmentation + Localization<br/>
  <span style='color:#1f2937;'>For clinical research only. Always verify AI findings with a qualified ophthalmologist.</span>
</div>
""", unsafe_allow_html=True)
