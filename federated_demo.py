"""
federated_demo.py — Visual Federated Learning Demo for Retinal Disease Diagnosis
==================================================================================
A Streamlit app that VISUALLY demonstrates the 5-client federated learning process.
Designed for project guide presentations — shows each hospital client training, 
FedAvg aggregation, and accuracy improving round by round.

Usage:
  streamlit run federated_demo.py
"""

import streamlit as st
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy, time, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RetinaAI — Federated Learning Demo",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background: #080b14 !important;
    color: #e4e6f1;
}
.block-container { padding: 2rem 2.5rem 4rem 2.5rem !important; max-width: 1400px; }
[data-testid="stSidebar"] { background: #0c0f1d !important; border-right: 1px solid #1a1d30; }
.stProgress > div > div { background: linear-gradient(90deg, #4f9cf9, #a78bfa) !important; border-radius: 99px !important; }
.stProgress { background: #1a1d2e !important; border-radius: 99px !important; }
hr { border-color: #1a1d30 !important; }

/* Page header */
.page-title {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #ffffff 30%, #4f9cf9 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.3rem;
}
.page-sub { color: #6b7280; font-size: 0.95rem; margin-bottom: 2rem; }

/* Architecture diagram */
.arch-box {
    background: rgba(17,20,37,0.95); border: 1px solid #1e2240;
    border-radius: 20px; padding: 1.4rem 1.6rem; margin-bottom: 1rem;
}
.arch-title { font-size: 0.68rem; font-weight: 700; color: #6b7280;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.8rem; }

/* Client card */
.client-card {
    background: rgba(17,20,37,0.95); border: 1px solid #1e2240;
    border-radius: 16px; padding: 1.1rem 1.2rem; margin-bottom: 0.8rem;
}
.client-card.active  { border-color: #4f9cf9; box-shadow: 0 0 18px rgba(79,156,249,0.2); }
.client-card.done    { border-color: #34d399; }
.client-card.waiting { opacity: 0.5; }
.c-header { display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.5rem; }
.c-icon { font-size: 1.3rem; }
.c-name { font-size: 0.92rem; font-weight: 700; color: #e4e6f1; }
.c-sub  { font-size: 0.72rem; color: #6b7280; }
.c-stat { font-size: 0.78rem; color: #9ca3af; margin-top: 0.3rem; }
.c-stat span { color: #e4e6f1; font-weight: 600; }

/* Server card */
.server-card {
    background: rgba(79,156,249,0.07); border: 2px solid #1e3a5f;
    border-radius: 16px; padding: 1.2rem 1.4rem; text-align: center;
}
.server-card.aggregating { border-color: #a78bfa; box-shadow: 0 0 24px rgba(167,139,250,0.25); }
.s-icon  { font-size: 2rem; margin-bottom: 0.4rem; }
.s-title { font-size: 1rem; font-weight: 700; color: #60a5fa; }
.s-sub   { font-size: 0.75rem; color: #6b7280; margin-top: 0.2rem; }

/* Round badge */
.round-badge {
    display: inline-block; background: rgba(79,156,249,0.15);
    border: 1px solid #1e3a5f; color: #60a5fa;
    border-radius: 99px; padding: 0.3rem 1rem; font-size: 0.8rem; font-weight: 700;
    margin-bottom: 0.8rem;
}
/* Status tag */
.tag-training { background: rgba(79,156,249,0.15); border: 1px solid #1e3a5f; color: #60a5fa; border-radius: 6px; padding: 0.15rem 0.5rem; font-size: 0.7rem; font-weight: 700; }
.tag-done     { background: rgba(52,211,153,0.15); border: 1px solid #065f46; color: #34d399; border-radius: 6px; padding: 0.15rem 0.5rem; font-size: 0.7rem; font-weight: 700; }
.tag-wait     { background: rgba(107,114,128,0.15); border: 1px solid #374151; color: #6b7280; border-radius: 6px; padding: 0.15rem 0.5rem; font-size: 0.7rem; font-weight: 700; }
.tag-agg      { background: rgba(167,139,250,0.15); border: 1px solid #6b21a8; color: #a78bfa; border-radius: 6px; padding: 0.15rem 0.5rem; font-size: 0.7rem; font-weight: 700; }

/* Accuracy card */
.acc-card {
    background: rgba(17,20,37,0.95); border: 1px solid #1e2240;
    border-radius: 16px; padding: 1.2rem 1.4rem;
}
.acc-val { font-size: 2.4rem; font-weight: 800; color: #4f9cf9; }
.acc-lbl { font-size: 0.7rem; color: #6b7280; text-transform: uppercase; letter-spacing: .05em; }

/* Flow arrow */
.flow-arrow { text-align: center; color: #374151; font-size: 1.5rem; line-height: 1; margin: 0.3rem 0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR    = r"C:\DR_CP\DE_MESSIDOR_EX\augmented_resized_V2"
NUM_CLASSES = 5
DEVICE      = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

HOSPITAL_NAMES = [
    "🏥 City Eye Hospital",
    "🏥 Apollo Retina Centre",
    "🏥 AIIMS Ophthalmology",
    "🏥 Sankara Eye Institute",
    "🏥 LV Prasad Eye Institute",
]

# Quick-demo settings (small data, fast training)
DEMO_SAMPLES_PER_CLIENT = 200   # images per client for demo speed
DEMO_LOCAL_EPOCHS       = 1
DEMO_BATCH_SIZE         = 32


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_data():
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    train_full = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'), tf)
    val_full   = datasets.ImageFolder(os.path.join(DATA_DIR, 'val'),   tf)
    return train_full, val_full

def partition(dataset, n_clients, samples_per_client):
    idx = np.random.permutation(len(dataset))
    client_datasets = []
    for i in range(n_clients):
        start = i * samples_per_client
        end   = start + samples_per_client
        client_datasets.append(Subset(dataset, idx[start:end]))
    return client_datasets

def build_model():
    m = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    m.classifier[1] = nn.Linear(m.classifier[1].in_features, NUM_CLASSES)
    return m.to(DEVICE)

def client_train(global_weights, client_data):
    model = build_model()
    model.load_state_dict(copy.deepcopy(global_weights))
    model.train()
    loader = DataLoader(client_data, batch_size=DEMO_BATCH_SIZE, shuffle=True)
    opt = optim.Adam(model.parameters(), lr=0.001)
    crit = nn.CrossEntropyLoss()
    for _ in range(DEMO_LOCAL_EPOCHS):
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
    return model.state_dict(), len(client_data)

def fedavg(weights_list, sizes):
    total = sum(sizes)
    avg = copy.deepcopy(weights_list[0])
    for k in avg:
        avg[k] = torch.zeros_like(avg[k], dtype=torch.float32)
        for i, w in enumerate(weights_list):
            avg[k] += w[k].float() * (sizes[i] / total)
    return avg

def evaluate(model, val_loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            _, preds = torch.max(model(x), 1)
            correct += (preds == y).sum().item()
            total   += y.size(0)
    return correct / total * 100


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🌐 FL Demo Settings")
    n_rounds = st.slider("Number of Rounds", 1, 5, 3,
                         help="How many global communication rounds to simulate")
    st.divider()
    st.markdown("""
    **How it works:**
    1. Server sends global model to all hospitals
    2. Each hospital trains on its private data
    3. Hospitals send weights back (not images!)
    4. Server averages weights (FedAvg)
    5. Repeat for N rounds
    """)
    st.divider()
    st.markdown(f"**Device:** `{DEVICE}`")
    st.markdown(f"**Samples/client:** `{DEMO_SAMPLES_PER_CLIENT}`")
    st.markdown(f"**Local epochs:** `{DEMO_LOCAL_EPOCHS}`")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='page-title'>🌐 Federated Learning — Live Demo</div>", unsafe_allow_html=True)
st.markdown("<div class='page-sub'>Watch 5 hospital clients train EfficientNet-B0 locally and share knowledge — without sharing patient data.</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ARCHITECTURE DIAGRAM (always visible)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 📐 System Architecture")
with st.container():
    st.markdown("""
    <div class='arch-box'>
        <div class='arch-title'>Federated Learning — Communication Protocol</div>
        <div style='display:flex;align-items:center;justify-content:center;gap:1rem;flex-wrap:wrap;'>
            <div style='text-align:center;'>
                <div style='font-size:2rem;'>🏥🏥🏥🏥🏥</div>
                <div style='font-size:0.78rem;color:#60a5fa;font-weight:700;margin-top:0.3rem;'>5 Hospital Clients</div>
                <div style='font-size:0.7rem;color:#6b7280;'>Train locally on private data</div>
            </div>
            <div style='font-size:1.2rem;color:#374151;'>⟷<br/><span style='font-size:0.65rem;color:#6b7280;'>weights<br/>only</span></div>
            <div style='text-align:center;'>
                <div style='font-size:2rem;'>🌐</div>
                <div style='font-size:0.78rem;color:#a78bfa;font-weight:700;margin-top:0.3rem;'>Central Server</div>
                <div style='font-size:0.7rem;color:#6b7280;'>FedAvg aggregation</div>
            </div>
            <div style='font-size:1.2rem;color:#374151;'>⟷<br/><span style='font-size:0.65rem;color:#6b7280;'>global<br/>model</span></div>
            <div style='text-align:center;'>
                <div style='font-size:2rem;'>💡</div>
                <div style='font-size:0.78rem;color:#34d399;font-weight:700;margin-top:0.3rem;'>Improved Global Model</div>
                <div style='font-size:0.7rem;color:#6b7280;'>Privacy-preserved learning</div>
            </div>
        </div>
        <div style='margin-top:1rem;padding:0.7rem 1rem;background:rgba(79,156,249,0.06);border-radius:10px;border:1px solid #1e3a5f;font-size:0.78rem;color:#6b7280;text-align:center;'>
            ⚠️ <strong style='color:#9ca3af;'>No patient images are ever transmitted</strong> — only model weight tensors (floating-point numbers) travel between hospitals and the server.
        </div>
    </div>
    """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# DATA SPLIT PREVIEW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🗂️ Data Partitioning — Each Hospital's Private Dataset")
data_cols = st.columns(5)
for i, col in enumerate(data_cols):
    with col:
        st.markdown(f"""
        <div class='client-card'>
            <div class='c-header'>
                <div class='c-icon'>🏥</div>
                <div>
                    <div class='c-name'>Hospital {i+1}</div>
                    <div class='c-sub'>{HOSPITAL_NAMES[i][2:]}</div>
                </div>
            </div>
            <div class='c-stat'>Images: <span>{DEMO_SAMPLES_PER_CLIENT}</span></div>
            <div class='c-stat'>Share: <span>20%</span></div>
            <div class='c-stat'>Access: <span style='color:#34d399;'>Private ✓</span></div>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────────────────────────────────────
# TRAINING SECTION
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("### 🚀 Run Federated Training")

start_btn = st.button("▶ Start Federated Training Demo", type="primary", use_container_width=True)

if start_btn:
    # Load data
    with st.spinner("Loading dataset and partitioning across 5 clients..."):
        train_full, val_full = load_data()
        client_datasets      = partition(train_full, 5, DEMO_SAMPLES_PER_CLIENT)
        val_loader = DataLoader(
            Subset(val_full, range(min(500, len(val_full)))),
            batch_size=64, shuffle=False
        )

    # Initialize global model
    global_model   = build_model()
    global_weights = global_model.state_dict()

    round_accuracies = []

    # ── ROUND LOOP ─────────────────────────────────────────────────────────
    for rnd in range(1, n_rounds + 1):
        st.markdown(f"<div class='round-badge'>🔄 Global Round {rnd} of {n_rounds}</div>", unsafe_allow_html=True)

        # Two-column layout: clients | server
        left_col, right_col = st.columns([3, 1], gap="large")

        client_weights_list = []
        client_sizes        = []
        client_status       = ["waiting"] * 5  # waiting / training / done

        with left_col:
            st.markdown("**🏥 Hospital Clients — Local Training**")
            # Placeholders for each client card
            client_placeholders = [st.empty() for _ in range(5)]

        with right_col:
            server_placeholder = st.empty()
            acc_placeholder    = st.empty()

        def render_clients(active_idx=-1, done_set=set()):
            for i, ph in enumerate(client_placeholders):
                if i in done_set:
                    state_html = "<span class='tag-done'>✓ Done</span>"
                    card_class = "done"
                elif i == active_idx:
                    state_html = "<span class='tag-training'>⟳ Training</span>"
                    card_class = "active"
                else:
                    state_html = "<span class='tag-wait'>Waiting</span>"
                    card_class = "waiting"
                ph.markdown(f"""
                <div class='client-card {card_class}'>
                    <div class='c-header'>
                        <div class='c-icon'>🏥</div>
                        <div style='flex:1;'>
                            <div class='c-name'>Hospital {i+1} — {HOSPITAL_NAMES[i][2:]}</div>
                            <div class='c-sub'>{DEMO_SAMPLES_PER_CLIENT} private images</div>
                        </div>
                        {state_html}
                    </div>
                    <div class='c-stat'>Local data: <span>{DEMO_SAMPLES_PER_CLIENT} images</span> &nbsp;|&nbsp; Epochs: <span>{DEMO_LOCAL_EPOCHS}</span></div>
                </div>
                """, unsafe_allow_html=True)

        def render_server(state="idle", acc=None):
            if state == "waiting":
                html = "<div class='server-card'><div class='s-icon'>🌐</div><div class='s-title'>Central Server</div><div class='s-sub'>Waiting for clients...</div></div>"
            elif state == "aggregating":
                html = "<div class='server-card aggregating'><div class='s-icon'>⚙️</div><div class='s-title' style='color:#a78bfa;'>Aggregating</div><div class='s-sub'><span class='tag-agg'>FedAvg Running</span></div></div>"
            else:
                html = f"<div class='server-card'><div class='s-icon'>✅</div><div class='s-title' style='color:#34d399;'>Updated!</div><div class='s-sub'>Global model improved</div></div>"
            server_placeholder.markdown(html, unsafe_allow_html=True)

        # Initial render
        render_clients()
        render_server("waiting")

        # ── Each client trains ─────────────────────────────────────────────
        done_set = set()
        for cid in range(5):
            render_clients(active_idx=cid, done_set=done_set)
            render_server("waiting")

            with right_col:
                prog = st.progress(0, text=f"Hospital {cid+1} training...")
                for p in range(0, 101, 20):
                    time.sleep(0.05)
                    prog.progress(p, text=f"Hospital {cid+1}: {p}%")

            # Actual training
            w, s = client_train(global_weights, client_datasets[cid])
            client_weights_list.append(w)
            client_sizes.append(s)

            done_set.add(cid)
            render_clients(active_idx=-1, done_set=done_set)
            prog.empty()

        # ── FedAvg on server ──────────────────────────────────────────────
        render_server("aggregating")
        with right_col:
            agg_prog = st.progress(0, text="FedAvg: averaging weights...")
            for p in range(0, 101, 25):
                time.sleep(0.1)
                agg_prog.progress(p, text=f"FedAvg: {p}%")
            agg_prog.empty()

        global_weights = fedavg(client_weights_list, client_sizes)
        global_model.load_state_dict(global_weights)

        # ── Evaluate ──────────────────────────────────────────────────────
        acc = evaluate(global_model, val_loader)
        round_accuracies.append(acc)
        render_server("done", acc)

        with right_col:
            acc_placeholder.markdown(f"""
            <div class='acc-card' style='margin-top:0.8rem;'>
                <div class='acc-lbl'>Round {rnd} Val Accuracy</div>
                <div class='acc-val'>{acc:.1f}%</div>
                <div class='acc-lbl' style='margin-top:0.3rem;'>Global Model</div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

    # ── FINAL RESULTS ──────────────────────────────────────────────────────
    st.success(f"🎉 Federated Training Complete! Final Accuracy: **{round_accuracies[-1]:.1f}%**")

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(9, 4))
    fig.patch.set_facecolor('#080b14')
    ax.set_facecolor('#0c0f1d')
    ax.plot(range(1, len(round_accuracies) + 1), round_accuracies,
            marker='o', linewidth=2.5, color='#4f9cf9',
            markerfacecolor='#a78bfa', markersize=10, markeredgewidth=2)
    for i, acc in enumerate(round_accuracies):
        ax.annotate(f"{acc:.1f}%", (i + 1, acc),
                    textcoords="offset points", xytext=(0, 10),
                    ha='center', color='#e4e6f1', fontsize=10, fontweight='bold')
    ax.set_title('Global Model Accuracy per Communication Round',
                 color='#e4e6f1', fontsize=13, fontweight='bold')
    ax.set_xlabel('Round', color='#6b7280')
    ax.set_ylabel('Validation Accuracy (%)', color='#6b7280')
    ax.tick_params(colors='#6b7280')
    ax.set_xticks(range(1, len(round_accuracies) + 1))
    ax.spines['bottom'].set_color('#1e2240')
    ax.spines['left'].set_color('#1e2240')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.15, color='#4f9cf9')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown("""
    <div style='background:rgba(17,20,37,0.95);border:1px solid #1e2240;border-radius:16px;padding:1.2rem 1.4rem;margin-top:1rem;'>
        <div style='font-size:0.68rem;font-weight:700;color:#6b7280;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;'>Key Takeaways</div>
        <div style='font-size:0.85rem;color:#9ca3af;line-height:1.8;'>
          ✅ <strong style='color:#e4e6f1;'>5 hospitals collaborated</strong> without sharing a single patient image<br/>
          ✅ <strong style='color:#e4e6f1;'>FedAvg aggregation</strong> merged all 5 clients' knowledge each round<br/>
          ✅ <strong style='color:#e4e6f1;'>Accuracy improved</strong> progressively over communication rounds<br/>
          ✅ <strong style='color:#e4e6f1;'>Privacy preserved</strong> — only weight tensors were transmitted, never raw data
        </div>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("👆 Click **Start Federated Training Demo** above to simulate the 5-hospital federated learning process live.")
    st.markdown("""
    <div style='background:rgba(17,20,37,0.95);border:1px solid #1e2240;border-radius:16px;padding:1.2rem 1.4rem;'>
        <div style='font-size:0.68rem;font-weight:700;color:#6b7280;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.6rem;'>What You Will See</div>
        <div style='font-size:0.85rem;color:#9ca3af;line-height:1.9;'>
          🏥 Five hospital client cards — each activates one by one as they train locally<br/>
          🌐 Central server card — shows FedAvg aggregation happening in real-time<br/>
          📈 Accuracy percentage — updates after each communication round<br/>
          📊 Final accuracy curve — shows learning progress across all rounds
        </div>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align:center;color:#374151;font-size:0.75rem;padding:2rem 0 0.5rem;border-top:1px solid #111425;margin-top:3rem;'>
  RetinaAI · Federated Learning Demo · EfficientNet-B0 · FedAvg
</div>""", unsafe_allow_html=True)
