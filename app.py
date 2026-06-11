"""
app.py - UHF-ECG Activation Mapper - Streamlit Application
Research & Education Tool for Ventricular Dyssynchrony Analysis
"""
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, os, sys

st.set_page_config(page_title="UHF-ECG Activation Mapper", page_icon="🫀", layout="wide",
                   initial_sidebar_state="expanded")

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main { background-color: #0a0a2e; }
.stApp { background: linear-gradient(135deg, #0a0a2e 0%, #1a1a4e 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #12122e 0%, #1e1e4a 100%); }
h1, h2, h3 { color: #e0e0ff !important; }
.metric-card { background: rgba(255,255,255,0.05); border-radius: 12px;
    padding: 16px; border: 1px solid rgba(255,255,255,0.1); text-align: center; }
.metric-value { font-size: 28px; font-weight: 700; }
.metric-label { font-size: 12px; color: #888; margin-top: 4px; }
.disclaimer { background: rgba(255,50,50,0.1); border: 1px solid #ff4444;
    border-radius: 8px; padding: 12px; margin: 10px 0; color: #ff8888; font-size: 12px; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] { background: rgba(255,255,255,0.05);
    border-radius: 8px; color: white; padding: 8px 16px; }
.stTabs [aria-selected="true"] { background: rgba(100,100,255,0.3); }
</style>
""", unsafe_allow_html=True)

from signal_processing import (load_ecg_csv, preprocess_ecg, detect_qrs,
                                compute_averaged_beat, estimate_qrs_duration,
                                estimate_qrs_onset)
from uhf_mapping import (compute_uhf_envelopes, compute_uhfat, compute_vd,
                          compute_edys, build_uhf_heatmap_matrix)
from nd_ecg import (compute_negative_derivative, compute_ndat,
                     compute_nd_dys, build_nd_heatmap_matrix)
from visualization import (plot_activation_map, plot_ved_scale, plot_ecg_traces,
                            plot_averaged_qrs, plot_combined_maps,
                            plot_summary_metrics, fig_to_bytes)
from report_generator import generate_pdf_report
from sample_data_generator import generate_ecg_signal, save_ecg_csv

def get_leads_order(lead_names):
    order = ['V1','V2','V3','V4','V5','V6','V7','V8']
    return [l for l in order if l in lead_names]

# ─── SIDEBAR ───
with st.sidebar:
    st.markdown("# 🫀 UHF-ECG Mapper")
    st.markdown("##### Ventricular Activation Analysis")
    st.markdown("---")
    
    mode = st.radio("Analysis Mode", ["📊 Digital ECG Signal", "🖼️ ECG Image (Phase 2)"],
                    index=0, help="Mode 2 will be available in a future update")
    if "Image" in mode:
        st.warning("⚠️ ECG Image mode is planned for Phase 2.")
        st.stop()
    
    st.markdown("### 📁 Data Input")
    data_source = st.radio("Source", ["Upload CSV/TXT", "Sample Data"], horizontal=True)
    
    uploaded_file = None
    sample_pattern = None
    
    if data_source == "Upload CSV/TXT":
        uploaded_file = st.file_uploader("Upload ECG file", type=['csv', 'txt'],
                                          help="CSV with columns: time_s, V1, V2, ..., V6")
    else:
        sample_pattern = st.selectbox("Pattern", ["Normal Sinus", "LBBB", "RBBB"])
    
    st.markdown("### ⚙️ Parameters")
    fs_input = st.number_input("Sampling Rate (Hz)", min_value=100, max_value=10000,
                                value=2000, step=100)
    notch_freq = st.selectbox("Powerline (Hz)", [50, 60], index=0)
    
    with st.expander("Advanced Settings"):
        qrs_pre = st.slider("QRS window pre (ms)", 50, 200, 100)
        qrs_post = st.slider("QRS window post (ms)", 100, 300, 200)
        uhf_threshold = st.slider("UHF threshold", 0.3, 0.7, 0.5, 0.05)
        corr_threshold = st.slider("Beat reject threshold", 0.5, 0.95, 0.85, 0.05)
    
    st.markdown("---")
    run_btn = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    st.markdown("---")
    st.markdown('<div class="disclaimer">⚠️ <b>RESEARCH / EDUCATION ONLY</b><br>'
                'Not for clinical diagnosis or treatment decisions. '
                'Does not replace invasive mapping or CE/FDA-approved devices.</div>',
                unsafe_allow_html=True)

# ─── MAIN ───
st.markdown("# 🫀 UHF-ECG Ventricular Activation Mapper")
st.markdown("*Research & Education Tool — Inspired by open-literature UHF-ECG methods*")

if not run_btn:
    # Landing page
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-value" style="color:#4ECDC4">📊</div>
        <div class="metric-label">UHF-ECG<br>Activation Map</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-value" style="color:#FFE66D">📈</div>
        <div class="metric-label">ND-ECG<br>Epicardial Map</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="metric-card">
        <div class="metric-value" style="color:#FF6B6B">🔬</div>
        <div class="metric-label">e-DYS / nd-DYS<br>Dyssynchrony</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("---")
    st.info("👈 Upload an ECG file or select sample data, then click **Run Analysis**")
    st.stop()

# ─── ANALYSIS PIPELINE ───
with st.spinner("Loading ECG data..."):
    try:
        if data_source == "Upload CSV/TXT" and uploaded_file is not None:
            ecg_data = load_ecg_csv(uploaded_file, sampling_rate=fs_input)
        elif sample_pattern:
            pat_map = {"Normal Sinus": "normal", "LBBB": "lbbb", "RBBB": "rbbb"}
            synth = generate_ecg_signal(fs=fs_input, duration_s=10,
                                         pattern=pat_map[sample_pattern])
            ecg_data = {
                'signals': synth['signals'], 'fs': float(fs_input),
                'time': synth['time_s'], 'duration_s': 10.0,
                'n_samples': len(synth['time_s']),
                'lead_names': list(synth['signals'].keys())
            }
        else:
            st.error("Please upload a file or select sample data.")
            st.stop()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

fs = ecg_data['fs']
leads_order = get_leads_order(ecg_data['lead_names'])

if not leads_order:
    st.error("No precordial leads (V1-V8) found in data.")
    st.stop()

# Quality warnings
if fs < 500:
    st.error(f"⚠️ Sampling rate ({fs:.0f} Hz) is too low for meaningful analysis.")
    st.stop()

uhf_possible = fs >= 1000
if not uhf_possible:
    st.warning(f"⚠️ Sampling rate ({fs:.0f} Hz) < 1000 Hz. "
               "UHF-ECG analysis disabled. Only ND-ECG available.")

# Preprocessing
with st.spinner("Preprocessing signals..."):
    signals_clean = preprocess_ecg(ecg_data['signals'], fs, notch_freq=notch_freq)

# QRS Detection
with st.spinner("Detecting QRS complexes..."):
    qrs_info = detect_qrs(signals_clean, fs)
    r_peaks = qrs_info['r_peaks']

if qrs_info['n_beats'] < 3:
    st.error(f"Only {qrs_info['n_beats']} beats detected. Need at least 3.")
    st.stop()

# Beat Averaging
with st.spinner("Averaging beats..."):
    avg_result = compute_averaged_beat(signals_clean, r_peaks, fs,
                                        window_ms=(-qrs_pre, qrs_post))
    avg_beats = avg_result['averaged_beats']
    time_ms = avg_result['time_axis_ms']

# QRS Duration
qrs_dur_info = estimate_qrs_duration(avg_beats, time_ms, fs)

# QRS Onset (UHFAT/NDAT için ortak referans)
qrs_onset_info = estimate_qrs_onset(avg_beats, time_ms, fs)
qrs_onset_ms = qrs_onset_info['qrs_onset_ms']

# ND-ECG Analysis
with st.spinner("Computing ND-ECG..."):
    nd_signals = compute_negative_derivative(avg_beats, fs)
    ndat_values = compute_ndat(nd_signals, time_ms, qrs_onset_ms=qrs_onset_ms)
    nd_dys_info = compute_nd_dys(ndat_values, leads_order=leads_order)
    nd_matrix, nd_leads = build_nd_heatmap_matrix(nd_signals, leads_order)

# UHF Analysis
uhf_envelopes = None
uhfat_values = None
edys_info = {'e_dys': 0, 'earliest_lead': '', 'latest_lead': ''}
vd_uhf = {}
uhf_matrix = None

if uhf_possible:
    with st.spinner("Computing UHF-ECG envelopes..."):
        uhf_envelopes = compute_uhf_envelopes(avg_beats, fs, leads_order)
        if uhf_envelopes:
            uhfat_values = compute_uhfat(uhf_envelopes, time_ms, threshold=uhf_threshold,
                                         qrs_onset_ms=qrs_onset_ms)
            vd_uhf = compute_vd(uhf_envelopes, time_ms, threshold=uhf_threshold)
            edys_info = compute_edys(uhfat_values, leads_order=leads_order)
            uhf_matrix, _ = build_uhf_heatmap_matrix(uhf_envelopes, leads_order)

# Compute ND Vd equivalent
vd_nd = {}
for lead in leads_order:
    if lead in nd_signals:
        nd = nd_signals[lead]['normalized']
        nd_pos = np.clip(nd, 0, None)
        mx = np.max(nd_pos)
        if mx > 0:
            mask = nd_pos >= 0.5 * mx
            dt = np.mean(np.diff(time_ms)) if len(time_ms) > 1 else 1
            vd_nd[lead] = float(np.sum(mask) * dt)
        else:
            vd_nd[lead] = 0

# ─── DISPLAY RESULTS ───
st.success(f"✅ Analysis complete — {qrs_info['n_beats']} beats, "
           f"{avg_result['n_beats_used']} used, quality {avg_result['quality_score']:.0%}")

# Metrics row
mc1, mc2, mc3, mc4, mc5, mc6 = st.columns(6)
with mc1:
    st.metric("e-DYS", f"{edys_info['e_dys']:.1f} ms" if uhf_possible else "N/A")
with mc2:
    st.metric("nd-DYS", f"{nd_dys_info['nd_dys']:.1f} ms")
with mc3:
    st.metric("QRSd", f"{qrs_dur_info['qrs_duration_ms']:.0f} ms")
with mc4:
    st.metric("Heart Rate", f"{qrs_info['heart_rate_bpm']:.0f} bpm")
with mc5:
    st.metric("Beats Used", f"{avg_result['n_beats_used']}")
with mc6:
    st.metric("Quality", f"{avg_result['quality_score']:.0%}")

# Tabs
if uhf_possible and uhf_matrix is not None:
    tabs = st.tabs(["🔥 UHF-ECG Map", "📈 ND-ECG Map", "🔄 Combined",
                     "📊 ECG Signal", "📋 Report"])
else:
    tabs = st.tabs(["📈 ND-ECG Map", "📊 ECG Signal", "📋 Report"])

tab_idx = 0

# UHF Map Tab
if uhf_possible and uhf_matrix is not None:
    with tabs[tab_idx]:
        st.markdown("### UHF-ECG Activation Map")
        st.markdown("*Volumetric ventricular depolarization pattern (150–1050 Hz)*")
        
        fig_uhf = plot_activation_map(uhf_matrix, time_ms, leads_order,
                                       activation_times=uhfat_values,
                                       vd_values=vd_uhf,
                                       title='UHF-ECG Activation Map')
        st.pyplot(fig_uhf)
        
        st.markdown("### Ventricular Electrical Dyssynchrony (VED)")
        fig_ved = plot_ved_scale(edys_info['e_dys'])
        st.pyplot(fig_ved)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**UHFAT Values (ms from QRS onset):**")
            if uhfat_values:
                for l in leads_order:
                    v = uhfat_values.get(l, 0)
                    st.markdown(f"- **{l}**: {v:.1f} ms")
        with c2:
            st.markdown("**Local Depolarization Duration Vd (ms):**")
            for l in leads_order:
                v = vd_uhf.get(l, 0)
                st.markdown(f"- **{l}**: {v:.0f} ms")
        
        plt.close(fig_uhf)
        plt.close(fig_ved)
    tab_idx += 1

# ND-ECG Tab
with tabs[tab_idx]:
    st.markdown("### ND-ECG Activation Map")
    st.markdown("*Epicardial activation pattern via negative derivative (-dV/dt)*")
    
    fig_nd = plot_activation_map(nd_matrix, time_ms, leads_order,
                                  activation_times=ndat_values,
                                  vd_values=vd_nd,
                                  title='ND-ECG Activation Map')
    st.pyplot(fig_nd)
    
    if not uhf_possible:
        st.markdown("### Ventricular Electrical Dyssynchrony (VED)")
        fig_ved2 = plot_ved_scale(nd_dys_info['nd_dys'])
        st.pyplot(fig_ved2)
        plt.close(fig_ved2)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**NDAT Values (ms from R-peak):**")
        for l in leads_order:
            v = ndat_values.get(l, 0)
            st.markdown(f"- **{l}**: {v:.1f} ms")
    with c2:
        st.markdown("**nd-DYS Details:**")
        st.markdown(f"- Earliest: **{nd_dys_info['earliest_lead']}** "
                    f"({nd_dys_info.get('earliest_time_ms',0):.1f} ms)")
        st.markdown(f"- Latest: **{nd_dys_info['latest_lead']}** "
                    f"({nd_dys_info.get('latest_time_ms',0):.1f} ms)")
    
    plt.close(fig_nd)
tab_idx += 1

# Combined Tab
if uhf_possible and uhf_matrix is not None:
    with tabs[tab_idx]:
        st.markdown("### Combined UHF-ECG & ND-ECG Comparison")
        fig_comb = plot_combined_maps(uhf_matrix, nd_matrix, time_ms, leads_order,
                                       uhfat=uhfat_values, ndat=ndat_values,
                                       vd_uhf=vd_uhf, vd_nd=vd_nd)
        st.pyplot(fig_comb)
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**e-DYS (UHF):** {edys_info['e_dys']:.1f} ms")
            st.markdown(f"Earliest: {edys_info.get('earliest_lead','')} → "
                        f"Latest: {edys_info.get('latest_lead','')}")
        with c2:
            st.markdown(f"**nd-DYS (ND-ECG):** {nd_dys_info['nd_dys']:.1f} ms")
            st.markdown(f"Earliest: {nd_dys_info['earliest_lead']} → "
                        f"Latest: {nd_dys_info['latest_lead']}")
        plt.close(fig_comb)
    tab_idx += 1

# ECG Signal Tab
with tabs[tab_idx]:
    st.markdown("### ECG Traces")
    # Show 2 seconds of data
    t_show = min(2.0, ecg_data['duration_s'])
    n_show = int(t_show * fs)
    show_sigs = {l: signals_clean[l][:n_show] for l in leads_order if l in signals_clean}
    show_time = ecg_data['time'][:n_show]
    fig_ecg = plot_ecg_traces(show_sigs, show_time, leads_order)
    st.pyplot(fig_ecg)
    plt.close(fig_ecg)
    
    st.markdown("### Averaged QRS Complex")
    fig_avg = plot_averaged_qrs(avg_beats, time_ms, leads_order)
    st.pyplot(fig_avg)
    plt.close(fig_avg)
tab_idx += 1

# Report Tab
with tabs[tab_idx]:
    st.markdown("### 📋 Export Report")
    
    metrics = {
        'e_dys': edys_info['e_dys'] if uhf_possible else 0,
        'nd_dys': nd_dys_info['nd_dys'],
        'qrs_dur': qrs_dur_info['qrs_duration_ms'],
        'n_beats': avg_result['n_beats_used'],
        'quality': avg_result['quality_score'],
        'fs': fs
    }
    
    fig_sum = plot_summary_metrics(**metrics)
    st.pyplot(fig_sum)
    
    st.markdown("---")
    
    # Generate downloadable figures
    figs_for_report = {}
    
    if uhf_possible and uhf_matrix is not None:
        f1 = plot_activation_map(uhf_matrix, time_ms, leads_order,
                                  uhfat_values, vd_uhf, 'UHF-ECG Activation Map')
        figs_for_report['UHF-ECG Activation Map'] = f1
    
    f2 = plot_activation_map(nd_matrix, time_ms, leads_order,
                              ndat_values, vd_nd, 'ND-ECG Activation Map')
    figs_for_report['ND-ECG Activation Map'] = f2
    
    f3 = plot_averaged_qrs(avg_beats, time_ms, leads_order)
    figs_for_report['Averaged QRS'] = f3
    
    c1, c2 = st.columns(2)
    with c1:
        # PDF
        try:
            pdf_buf = io.BytesIO()
            generate_pdf_report(figs_for_report, metrics, pdf_buf)
            pdf_buf.seek(0)
            st.download_button("📄 Download PDF Report", pdf_buf.getvalue(),
                              "uhf_ecg_report.pdf", "application/pdf",
                              use_container_width=True)
        except Exception as e:
            st.error(f"PDF generation error: {e}")
    
    with c2:
        # PNG of main map
        main_fig = list(figs_for_report.values())[0] if figs_for_report else None
        if main_fig:
            png_bytes = fig_to_bytes(main_fig, 'png', 200)
            st.download_button("🖼️ Download PNG Map", png_bytes,
                              "activation_map.png", "image/png",
                              use_container_width=True)
    
    for f in figs_for_report.values():
        plt.close(f)
    plt.close(fig_sum)
    
    st.markdown("---")
    st.markdown('<div class="disclaimer">⚠️ <b>RESEARCH / EDUCATION USE ONLY</b><br>'
                'These results do not replace invasive electrophysiology mapping, '
                'echocardiographic assessment, or CE/FDA-approved diagnostic devices.<br>'
                'Based on open-literature methods (Jurak, Plesinger, Curila et al.). '
                'Not intended for clinical diagnosis or treatment decisions.</div>',
                unsafe_allow_html=True)
