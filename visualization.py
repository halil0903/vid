"""
visualization.py - VDI-style UHF-ECG Activation Map Visualization
Produces heatmaps matching the VDI Demonstrator aesthetic.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
import io

# VDI-style colormap: dark blue -> cyan -> yellow -> red
def get_vdi_colormap():
    colors = [
        (0.0, '#000033'),   # very dark blue
        (0.05, '#000066'),  # dark blue
        (0.15, '#0000CC'),  # blue
        (0.3, '#0066FF'),   # light blue
        (0.45, '#00CCFF'),  # cyan
        (0.6, '#FFFF00'),   # yellow
        (0.75, '#FF9900'),  # orange
        (0.9, '#FF0000'),   # red
        (1.0, '#CC0000'),   # dark red
    ]
    positions = [c[0] for c in colors]
    color_list = [c[1] for c in colors]
    cmap = mcolors.LinearSegmentedColormap.from_list('vdi_uhf', list(zip(positions, color_list)))
    return cmap

VDI_CMAP = get_vdi_colormap()

def plot_activation_map(matrix, time_axis_ms, leads_order, activation_times=None,
                         vd_values=None, title='UHF-ECG Activation Map',
                         qrs_onset_ms=None, qrs_offset_ms=None,
                         figsize=(10, 6), dpi=150):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('#0a0a2e')
    ax.set_facecolor('#000033')

    from scipy.ndimage import zoom
    zoom_y = max(4, 80 // max(matrix.shape[0],1))
    zoom_x = max(1, 400 // max(matrix.shape[1],1))
    matrix_smooth = zoom(matrix, (zoom_y, zoom_x), order=3)
    matrix_smooth = np.clip(matrix_smooth, 0, 1)

    t_min, t_max = time_axis_ms[0], time_axis_ms[-1]
    n_leads = len(leads_order)

    im = ax.imshow(matrix_smooth, aspect='auto', cmap=VDI_CMAP, vmin=0, vmax=1,
                   extent=[t_min, t_max, n_leads-0.5, -0.5],
                   interpolation='bilinear')

    if activation_times:
        at_x = [activation_times.get(l, 0) for l in leads_order]
        at_y = list(range(len(leads_order)))
        ax.plot(at_x, at_y, 'k-', linewidth=2.5, zorder=5)
        ax.plot(at_x, at_y, 'wo', markersize=5, zorder=6, markeredgecolor='black', markeredgewidth=0.5)

    ax.set_yticks(range(n_leads))
    ax.set_yticklabels(leads_order, fontsize=12, fontweight='bold', color='white')
    ax.set_xlabel('t [ms]', fontsize=12, color='white', fontweight='bold')
    ax.tick_params(colors='white', labelsize=10)

    for i in range(n_leads):
        ax.axhline(y=i+0.5, color='white', linewidth=0.3, alpha=0.3)

    if qrs_onset_ms is not None:
        ax.axvline(x=0, color='white', linewidth=0.8, alpha=0.5, linestyle='--')

    ax.set_title(title, fontsize=14, fontweight='bold', color='white', pad=10)

    if vd_values:
        ax2 = ax.twinx()
        ax2.set_ylim(ax.get_ylim())
        ax2.set_yticks(range(n_leads))
        vd_labels = [f"{int(vd_values.get(l,0))}" for l in leads_order]
        ax2.set_yticklabels(vd_labels, fontsize=11, color='#FFD700', fontweight='bold')
        ax2.tick_params(colors='#FFD700')
        ax2.set_ylabel('Vd', fontsize=12, color='#FFD700', fontweight='bold', rotation=0, labelpad=15)
        ax2.set_facecolor('none')

    scale_y = n_leads - 0.3
    scale_x = t_min + 5
    scale_w_ms = 10
    ax.plot([scale_x, scale_x+scale_w_ms], [scale_y, scale_y], 'w-', linewidth=3)
    ax.text(scale_x+scale_w_ms+2, scale_y, '10 ms', color='white', fontsize=8, va='center')

    fig.tight_layout()
    return fig

def plot_ved_scale(e_dys_value, figsize=(8, 1.2), dpi=150):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('#1a1a3e')
    ax.set_facecolor('#1a1a3e')
    ax.set_xlim(-110, 110)
    ax.set_ylim(-0.5, 1.5)
    ax.axhline(y=0.5, color='white', linewidth=2, xmin=0.05, xmax=0.95)
    for v in [-100,-50,0,50,100]:
        ax.plot([v,v],[0.35,0.65],'w-',linewidth=1.5)
        ax.text(v, 0.1, str(v), ha='center', fontsize=9, color='white')
    sync_rect = Rectangle((-20, 0.3), 40, 0.4, facecolor='#44cc44', alpha=0.6, edgecolor='white', linewidth=1)
    ax.add_patch(sync_rect)
    ax.text(0, 1.2, 'Synchronous', ha='center', fontsize=10, fontweight='bold', color='#44cc44')
    ax.text(-100, 1.2, 'Delayed RV', ha='center', fontsize=9, color='#ff6666')
    ax.text(100, 1.2, 'Delayed LV', ha='center', fontsize=9, color='#ff6666')
    clamped = max(-100, min(100, e_dys_value))
    ax.annotate('', xy=(clamped,0.5), xytext=(clamped,1.05),
                arrowprops=dict(arrowstyle='->', color='#FFD700', lw=2.5))
    ax.text(clamped, -0.3, f'VED: {e_dys_value:.0f}ms', ha='center', fontsize=11, fontweight='bold', color='#FFD700')
    ax.set_axis_off()
    fig.tight_layout()
    return fig

def plot_ecg_traces(signals_dict, time_array, leads_order=None, figsize=(12,8), dpi=100):
    if leads_order is None:
        leads_order = sorted(signals_dict.keys(), key=lambda x: int(x.replace('V','')) if x.startswith('V') and x[1:].isdigit() else 99)
    n = len(leads_order)
    fig, axes = plt.subplots(n, 1, figsize=figsize, dpi=dpi, sharex=True)
    fig.patch.set_facecolor('#fef6f0')
    if n == 1: axes = [axes]
    for i, lead in enumerate(leads_order):
        ax = axes[i]
        ax.set_facecolor('#fef6f0')
        for x in np.arange(0, time_array[-1], 0.2):
            ax.axvline(x=x, color='#ffcccc', linewidth=0.3)
        for x in np.arange(0, time_array[-1], 1.0):
            ax.axvline(x=x, color='#ff9999', linewidth=0.6)
        sig = signals_dict.get(lead, np.zeros(len(time_array)))
        ax.plot(time_array, sig, 'k-', linewidth=0.8)
        ax.set_ylabel(lead, fontsize=11, fontweight='bold', rotation=0, labelpad=25)
        ax.tick_params(labelsize=8)
        if i < n-1: ax.set_xticklabels([])
    axes[-1].set_xlabel('Time (s)', fontsize=10)
    fig.suptitle('ECG Traces', fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    return fig

def plot_averaged_qrs(averaged_beats, time_axis_ms, leads_order=None, figsize=(10,6), dpi=120):
    if leads_order is None:
        leads_order = sorted(averaged_beats.keys(), key=lambda x: int(x.replace('V','')) if x.startswith('V') and x[1:].isdigit() else 99)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('#0f0f2d')
    ax.set_facecolor('#0f0f2d')
    colors = plt.cm.rainbow(np.linspace(0,1,len(leads_order)))
    for i, lead in enumerate(leads_order):
        if lead in averaged_beats:
            ax.plot(time_axis_ms, averaged_beats[lead], color=colors[i], linewidth=1.5, label=lead)
    ax.axvline(x=0, color='white', linewidth=0.8, linestyle='--', alpha=0.5, label='R-peak')
    ax.set_xlabel('Time (ms)', color='white', fontsize=11)
    ax.set_ylabel('Amplitude', color='white', fontsize=11)
    ax.set_title('Averaged QRS Complex', color='white', fontsize=13, fontweight='bold')
    ax.tick_params(colors='white')
    ax.legend(loc='upper right', fontsize=9, facecolor='#1a1a3e', edgecolor='white', labelcolor='white')
    ax.spines['bottom'].set_color('white'); ax.spines['left'].set_color('white')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    fig.tight_layout()
    return fig

def plot_combined_maps(uhf_matrix, nd_matrix, time_ms, leads_order,
                       uhfat=None, ndat=None, vd_uhf=None, vd_nd=None, dpi=150):
    fig = plt.figure(figsize=(16,6), dpi=dpi)
    fig.patch.set_facecolor('#0a0a2e')
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)
    from scipy.ndimage import zoom
    for idx, (mat, at, vd, ttl) in enumerate([
        (uhf_matrix, uhfat, vd_uhf, 'UHF-ECG Activation Map'),
        (nd_matrix, ndat, vd_nd, 'ND-ECG Activation Map')
    ]):
        ax = fig.add_subplot(gs[0, idx])
        ax.set_facecolor('#000033')
        zy = max(4, 80//max(mat.shape[0],1))
        zx = max(1, 400//max(mat.shape[1],1))
        ms = zoom(mat, (zy,zx), order=3)
        ms = np.clip(ms, 0, 1)
        t0, t1 = time_ms[0], time_ms[-1]
        nl = len(leads_order)
        ax.imshow(ms, aspect='auto', cmap=VDI_CMAP, vmin=0, vmax=1,
                  extent=[t0,t1,nl-0.5,-0.5], interpolation='bilinear')
        if at:
            ax_x = [at.get(l,0) for l in leads_order]
            ax_y = list(range(nl))
            ax.plot(ax_x, ax_y, 'k-', linewidth=2.5, zorder=5)
            ax.plot(ax_x, ax_y, 'wo', markersize=4, zorder=6, markeredgecolor='k', markeredgewidth=0.5)
        ax.set_yticks(range(nl))
        ax.set_yticklabels(leads_order, fontsize=10, fontweight='bold', color='white')
        ax.set_xlabel('t [ms]', fontsize=10, color='white')
        ax.set_title(ttl, fontsize=12, fontweight='bold', color='white')
        ax.tick_params(colors='white')
        for i in range(nl): ax.axhline(y=i+0.5, color='white', linewidth=0.2, alpha=0.3)
        if vd:
            ax2 = ax.twinx()
            ax2.set_ylim(ax.get_ylim()); ax2.set_yticks(range(nl))
            ax2.set_yticklabels([f"{int(vd.get(l,0))}" for l in leads_order], fontsize=9, color='#FFD700', fontweight='bold')
            ax2.tick_params(colors='#FFD700'); ax2.set_facecolor('none')
    fig.tight_layout()
    return fig

def plot_summary_metrics(e_dys, nd_dys, qrs_dur, n_beats, quality, fs, figsize=(8,3), dpi=120):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    fig.patch.set_facecolor('#1a1a3e')
    ax.set_facecolor('#1a1a3e')
    ax.set_axis_off()
    metrics = [
        ('e-DYS', f'{e_dys:.1f} ms', '#FF6B6B'),
        ('nd-DYS', f'{nd_dys:.1f} ms', '#4ECDC4'),
        ('QRSd', f'{qrs_dur:.0f} ms', '#FFE66D'),
        ('Beats', f'{n_beats}', '#A8E6CF'),
        ('Quality', f'{quality:.0%}', '#DDA0DD'),
        ('Fs', f'{fs:.0f} Hz', '#87CEEB'),
    ]
    n = len(metrics)
    for i, (label, value, color) in enumerate(metrics):
        x = (i + 0.5) / n
        ax.text(x, 0.7, value, ha='center', va='center', fontsize=16, fontweight='bold',
                color=color, transform=ax.transAxes)
        ax.text(x, 0.3, label, ha='center', va='center', fontsize=11,
                color='#aaaaaa', transform=ax.transAxes)
    fig.tight_layout()
    return fig

def fig_to_bytes(fig, fmt='png', dpi=150):
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches='tight', facecolor=fig.get_facecolor())
    buf.seek(0)
    return buf.getvalue()
