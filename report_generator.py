"""
report_generator.py - PDF/PNG Report Export
Generates clinical-research style reports with disclaimers.
"""
import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from datetime import datetime


def generate_pdf_report(figures_dict, metrics, filename_buf=None):
    """
    Generate multi-page PDF report.
    
    figures_dict: dict of name -> matplotlib figure
    metrics: dict with e_dys, nd_dys, qrs_dur, n_beats, quality, fs
    """
    buf = filename_buf if filename_buf else io.BytesIO()
    
    with PdfPages(buf) as pdf:
        # Page 1: Title & Disclaimer
        fig_title = plt.figure(figsize=(11, 8.5), dpi=150)
        fig_title.patch.set_facecolor('#0a0a2e')
        ax = fig_title.add_axes([0, 0, 1, 1])
        ax.set_facecolor('#0a0a2e')
        ax.set_axis_off()
        
        ax.text(0.5, 0.85, '🫀 UHF-ECG Activation Mapping Report', 
                ha='center', va='center', fontsize=22, fontweight='bold', color='white',
                transform=ax.transAxes)
        ax.text(0.5, 0.78, 'Ventricular Dyssynchrony Analysis',
                ha='center', va='center', fontsize=16, color='#87CEEB',
                transform=ax.transAxes)
        
        # Metrics
        info_lines = [
            f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            f"Sampling Rate: {metrics.get('fs', 0):.0f} Hz",
            f"Beats Analyzed: {metrics.get('n_beats', 0)}",
            f"Quality Score: {metrics.get('quality', 0):.1%}",
            f"QRS Duration: {metrics.get('qrs_dur', 0):.0f} ms",
            "",
            f"e-DYS (UHF): {metrics.get('e_dys', 0):.1f} ms",
            f"nd-DYS (ND-ECG): {metrics.get('nd_dys', 0):.1f} ms",
        ]
        
        for i, line in enumerate(info_lines):
            ax.text(0.5, 0.62 - i * 0.045, line,
                    ha='center', va='center', fontsize=13, color='#cccccc',
                    transform=ax.transAxes, fontfamily='monospace')
        
        # Disclaimer
        disclaimer = (
            "⚠️ RESEARCH / EDUCATION USE ONLY\n"
            "These results do not replace invasive electrophysiology mapping,\n"
            "echocardiographic assessment, or CE/FDA-approved diagnostic devices.\n"
            "Not intended for clinical diagnosis, treatment decisions,\n"
            "or device optimization."
        )
        ax.text(0.5, 0.12, disclaimer,
                ha='center', va='center', fontsize=10, color='#FF6B6B',
                transform=ax.transAxes, fontstyle='italic',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='#1a0000', 
                         edgecolor='#FF6B6B', alpha=0.9))
        
        ax.text(0.5, 0.02, 'Based on open-literature methods (Jurak, Plesinger, Curila et al.)',
                ha='center', va='center', fontsize=8, color='#666666',
                transform=ax.transAxes)
        
        pdf.savefig(fig_title, facecolor=fig_title.get_facecolor())
        plt.close(fig_title)
        
        # Remaining pages: each figure
        for name, fig in figures_dict.items():
            if fig is not None:
                # Add disclaimer footer to each page
                fig.text(0.5, 0.01, '⚠️ Research/Education Only — Not for clinical use',
                        ha='center', fontsize=7, color='#FF6B6B', fontstyle='italic',
                        transform=fig.transFigure)
                pdf.savefig(fig, facecolor=fig.get_facecolor(), bbox_inches='tight')
    
    if filename_buf is None:
        buf.seek(0)
        return buf.getvalue()
    return None


def generate_png_composite(figures_dict, metrics, dpi=150):
    """Generate a single composite PNG with all key visualizations."""
    n_figs = len([f for f in figures_dict.values() if f is not None])
    if n_figs == 0:
        return None
    
    fig_composite = plt.figure(figsize=(16, 5 * n_figs), dpi=dpi)
    fig_composite.patch.set_facecolor('#0a0a2e')
    
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(n_figs + 1, 1, figure=fig_composite, hspace=0.3,
                  height_ratios=[0.3] + [1] * n_figs)
    
    # Header
    ax_header = fig_composite.add_subplot(gs[0, 0])
    ax_header.set_facecolor('#0a0a2e')
    ax_header.set_axis_off()
    ax_header.text(0.5, 0.7, 'UHF-ECG Activation Mapping Report',
                   ha='center', fontsize=18, fontweight='bold', color='white',
                   transform=ax_header.transAxes)
    
    m_text = (f"e-DYS: {metrics.get('e_dys',0):.1f} ms  |  "
              f"nd-DYS: {metrics.get('nd_dys',0):.1f} ms  |  "
              f"QRSd: {metrics.get('qrs_dur',0):.0f} ms  |  "
              f"Beats: {metrics.get('n_beats',0)}  |  "
              f"Fs: {metrics.get('fs',0):.0f} Hz")
    ax_header.text(0.5, 0.2, m_text,
                   ha='center', fontsize=11, color='#87CEEB',
                   transform=ax_header.transAxes, fontfamily='monospace')
    
    # Embed figures as images
    for i, (name, fig) in enumerate(figures_dict.items()):
        if fig is None:
            continue
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                    facecolor=fig.get_facecolor())
        buf.seek(0)
        from PIL import Image
        try:
            img = plt.imread(buf)
            ax = fig_composite.add_subplot(gs[i + 1, 0])
            ax.imshow(img)
            ax.set_axis_off()
            ax.set_title(name, color='white', fontsize=12, fontweight='bold')
        except Exception:
            pass
    
    # Disclaimer
    fig_composite.text(0.5, 0.005,
                       '⚠️ RESEARCH/EDUCATION ONLY — Not for clinical diagnosis or treatment',
                       ha='center', fontsize=9, color='#FF6B6B', fontstyle='italic')
    
    buf_out = io.BytesIO()
    fig_composite.savefig(buf_out, format='png', dpi=dpi, bbox_inches='tight',
                          facecolor=fig_composite.get_facecolor())
    plt.close(fig_composite)
    buf_out.seek(0)
    return buf_out.getvalue()
