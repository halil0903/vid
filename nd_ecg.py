"""
nd_ecg.py - Negative Derivative ECG Analysis
Computes -dV/dt based epicardial activation times (NDAT).
Intrinsicoid deflection method from open literature.
"""
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter

def _bandpass(sig, fs, low, high, order=4):
    nyq = fs/2.0
    lo, hi = max(low/nyq,0.001), min(high/nyq,0.999)
    if lo >= hi: return sig
    b, a = butter(order, [lo, hi], btype='band')
    return filtfilt(b, a, sig, axis=0)

def compute_negative_derivative(averaged_beats, fs, filter_band=(1,40)):
    nd_signals = {}
    for lead, beat in averaged_beats.items():
        filt = _bandpass(beat, fs, filter_band[0], filter_band[1], order=3)
        dt = 1.0 / fs
        deriv = -np.gradient(filt, dt)
        wl = max(int(0.005*fs),5)
        if wl % 2 == 0: wl += 1
        if wl >= len(deriv): wl = len(deriv) if len(deriv)%2==1 else len(deriv)-1
        if wl >= 4:
            try:
                deriv_smooth = savgol_filter(deriv, wl, 2)
            except Exception:
                deriv_smooth = deriv
        else:
            deriv_smooth = deriv
        mx = np.max(np.abs(deriv_smooth))
        if mx > 0:
            deriv_norm = deriv_smooth / mx
        else:
            deriv_norm = deriv_smooth
        nd_signals[lead] = {'raw_derivative': deriv, 'smooth_derivative': deriv_smooth, 'normalized': deriv_norm}
    return nd_signals

def compute_ndat(nd_signals, time_axis_ms):
    ndat = {}
    for lead, data in nd_signals.items():
        nd = data['smooth_derivative']
        peak_idx = np.argmax(nd)
        ndat[lead] = float(time_axis_ms[peak_idx])
    return ndat

def compute_nd_dys(ndat_values):
    if not ndat_values: return {'nd_dys':0.0,'earliest_lead':'','latest_lead':''}
    leads = list(ndat_values.keys())
    times = [ndat_values[l] for l in leads]
    ei, li = int(np.argmin(times)), int(np.argmax(times))
    nd_dys = times[li] - times[ei]
    po = ['V1','V2','V3','V4','V5','V6','V7','V8']
    el, ll = leads[ei], leads[li]
    ep = po.index(el) if el in po else 0
    lp = po.index(ll) if ll in po else 0
    sign = 1.0 if lp >= ep else -1.0
    return {'nd_dys': float(sign*nd_dys), 'nd_dys_abs': float(nd_dys),
            'earliest_lead': el, 'latest_lead': ll,
            'earliest_time_ms': float(times[ei]), 'latest_time_ms': float(times[li])}

def build_nd_heatmap_matrix(nd_signals, leads_order=None):
    if leads_order is None:
        leads_order = sorted(nd_signals.keys(), key=lambda x: int(x.replace('V','')) if x.startswith('V') and x[1:].isdigit() else 99)
    n = len(next(iter(nd_signals.values()))['normalized'])
    matrix = np.zeros((len(leads_order), n))
    for i, lead in enumerate(leads_order):
        if lead in nd_signals:
            nd = nd_signals[lead]['normalized']
            nd_pos = np.clip(nd, 0, None)
            mx = np.max(nd_pos)
            if mx > 0: nd_pos = nd_pos / mx
            matrix[i,:] = nd_pos
    return matrix, leads_order
