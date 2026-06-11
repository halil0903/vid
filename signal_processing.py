"""
signal_processing.py - ECG Signal Preprocessing & QRS Detection
"""
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, find_peaks
from scipy.stats import pearsonr
import pandas as pd

def load_ecg_csv(file_path_or_buffer, sampling_rate=None, lead_columns=None):
    try:
        df = pd.read_csv(file_path_or_buffer, sep=None, engine='python')
    except Exception:
        if hasattr(file_path_or_buffer, 'seek'):
            file_path_or_buffer.seek(0)
        df = pd.read_csv(file_path_or_buffer, sep='\t')
    df.columns = [c.strip() for c in df.columns]
    time_col = None
    for col in df.columns:
        if col.lower() in ['time', 'time_s', 't', 'time_ms', 'sample']:
            time_col = col
            break
    if time_col and sampling_rate is None:
        tv = df[time_col].values
        dt = np.median(np.diff(tv))
        if dt < 0.1:
            sampling_rate = 1.0 / dt
        else:
            sampling_rate = 1000.0 / dt
    if sampling_rate is None:
        sampling_rate = 1000.0
    precordial = ['V1','V2','V3','V4','V5','V6','V7','V8']
    signals = {}
    if lead_columns:
        for ln, cn in lead_columns.items():
            if cn in df.columns:
                signals[ln] = df[cn].values.astype(float)
    else:
        for lead in precordial:
            for col in df.columns:
                if col.upper().replace(' ','') == lead:
                    signals[lead] = df[col].values.astype(float)
                    break
        if not signals:
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if time_col and time_col in numeric_cols:
                numeric_cols.remove(time_col)
            for i, col in enumerate(numeric_cols[:8]):
                signals[precordial[i] if i < len(precordial) else f'Ch{i+1}'] = df[col].values.astype(float)
    n = len(next(iter(signals.values())))
    return {'signals': signals, 'fs': float(sampling_rate), 'time': np.arange(n)/sampling_rate,
            'duration_s': n/sampling_rate, 'n_samples': n, 'lead_names': list(signals.keys())}

def remove_baseline_wander(sig, fs, cutoff=0.5, order=3):
    nyq = fs / 2.0
    if cutoff >= nyq: return sig
    b, a = butter(order, cutoff/nyq, btype='high')
    return filtfilt(b, a, sig, axis=0)

def remove_powerline_noise(sig, fs, freq=50.0, Q=30.0):
    nyq = fs / 2.0
    filtered = sig.copy()
    h = freq
    while h < nyq and h <= 300:
        b, a = iirnotch(h, Q, fs)
        filtered = filtfilt(b, a, filtered, axis=0)
        h += freq
    return filtered

def bandpass_filter(sig, fs, low, high, order=4):
    nyq = fs / 2.0
    lo = max(low/nyq, 0.001)
    hi = min(high/nyq, 0.999)
    if lo >= hi: return sig
    b, a = butter(order, [lo, hi], btype='band')
    return filtfilt(b, a, sig, axis=0)

def preprocess_ecg(signals_dict, fs, notch_freq=50.0):
    out = {}
    for lead, s in signals_dict.items():
        c = np.nan_to_num(s, nan=0.0)
        c = remove_baseline_wander(c, fs)
        c = remove_powerline_noise(c, fs, freq=notch_freq)
        out[lead] = c
    return out

def detect_r_peaks(sig, fs, min_dist_ms=300):
    filt = bandpass_filter(sig, fs, 5.0, 30.0, order=2)
    d = np.diff(filt); d = np.append(d, d[-1])
    sq = d**2
    w = max(int(0.08*fs),1)
    integ = np.convolve(sq, np.ones(w)/w, mode='same')
    md = int(min_dist_ms*fs/1000)
    th = np.mean(integ) + 0.5*np.std(integ)
    peaks, _ = find_peaks(integ, height=th, distance=md)
    refined = []
    sw = int(0.05*fs)
    for p in peaks:
        s = max(0, p-sw); e = min(len(sig), p+sw)
        refined.append(s + np.argmax(np.abs(sig[s:e])))
    return np.array(refined, dtype=int)

def select_reference_lead(signals_dict, fs):
    best, mx = None, 0
    for lead, s in signals_dict.items():
        f = bandpass_filter(s, fs, 5.0, 30.0, order=2)
        a = np.max(np.abs(f))
        if a > mx: mx, best = a, lead
    return best

def detect_qrs(signals_dict, fs, ref_lead=None):
    if ref_lead is None: ref_lead = select_reference_lead(signals_dict, fs)
    if ref_lead is None or ref_lead not in signals_dict: ref_lead = list(signals_dict.keys())[0]
    rp = detect_r_peaks(signals_dict[ref_lead], fs)
    rr = np.diff(rp)/fs*1000
    hr = 60000.0/np.mean(rr) if len(rr)>0 else 0
    return {'r_peaks': rp, 'reference_lead': ref_lead, 'heart_rate_bpm': hr, 'n_beats': len(rp), 'rr_intervals_ms': rr}

def extract_beats(sig, r_peaks, fs, window_ms=(-100, 200)):
    pre = int(abs(window_ms[0])*fs/1000); post = int(window_ms[1]*fs/1000); total = pre+post
    beats, vp = [], []
    for p in r_peaks:
        s, e = p-pre, p+post
        if s>=0 and e<=len(sig):
            b = sig[s:e]
            if len(b)==total: beats.append(b); vp.append(p)
    return np.array(beats) if beats else np.array([]).reshape(0,total), np.array(vp)

def reject_noisy_beats(beats, thr=0.85):
    if len(beats)<3: return beats, 0, np.ones(len(beats))
    tmpl = np.median(beats, axis=0)
    sc = []
    for b in beats:
        try: r,_=pearsonr(b,tmpl); sc.append(r if not np.isnan(r) else 0)
        except: sc.append(0)
    sc = np.array(sc); gm = sc>=thr; gb = beats[gm]
    if len(gb)<3: return beats, 0, np.ones(len(beats))
    return gb, int(np.sum(~gm)), sc

def compute_averaged_beat(signals_dict, r_peaks, fs, window_ms=(-100,200)):
    avg = {}; tu, tr = 0, 0; qsa = []
    for lead, s in signals_dict.items():
        beats, _ = extract_beats(s, r_peaks, fs, window_ms)
        if len(beats)<2:
            avg[lead] = np.zeros(int((window_ms[1]-window_ms[0])*fs/1000)); continue
        gb, rej, sc = reject_noisy_beats(beats)
        avg[lead] = np.median(gb, axis=0)
        tu = max(tu, len(gb)); tr = max(tr, rej)
        qsa.extend(sc[sc>0])
    n = len(next(iter(avg.values()))); pre = int(abs(window_ms[0])*fs/1000)
    t = (np.arange(n)-pre)/fs*1000
    return {'averaged_beats': avg, 'n_beats_used': tu, 'n_beats_rejected': tr,
            'quality_score': float(np.mean(qsa)) if qsa else 0, 'time_axis_ms': t, 'window_ms': window_ms}

def estimate_qrs_duration(avg_beats, time_ms, fs, ref_lead=None):
    if ref_lead and ref_lead in avg_beats: beat = avg_beats[ref_lead]
    else:
        mx, beat = 0, None
        for l,b in avg_beats.items():
            a=np.max(np.abs(b))
            if a>mx: mx,beat=a,b
    if beat is None: return {'qrs_onset_ms':0,'qrs_offset_ms':0,'qrs_duration_ms':0}
    en = beat**2; w=max(int(0.005*fs),3);
    if w%2==0: w+=1
    es = np.convolve(en, np.ones(w)/w, mode='same')
    th = 0.05*np.max(es); idx = np.where(es>th)[0]
    if len(idx)<2: return {'qrs_onset_ms':-50,'qrs_offset_ms':50,'qrs_duration_ms':100}
    return {'qrs_onset_ms':float(time_ms[idx[0]]),'qrs_offset_ms':float(time_ms[idx[-1]]),
            'qrs_duration_ms':float(time_ms[idx[-1]]-time_ms[idx[0]])}
