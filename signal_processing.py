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
    """Çıkarılan beat'leri ve bunlara karşılık gelen r_peaks İNDEKSLERİNİ döndürür.
    (İndeks, lead'ler arası ortak beat maskesi için gereklidir.)"""
    pre = int(abs(window_ms[0])*fs/1000); post = int(window_ms[1]*fs/1000); total = pre+post
    beats, valid_idx = [], []
    for i, p in enumerate(r_peaks):
        s, e = p-pre, p+post
        if s>=0 and e<=len(sig):
            b = sig[s:e]
            if len(b)==total:
                beats.append(b); valid_idx.append(i)
    if beats:
        return np.array(beats), np.array(valid_idx, dtype=int)
    return np.array([]).reshape(0,total), np.array([], dtype=int)


def compute_common_beat_mask(signals_dict, r_peaks, fs, window_ms, ref_lead, thr=0.85):
    """Lead'ler arası TUTARLI beat seçimi.

    Referans lead üzerinde template-korelasyonu ile gürültülü beat'leri tespit
    eder ve aynı kabul kararını (ortak maske) TÜM lead'lere uygular. Böylece her
    lead aynı kalp atımlarından ortalanır; lead'ler arası zamanlama karşılaştırması
    (e-DYS / nd-DYS'nin temeli) tutarlı kalır.

    Döndürür: (kabul edilen r_peaks indeksleri, kalite skoru, n_rejected)
    """
    if ref_lead not in signals_dict:
        ref_lead = list(signals_dict.keys())[0]
    ref_beats, valid_idx = extract_beats(signals_dict[ref_lead], r_peaks, fs, window_ms)
    if len(ref_beats) < 3:
        return valid_idx, 1.0, 0
    template = np.median(ref_beats, axis=0)
    scores = []
    for b in ref_beats:
        try:
            r, _ = pearsonr(b, template)
            scores.append(r if not np.isnan(r) else 0.0)
        except Exception:
            scores.append(0.0)
    scores = np.array(scores)
    good = scores >= thr
    if np.sum(good) < 3:
        good = np.zeros_like(good, dtype=bool)
        good[np.argsort(scores)[-3:]] = True
    accepted_idx = valid_idx[good]
    quality = float(np.mean(scores[good])) if np.any(good) else 0.0
    n_rejected = int(np.sum(~good))
    return accepted_idx, quality, n_rejected

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

def compute_averaged_beat(signals_dict, r_peaks, fs, window_ms=(-100,200), ref_lead=None):
    """DÜZELTİLMİŞ beat averaging — lead'ler arası ORTAK beat reddi.

    Önceki sürüm her lead'i bağımsız reddediyordu; bu, lead'ler arasında farklı
    atım setlerinin ortalanmasına ve dolayısıyla lead'ler arası zamanlama
    karşılaştırmalarının (e-DYS / nd-DYS) güvenilmez olmasına yol açıyordu.
    Bu sürüm referans lead'den ortak bir kabul maskesi üretir ve tüm lead'lere
    aynı atım setini uygular.
    """
    if ref_lead is None or ref_lead not in signals_dict:
        ref_lead = max(signals_dict.keys(),
                       key=lambda l: np.max(np.abs(signals_dict[l])))
    accepted_idx, quality, n_rejected = compute_common_beat_mask(
        signals_dict, r_peaks, fs, window_ms, ref_lead, thr=0.85)
    accepted_peaks = np.array(r_peaks)[accepted_idx] if len(accepted_idx) else np.array([])

    pre = int(abs(window_ms[0])*fs/1000); post = int(window_ms[1]*fs/1000); total = pre+post
    avg = {}
    for lead, s in signals_dict.items():
        beats = []
        for p in accepted_peaks:
            st_, en_ = p-pre, p+post
            if st_>=0 and en_<=len(s):
                b = s[st_:en_]
                if len(b)==total: beats.append(b)
        avg[lead] = np.median(np.array(beats), axis=0) if beats else np.zeros(total)

    t = (np.arange(total)-pre)/fs*1000
    return {'averaged_beats': avg, 'n_beats_used': int(len(accepted_peaks)),
            'n_beats_rejected': n_rejected, 'quality_score': quality,
            'time_axis_ms': t, 'window_ms': window_ms, 'reference_lead': ref_lead}


def estimate_qrs_onset(avg_beats, time_ms, fs):
    """Tüm lead'lerin birleşik RMS enerjisinden TEK bir sağlam QRS onset.

    UHFAT/NDAT değerlerini 'QRS onset'ten ms' olarak raporlayabilmek için
    aktivasyon zamanlarının kaydırılacağı referansı sağlar. Tek lead yerine
    birleşik sinyal kullanmak, tek bir gürültülü lead'in onset'i kaydırmasını
    engeller.
    """
    leads = list(avg_beats.keys())
    if not leads:
        return {'qrs_onset_ms': 0.0, 'qrs_offset_ms': 0.0, 'qrs_duration_ms': 0.0}
    stack = np.vstack([avg_beats[l] for l in leads])
    rms = np.sqrt(np.mean(stack**2, axis=0))
    energy = rms**2
    w = max(int(0.005*fs), 3)
    if w % 2 == 0: w += 1
    smooth = np.convolve(energy, np.ones(w)/w, mode='same')
    thr = 0.05*np.max(smooth)
    idx = np.where(smooth > thr)[0]
    if len(idx) < 2:
        return {'qrs_onset_ms': float(time_ms[0]), 'qrs_offset_ms': float(time_ms[-1]),
                'qrs_duration_ms': float(time_ms[-1]-time_ms[0])}
    onset, offset = float(time_ms[idx[0]]), float(time_ms[idx[-1]])
    return {'qrs_onset_ms': onset, 'qrs_offset_ms': offset,
            'qrs_duration_ms': offset-onset}

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
