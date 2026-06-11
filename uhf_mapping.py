"""
uhf_mapping.py - UHF-ECG Envelope Analysis & UHFAT Computation
Based on open literature: Jurak, Plesinger, Curila et al.
150-1050 Hz, 16 bands, 100 Hz width, 50 Hz step.
"""
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

def _bandpass(sig, fs, low, high, order=4):
    nyq = fs/2.0
    lo, hi = max(low/nyq,0.001), min(high/nyq,0.999)
    if lo >= hi: return sig
    b, a = butter(order, [lo, hi], btype='band')
    return filtfilt(b, a, sig, axis=0)

def get_uhf_bands(f_low=150, f_high=1050, width=100, step=50):
    bands = []
    f = f_low
    while f + width <= f_high + step:
        bands.append((f, f+width))
        f += step
    return bands

def compute_envelope(sig):
    analytic = hilbert(sig)
    return np.abs(analytic)

def compute_uhf_envelopes(averaged_beats, fs, leads=None):
    if fs < 500:
        return None
    bands = get_uhf_bands()
    nyq = fs / 2.0
    valid_bands = [(lo,hi) for lo,hi in bands if hi < nyq]
    if not valid_bands:
        return None
    if leads is None:
        leads = list(averaged_beats.keys())
    result = {}
    for lead in leads:
        if lead not in averaged_beats: continue
        beat = averaged_beats[lead]
        band_envs = []
        for lo, hi in valid_bands:
            try:
                filtered = _bandpass(beat, fs, lo, hi, order=4)
                env = compute_envelope(filtered)
                mx = np.max(env)
                if mx > 0:
                    env_norm = env / mx
                else:
                    env_norm = env
                band_envs.append(env_norm)
            except Exception:
                continue
        if band_envs:
            avg_env = np.mean(band_envs, axis=0)
            mx = np.max(avg_env)
            if mx > 0:
                avg_env = avg_env / mx
            result[lead] = {
                'envelope': avg_env,
                'band_envelopes': band_envs,
                'n_bands': len(band_envs)
            }
    return result if result else None

def compute_uhfat(envelopes, time_axis_ms, threshold=0.5, qrs_onset_ms=0.0):
    """UHF aktivasyon zamanı (UHFAT) — QRS ONSET referanslı.

    Center-of-gravity yöntemi korunur; sonuç 'QRS onset = 0' eksenine kaydırılır.
    Önceki sürüm R-peak merkezli pencere eksenine göre değer veriyordu; bu,
    arayüzdeki 'ms from QRS onset' etiketiyle ve literatürdeki UHFAT tanımıyla
    tutarsızdı. qrs_onset_ms = estimate_qrs_onset()'ten gelen onset.
    """
    uhfat = {}
    for lead, data in envelopes.items():
        env = data['envelope']
        mx = np.max(env)
        if mx <= 0:
            uhfat[lead] = 0.0
            continue
        mask = env >= threshold * mx
        env_t = env.copy()
        env_t[~mask] = 0.0
        denom = np.sum(env_t)
        if denom > 0:
            cog = np.sum(time_axis_ms * env_t) / denom
        else:
            cog = time_axis_ms[np.argmax(env)]
        uhfat[lead] = float(cog - qrs_onset_ms)
    return uhfat

def compute_vd(envelopes, time_axis_ms, threshold=0.5):
    vd = {}
    dt = np.mean(np.diff(time_axis_ms)) if len(time_axis_ms)>1 else 1.0
    for lead, data in envelopes.items():
        env = data['envelope']
        mx = np.max(env)
        if mx <= 0:
            vd[lead] = 0.0
            continue
        mask = env >= threshold * mx
        dur = np.sum(mask) * dt
        vd[lead] = float(dur)
    return vd

def compute_edys(uhfat_values, leads_order=None):
    """e-DYS (elektriksel dissenkroni) — literatüre uygun, gürültüye dayanıklı.

    e-DYS = en geç UHFAT − en erken UHFAT. İşaret, keyfi tek lead pozisyonu
    yerine V1→V6 UHFAT dizisinin lineer regresyon EĞİMİNDEN belirlenir
    (pozitif eğim = sağdan sola geç aktivasyon, sol-gecikme paterni). Böylece
    tek gürültülü bir lead işareti ters çeviremez.
    """
    if not uhfat_values:
        return {'e_dys': 0.0, 'e_dys_abs': 0.0, 'earliest_lead': '', 'latest_lead': ''}
    if leads_order is None:
        precordial = ['V1','V2','V3','V4','V5','V6','V7','V8']
        leads_order = [l for l in precordial if l in uhfat_values]
    leads = [l for l in leads_order if l in uhfat_values]
    if len(leads) < 2:
        only = leads[0] if leads else ''
        return {'e_dys': 0.0, 'e_dys_abs': 0.0, 'earliest_lead': only, 'latest_lead': only}
    times = np.array([uhfat_values[l] for l in leads])
    earliest_idx = int(np.argmin(times)); latest_idx = int(np.argmax(times))
    e_dys_abs = float(times[latest_idx] - times[earliest_idx])
    positions = np.arange(len(leads), dtype=float)
    if np.std(times) > 0:
        slope = float(np.polyfit(positions, times, 1)[0])
        sign = 1.0 if slope >= 0 else -1.0
    else:
        slope = 0.0; sign = 1.0
    return {'e_dys': float(sign * e_dys_abs), 'e_dys_abs': e_dys_abs,
            'earliest_lead': leads[earliest_idx], 'latest_lead': leads[latest_idx],
            'earliest_time_ms': float(times[earliest_idx]),
            'latest_time_ms': float(times[latest_idx]),
            'activation_slope': slope}

def build_uhf_heatmap_matrix(envelopes, leads_order=None):
    if leads_order is None:
        leads_order = sorted(envelopes.keys(), key=lambda x: int(x.replace('V','')) if x.startswith('V') and x[1:].isdigit() else 99)
    n_samples = len(next(iter(envelopes.values()))['envelope'])
    matrix = np.zeros((len(leads_order), n_samples))
    for i, lead in enumerate(leads_order):
        if lead in envelopes:
            matrix[i, :] = envelopes[lead]['envelope']
    return matrix, leads_order
