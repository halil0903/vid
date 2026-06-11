"""
sample_data_generator.py - Synthetic ECG Signal Generator
Generates Normal, LBBB, RBBB patterns at configurable sampling rate.
Includes realistic high-frequency QRS components for UHF analysis.
"""
import numpy as np
import os

def _gaussian(t, center, width, amp):
    return amp * np.exp(-0.5 * ((t - center) / width) ** 2)

def _add_hf_component(t, center, width, amp, freq):
    """Add a high-frequency oscillation modulated by a Gaussian envelope."""
    env = np.exp(-0.5 * ((t - center) / width) ** 2)
    return amp * env * np.sin(2 * np.pi * freq * (t - center))

def generate_single_beat(t, qrs_center, params):
    """Generate a single heartbeat with P-QRS-T waves and HF components."""
    sig = np.zeros(len(t))
    p = params
    
    # P wave
    sig += _gaussian(t, qrs_center + p['p_offset'], p['p_width'], p['p_amp'])
    # Q wave
    sig += _gaussian(t, qrs_center + p['q_offset'], p['q_width'], p['q_amp'])
    # R wave
    sig += _gaussian(t, qrs_center + p['r_offset'], p['r_width'], p['r_amp'])
    # S wave
    sig += _gaussian(t, qrs_center + p['s_offset'], p['s_width'], p['s_amp'])
    # T wave
    sig += _gaussian(t, qrs_center + p['t_offset'], p['t_width'], p['t_amp'])
    
    # High-frequency QRS components (essential for UHF analysis)
    # These simulate the rapid depolarization wavefront
    hf_center = qrs_center + p.get('hf_offset', p['r_offset'])
    hf_width = p.get('hf_width', 0.012)
    hf_amp = p.get('hf_amp', 0.03)
    
    # Multiple HF components at different frequencies within UHF range
    for freq in [200, 350, 500, 650, 800]:
        phase = p.get('hf_phase', 0)
        sig += hf_amp * np.exp(-0.5*((t - hf_center)/hf_width)**2) * \
               np.sin(2*np.pi*freq*(t - hf_center) + phase)
        hf_amp *= 0.7  # Each higher freq has less amplitude
    
    return sig

def get_lead_params(lead, pattern='normal'):
    """Get morphology parameters for each precordial lead and pattern."""
    base = {
        'p_offset': -0.16, 'p_width': 0.04, 'p_amp': 0.15,
        'q_offset': -0.03, 'q_width': 0.01, 'q_amp': -0.1,
        'r_offset': 0.0, 'r_width': 0.015, 'r_amp': 1.0,
        's_offset': 0.03, 's_width': 0.012, 's_amp': -0.3,
        't_offset': 0.18, 't_width': 0.06, 't_amp': 0.3,
        'hf_offset': 0.0, 'hf_width': 0.012, 'hf_amp': 0.04,
        'hf_phase': 0,
    }
    
    lead_num = int(lead.replace('V', '')) if lead.startswith('V') and lead[1:].isdigit() else 1
    
    if pattern == 'normal':
        r_prog = {1:0.3, 2:0.5, 3:0.8, 4:1.2, 5:1.0, 6:0.8, 7:0.6, 8:0.5}
        s_prog = {1:-0.8, 2:-0.6, 3:-0.3, 4:-0.1, 5:-0.05, 6:-0.05, 7:-0.05, 8:-0.05}
        base['r_amp'] = r_prog.get(lead_num, 0.5)
        base['s_amp'] = s_prog.get(lead_num, -0.1)
        base['r_width'] = 0.012
        base['s_width'] = 0.010
        # Normal: narrow synchronous activation (~5-15ms spread)
        at_shift = {1:-0.005, 2:-0.003, 3:0.0, 4:0.002, 5:0.005, 6:0.007, 7:0.005, 8:0.003}
        shift = at_shift.get(lead_num, 0)
        base['r_offset'] += shift
        base['s_offset'] += shift
        base['hf_offset'] = shift
        base['hf_width'] = 0.010
        base['hf_amp'] = 0.04 * r_prog.get(lead_num, 0.5)
        
    elif pattern == 'lbbb':
        r_prog = {1:0.15, 2:0.2, 3:0.25, 4:0.8, 5:1.1, 6:1.0, 7:0.8, 8:0.7}
        s_prog = {1:-1.2, 2:-1.0, 3:-0.6, 4:-0.05, 5:-0.02, 6:-0.02, 7:-0.05, 8:-0.05}
        base['r_amp'] = r_prog.get(lead_num, 0.5)
        base['s_amp'] = s_prog.get(lead_num, -0.1)
        base['r_width'] = 0.025
        base['s_width'] = 0.020
        # LBBB: significant delay V1->V6 (~40-80ms spread)
        at_shift = {1:-0.010, 2:-0.005, 3:0.005, 4:0.025, 5:0.040, 6:0.050, 7:0.045, 8:0.035}
        shift = at_shift.get(lead_num, 0)
        base['r_offset'] += shift
        base['s_offset'] += shift
        base['q_amp'] = -0.02 if lead_num >= 4 else -0.1
        base['t_amp'] = -0.3 if lead_num >= 4 else 0.4
        # HF component follows the delayed activation
        base['hf_offset'] = shift
        base['hf_width'] = 0.015  # Broader HF due to slower conduction
        base['hf_amp'] = 0.035 * r_prog.get(lead_num, 0.5)
        base['hf_phase'] = lead_num * 0.5  # Phase shift across leads
        
    elif pattern == 'rbbb':
        r_prog = {1:0.9, 2:0.7, 3:0.5, 4:0.8, 5:0.9, 6:0.8, 7:0.6, 8:0.5}
        s_prog = {1:-0.3, 2:-0.5, 3:-0.4, 4:-0.2, 5:-0.5, 6:-0.6, 7:-0.4, 8:-0.3}
        base['r_amp'] = r_prog.get(lead_num, 0.5)
        base['s_amp'] = s_prog.get(lead_num, -0.1)
        base['r_width'] = 0.020
        base['s_width'] = 0.018
        # RBBB: delayed V1-V2, early lateral
        at_shift = {1:0.035, 2:0.030, 3:0.015, 4:0.0, 5:-0.005, 6:-0.005, 7:0.0, 8:0.005}
        shift = at_shift.get(lead_num, 0)
        base['r_offset'] += shift
        base['s_offset'] += shift
        base['t_amp'] = -0.25 if lead_num <= 2 else 0.3
        base['hf_offset'] = shift
        base['hf_width'] = 0.014
        base['hf_amp'] = 0.035 * r_prog.get(lead_num, 0.5)
        base['hf_phase'] = (8 - lead_num) * 0.5
        
    return base

def _baseline_wander(t, fs):
    """Gerçekçi taban çizgisi kayması: solunum (~0.25 Hz) + yavaş drift."""
    resp = 0.05 * np.sin(2 * np.pi * 0.25 * t + np.random.uniform(0, 2 * np.pi))
    drift = 0.03 * np.sin(2 * np.pi * 0.08 * t + np.random.uniform(0, 2 * np.pi))
    return resp + drift


def _emg_noise(t, fs, amp=0.02):
    """Kas (EMG) gürültüsü: 20-150 Hz bandında bant-sınırlı gürültü.
    Yüksek frekanslı olduğu için UHF analizini zorlar — gerçekçi test sağlar."""
    from scipy.signal import butter, filtfilt
    white = np.random.randn(len(t))
    nyq = fs / 2.0
    lo, hi = 20.0 / nyq, min(150.0 / nyq, 0.99)
    if lo < hi:
        b, a = butter(2, [lo, hi], btype='band')
        emg = filtfilt(b, a, white)
        return amp * emg / (np.std(emg) + 1e-9)
    return amp * white


def _motion_artifact(t, fs, n_events=2, amp=0.15):
    """Ani hareket artefaktları: rastgele zamanlarda geçici büyük sapmalar."""
    sig = np.zeros(len(t))
    if len(t) < 10:
        return sig
    for _ in range(n_events):
        center = np.random.uniform(t[0], t[-1])
        width = np.random.uniform(0.05, 0.2)
        sig += amp * np.random.choice([-1, 1]) * np.exp(-0.5 * ((t - center) / width) ** 2)
    return sig


def generate_ecg_signal(fs=2000, duration_s=10, pattern='normal', leads=None,
                        noise_level=0.015, realism='basic', hr=70):
    """Sentetik precordial ECG üret.

    realism:
      'clean'    - neredeyse gürültüsüz (algoritma doğrulaması için)
      'basic'    - hafif beyaz gürültü + yavaş drift (eski varsayılan davranış)
      'realistic'- baseline wander + EMG + ara sıra hareket artefaktı
      'noisy'    - 'realistic' + güçlendirilmiş gürültü (dayanıklılık testi)
    hr: kalp hızı (bpm)
    """
    if leads is None:
        leads = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

    n_samples = int(duration_s * fs)
    t = np.arange(n_samples) / fs

    rr_interval = 60.0 / max(hr, 1)
    beat_times = np.arange(0.3, duration_s - 0.3, rr_interval)
    # fizyolojik RR değişkenliği
    beat_times = beat_times + np.random.normal(0, 0.008, len(beat_times))

    # realism seviyesine göre gürültü ölçekleri
    cfg = {
        'clean':     dict(white=0.002, wander=0.0,  emg=0.0,   motion=0),
        'basic':     dict(white=noise_level, wander=0.03, emg=0.0, motion=0),
        'realistic': dict(white=0.01, wander=1.0,  emg=0.015, motion=2),
        'noisy':     dict(white=0.02, wander=1.5,  emg=0.035, motion=4),
    }.get(realism, dict(white=noise_level, wander=0.03, emg=0.0, motion=0))

    signals = {}
    for lead in leads:
        params = get_lead_params(lead, pattern)
        sig = np.zeros(n_samples)
        for bt in beat_times:
            sig += generate_single_beat(t, bt, params)
        # beyaz gürültü
        sig += np.random.normal(0, cfg['white'], n_samples)
        # taban çizgisi
        if cfg['wander'] > 0:
            sig += cfg['wander'] * _baseline_wander(t, fs)
        # EMG
        if cfg['emg'] > 0:
            sig += _emg_noise(t, fs, amp=cfg['emg'])
        # hareket artefaktı
        if cfg['motion'] > 0:
            sig += _motion_artifact(t, fs, n_events=cfg['motion'])
        signals[lead] = sig

    return {'time_s': t, 'signals': signals, 'fs': fs,
            'beat_times': beat_times, 'realism': realism}

def save_ecg_csv(ecg_data, filepath):
    import pandas as pd
    data = {'time_s': ecg_data['time_s']}
    for lead, sig in ecg_data['signals'].items():
        data[lead] = sig
    df = pd.DataFrame(data)
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    df.to_csv(filepath, index=False, float_format='%.6f')
    return filepath

def generate_all_samples(output_dir, fs=2000):
    os.makedirs(output_dir, exist_ok=True)
    patterns = {
        'normal_v1v6_2000hz.csv': 'normal',
        'lbbb_v1v6_2000hz.csv': 'lbbb',
        'rbbb_v1v6_2000hz.csv': 'rbbb',
    }
    files = []
    for fname, pat in patterns.items():
        np.random.seed(42)
        ecg = generate_ecg_signal(fs=fs, duration_s=10, pattern=pat)
        fp = os.path.join(output_dir, fname)
        save_ecg_csv(ecg, fp)
        files.append(fp)
        print(f"Generated: {fp}")
    return files

if __name__ == '__main__':
    generate_all_samples('sample_data')
