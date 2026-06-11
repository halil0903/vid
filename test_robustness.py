"""
test_robustness.py - Sağlamlık ve güven değerlendirmesi testleri

Doğrular:
1. Gerçekçi gürültü (realistic/noisy) altında e-DYS yönü korunuyor mu
2. Güven değerlendirmesi gürültü/lead/fs koşullarına doğru tepki veriyor mu
3. Gürültü pipeline'ı çökertmiyor
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sample_data_generator import generate_ecg_signal
from signal_processing import (preprocess_ecg, detect_qrs, compute_averaged_beat,
                               estimate_qrs_onset)
from uhf_mapping import compute_uhf_envelopes, compute_uhfat, compute_edys
from quality_checks import (assess_sampling_rate, assess_lead_configuration,
                            assess_averaging_quality, overall_confidence)

FS = 2000
LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

ok = True
def check(name, cond):
    global ok
    if not cond: ok = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def pipeline(pattern, realism, seed=0):
    np.random.seed(seed)
    synth = generate_ecg_signal(fs=FS, duration_s=10, pattern=pattern,
                                realism=realism, leads=LEADS)
    clean = preprocess_ecg(synth['signals'], FS, notch_freq=50)
    qrs = detect_qrs(clean, FS)
    if qrs['n_beats'] < 3:
        return None
    avg = compute_averaged_beat(clean, qrs['r_peaks'], FS, window_ms=(-100, 200))
    onset = estimate_qrs_onset(avg['averaged_beats'], avg['time_axis_ms'], FS)['qrs_onset_ms']
    env = compute_uhf_envelopes(avg['averaged_beats'], FS, LEADS)
    uhfat = compute_uhfat(env, avg['time_axis_ms'], 0.5, qrs_onset_ms=onset)
    edys = compute_edys(uhfat, leads_order=LEADS)
    return dict(edys=edys, n=avg['n_beats_used'], q=avg['quality_score'])


print("=" * 60)
print("SAĞLAMLIK & GÜVEN TESTLERİ")
print("=" * 60)

print("\n--- 1. Gürültü altında e-DYS yön korunumu (5 seed çoğunluk) ---")
# 'clean' ve 'realistic'/'noisy' UHF için anlamlı; her birinde 5 seed'in
# çoğunluğu doğru yönü vermeli. Tek tük sapan vakalar güven bayrağıyla
# kullanıcıya zaten 'düşük güven' olarak bildirilir.
for realism in ['clean', 'realistic', 'noisy']:
    lbbb_signs = []
    rbbb_signs = []
    for seed in range(5):
        lbbb = pipeline('lbbb', realism, seed=seed)
        rbbb = pipeline('rbbb', realism, seed=seed + 100)
        if lbbb:
            lbbb_signs.append(lbbb['edys']['e_dys'] > 0)
        if rbbb:
            rbbb_signs.append(rbbb['edys']['e_dys'] < 0)
    lpos = sum(lbbb_signs)
    rneg = sum(rbbb_signs)
    print(f"  {realism:10s}: LBBB pozitif {lpos}/5  ·  RBBB negatif {rneg}/5")
    check(f"{realism}: LBBB yönü çoğunlukla doğru (>=4/5)", lpos >= 4)
    check(f"{realism}: RBBB yönü çoğunlukla doğru (>=4/5)", rneg >= 4)

print("\n--- 2. Sampling rate değerlendirmesi ---")
check("400 Hz -> error", assess_sampling_rate(400)['level'] == 'error')
check("800 Hz -> warning, uhf kapalı",
      assess_sampling_rate(800)['level'] == 'warning' and not assess_sampling_rate(800)['uhf_possible'])
check("1500 Hz -> warning, uhf açık",
      assess_sampling_rate(1500)['level'] == 'warning' and assess_sampling_rate(1500)['uhf_possible'])
check("2000 Hz -> ok", assess_sampling_rate(2000)['level'] == 'ok')

print("\n--- 3. Lead konfigürasyonu değerlendirmesi ---")
check("Tek lead -> error", assess_lead_configuration(['V1'])['level'] == 'error')
check("V1,V2 -> warning", assess_lead_configuration(['V1', 'V2'])['level'] == 'warning')
check("V1-V6 -> ok", assess_lead_configuration(LEADS)['level'] == 'ok')
check("V1-V4 eksik -> warning",
      assess_lead_configuration(['V1', 'V2', 'V3', 'V4'])['level'] == 'warning')

print("\n--- 4. Averaging güven değerlendirmesi ---")
check("2 atım -> error", assess_averaging_quality(2, 0.9)['level'] == 'error')
check("4 atım -> moderate güven",
      assess_averaging_quality(4, 0.9)['confidence'] == 'moderate')
check("8 atım kalite 0.95 -> high",
      assess_averaging_quality(8, 0.95)['confidence'] == 'high')
check("8 atım kalite 0.4 -> low",
      assess_averaging_quality(8, 0.4)['confidence'] == 'low')

print("\n--- 5. Birleşik güven ---")
good_fs = assess_sampling_rate(2000)
good_lead = assess_lead_configuration(LEADS)
good_avg = assess_averaging_quality(8, 0.95)
check("Hepsi iyi -> Yüksek",
      overall_confidence(good_fs, good_lead, good_avg)['label'] == 'Yüksek')
bad_fs = assess_sampling_rate(1500)  # warning
check("fs warning -> en fazla Orta",
      overall_confidence(bad_fs, good_lead, good_avg)['label'] in ('Orta', 'Düşük'))

print("\n" + "=" * 60)
print("SONUÇ:", "TÜM TESTLER GEÇTİ ✓" if ok else "BAZI TESTLER BAŞARISIZ ✗")
print("=" * 60)
sys.exit(0 if ok else 1)
