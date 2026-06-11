"""
test_literature_validation.py - Yayınlanmış klinik değerlere karşı validasyon

Uygulamanın sentetik normal/LBBB/RBBB çıktılarının, hakemli UHF-ECG
literatüründe bildirilen e-DYS / nd-DYS aralıklarına düşüp düşmediğini doğrular.

Referanslar (reference_values.py içinde tam atıf):
  [1] Roubicek Europace 2022 — LBBB 86±20, RBBB -52±22, IVCD 26±17 ms
  [2] Curila Heart Rhythm 2024 — trueLBBB 89±15 ms, eşik >61 ms
  [3] Leinveber Europace 2023 — CRT eşiği 47 ms
  [4] Curila Sci Rep 2024 — nd-DYS > e-DYS; Normal düşük, LBBB+, RBBB-

NOT: Bu test, sentetik sinyal jeneratörünün klinik gerçekçiliğini doğrular.
Gerçek hasta verisiyle tanısal validasyon DEĞİLDİR — yayınlanmış makaleler ham
UHF sinyali paylaşmadığı için bu mümkün değildir.
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sample_data_generator import generate_ecg_signal
from signal_processing import (preprocess_ecg, detect_qrs, compute_averaged_beat,
                               estimate_qrs_onset)
from uhf_mapping import compute_uhf_envelopes, compute_uhfat, compute_edys
from nd_ecg import compute_negative_derivative, compute_ndat, compute_nd_dys
from reference_values import (validate_against_reference, check_nd_edys_consistency,
                              classify_edys, EDYS_REFERENCES, EDYS_THRESHOLDS)

FS = 2000
LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

ok = True
def check(name, cond):
    global ok
    if not cond: ok = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


def run_pipeline(pattern, realism='clean', seed=0):
    np.random.seed(seed)
    synth = generate_ecg_signal(fs=FS, duration_s=10, pattern=pattern,
                                realism=realism, leads=LEADS)
    clean = preprocess_ecg(synth['signals'], FS, 50)
    qrs = detect_qrs(clean, FS)
    avg = compute_averaged_beat(clean, qrs['r_peaks'], FS, window_ms=(-100, 200))
    onset = estimate_qrs_onset(avg['averaged_beats'], avg['time_axis_ms'], FS)['qrs_onset_ms']
    env = compute_uhf_envelopes(avg['averaged_beats'], FS, LEADS)
    uhfat = compute_uhfat(env, avg['time_axis_ms'], 0.5, qrs_onset_ms=onset)
    edys = compute_edys(uhfat, leads_order=LEADS)
    nd = compute_negative_derivative(avg['averaged_beats'], FS)
    ndat = compute_ndat(nd, avg['time_axis_ms'], qrs_onset_ms=onset)
    nddys = compute_nd_dys(ndat, leads_order=LEADS)
    return edys['e_dys'], nddys['nd_dys']


print("=" * 64)
print("LİTERATÜR VALİDASYONU (yayınlanmış UHF-ECG klinik değerleri)")
print("=" * 64)

# Her patern için 5 seed ortalaması (gürültüsüz, generator gerçekçiliği testi)
print("\n--- 1. e-DYS literatür aralığı validasyonu ---")
means = {}
for pat in ['normal', 'lbbb', 'rbbb']:
    vals = [run_pipeline(pat, 'clean', s)[0] for s in range(5)]
    me = float(np.mean(vals))
    means[pat] = me
    ref = EDYS_REFERENCES[pat]
    val = validate_against_reference(me, pat)
    print(f"  {pat:7s}: e-DYS={me:+6.1f} ms  (lit. {ref['mean']:+d}±{ref['sd']} ms "
          f"{ref['src']}, aralık {ref['range']})  ±SD={val['deviation_sd']:.2f}")
    check(f"{pat} e-DYS literatür aralığında", val['in_range'])
    check(f"{pat} e-DYS ortalamanın 2 SD içinde", val['within_2sd'])

print("\n--- 2. İşaret konvansiyonu (literatürle uyumlu) ---")
check("LBBB pozitif (sol gecikme)", means['lbbb'] > 0)
check("RBBB negatif (sağ gecikme)", means['rbbb'] < 0)
check("Normal ~ senkron (|e-DYS| < 20)", abs(means['normal']) < 20)
check("LBBB > Normal > RBBB sıralaması",
      means['lbbb'] > means['normal'] > means['rbbb'])

print("\n--- 3. Klinik eşikler ---")
check(f"LBBB, trueLBBB eşiğinin üzerinde (>{EDYS_THRESHOLDS['true_lbbb']} ms)",
      means['lbbb'] > EDYS_THRESHOLDS['true_lbbb'])
check(f"LBBB, CRT eşiğinin üzerinde (>{EDYS_THRESHOLDS['crt_response']} ms)",
      means['lbbb'] > EDYS_THRESHOLDS['crt_response'])
check("Normal, CRT eşiğinin altında",
      means['normal'] < EDYS_THRESHOLDS['crt_response'])

print("\n--- 4. nd-DYS vs e-DYS tutarlılığı (lit. [4]: nd-DYS >= e-DYS) ---")
for pat in ['lbbb', 'rbbb']:
    e, n = run_pipeline(pat, 'clean', 0)
    chk = check_nd_edys_consistency(e, n)
    print(f"  {pat}: e-DYS={e:+.1f}  nd-DYS={n:+.1f}  -> {chk['consistent']}")
    check(f"{pat}: nd-DYS/e-DYS ilişkisi literatürle uyumlu", chk['consistent'])

print("\n--- 5. Sınıflandırma çıktısı (klinik yorum) ---")
cls_lbbb = classify_edys(means['lbbb'])
cls_rbbb = classify_edys(means['rbbb'])
cls_norm = classify_edys(means['normal'])
print(f"  LBBB  -> {cls_lbbb['pattern']}")
print(f"  RBBB  -> {cls_rbbb['pattern']}")
print(f"  Normal-> {cls_norm['pattern']}")
check("LBBB sol-taraf gecikmesi olarak sınıflandı", 'sol' in cls_lbbb['pattern'].lower())
check("RBBB sağ-taraf gecikmesi olarak sınıflandı", 'sağ' in cls_rbbb['pattern'].lower())
check("LBBB CRT-relevant işaretlendi", cls_lbbb['crt_relevant'])

print("\n" + "=" * 64)
print("SONUÇ:", "TÜM TESTLER GEÇTİ ✓" if ok else "BAZI TESTLER BAŞARISIZ ✗")
print("=" * 64)
sys.exit(0 if ok else 1)
