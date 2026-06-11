"""
test_scientific_fixes.py - Bilimsel doğruluk düzeltmelerinin regresyon testi

Gerçek repo modüllerini (signal_processing, uhf_mapping, nd_ecg) kullanarak
sentetik normal/LBBB/RBBB sinyallerinde e-DYS ve nd-DYS'nin fizyolojik olarak
beklenen yön ve büyüklükte çıktığını doğrular.

Çalıştır:  python test_scientific_fixes.py
"""
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from signal_processing import compute_averaged_beat, estimate_qrs_onset
from uhf_mapping import compute_uhf_envelopes, compute_uhfat, compute_edys
from nd_ecg import compute_negative_derivative, compute_ndat, compute_nd_dys

np.random.seed(0)
FS = 2000
LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def make_beat(t, delay_ms, center_s):
    sig = np.zeros(len(t))
    c = center_s + delay_ms / 1000.0
    sig += 1.0 * np.exp(-0.5 * ((t - c) / 0.015) ** 2)
    amp = 0.06
    for f in [200, 350, 500, 650, 800]:
        sig += amp * np.exp(-0.5 * ((t - c) / 0.010) ** 2) * np.sin(2 * np.pi * f * (t - c))
        amp *= 0.7
    return sig


def make_signals(pattern, n_beats=8, hr=60):
    if pattern == 'normal':
        delays = {l: i * 2.0 for i, l in enumerate(LEADS)}
    elif pattern == 'lbbb':
        delays = {l: i * 8.0 for i, l in enumerate(LEADS)}
    elif pattern == 'rbbb':
        delays = {l: (len(LEADS) - 1 - i) * 7.0 for i, l in enumerate(LEADS)}
    else:
        delays = {l: 0.0 for l in LEADS}
    rr = 60.0 / hr
    duration = n_beats * rr + 1.0
    t = np.arange(0, duration, 1.0 / FS)
    signals = {l: np.zeros(len(t)) for l in LEADS}
    for k in range(n_beats):
        center = 0.5 + k * rr
        for l in LEADS:
            signals[l] += make_beat(t, delays[l], center)
    for l in LEADS:
        signals[l] += 0.01 * np.random.randn(len(t))
    r_peaks = np.array([int(0.5 * FS) + k * int(FS * rr) for k in range(n_beats)])
    return signals, r_peaks


def run(pattern):
    signals, r_peaks = make_signals(pattern)
    avg = compute_averaged_beat(signals, r_peaks, FS, window_ms=(-100, 200))
    onset = estimate_qrs_onset(avg['averaged_beats'], avg['time_axis_ms'], FS)['qrs_onset_ms']
    env = compute_uhf_envelopes(avg['averaged_beats'], FS, LEADS)
    uhfat = compute_uhfat(env, avg['time_axis_ms'], threshold=0.5, qrs_onset_ms=onset)
    edys = compute_edys(uhfat, leads_order=LEADS)
    nd = compute_negative_derivative(avg['averaged_beats'], FS)
    ndat = compute_ndat(nd, avg['time_axis_ms'], qrs_onset_ms=onset)
    nddys = compute_nd_dys(ndat, leads_order=LEADS)
    return avg, onset, uhfat, edys, nddys


ok = True
def check(name, cond):
    global ok
    if not cond: ok = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


print("=" * 60)
print("BİLİMSEL DOĞRULUK DÜZELTMELERİ - REGRESYON TESTİ")
print("=" * 60)

res = {}
for p in ['normal', 'lbbb', 'rbbb']:
    avg, onset, uhfat, edys, nddys = run(p)
    res[p] = dict(onset=onset, uhfat=uhfat, edys=edys, nddys=nddys,
                  n=avg['n_beats_used'], q=avg['quality_score'])
    print(f"\n--- {p.upper()} ---")
    print(f"  onset={onset:.1f}ms  beats={avg['n_beats_used']}  quality={avg['quality_score']:.2f}")
    print("  UHFAT: " + ", ".join(f"{l}={uhfat[l]:.1f}" for l in LEADS))
    print(f"  e-DYS={edys['e_dys']:.1f}ms ({edys['earliest_lead']}->{edys['latest_lead']})  "
          f"nd-DYS={nddys['nd_dys']:.1f}ms")

print("\n" + "=" * 60)
print("KONTROLLER")
print("=" * 60)
check("Onset-referanslı UHFAT (en erken < 30ms, R-peak merkezli değil)",
      min(res['normal']['uhfat'].values()) < 30)
check("LBBB e-DYS pozitif", res['lbbb']['edys']['e_dys'] > 0)
check("LBBB > Normal e-DYS büyüklüğü",
      res['lbbb']['edys']['e_dys_abs'] > res['normal']['edys']['e_dys_abs'])
check("LBBB V1->V6 yönü",
      res['lbbb']['edys']['earliest_lead'] == 'V1' and res['lbbb']['edys']['latest_lead'] == 'V6')
check("RBBB e-DYS negatif (ters yön)", res['rbbb']['edys']['e_dys'] < 0)
check("RBBB V6->V1 yönü",
      res['rbbb']['edys']['earliest_lead'] == 'V6' and res['rbbb']['edys']['latest_lead'] == 'V1')
check("nd-DYS e-DYS ile aynı işaret (LBBB)", res['lbbb']['nddys']['nd_dys'] > 0)
check("nd-DYS e-DYS ile aynı işaret (RBBB)", res['rbbb']['nddys']['nd_dys'] < 0)
check("Tüm pattern'lerde >=5 beat kullanıldı (ortak maske)",
      all(res[p]['n'] >= 5 for p in res))

print("\n" + "=" * 60)
print("SONUÇ:", "TÜM TESTLER GEÇTİ ✓" if ok else "BAZI TESTLER BAŞARISIZ ✗")
print("=" * 60)
sys.exit(0 if ok else 1)
