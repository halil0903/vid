"""
test_image_digitizer.py - ECG görüntü dijitizasyonu doğrulama testi

Bilinen sinyalden sentetik bir ECG görüntüsü (grid + iz) çizer, dijitize eder
ve geri çıkarılan sinyalin orijinaliyle uyumunu (korelasyon, zamanlama) ölçer.
"""
import numpy as np
from PIL import Image, ImageDraw
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ecg_image_digitizer import (detect_grid_spacing, calibration_from_grid,
                                 digitize_ecg_image)

# ── Sentetik görüntü parametreleri (bilinen ground truth) ──
PAPER_SPEED = 25.0   # mm/s
GAIN = 10.0          # mm/mV
PX_PER_MM = 8        # 8 piksel = 1 mm = 1 küçük kare
DUR_S = 2.5          # her lead 2.5 s
LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Ground-truth fs (görüntü çözünürlüğünden): px_per_mm * paper_speed
FS_GT = PX_PER_MM * PAPER_SPEED   # 8 * 25 = 200 Hz efektif


def make_signal(lead, fs, dur):
    """Bilinen morfolojide basit QRS dizisi (lead'e göre kaymış aktivasyon)."""
    t = np.arange(0, dur, 1.0 / fs)
    sig = np.zeros(len(t))
    delay = {'V1': 0.0, 'V2': 0.004, 'V3': 0.008, 'V4': 0.012,
             'V5': 0.016, 'V6': 0.020}.get(lead, 0)  # LBBB benzeri gradyan
    amp = {'V1': 0.3, 'V2': 0.6, 'V3': 1.0, 'V4': 1.2, 'V5': 1.0, 'V6': 0.8}.get(lead, 0.8)
    for beat_c in np.arange(0.4, dur, 0.8):  # ~75 bpm
        c = beat_c + delay
        sig += amp * np.exp(-0.5 * ((t - c) / 0.012) ** 2)       # R
        sig += -0.25 * amp * np.exp(-0.5 * ((t - c - 0.03) / 0.012) ** 2)  # S
        sig += 0.15 * np.exp(-0.5 * ((t - c + 0.18) / 0.05) ** 2)  # T
    return t, sig


def render_ecg_image():
    """Bilinen sinyallerden grid'li ECG görüntüsü çiz. Lead bbox'larını döndür."""
    px_mm = PX_PER_MM
    w = int(DUR_S * PAPER_SPEED * px_mm) + 40        # zaman ekseni
    row_h = int(30 * px_mm)                           # her lead 30 mm yüksek band (taşmayı önler)
    h = row_h * len(LEADS) + 40
    img = Image.new('L', (w, h), 255)
    draw = ImageDraw.Draw(img)

    # Grid çiz (küçük kareler açık gri, büyük kareler koyu gri)
    for x in range(0, w, px_mm):
        shade = 180 if (x // px_mm) % 5 else 120
        draw.line([(x, 0), (x, h)], fill=shade, width=1)
    for y in range(0, h, px_mm):
        shade = 180 if (y // px_mm) % 5 else 120
        draw.line([(0, y), (w, y)], fill=shade, width=1)

    bboxes = {}
    ground_truth = {}
    for i, lead in enumerate(LEADS):
        t, sig = make_signal(lead, FS_GT, DUR_S)
        ground_truth[lead] = sig
        y_center = 20 + i * row_h + row_h // 2
        x0 = 20
        # sinyali piksel koordinatına çiz (mV -> px: gain mm/mV * px_mm)
        px_per_mv = GAIN * px_mm
        pts = []
        for k in range(len(sig)):
            px = x0 + int(k * (PAPER_SPEED * px_mm) / FS_GT)
            py = int(y_center - sig[k] * px_per_mv)
            pts.append((px, py))
        draw.line(pts, fill=0, width=2)
        # bbox: bu lead'in band'ı
        bboxes[lead] = (x0, 20 + i * row_h, x0 + int(DUR_S * PAPER_SPEED * px_mm),
                        20 + (i + 1) * row_h)

    return np.asarray(img, dtype=np.float64), bboxes, ground_truth


ok = True
def check(name, cond):
    global ok
    if not cond: ok = False
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")


print("=" * 60)
print("ECG GÖRÜNTÜ DİJİTİZASYONU - DOĞRULAMA")
print("=" * 60)

gray, bboxes, gt = render_ecg_image()
print(f"\nSentetik görüntü: {gray.shape[1]}x{gray.shape[0]} px, "
      f"ground-truth fs={FS_GT:.0f} Hz, {len(LEADS)} lead")

# 1. Grid tespiti
grid = detect_grid_spacing(gray)
print(f"\nGrid tespiti: x={grid['px_per_small_box_x']} px, "
      f"y={grid['px_per_small_box_y']} px (gerçek={PX_PER_MM} px), "
      f"güven={grid['confidence']:.2f}")
check("Grid x ~ gerçek küçük kare (±2px)",
      grid['px_per_small_box_x'] and abs(grid['px_per_small_box_x'] - PX_PER_MM) <= 2)
check("Grid y ~ gerçek küçük kare (±2px)",
      grid['px_per_small_box_y'] and abs(grid['px_per_small_box_y'] - PX_PER_MM) <= 2)

# 2. Kalibrasyon -> efektif fs
calib = calibration_from_grid(PX_PER_MM, PX_PER_MM, PAPER_SPEED, GAIN)
print(f"\nKalibrasyon: fs_eff={calib['fs_effective']:.0f} Hz "
      f"(gerçek={FS_GT:.0f}), mv/px={calib['mv_per_px']:.4f}")
check("Efektif fs ~ ground truth", abs(calib['fs_effective'] - FS_GT) < 1)

# 3. Tam dijitizasyon (grid'i elle veriyoruz ki tespit hatası izolasyon dışı kalsın)
result = digitize_ecg_image(gray, bboxes, PAPER_SPEED, GAIN,
                            grid_spacing={'px_per_small_box_x': PX_PER_MM,
                                          'px_per_small_box_y': PX_PER_MM,
                                          'confidence': 1.0},
                            fs_target=FS_GT)
print(f"\nDijitize: fs={result['fs']:.0f} Hz, {result['n_samples']} örnek, "
      f"lead'ler={result['lead_names']}")

# 4. Geri çıkarılan sinyal orijinaliyle ne kadar uyumlu?
print("\nLead bazında korelasyon (dijitize vs ground truth):")
corrs = []
for lead in LEADS:
    dig = result['signals'][lead]
    truth = gt[lead]
    n = min(len(dig), len(truth))
    d, tr = dig[:n], truth[:n]
    if np.std(d) > 0 and np.std(tr) > 0:
        c = np.corrcoef(d, tr)[0, 1]
    else:
        c = 0
    corrs.append(c)
    print(f"  {lead}: r={c:.3f}")
    check(f"{lead} korelasyon > 0.85", c > 0.85)

print(f"\nOrtalama korelasyon: {np.mean(corrs):.3f}")
check("Ortalama korelasyon > 0.90", np.mean(corrs) > 0.90)

print("\n" + "=" * 60)
print("SONUÇ:", "TÜM TESTLER GEÇTİ ✓" if ok else "BAZI TESTLER BAŞARISIZ ✗")
print("=" * 60)
sys.exit(0 if ok else 1)
