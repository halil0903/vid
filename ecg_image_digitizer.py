"""
ecg_image_digitizer.py - ECG Görüntü Dijitizasyonu (Phase 2)

Temiz dijital ECG görüntülerinden (PNG/PDF cihaz çıktısı) sinyal geri çıkarımı.

ÖNEMLİ FİZİKSEL SINIR:
  Görüntüden dijitize edilen ECG'nin efektif örnekleme oranı, görüntünün
  piksel/saniye çözünürlüğüyle sınırlıdır (tipik ~100-250 Hz). UHF-ECG analizi
  150-1050 Hz bileşenler gerektirir; bu frekanslar kağıt/görüntü ECG'de FİZİKSEL
  OLARAK BULUNMAZ. Bu nedenle dijitize sinyalle yalnızca ND-ECG ve temel
  zamanlama analizi yapılabilir, UHF-ECG yapılamaz.

Yaklaşım: yarı-otomatik, kullanıcı-kalibrasyonlu dijitizasyon.
  - Grid kalibrasyonu: standart ECG grid'i (1 küçük kare = 0.04 s × 0.1 mV @ 25mm/s)
  - İz çıkarımı: sütun bazında koyu piksel takibi
"""
import numpy as np
from PIL import Image
import io


def load_image(file_bytes_or_path, target_dpi=200):
    """PNG/JPG veya PDF'i grayscale numpy dizisine yükle.
    PDF ise ilk sayfayı pdftoppm ile rasterize eder."""
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        data = bytes(file_bytes_or_path)
        is_pdf = data[:4] == b'%PDF'
        if is_pdf:
            return _pdf_to_array(data, target_dpi)
        img = Image.open(io.BytesIO(data)).convert('L')
        return np.asarray(img, dtype=np.float64)
    # path
    path = str(file_bytes_or_path)
    if path.lower().endswith('.pdf'):
        with open(path, 'rb') as f:
            return _pdf_to_array(f.read(), target_dpi)
    img = Image.open(path).convert('L')
    return np.asarray(img, dtype=np.float64)


def _pdf_to_array(pdf_bytes, dpi=200):
    """PDF ilk sayfasını pdftoppm ile PNG'e çevirip yükler."""
    import subprocess, tempfile, os
    with tempfile.TemporaryDirectory() as td:
        pdf_path = os.path.join(td, 'in.pdf')
        with open(pdf_path, 'wb') as f:
            f.write(pdf_bytes)
        out_prefix = os.path.join(td, 'page')
        subprocess.run(['pdftoppm', '-png', '-r', str(dpi), '-f', '1', '-l', '1',
                        pdf_path, out_prefix], check=True, capture_output=True)
        pngs = sorted(p for p in os.listdir(td) if p.endswith('.png'))
        if not pngs:
            raise RuntimeError("PDF rasterize edilemedi.")
        img = Image.open(os.path.join(td, pngs[0])).convert('L')
        return np.asarray(img, dtype=np.float64)


def detect_grid_spacing(gray, color_img=None):
    """ECG grid'inin küçük kare boyutunu (piksel) otomatik tahmin et.

    Yöntem: yatay/dikey projeksiyonların otokorelasyonundan baskın periyot.
    ECG grid'inde büyük kareler (5 küçük kare) daha koyudur ve otokorelasyonda
    baskın çıkar; bu yüzden bulunan periyot büyük kare olabilir. Küçük kareyi
    ayrıca daha kısa periyot aralığında arar ve büyük/küçük tutarlılığını kontrol
    eder.

    Döndürür: dict(px_per_small_box_x, px_per_small_box_y, confidence)
    """
    inv = 255.0 - gray
    col_proj = np.mean(inv, axis=0)
    row_proj = np.mean(inv, axis=1)

    def autocorr(proj):
        p = proj - np.mean(proj)
        ac = np.correlate(p, p, mode='full')[len(p)-1:]
        return ac

    def find_period(proj, lo, hi):
        ac = autocorr(proj)
        hi = min(hi, len(ac) - 1)
        if hi <= lo:
            return None, 0.0
        seg = ac[lo:hi]
        if len(seg) == 0 or np.max(seg) <= 0:
            return None, 0.0
        peak = lo + int(np.argmax(seg))
        conf = float(ac[peak] / (ac[0] + 1e-9))
        return peak, conf

    def small_box(proj):
        # Önce geniş aralıkta baskın periyodu bul (büyük veya küçük kare olabilir)
        dom, cdom = find_period(proj, 3, 80)
        if dom is None:
            return None, 0.0
        # Küçük kareyi dar aralıkta ara: 3..(dom civarı)
        small, csmall = find_period(proj, 3, max(8, int(dom * 0.7)))
        # Eğer dom, small'ın ~5 katıysa büyük kare yakalanmış demektir -> dom/5
        if small and abs(dom / small - 5.0) < 1.0:
            return small, max(cdom, csmall)
        if dom >= 20:  # muhtemelen büyük kare
            cand = dom / 5.0
            if cand >= 3:
                return cand, cdom
        return (small if small else dom), max(cdom, csmall)

    px_x, cx = small_box(col_proj)
    px_y, cy = small_box(row_proj)
    return {
        'px_per_small_box_x': px_x,
        'px_per_small_box_y': px_y,
        'confidence': float(np.mean([cx, cy])),
    }


def calibration_from_grid(px_small_x, px_small_y, paper_speed=25.0, gain=10.0):
    """Grid kalibrasyonundan ölçek faktörleri.

    Standart: 1 küçük kare = 0.04 s (x) ve 0.1 mV (y) @ 25 mm/s, 10 mm/mV.
    paper_speed: mm/s (25 veya 50)
    gain: mm/mV (10 standart)

    Döndürür: dict(s_per_px, mv_per_px, fs_effective)
    """
    # 1 küçük kare = 1 mm. Yani px_small = mm başına piksel.
    px_per_mm_x = px_small_x  # 1 küçük kare = 1 mm
    px_per_mm_y = px_small_y
    s_per_px = (1.0 / px_per_mm_x) / paper_speed     # mm/px / (mm/s) = s/px
    mv_per_px = (1.0 / px_per_mm_y) / gain           # mm/px / (mm/mV) = mV/px
    fs_effective = 1.0 / s_per_px
    return {
        's_per_px': s_per_px,
        'mv_per_px': mv_per_px,
        'fs_effective': float(fs_effective),
        'paper_speed': paper_speed,
        'gain': gain,
    }


def digitize_trace(gray, bbox, mv_per_px, s_per_px, baseline_px=None,
                   grid_suppress=True):
    """Tek bir lead izini dijitize et.

    bbox: (x0, y0, x1, y1) izin bulunduğu dikdörtgen (piksel).
    Yöntem: her piksel-sütununda en koyu (iz) pikselin y konumunu bul,
    grid çizgilerini bastır, baseline'a göre mV'ye çevir.

    Döndürür: dict(time_s, voltage_mv, n_samples)
    """
    x0, y0, x1, y1 = [int(v) for v in bbox]
    sub = gray[y0:y1, x0:x1]
    h, w = sub.shape
    if w < 2 or h < 2:
        return None

    inv = 255.0 - sub  # iz koyu -> yüksek
    # Grid bastırma: her sütunda median'ı çıkar (grid sütun boyunca ~sabit zemin)
    if grid_suppress:
        inv = inv - np.median(inv, axis=0, keepdims=True)
        inv = np.clip(inv, 0, None)

    y_idx = np.zeros(w)
    prev_y = None
    for c in range(w):
        col = inv[:, c]
        mx = np.max(col)
        if mx <= 0:
            y_idx[c] = prev_y if prev_y is not None else (
                baseline_px if baseline_px is not None else h / 2.0)
            continue
        # En koyu bölgenin ağırlık merkezi (anti-aliasing yumuşatma)
        mask = col >= 0.5 * mx
        ys = np.where(mask)[0]
        if len(ys) == 0:
            cand = float(np.argmax(col))
        else:
            # Süreklilik: birden fazla ayrık koyu bölge varsa (grid kalıntısı),
            # önceki örneğe en yakın olanı seç (iz sürekliliği)
            if prev_y is not None and len(ys) > 1:
                groups = np.split(ys, np.where(np.diff(ys) > 1)[0] + 1)
                centers = [np.mean(g) for g in groups]
                weights = [np.sum(col[g.astype(int)]) for g in groups]
                # önceki konuma yakınlık + ağırlık birleşik skoru
                best = min(range(len(groups)),
                           key=lambda i: abs(centers[i] - prev_y) - 0.3 * weights[i])
                cand = float(centers[best])
            else:
                cand = float(np.mean(ys))
        y_idx[c] = cand
        prev_y = cand

    # baseline: tüm izin medyanı (izoelektrik çizgi tahmini)
    base = baseline_px if baseline_px is not None else np.median(y_idx)
    # görüntüde y aşağı doğru artar; voltaj yukarı pozitif -> işaret ters
    voltage_mv = (base - y_idx) * mv_per_px
    time_s = np.arange(w) * s_per_px
    return {'time_s': time_s, 'voltage_mv': voltage_mv, 'n_samples': w}


def resample_uniform(time_s, voltage, fs_target):
    """Dijitize sinyali hedef fs'e eşit aralıklı yeniden örnekle."""
    if len(time_s) < 2:
        return time_s, voltage
    t_uniform = np.arange(time_s[0], time_s[-1], 1.0 / fs_target)
    v_uniform = np.interp(t_uniform, time_s, voltage)
    return t_uniform, v_uniform


def digitize_ecg_image(gray, lead_bboxes, paper_speed=25.0, gain=10.0,
                       grid_spacing=None, fs_target=None):
    """Tam dijitizasyon pipeline'ı.

    lead_bboxes: {lead_name: (x0,y0,x1,y1)}  her lead'in iz bölgesi
    grid_spacing: None ise otomatik tespit
    fs_target: None ise efektif fs kullanılır

    Döndürür: pipeline'ın beklediği ecg_data sözlüğü + dijitizasyon metası
    """
    if grid_spacing is None:
        grid_spacing = detect_grid_spacing(gray)
    px_x = grid_spacing['px_per_small_box_x']
    px_y = grid_spacing['px_per_small_box_y']
    if not px_x or not px_y or px_x < 2 or px_y < 2:
        raise ValueError("Grid tespit edilemedi. Grid boyutunu elle girin.")

    calib = calibration_from_grid(px_x, px_y, paper_speed, gain)
    fs_eff = calib['fs_effective']
    fs_use = fs_target if fs_target else fs_eff

    signals = {}
    for lead, bbox in lead_bboxes.items():
        tr = digitize_trace(gray, bbox, calib['mv_per_px'], calib['s_per_px'])
        if tr is None:
            continue
        t_u, v_u = resample_uniform(tr['time_s'], tr['voltage_mv'], fs_use)
        signals[lead] = v_u

    if not signals:
        raise ValueError("Hiçbir lead dijitize edilemedi.")

    n = len(next(iter(signals.values())))
    return {
        'signals': signals,
        'fs': float(fs_use),
        'time': np.arange(n) / fs_use,
        'duration_s': n / fs_use,
        'n_samples': n,
        'lead_names': list(signals.keys()),
        'digitization': {
            'fs_effective': fs_eff,
            'grid_confidence': grid_spacing.get('confidence', 0),
            'px_per_mm_x': px_x,
            'px_per_mm_y': px_y,
            'calibration': calib,
        }
    }
