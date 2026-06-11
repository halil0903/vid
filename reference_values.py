"""
reference_values.py - UHF-ECG / ND-ECG literatür referans aralıkları

Hesaplanan dissenkroni metriklerini yayınlanmış klinik değerlerle karşılaştırır
ve kaynak atıflı yorum üretir. Değerler aşağıdaki hakemli kaynaklardan alınmıştır:

[1] Roubicek T ve ark. Europace 2022;24(Suppl_1):euac053.480
    - LBBB e-DYS: 86±20 ms;  IVCD e-DYS: 26±17 ms;  RBBB e-DYS: -52±22 ms
[2] Curila K ve ark. Heart Rhythm 2024 (PO-04-181)
    - trueLBBB spontan e-DYS: 89±15 ms;  IVCD: 50±23 ms
    - e-DYS > 61 ms trueLBBB'yi ayırt eder (sens %100, spes %86)
[3] Leinveber P ve ark. Europace 2023;25(Suppl_1):euad122.461
    - CRT yanıtı için optimal e-DYS eşiği: 47 ms (sens %78, spes %79)
[4] Curila K ve ark. Sci Rep 2024;14:5566 (doi:10.1038/s41598-024-55789-w)
    - nd-DYS değerleri e-DYS'den tutarlı şekilde DAHA BÜYÜK
    - Normal: çok düşük dissenkroni; LBBB: yüksek pozitif; RBBB: yüksek negatif

İŞARET KONVANSİYONU (literatürle uyumlu):
    e-DYS > 0  -> sol-taraf gecikmesi (LBBB benzeri; V1 erken, V6 geç)
    e-DYS < 0  -> sağ-taraf gecikmesi (RBBB benzeri; V6 erken, V1 geç)
    |e-DYS| ~ 0 -> senkron aktivasyon (normal)

NOT: Bu referanslar 14-lead UHF-ECG (≥5 kHz) ile elde edilmiştir. Bu uygulama
standart örnekleme ile çalıştığında değerler yalnızca yaklaşık karşılaştırma
amaçlıdır; tanısal kullanım için tasarlanmamıştır.
"""

# Literatür e-DYS referansları (ms): (ortalama, std, kaynak)
EDYS_REFERENCES = {
    'normal':   {'mean': 0,    'sd': 15, 'range': (-20, 20),  'src': '[4]'},
    'lbbb':     {'mean': 86,   'sd': 20, 'range': (61, 110),  'src': '[1][2]'},
    'rbbb':     {'mean': -52,  'sd': 22, 'range': (-75, -25), 'src': '[1]'},
    'ivcd':     {'mean': 26,   'sd': 17, 'range': (10, 50),   'src': '[1]'},
}

# Klinik karar eşikleri (ms)
EDYS_THRESHOLDS = {
    'crt_response': 47,    # CRT yanıtını öngören eşik [3]
    'true_lbbb':    61,    # trueLBBB vs IVCD ayrımı [2]
}

# QRS süresi referansları (ms)
QRSD_REFERENCES = {
    'normal': (60, 100),
    'lbbb':   (120, 180),   # genelde ≥150 ms CRT bağlamında
    'rbbb':   (120, 180),
}


def classify_edys(e_dys):
    """e-DYS değerini literatür aralıklarına göre sınıflandır.

    Döndürür: dict(pattern, interpretation, matched_refs, crt_relevant)
    """
    abs_e = abs(e_dys)

    # Yön/örüntü
    if abs_e <= 20:
        pattern = 'Senkron / normale yakın'
        interp = ("Dissenkroni minimal (|e-DYS| ≤ 20 ms). Normal ventriküler "
                  "aktivasyona uygun.")
    elif e_dys > 20:
        if e_dys >= EDYS_THRESHOLDS['true_lbbb']:
            pattern = 'Belirgin sol-taraf gecikmesi'
            interp = (f"e-DYS = {e_dys:.0f} ms, trueLBBB eşiğinin "
                      f"(>{EDYS_THRESHOLDS['true_lbbb']} ms) üzerinde. LBBB benzeri "
                      f"belirgin interventriküler dissenkroni paterni.")
        else:
            pattern = 'Hafif-orta sol-taraf gecikmesi'
            interp = (f"e-DYS = {e_dys:.0f} ms, pozitif (sol gecikme) ancak "
                      f"trueLBBB eşiğinin altında — IVCD veya hafif LBBB ile uyumlu "
                      f"olabilir.")
    else:  # e_dys < -20
        pattern = 'Sağ-taraf gecikmesi'
        interp = (f"e-DYS = {e_dys:.0f} ms, negatif. RBBB benzeri patern (sağ "
                  f"ventrikül gecikmesi; V6 erken, V1 geç).")

    # CRT bağlamı
    crt_relevant = e_dys > EDYS_THRESHOLDS['crt_response']
    crt_note = ""
    if crt_relevant:
        crt_note = (f"e-DYS > {EDYS_THRESHOLDS['crt_response']} ms — literatürde "
                    f"CRT yanıtıyla ilişkili eşiğin üzerinde [3].")

    # Hangi referans gruplara yakın?
    matched = []
    for grp, ref in EDYS_REFERENCES.items():
        lo, hi = ref['range']
        if lo <= e_dys <= hi:
            matched.append(f"{grp.upper()} ({ref['mean']}±{ref['sd']} ms {ref['src']})")

    return {
        'pattern': pattern,
        'interpretation': interp,
        'matched_refs': matched,
        'crt_relevant': crt_relevant,
        'crt_note': crt_note,
    }


def check_nd_edys_consistency(e_dys, nd_dys):
    """Literatür [4]: nd-DYS, e-DYS'den daha büyük olmalı (mutlak değerce).

    Döndürür: dict(consistent, message)
    """
    if abs(nd_dys) >= abs(e_dys) * 0.8:  # %20 tolerans
        return {
            'consistent': True,
            'message': (f"nd-DYS ({nd_dys:.0f} ms) ile e-DYS ({e_dys:.0f} ms) "
                        f"ilişkisi literatürle uyumlu (nd-DYS genelde ≥ e-DYS [4]).")
        }
    return {
        'consistent': False,
        'message': (f"Beklenmeyen ilişki: |nd-DYS| ({abs(nd_dys):.0f}) < |e-DYS| "
                    f"({abs(e_dys):.0f}). Literatürde nd-DYS genelde daha büyüktür "
                    f"[4]; sinyal kalitesini veya kalibrasyonu kontrol edin.")
    }


def validate_against_reference(e_dys, expected_pattern, tolerance_sd=2.0):
    """Bilinen patern (normal/lbbb/rbbb/ivcd) için hesaplanan e-DYS'nin literatür
    aralığında olup olmadığını kontrol et. Validasyon testleri için kullanılır.

    Döndürür: dict(in_range, ref, deviation_sd)
    """
    ref = EDYS_REFERENCES.get(expected_pattern.lower())
    if ref is None:
        return {'in_range': None, 'ref': None, 'deviation_sd': None}
    dev = abs(e_dys - ref['mean']) / ref['sd'] if ref['sd'] > 0 else 0
    lo, hi = ref['range']
    return {
        'in_range': lo <= e_dys <= hi,
        'within_2sd': dev <= tolerance_sd,
        'ref': ref,
        'deviation_sd': dev,
    }
