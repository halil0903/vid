"""
quality_checks.py - Analiz güvenilirliği değerlendirmesi

Sampling rate yeterliliği, lead konfigürasyonu uygunluğu ve averaging kalitesini
değerlendirir; kullanıcıya gösterilecek uyarı/güven bayraklarını üretir.
"""

# UHF-ECG için önerilen örnekleme oranları (Hz)
UHF_FS_MIN = 1000      # teknik minimum (Nyquist > ~500 Hz)
UHF_FS_RECOMMENDED = 2000  # güvenilir UHF için önerilen
UHF_FS_IDEAL = 5000    # ideal (1050 Hz banda geniş pay)

# Güvenilir e-DYS için önerilen precordial lead seti
RECOMMENDED_LEADS = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def assess_sampling_rate(fs):
    """Örnekleme oranını değerlendir.

    Döndürür: dict(level, uhf_possible, message)
      level: 'error' | 'warning' | 'ok'
    """
    if fs < 500:
        return {
            'level': 'error',
            'uhf_possible': False,
            'message': (f"Örnekleme oranı ({fs:.0f} Hz) çok düşük. Anlamlı analiz "
                        f"için en az 500 Hz (ND-ECG) gerekir.")
        }
    if fs < UHF_FS_MIN:
        return {
            'level': 'warning',
            'uhf_possible': False,
            'message': (f"Örnekleme oranı ({fs:.0f} Hz) < {UHF_FS_MIN} Hz. "
                        f"UHF-ECG analizi devre dışı; yalnızca ND-ECG mevcut.")
        }
    if fs < UHF_FS_RECOMMENDED:
        return {
            'level': 'warning',
            'uhf_possible': True,
            'message': (f"Örnekleme oranı ({fs:.0f} Hz) teknik minimumun üzerinde "
                        f"ancak {UHF_FS_RECOMMENDED} Hz altında. UHF zarfları "
                        f"150–1050 Hz bandının üst kısmında aliasing'e duyarlı "
                        f"olabilir; UHFAT/e-DYS değerlerini dikkatle yorumlayın.")
        }
    return {
        'level': 'ok',
        'uhf_possible': True,
        'message': f"Örnekleme oranı ({fs:.0f} Hz) UHF-ECG için uygun."
    }


def assess_lead_configuration(leads_order):
    """Lead setinin e-DYS güvenilirliği açısından uygunluğunu değerlendir."""
    present = [l for l in RECOMMENDED_LEADS if l in leads_order]
    n = len(present)
    missing = [l for l in RECOMMENDED_LEADS if l not in leads_order]

    if n < 2:
        return {
            'level': 'error',
            'message': (f"Yetersiz precordial lead ({n}). e-DYS için en az V1 ve V6 "
                        f"gibi yatay düzlemde ayrık iki lead gerekir.")
        }
    if n < 4:
        return {
            'level': 'warning',
            'message': (f"Sınırlı lead seti ({n}/6 precordial). Eksik: "
                        f"{', '.join(missing)}. e-DYS hesaplanır ancak yatay "
                        f"aktivasyon gradyanı az sayıda lead'e dayanır; düşük "
                        f"güvenle yorumlayın.")
        }
    if missing:
        return {
            'level': 'warning',
            'message': (f"{n}/6 precordial lead mevcut. Eksik: "
                        f"{', '.join(missing)}. Sonuçlar kullanılabilir; tam "
                        f"V1–V6 seti en güvenilir e-DYS'yi verir.")
        }
    return {
        'level': 'ok',
        'message': f"Tam precordial set (V1–V6) mevcut — e-DYS için ideal."
    }


def assess_averaging_quality(n_beats_used, quality_score,
                             min_beats=5, low_quality=0.70):
    """Beat averaging güvenilirliğini değerlendir.

    Döndürür: dict(level, confidence, message)
      confidence: 'high' | 'moderate' | 'low'
    """
    reasons = []
    level = 'ok'
    confidence = 'high'

    if n_beats_used < 3:
        return {
            'level': 'error', 'confidence': 'low',
            'message': (f"Yalnızca {n_beats_used} atım kullanıldı. Güvenilir "
                        f"ortalama için en az 3 (tercihen ≥{min_beats}) gerekir.")
        }

    if n_beats_used < min_beats:
        reasons.append(f"az sayıda atım ({n_beats_used} < {min_beats})")
        level = 'warning'
        confidence = 'moderate'

    if quality_score < low_quality:
        reasons.append(f"düşük şablon uyumu (kalite {quality_score:.0%})")
        level = 'warning'
        confidence = 'low' if quality_score < 0.5 else 'moderate'

    if reasons:
        return {
            'level': level, 'confidence': confidence,
            'message': ("Düşük güven: " + ", ".join(reasons) +
                        ". Metrikleri temkinli yorumlayın.")
        }
    return {
        'level': 'ok', 'confidence': 'high',
        'message': (f"İyi averaging: {n_beats_used} atım, "
                    f"kalite {quality_score:.0%}.")
    }


def overall_confidence(sampling_assess, lead_assess, avg_assess):
    """Üç değerlendirmeden birleşik güven seviyesi.

    En kötü bileşen sonucu belirler.
    """
    order = {'high': 2, 'moderate': 1, 'low': 0}
    conf = avg_assess.get('confidence', 'high')

    # sampling / lead 'warning' ise güveni en fazla 'moderate'e indir
    if sampling_assess['level'] == 'warning' and order[conf] > 1:
        conf = 'moderate'
    if lead_assess['level'] == 'warning' and order[conf] > 1:
        conf = 'moderate'
    if sampling_assess['level'] == 'error' or lead_assess['level'] == 'error':
        conf = 'low'

    label = {'high': 'Yüksek', 'moderate': 'Orta', 'low': 'Düşük'}[conf]
    return {'confidence': conf, 'label': label}
