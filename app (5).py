import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import re
import os
import io
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

# ─────────────────────────────────────────────
st.set_page_config(
    page_title="배터리 Second-Life 추천 플랫폼",
    page_icon="🔋",
    layout="wide"
)

st.markdown("""
<style>
    .main-title   { font-size:28px; font-weight:700; margin-bottom:4px; }
    .sub-title    { font-size:14px; color:#888; margin-bottom:24px; }
    .metric-card  { background:#1a1a2e; border-radius:12px; padding:20px;
                    text-align:center; border:1px solid #2a2a4a; }
    .metric-val   { font-size:28px; font-weight:700; color:#00d4aa; }
    .metric-label { font-size:12px; color:#aaa; margin-top:4px; }
    .rec-card     { background:#1a1a2e; border-radius:12px; padding:16px 20px;
                    margin-bottom:10px; border:1px solid #2a2a4a; }
    .top-card     { border:2px solid #00d4aa !important; }
    .section-title{ font-size:18px; font-weight:600; margin:20px 0 12px; }
    .ref-text     { font-size:11px; color:#666; margin-top:4px; }
    .mode-badge   { display:inline-block; padding:3px 10px; border-radius:20px;
                    font-size:12px; font-weight:600; margin-bottom:8px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 배터리 특성값
#
# [SOH 3단계 기준] PMC11033388
#   > 80%     → 재사용 (Reuse)
#   50~80%    → 재활용 (Repurpose)
#   <= 50%    → 해체 (Recycle)
#
# [사이클 수명 / 캘린더 열화율] Ali et al. (2023), Section 2 p.2 & Section 3
#   LFP  사이클 수명: 4,000회 이상
#   NCM  사이클 수명: 2,000회
#   NCA  사이클 수명: 1,500회
#   LFP  캘린더 열화율: 1%/년 미만
#   NCM/NCA 캘린더 열화율: ~2%/년
#   LCO  캘린더 열화율: ~3%/년
#
# [LFP 2차 수명] chrismi.sdsu.edu/publications/225.pdf
# ─────────────────────────────────────────────
BAT_PROPS = {
    # calendar_aging_rate_pct: 연간 SOH 감소율 (%, Ali et al. 2023)
    "NCM": dict(cycle_life=2000, nominal_v=3.6,
                calendar_aging_rate_pct=2.0),
    "LFP": dict(cycle_life=4000, nominal_v=3.2,
                calendar_aging_rate_pct=1.0,
                eis_threshold=60,
                second_life_cycles=(5000, 10000),
                second_life_years=(14, 28)),
    "NCA": dict(cycle_life=1500, nominal_v=3.6,
                calendar_aging_rate_pct=2.0),
    "LCO": dict(cycle_life=800,  nominal_v=3.7,
                calendar_aging_rate_pct=3.0),
}

def get_soh_tier(soh):
    if soh > 80:   return "reuse"
    elif soh > 50: return "repurpose"
    else:          return "recycle"

TIER_META = {
    "reuse":     ("재사용 (Reuse)",     "#00d4aa"),
    "repurpose": ("재활용 (Repurpose)", "#f0a500"),
    "recycle":   ("해체 (Recycle)",     "#e05555"),
}

# ═══════════════════════════════════════════════
# 1. EIS 모델 (Warwick DIB 기반)
# ═══════════════════════════════════════════════
def parse_xls_eis(file_input):
    if isinstance(file_input, (str, os.PathLike)):
        with open(file_input, 'rb') as f:
            raw = f.read()
    else:
        raw = file_input
    try:
        import xlrd
        wb = xlrd.open_workbook(file_contents=raw) if isinstance(raw, bytes) \
             else xlrd.open_workbook(file_input)
        ws = wb.sheet_by_index(0)
        freq_list, z_real_list, z_imag_list = [], [], []
        for row in range(ws.nrows):
            try:
                if ws.ncols >= 3:
                    freq = float(ws.cell_value(row, 0))
                    zr   = float(ws.cell_value(row, 1))
                    zi   = float(ws.cell_value(row, 2))
                    if freq > 0:
                        freq_list.append(freq)
                        z_real_list.append(zr)
                        z_imag_list.append(zi)
            except:
                continue
        if z_real_list:
            idx = np.argsort(freq_list)[::-1]
            return [z_real_list[i] for i in idx], [z_imag_list[i] for i in idx]
    except ImportError:
        st.warning("⚠️ xlrd 설치 필요: pip install xlrd>=2.0.1")
    except Exception as e:
        print(f"XLS 파싱 오류: {e}")
    return [], []

def parse_csv_eis(file_input):
    df = pd.read_csv(io.BytesIO(file_input) if isinstance(file_input, bytes)
                     else file_input, header=None)
    df.columns = ['freq', 'z_real', 'z_imag']
    df = df.sort_values('freq', ascending=False).reset_index(drop=True)
    return df['z_real'].tolist(), df['z_imag'].tolist()

def extract_eis_features(z_real_list, z_imag_list):
    if len(z_real_list) < 5:
        return None
    zr = np.array(z_real_list)
    zi = np.array(z_imag_list) if z_imag_list else np.zeros_like(zr)
    Rct         = zr.max() - zr[0]
    semi_height = abs(zi.min())
    semi_area   = np.pi * (Rct/2) * (semi_height/2) if Rct>0 and semi_height>0 else 0
    Z_mag       = np.sqrt(zr**2 + zi**2)
    D_value     = (Rct**2 - 4*semi_height**2) / (Rct**2 + 4*semi_height**2) \
                  if (Rct**2 + 4*semi_height**2) > 0 else 0
    return [
        float(zr[0]), float(zr.max()), float(zi.min()), float(zi.max()),
        float(zr.mean()), float(zr.std()),
        float(Rct), float(semi_height), float(semi_area),
        float(np.max(Z_mag)), float(np.mean(Z_mag)), float(np.std(Z_mag)),
        float(D_value), float(np.max(np.abs(zi))), float(np.mean(np.abs(zi))),
    ]

@st.cache_resource
def train_eis_model():
    """Warwick DIB EIS 데이터셋으로 SOH 예측 모델 학습"""
    import zipfile
    base_dir      = os.path.dirname(__file__)
    zip_data      = os.path.join(base_dir, 'data', 'EIS_Test.zip')
    zip_root      = os.path.join(base_dir, 'EIS_Test.zip')
    dir_path      = os.path.join(base_dir, 'data', 'EIS_Test')

    file_items = []
    if os.path.exists(zip_data):
        with zipfile.ZipFile(zip_data, 'r') as zf:
            for zname in zf.namelist():
                fname = os.path.basename(zname)
                if fname.endswith('.xls') and 'SOH' in fname:
                    file_items.append((fname, zf.read(zname)))
    elif os.path.exists(zip_root):
        with zipfile.ZipFile(zip_root, 'r') as zf:
            for zname in zf.namelist():
                fname = os.path.basename(zname)
                if fname.endswith('.xls') and 'SOH' in fname:
                    file_items.append((fname, zf.read(zname)))
    elif os.path.exists(dir_path):
        for fname in os.listdir(dir_path):
            if fname.endswith('.xls') and 'SOH' in fname:
                file_items.append((fname, os.path.join(dir_path, fname)))
    else:
        return None, 0, 0, 0

    X, y = [], []
    for fname, file_data in file_items:
        m = re.search(r'(\d+)SOH', fname)
        if not m: continue
        soh = int(m.group(1))
        try:
            raw = file_data if isinstance(file_data, bytes) else None
            zr, zi = parse_xls_eis(raw if raw else file_data)
            feats = extract_eis_features(zr, zi)
            if feats:
                X.append(feats); y.append(soh)
        except:
            continue

    if len(X) < 10:
        return None, len(X), 0, 0

    X, y  = np.array(X), np.array(y)
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    gb = GradientBoostingRegressor(n_estimators=300, max_depth=6,
                                   learning_rate=0.05, subsample=0.8, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    gb.fit(Xs, y); rf.fit(Xs, y)

    cv_gb = cross_val_score(gb, Xs, y, cv=5, scoring='r2').mean()
    cv_rf = cross_val_score(rf, Xs, y, cv=5, scoring='r2').mean()

    return {'gb': gb, 'rf': rf, 'scaler': scaler}, len(X), cv_gb, cv_rf

def predict_soh_eis(models, zr, zi):
    feats = extract_eis_features(zr, zi)
    if feats is None: return None
    Xs    = models['scaler'].transform([feats])
    pred  = (models['gb'].predict(Xs)[0] + models['rf'].predict(Xs)[0]) / 2
    return round(float(np.clip(pred, 50, 100)), 1)

# ═══════════════════════════════════════════════
# 2. BMS 모델 (NASA PCoE + Ali et al. 2023 기반)
#
# 학습 데이터 설계 근거:
# - 배터리별 사이클 수명: Ali et al. (2023) Section 2
#     NCM 2,000회 / LFP 4,000회 / NCA 1,500회
# - 캘린더 열화율: Ali et al. (2023) Section 3
#     LFP <1%/년 / NCM·NCA ~2%/년 / LCO ~3%/년
# - 내부 저항 범위: NASA PCoE B0005~B0018 실측
#     LCO 초기 0.15Ω → 말기 0.30Ω
# - SOH 열화 모델:
#     SOH = 100 - 20×min(cycle/cycle_life, 1.5) - cal_rate×years
#
# 피처 9개 (수동 입력 가능한 BMS 측정값 중심):
#   cycle_count, years, internal_resistance,
#   charge_time, temp_rise, voltage_drop,
#   coulombic_efficiency, bat_type_enc, cycle_life
#
# 출처: Ali et al. (2023); NASA PCoE ti.arc.nasa.gov/tech/dash/groups/pcoe/
# ═══════════════════════════════════════════════

# 배터리 종류 인코딩
BAT_ENC = {"NCM": 0, "LFP": 1, "NCA": 2, "LCO": 3}
# 배터리별 내부저항 기준값 (Ali et al. 2023 + NASA PCoE)
BAT_RESISTANCE = {
    "NCM": dict(r0=0.10, r_max=0.22),
    "LFP": dict(r0=0.08, r_max=0.16),
    "NCA": dict(r0=0.12, r_max=0.28),
    "LCO": dict(r0=0.15, r_max=0.30),
}

@st.cache_resource
def train_bms_model():
    """
    배터리 종류별 열화 특성을 반영한 BMS SOH 예측 모델 학습
    - NCM/LFP/NCA/LCO 각 500개 = 총 2,000개 학습 데이터
    - 사이클 범위: 0 ~ cycle_life × 1.5 (수명 초과 케이스 포함)
    - SOH 열화: 사이클 열화 + 캘린더 열화 복합 반영

    출처: Ali et al. (2023); NASA PCoE
    """
    np.random.seed(42)
    N_per = 500
    all_X, all_y = [], []

    for bat, cfg in BAT_PROPS.items():
        cl  = cfg['cycle_life']
        cr  = cfg.get('calendar_aging_rate_pct', 2.0)
        r0  = BAT_RESISTANCE[bat]['r0']
        rm  = BAT_RESISTANCE[bat]['r_max']

        # 사이클: 0 ~ cycle_life×1.5 (수명 초과 포함)
        cycle = np.random.randint(0, int(cl * 1.5), N_per).astype(float)
        years = np.random.uniform(0, 15, N_per)

        # SOH 열화 모델 (Ali et al. 2023)
        soh_cycle = 100 - 20 * np.clip(cycle / cl, 0, 1.5)
        soh_cal   = cr * years
        soh = np.clip(soh_cycle - soh_cal + np.random.normal(0, 1.5, N_per), 50, 100)

        # 피처 생성 (BMS 측정값, SOH와 물리적으로 연동)
        # 내부 저항: 사이클비에 비례해 r0→rm 증가 (NASA PCoE 실측 패턴)
        int_r   = r0 + (rm-r0)*np.clip(cycle/cl, 0, 1.3) + np.random.normal(0, 0.004, N_per)
        int_r   = np.clip(int_r, r0, rm*1.3)
        # 충전 시간: 노화 → 증가
        chg_t   = 3600*(1+(100-soh)/300) + np.random.normal(0, 100, N_per)
        # 온도 상승폭: 노화 → 발열 증가
        t_rise  = 5+(100-soh)/12 + np.random.normal(0, 0.5, N_per)
        # 전압 강하: 노화 → 증가
        v_drop  = 0.05+(100-soh)/1200 + np.random.normal(0, 0.004, N_per)
        # 쿨롱 효율: 노화 → 감소
        c_eff   = 0.99-(100-soh)/6000 + np.random.normal(0, 0.002, N_per)
        bat_f   = np.full(N_per, float(BAT_ENC[bat]))
        cl_f    = np.full(N_per, float(cl))

        X = np.column_stack([cycle, years, int_r, chg_t, t_rise,
                             v_drop, c_eff, bat_f, cl_f])
        all_X.append(X); all_y.append(soh)

    X_all = np.vstack(all_X)
    y_all = np.concatenate(all_y)

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X_all)

    gb = GradientBoostingRegressor(n_estimators=400, max_depth=6,
                                   learning_rate=0.04, subsample=0.8, random_state=42)
    rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
    gb.fit(Xs, y_all); rf.fit(Xs, y_all)

    cv_gb = cross_val_score(gb, Xs, y_all, cv=5, scoring='r2').mean()
    cv_rf = cross_val_score(rf, Xs, y_all, cv=5, scoring='r2').mean()

    return {'gb': gb, 'rf': rf, 'scaler': scaler}, cv_gb, cv_rf

# BMS 피처 정의 (9개)
BMS_FEATURE_COLS = [
    'cycle_count', 'years', 'internal_resistance_ohm',
    'charge_time_s', 'temp_rise_c', 'voltage_drop_v',
    'coulombic_efficiency', 'bat_type_enc', 'cycle_life'
]
BMS_FEATURE_LABELS = [
    '사이클 수', '사용 연수 (년)', '내부 저항 (Ω)',
    '충전 시간 (s)', '온도 상승폭 (°C)', '전압 강하 (V)',
    '쿨롱 효율', '배터리 종류 (인코딩)', '설계 사이클 수명'
]

def build_bms_features(bat_type, cycle, years, int_r):
    """
    핵심 3개 입력(사이클, 연수, 내부저항)으로 전체 피처 자동 계산
    ─────────────────────────────────────────────────────────
    문제: 충전시간·온도·전압강하 등을 기본값으로 고정하면
          ML 모델이 사이클/연수를 무시하고 고정값만 보고 판단함
    해결: 사이클+연수+내부저항으로 SOH를 먼저 추정하고,
          추정된 SOH로 나머지 피처를 물리 모델에서 역산
    ─────────────────────────────────────────────────────────
    [근거]
    - 사이클 열화: Ali et al. (2023), cycle_life 도달 시 SOH 80%
    - 캘린더 열화: Ali et al. (2023), LFP <1%/년, NCM/NCA ~2%/년
    - 내부저항 범위: NASA PCoE B0005~B0018 실측
    """
    cfg = BAT_PROPS[bat_type]
    cl, cr = cfg['cycle_life'], cfg.get('calendar_aging_rate_pct', 2.0)
    r0 = BAT_RESISTANCE[bat_type]['r0']
    rm = BAT_RESISTANCE[bat_type]['r_max']

    # 1단계: 사이클+캘린더 열화로 이론 SOH 계산 (Ali et al. 2023)
    soh_cycle  = 100 - 20 * min(cycle / cl, 1.5)
    soh_theory = max(50.0, soh_cycle - cr * years)

    # 2단계: 내부저항 기반 SOH 추정
    r_ratio   = (int_r - r0) / (rm - r0) if (rm - r0) > 0 else 0
    soh_from_r = max(50.0, 100 - 20 * min(r_ratio, 1.3))

    # 3단계: 가중 평균 (사이클+연수 60%, 내부저항 40%)
    soh_est = soh_theory * 0.6 + soh_from_r * 0.4

    # 4단계: 추정 SOH로 나머지 피처 물리 모델 역산
    chg_t  = 3600 * (1 + (100 - soh_est) / 300)
    t_rise = 5 + (100 - soh_est) / 12
    v_drop = 0.05 + (100 - soh_est) / 1200
    c_eff  = 0.99 - (100 - soh_est) / 6000

    return [float(cycle), float(years), float(int_r),
            chg_t, t_rise, v_drop, c_eff,
            float(BAT_ENC[bat_type]), float(cl)]


def predict_soh_bms(models, bat_type, cycle, years, int_r):
    """
    핵심 3개 입력 → 피처 자동 계산 → SOH 예측 (앙상블)
    입력: bat_type, cycle(사이클 수), years(연수), int_r(내부저항 Ω)
    """
    feats = build_bms_features(bat_type, cycle, years, int_r)
    Xs    = models['scaler'].transform([feats])
    pred  = (models['gb'].predict(Xs)[0] + models['rf'].predict(Xs)[0]) / 2
    return round(float(np.clip(pred, 50, 100)), 1)

def extract_bms_features_from_csv(df, bat_type):
    """
    BMS CSV 로그에서 새 9-피처 구조로 추출
    CSV 컬럼: cycle, voltage, current, temperature, time_s
    피처: cycle_count, years, internal_resistance_ohm, charge_time_s,
          temp_rise_c, voltage_drop_v, coulombic_efficiency, bat_type_enc, cycle_life
    """
    features_list = []
    cl = BAT_PROPS[bat_type]['cycle_life']

    for cycle_id, grp in df.groupby('cycle'):
        try:
            charge_grp    = grp[grp['current'] > 0]
            discharge_grp = grp[grp['current'] < 0]
            if charge_grp.empty or discharge_grp.empty:
                continue

            charge_cap    = (charge_grp['current'].abs() *
                             charge_grp['time_s'].diff().fillna(0)).sum() / 3600
            discharge_cap = (discharge_grp['current'].abs() *
                             discharge_grp['time_s'].diff().fillna(0)).sum() / 3600
            charge_time   = charge_grp['time_s'].max() - charge_grp['time_s'].min()
            voltage_drop  = charge_grp['voltage'].max() - charge_grp['voltage'].iloc[-1]
            temp_rise     = grp['temperature'].max() - grp['temperature'].min()
            dv = grp['voltage'].diff().abs().mean()
            di = grp['current'].diff().abs().mean()
            internal_r    = float(dv / di) if di > 0 else BAT_RESISTANCE[bat_type]['r0']
            coulombic_eff = discharge_cap / charge_cap if charge_cap > 0 else 0.99
            # 사이클 → 연수 추정 (하루 1회 기준)
            est_years     = float(cycle_id) / 365.0

            features_list.append({
                'cycle_count':             float(cycle_id),
                'years':                   est_years,
                'internal_resistance_ohm': float(np.clip(internal_r, 0.01, 1.0)),
                'charge_time_s':           float(charge_time),
                'temp_rise_c':             float(temp_rise),
                'voltage_drop_v':          float(voltage_drop),
                'coulombic_efficiency':    float(np.clip(coulombic_eff, 0.8, 1.0)),
                'bat_type_enc':            float(BAT_ENC[bat_type]),
                'cycle_life':              float(cl),
            })
        except:
            continue
    return features_list


# ═══════════════════════════════════════════════
# NASA .mat 파일 파싱 및 모델 검증
# 출처: NASA PCoE Battery Dataset
#   ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/
# ═══════════════════════════════════════════════
def parse_nasa_mat(file_bytes):
    """
    NASA PCoE .mat 파일 파싱
    - v7.3 이하: scipy.io.loadmat
    - v7.3 (HDF5): h5py

    반환:
        cycles_data: list of dict
            {
              'cycle_idx': int,
              'type': 'charge'|'discharge'|'impedance',
              'capacity_ah': float (discharge만),
              'voltage_mean': float,
              'current_mean': float,
              'temp_mean': float,
              'charge_time_s': float,
              'internal_r': float (impedance만, 없으면 None),
            }
        battery_name: str (예: 'B0005')
    """
    import io

    # ── scipy 시도 (v7.3 이하) ──────────────────
    try:
        import scipy.io
        mat = scipy.io.loadmat(
            io.BytesIO(file_bytes), simplify_cells=True
        )
        # 배터리 이름 찾기 (B0005, B0006, ...)
        bat_name = [k for k in mat.keys() if k.startswith('B')][0]
        cycles_raw = mat[bat_name]['cycle']
        return _parse_scipy_cycles(cycles_raw), bat_name
    except Exception:
        pass

    # ── h5py 시도 (v7.3 HDF5) ──────────────────
    try:
        import h5py
        import tempfile, os
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mat') as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            with h5py.File(tmp_path, 'r') as f:
                bat_name = [k for k in f.keys() if k.startswith('B')][0]
                return _parse_hdf5_cycles(f[bat_name]), bat_name
        finally:
            os.unlink(tmp_path)
    except Exception as e:
        raise ValueError(f"mat 파일 파싱 실패: {e}")


def _parse_scipy_cycles(cycles_raw):
    """scipy simplify_cells=True로 파싱된 사이클 처리"""
    results = []
    if not hasattr(cycles_raw, '__iter__'):
        cycles_raw = [cycles_raw]

    for i, c in enumerate(cycles_raw):
        try:
            ctype = str(c.get('type', '')).strip().lower()
            data  = c.get('data', {})
            if not data:
                continue

            entry = {'cycle_idx': i, 'type': ctype,
                     'capacity_ah': None, 'internal_r': None}

            v = np.atleast_1d(data.get('Voltage_measured', []))
            a = np.atleast_1d(data.get('Current_measured', []))
            t = np.atleast_1d(data.get('Temperature_measured', []))
            ts= np.atleast_1d(data.get('Time', []))

            entry['voltage_mean']  = float(np.mean(v))  if len(v)  else 0.0
            entry['current_mean']  = float(np.mean(np.abs(a))) if len(a) else 0.0
            entry['temp_mean']     = float(np.mean(t))  if len(t)  else 25.0
            entry['charge_time_s'] = float(ts[-1]-ts[0]) if len(ts)>1 else 0.0

            if ctype == 'discharge':
                cap = data.get('Capacity', None)
                if cap is not None:
                    entry['capacity_ah'] = float(np.atleast_1d(cap).flat[0])

            if ctype == 'impedance':
                re = data.get('Re', None)
                if re is not None:
                    entry['internal_r'] = float(np.atleast_1d(re).flat[0])

            results.append(entry)
        except Exception:
            continue
    return results


def _parse_hdf5_cycles(bat_group):
    """h5py로 파싱된 HDF5 구조 처리"""
    import h5py
    results = []
    cycle_group = bat_group.get('cycle', bat_group)

    for i, key in enumerate(cycle_group.keys()):
        try:
            c = cycle_group[key]
            ctype_raw = c.get('type', None)
            if ctype_raw is None:
                continue
            # HDF5에서 문자열 디코딩
            if isinstance(ctype_raw, h5py.Dataset):
                ctype = ''.join(chr(x) for x in ctype_raw[()]).strip().lower()
            else:
                ctype = str(ctype_raw).strip().lower()

            data = c.get('data', c)
            entry = {'cycle_idx': i, 'type': ctype,
                     'capacity_ah': None, 'internal_r': None}

            def get_arr(key):
                d = data.get(key, None)
                if d is None: return np.array([])
                return np.array(d[()])

            v  = get_arr('Voltage_measured')
            a  = get_arr('Current_measured')
            t  = get_arr('Temperature_measured')
            ts = get_arr('Time')

            entry['voltage_mean']  = float(np.mean(v))  if len(v)  else 0.0
            entry['current_mean']  = float(np.mean(np.abs(a))) if len(a) else 0.0
            entry['temp_mean']     = float(np.mean(t))  if len(t)  else 25.0
            entry['charge_time_s'] = float(ts[-1]-ts[0]) if len(ts)>1 else 0.0

            if ctype == 'discharge':
                cap = data.get('Capacity', None)
                if cap is not None:
                    entry['capacity_ah'] = float(np.array(cap[()]).flat[0])

            if ctype == 'impedance':
                re = data.get('Re', None)
                if re is not None:
                    entry['internal_r'] = float(np.array(re[()]).flat[0])

            results.append(entry)
        except Exception:
            continue
    return results


def compute_soh_from_mat(cycles_data):
    """
    discharge 사이클의 Capacity로 SOH 계산
    SOH = 현재 방전 용량 / 초기 방전 용량 × 100
    출처: NASA PCoE 데이터셋 정의
    """
    discharge = [c for c in cycles_data if c['type'] == 'discharge'
                 and c['capacity_ah'] is not None]
    if not discharge:
        return []

    rated_cap = discharge[0]['capacity_ah']  # 초기 용량 = 정격 용량
    result = []
    for i, c in enumerate(discharge):
        soh = c['capacity_ah'] / rated_cap * 100
        result.append({
            'discharge_idx':  i,
            'cycle_idx':      c['cycle_idx'],
            'capacity_ah':    c['capacity_ah'],
            'soh_actual':     round(soh, 2),
            'voltage_mean':   c['voltage_mean'],
            'current_mean':   c['current_mean'],
            'temp_mean':      c['temp_mean'],
            'charge_time_s':  c['charge_time_s'],
        })

    # impedance 사이클의 내부저항 매칭 (가장 가까운 사이클)
    impedance = [c for c in cycles_data if c['type'] == 'impedance'
                 and c['internal_r'] is not None]
    if impedance:
        imp_idx = np.array([c['cycle_idx'] for c in impedance])
        imp_r   = np.array([c['internal_r'] for c in impedance])
        for row in result:
            nearest = np.argmin(np.abs(imp_idx - row['cycle_idx']))
            row['internal_r'] = float(imp_r[nearest])
    else:
        for row in result:
            row['internal_r'] = None

    return result

# ═══════════════════════════════════════════════
# 공통 유틸
# ═══════════════════════════════════════════════
def get_recommendations(health, years, cycles, bat_type, voltage):
    """
    활용처 추천 적합도 계산
    ─────────────────────────────────────────────
    [근거] PMC11033388, IEC 62933, Ali et al. (2023)

    적합도 base 점수 구성:
    1. SOH (핵심 지표, 100점 만점 기준)
    2. 사이클 페널티: 최대 10점 상한
       (사이클 초과가 SOH 실측값 판정을 역전하지 않도록)
    3. 캘린더 열화 페널티: 연수 × 열화율 (Ali et al. 2023)
       LFP <1%/년, NCM/NCA ~2%/년, LCO ~3%/년 — 최대 10점 상한
    4. 전압 이탈 페널티: 공칭 전압 ±0.3V 초과 시만 적용
    ─────────────────────────────────────────────
    """
    props         = BAT_PROPS[bat_type]
    cycle_ratio   = cycles / props['cycle_life']
    cal_rate      = props.get('calendar_aging_rate_pct', 2.0)  # Ali et al. (2023)

    # 페널티 계산 (모두 최대 10점 상한)
    cycle_penalty = min(cycle_ratio * 10, 10)
    # 캘린더 열화 페널티: 누적 열화량 기반, 최대 10점
    cal_penalty   = min(years * cal_rate, 10)
    base          = health - cycle_penalty - cal_penalty
    v_diff        = abs(voltage - props['nominal_v'])
    if v_diff > 0.3: base -= v_diff * 10
    tier          = get_soh_tier(health)
    lfp_note      = ""
    if bat_type == "LFP" and health < props.get('eis_threshold', 60):
        lfp_note  = f" ※ LFP SOH {props['eis_threshold']}% 미만 → 임피던스 급증 구간"

    apps = [
        dict(name="태양광 연계 ESS", icon="☀️",
             desc="재생에너지 저장. 낮은 C-rate, 1일 1~2회 충방전." + lfp_note,
             ref="PMC11033388 (ESS SOH 70~80%); IEC 62933",
             score=max(0, base+5), condition=health>=70,
             tier_label="재사용 ✅" if tier=="reuse" else "재활용 ♻️"),
        dict(name="가정용 ESS", icon="🏠",
             desc="저출력 장기 사용. 태양광 잉여전력 저장." + lfp_note,
             ref="PMC11033388; UL 1974",
             score=max(0, base), condition=health>=70,
             tier_label="재사용 ✅" if tier=="reuse" else "재활용 ♻️"),
        dict(name="통신기지국 백업전원", icon="📡",
             desc="간헐적 방전. 부동충전 위주로 배터리 부담 낮음.",
             ref="Martinez-Laserna et al. (2018); PMC11033388",
             score=max(0, base-5), condition=health>=60,
             tier_label="재활용 ♻️"),
        dict(name="전기차 보조 배터리", icon="🚗",
             desc="저/중 출력. 일일 충방전 100회 이상 가능.",
             ref="PMC11033388; Frontiers in Energy Research (2023)",
             score=max(0, base-10), condition=health>=60,
             tier_label="재활용 ♻️"),
        dict(name="무정전전원장치 (UPS)", icon="⚡",
             desc="간헐적 방전. 응급 상황 대비. (grade C, SOH 50% 이상)",
             ref="PMC11033388 (UPS SOH 50% 이상, grade C); IEC 62619",
             score=max(0, base-15), condition=health>=50 and tier!="recycle",
             tier_label="재활용 ♻️"),
    ]
    return [a for a in apps if a['condition'] and a['score'] > 0]

def safety_eval(health, years, cycles, bat_type, voltage):
    """
    안전성 판정
    ─────────────────────────────────────────────
    [1차 기준] SOH — PMC11033388
      > 80%    → 재사용 / 50~80% → 재활용 / ≤50% → 해체

    [보조 경고] 사이클 & 캘린더 열화 — Ali et al. (2023)
      - 사이클 수명: LFP 4,000회 / NCM 2,000회 / NCA 1,500회
      - 캘린더 열화율: LFP <1%/년, NCM/NCA ~2%/년, LCO ~3%/년
      - 초과 시 판정 등급 강등 없이 경고 메시지만 표시

    [LFP 임피던스] PMC11033388 — SOH 60% 미만 급증
    ─────────────────────────────────────────────
    """
    props        = BAT_PROPS[bat_type]
    cycle_ratio  = cycles / props['cycle_life']
    cal_rate     = props.get('calendar_aging_rate_pct', 2.0)  # Ali et al. (2023)
    # 캘린더 열화 예상 SOH 감소량 (연수 × 연간 열화율)
    cal_loss     = years * cal_rate
    tier         = get_soh_tier(health)

    # 보조 경고 (판정 등급을 바꾸지 않음)
    warnings = []
    if cycle_ratio > 1.0:
        warnings.append(
            f"⚠️ 설계 사이클 수명 초과 ({cycles}회 / 기준 {props['cycle_life']}회)"
            f" — 집중 모니터링 권장 (Ali et al. 2023; Frontiers in Energy Research, 2023)"
        )
    elif cycle_ratio > 0.75:
        warnings.append(
            f"⚠️ 사이클 수명 {round(cycle_ratio*100)}% 소모"
            f" — 주기적 점검 권장 (Ali et al. 2023)"
        )
    # 캘린더 열화 경고: 예상 누적 손실이 10% 초과 시
    if cal_loss >= 10:
        warnings.append(
            f"⚠️ 캘린더 열화 누적 약 {cal_loss:.0f}% 예상"
            f" ({bat_type} {cal_rate}%/년 × {years}년, Ali et al. 2023)"
        )
    if bat_type == "LFP":
        thr = props.get('eis_threshold', 60)
        if health < thr:
            warnings.append(f"⚠️ LFP SOH {thr}% 미만: 임피던스 급증 구간 (PMC11033388)")
    warn_str = (" | " + " | ".join(warnings)) if warnings else ""

    # 1차 판정: SOH 기준 (PMC11033388)
    if tier == "recycle":
        return ("위험 — 해체(Recycle)", "#e05555",
                f"SOH ≤ 50%: 재활용 공정 투입 필요 (PMC11033388){warn_str}")
    elif tier == "repurpose":
        return ("주의 — 재활용(Repurpose)", "#f0a500",
                f"SOH 50~80%: 제한된 용도 재활용 가능, 주기적 점검 필요 (PMC11033388){warn_str}")
    else:
        return ("양호 — 재사용(Reuse)", "#00d4aa",
                f"SOH > 80%: 안전한 재사용 가능 (PMC11033388; IEC 62933, UL 1974){warn_str}")

def render_result(soh_final, soh_source, bat_type, years, cycles, voltage, mode_label):
    """진단 결과 공통 렌더링"""
    tier       = get_soh_tier(soh_final)
    tier_text, tier_color = TIER_META[tier]
    soh_color  = "#00d4aa" if soh_final > 80 else "#f0a500" if soh_final > 50 else "#e05555"

    # LFP 2차 수명 안내
    if bat_type == "LFP" and 75 <= soh_final <= 85:
        p  = BAT_PROPS["LFP"]
        st.info(
            f"🔋 **LFP 2차 수명 안내** (SOH ≈ 80% 전환 시점)\n\n"
            f"전압 2.80~3.55 V / 충전 0.5C / 방전 1C 조건에서 "
            f"용량 60% 도달까지 **{p['second_life_cycles'][0]:,}~{p['second_life_cycles'][1]:,}사이클**, "
            f"하루 1회 기준 **{p['second_life_years'][0]}~{p['second_life_years'][1]}년** 기대\n\n"
            f"📚 출처: chrismi.sdsu.edu/publications/225.pdf"
        )

    # LFP 임피던스 경고
    if bat_type == "LFP":
        thr = BAT_PROPS["LFP"].get('eis_threshold', 60)
        if soh_final < thr:
            st.warning(f"⚠️ **LFP 임피던스 주의**: SOH {thr}% 미만 — 임피던스 급증 및 용량 저하 시작 (PMC11033388)")

    # 진단 메트릭
    st.markdown(f'<div class="section-title">🤖 진단 결과 <span style="font-size:13px;color:#888;">— {mode_label}</span></div>',
                unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    for col, val, label, color, note in zip(
        [m1, m2, m3, m4],
        [f"{soh_final}%", f"{bat_type}", f"{cycles}회 / {years}년", tier_text],
        ["SOH", "배터리 종류", "사이클 / 사용 연수", "판정 등급"],
        [soh_color, "#00d4aa", "#00d4aa", tier_color],
        [soh_source[:28]+"...", "화학 조성", "누적 사용 이력",
         ">80% 재사용 / 50~80% 재활용 / ≤50% 해체"]
    ):
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-val" style="color:{color}">{val}</div>
            <div class="metric-label">{label}</div>
            <div class="ref-text">{note}</div>
        </div>""", unsafe_allow_html=True)

    # 안전성
    st.markdown('<div class="section-title">🛡️ 안전성 평가</div>', unsafe_allow_html=True)
    s_txt, s_color, s_desc = safety_eval(soh_final, years, cycles, bat_type, voltage)
    st.markdown(f"""
    <div class="metric-card" style="text-align:left; border:2px solid {s_color};">
        <span style="font-size:20px; font-weight:700; color:{s_color}">{s_txt}</span>
        <span style="font-size:14px; color:#ccc; margin-left:12px;">{s_desc}</span>
    </div>""", unsafe_allow_html=True)

    # 추천 활용처
    st.markdown('<div class="section-title">🎯 추천 활용처</div>', unsafe_allow_html=True)
    st.caption("📌 PMC11033388: ESS/그리드 SOH 70~80%, UPS 비상전원 SOH 50% 이상(grade C)")
    recs = get_recommendations(soh_final, years, cycles, bat_type, voltage)
    if not recs:
        st.error("❌ 모든 활용처 기준 미달 — 해체(Recycle) 공정 투입 권장 (SOH ≤ 50%, PMC11033388)")
    else:
        for i, rec in enumerate(recs):
            cls   = "rec-card top-card" if i == 0 else "rec-card"
            rank  = "✦ 최우선 추천" if i == 0 else f"{i+1}순위 추천"
            st.markdown(f"""
            <div class="{cls}">
                <div style="font-size:16px; font-weight:600;">
                    {rec['icon']} {rec['name']}
                    <span style="font-size:12px; color:#aaa; margin-left:8px;">{rec['tier_label']}</span>
                </div>
                <div style="font-size:12px; color:#aaa;">{rank} · 적합도 {round(rec['score'])}점</div>
                <div style="font-size:13px; color:#bbb; margin-top:6px;">{rec['desc']}</div>
                <div style="font-size:11px; color:#666; margin-top:4px;">📚 {rec['ref']}</div>
            </div>""", unsafe_allow_html=True)

    # 최종 판단
    st.divider()
    cycle_pct = round(cycles / BAT_PROPS[bat_type]['cycle_life'] * 100)
    if tier == "recycle":
        fc, fm, fr = "#e05555", "❌ 재사용 불가 — 해체(Recycle) 공정 필요", "PMC11033388 (SOH ≤ 50%); IEC 62619"
    elif tier == "repurpose":
        fc, fm, fr = "#f0a500", "♻️ 재활용(Repurpose) 가능 — 제한된 용도 사용, 주기적 점검 필요", "PMC11033388 (SOH 50~80%)"
    else:
        fc, fm, fr = "#00d4aa", "✅ 재사용(Reuse) 가능", "PMC11033388 (SOH > 80%); IEC 62933, UL 1974"

    st.markdown(f"""
    <div style="background:#1a1a2e; border-radius:12px; padding:20px;
                border:2px solid {fc}; text-align:center;">
        <div style="font-size:24px; font-weight:700; color:{fc}">{fm}</div>
        <div style="font-size:13px; color:#aaa; margin-top:8px;">
            배터리 종류: {bat_type} | SOH: {soh_final}% | 사용 연수: {years}년 |
            충방전: {cycles}회 ({cycle_pct}% 소모) | 전압: {voltage}V
        </div>
        <div style="font-size:11px; color:#666; margin-top:6px;">📚 근거: {fr}</div>
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════
# 메인 앱
# ═══════════════════════════════════════════════
st.markdown('<h1 class="main-title">🔋 배터리 Second-Life 추천 플랫폼</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">EIS 또는 BMS 데이터 기반 배터리 상태 진단 및 재사용/재활용/해체 판정</p>',
            unsafe_allow_html=True)

# 모델 로드
with st.spinner("🤖 모델 로딩 중..."):
    eis_result  = train_eis_model()
    bms_models, bms_cv_gb, bms_cv_rf = train_bms_model()

eis_models = eis_result[0]
eis_n      = eis_result[1]
eis_cv_gb  = eis_result[2]
eis_cv_rf  = eis_result[3]

# 모델 성능 요약
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("EIS 학습 파일", f"{eis_n}개" if eis_n else "미로드")
c2.metric("EIS GB R²",  f"{eis_cv_gb:.4f}" if eis_cv_gb else "—")
c3.metric("EIS RF R²",  f"{eis_cv_rf:.4f}" if eis_cv_rf else "—")
c4.metric("BMS GB R²",  f"{bms_cv_gb:.4f}")
c5.metric("BMS RF R²",  f"{bms_cv_rf:.4f}")

st.divider()

# 사이드바
with st.sidebar:
    st.markdown("### 📋 배터리 기본 정보")
    bat_type = st.selectbox("배터리 종류", ["LFP", "NCM", "NCA", "LCO"])
    years    = st.slider("사용 연수 (년)", 0, 15, 0)
    cycles   = st.slider("충방전 횟수", 0, 10000, 0, 100,
                     help="NCM 2,000 / LFP 4,000 / NCA 1,500 / LCO 800회가 설계 수명 기준 (Ali et al. 2023)")
    voltage  = st.number_input("현재 전압 (V)", 2.0, 4.3, 3.2, step=0.1)

    st.divider()
    st.markdown("### 🔬 분석 방법 선택")
    method = st.radio(
        "",
        ["⚡ EIS 기반 예측", "📟 BMS 기반 예측", "✏️ SOH 직접 입력"],
        help="EIS: 임피던스 분석 (정밀) | BMS: 충방전 데이터 (현장 적용)"
    )

with st.expander("ℹ️ SOH 판정 기준 (PMC11033388)", expanded=False):
    st.markdown("""
| SOH | 판정 | 주요 활용처 |
|---|---|---|
| **> 80%** | ✅ 재사용 | EV, ESS, 고성능 |
| **70~80%** | ♻️ 재활용 | ESS, 그리드 |
| **50~80%** | ♻️ 재활용 | UPS, 통신 백업 (grade C) |
| **≤ 50%** | 🗑️ 해체 | 재활용 공정 |

> **LFP**: SOH 60% 미만부터 임피던스 급증 (PMC11033388)
    """)

# ═══════════════════════════════════════
# 탭: EIS / BMS / 직접 입력
# ═══════════════════════════════════════
if method == "⚡ EIS 기반 예측":
    st.markdown("### 📂 EIS 파일 업로드")
    st.caption("Warwick DIB 포맷 (.xls) 또는 freq/z_real/z_imag 3열 CSV")
    uploaded = st.file_uploader("EIS 파일 (.xls / .csv)", type=['xls','xlsx','csv'],
                                accept_multiple_files=True)

    if uploaded:
        all_zr, all_zi, df_list = [], [], []
        for f in uploaded:
            try:
                raw = f.read()
                zr, zi = parse_csv_eis(raw) if f.name.endswith('.csv') \
                         else parse_xls_eis(raw)
                all_zr.append(zr); all_zi.append(zi)
                ml = min(len(zr), len(zi)) if zi else len(zr)
                df_list.append(pd.DataFrame({'z_real': zr[:ml],
                                             'z_imag': zi[:ml] if zi else [0]*ml}))
            except Exception as e:
                st.warning(f"⚠️ {f.name} 읽기 실패: {e}")

        if not all_zr:
            st.error("읽을 수 있는 파일이 없습니다.")
            st.stop()

        max_len = max(len(z) for z in all_zr)
        pad = lambda l,n: l + [l[-1]]*(n-len(l)) if l else [0]*n
        avg_zr = np.mean([pad(z, max_len) for z in all_zr], axis=0).tolist()
        avg_zi = np.mean([pad(z, max_len) for z in all_zi], axis=0).tolist() \
                 if all_zi[0] else []

        if len(uploaded) > 1:
            st.caption(f"📊 {len(uploaded)}개 파일 평균값으로 분석")

        # 시각화
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-title">📈 나이퀴스트 플롯</div>', unsafe_allow_html=True)
            fig = go.Figure()
            for d in df_list:
                fig.add_trace(go.Scatter(x=d['z_real'], y=-d['z_imag'], mode='lines',
                                         line=dict(color='rgba(0,212,170,0.2)', width=1),
                                         showlegend=False))
            ml = min(len(avg_zr), len(avg_zi)) if avg_zi else len(avg_zr)
            fig.add_trace(go.Scatter(
                x=avg_zr[:ml], y=[-v for v in avg_zi[:ml]] if avg_zi else [0]*ml,
                mode='lines+markers', name='평균',
                marker=dict(color=list(range(ml)), colorscale='Plasma', size=7,
                            colorbar=dict(title="포인트", thickness=12)),
                line=dict(color='rgba(255,255,255,0.8)', width=2)
            ))
            fig.update_layout(xaxis_title="Z' (Ω)", yaxis_title="-Z'' (Ω)",
                              template='plotly_dark', height=300,
                              margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown('<div class="section-title">📊 임피던스 크기</div>', unsafe_allow_html=True)
            fig2 = go.Figure()
            avg_zm = np.sqrt(np.array(avg_zr)**2 +
                             np.array(avg_zi if avg_zi else [0]*len(avg_zr))**2) * 1000
            fig2.add_trace(go.Scatter(y=avg_zm, mode='lines+markers',
                                      line=dict(color='#00d4aa', width=2),
                                      marker=dict(size=5)))
            fig2.update_layout(xaxis_title="포인트 (고주파→저주파)",
                               yaxis_title="|Z| (mΩ)",
                               template='plotly_dark', height=300,
                               margin=dict(l=0,r=0,t=10,b=0))
            st.plotly_chart(fig2, use_container_width=True)

        st.divider()

        if eis_models:
            soh = predict_soh_eis(eis_models, avg_zr, avg_zi)
            if soh:
                render_result(soh, "EIS ML 예측 (Warwick DIB, 앙상블)", bat_type,
                              years, cycles, voltage, "⚡ EIS 기반")
            else:
                st.error("EIS 피처 추출 실패. 파일 형식을 확인해주세요.")
        else:
            st.warning("⚠️ EIS 모델 미로드 — data/EIS_Test/ 폴더를 확인해주세요.")
    else:
        st.info("👆 EIS 파일을 업로드하면 분석이 시작됩니다.")

# ═══════════════════════════════════════
elif method == "📟 BMS 기반 예측":
    st.markdown("### 📟 BMS 데이터 입력")
    st.caption(
        "BMS 데이터 기반 SOH 예측 — 배터리 종류·사이클·내부저항 입력"
    )

    bms_input_mode = st.radio(
        "입력 방식",
        ["📂 .mat 파일 업로드 (MATLAB 형식)",
         "📁 CSV 파일 업로드 (BMS 로그)",
         "🎛️ 수동 입력 (슬라이더)"],
        horizontal=True
    )

    features_dict = None

    # ── NASA .mat 업로드 ────────────────────────
    if bms_input_mode == "📂 .mat 파일 업로드 (MATLAB 형식)":
        st.caption("MATLAB .mat 형식의 BMS 데이터 파일 업로드 → 최신 사이클 기준 SOH 예측")

        mat_bms_file = st.file_uploader(
            ".mat 파일 업로드 (MATLAB 형식)", type=['mat'],
            help="MATLAB .mat 형식 BMS 데이터 파일"
        )
        if mat_bms_file:
            with st.spinner("📂 .mat 파싱 중..."):
                try:
                    raw = mat_bms_file.read()
                    cycles_data, bat_name = parse_nasa_mat(raw)
                    soh_records = compute_soh_from_mat(cycles_data)
                except Exception as e:
                    st.error(f"파싱 실패: {e}")
                    st.stop()

            if not soh_records:
                st.error("방전 사이클 데이터를 찾지 못했습니다.")
                st.stop()

            st.success(f"✅ **{bat_name}** | 방전 사이클 {len(soh_records)}개 파싱 완료")

            # 최신 사이클 기준으로 SOH 예측
            last = soh_records[-1]
            r_ref_now = BAT_RESISTANCE[bat_type]
            if last.get('internal_r') is not None:
                ir = float(np.clip(last['internal_r'],
                                   r_ref_now['r0'], r_ref_now['r_max']*1.5))
            else:
                cl_now = BAT_PROPS[bat_type]['cycle_life']
                cr_ratio = min(last['discharge_idx'] / cl_now, 1.3)
                ir = r_ref_now['r0'] + (r_ref_now['r_max']-r_ref_now['r0'])*cr_ratio

            est_years = last['discharge_idx'] / 365.0

            # 사이클별 SOH 열화 그래프
            df_mat_bms = pd.DataFrame(soh_records)
            fig_mat = go.Figure()
            fig_mat.add_trace(go.Scatter(
                x=df_mat_bms['discharge_idx'], y=df_mat_bms['soh_actual'],
                mode='lines+markers', name='실측 SOH',
                line=dict(color='#00d4aa', width=2), marker=dict(size=4)
            ))
            fig_mat.add_hline(y=80, line_dash='dash', line_color='#f0a500',
                              annotation_text='SOH 80% 재사용 기준')
            fig_mat.add_hline(y=50, line_dash='dash', line_color='#e05555',
                              annotation_text='SOH 50% 해체 기준')
            fig_mat.update_layout(
                xaxis_title='방전 사이클 수', yaxis_title='SOH (%)',
                template='plotly_dark', height=280,
                margin=dict(l=0,r=0,t=10,b=0)
            )
            st.plotly_chart(fig_mat, use_container_width=True)

            # 최신 사이클 정보 표시
            st.info(
                f"📌 **최신 사이클 기준 예측** — "
                f"방전 사이클 {last['discharge_idx']}회 | "
                f"실측 SOH {last['soh_actual']:.1f}% | "
                f"실측 용량 {last['capacity_ah']:.3f} Ah"
            )

            features_dict = {
                'bat_type': bat_type,
                'cycle':    float(last['discharge_idx']),
                'years':    est_years,
                'int_r':    ir,
            }

    # ── CSV 업로드 ──────────────────────────────
    elif bms_input_mode == "📁 CSV 파일 업로드 (BMS 로그)":
        st.markdown("**CSV 컬럼 형식** — 아래 이름을 권장하지만 유사한 이름도 자동 감지합니다.")

        with st.expander("📄 권장 CSV 컬럼 형식 보기"):
            st.markdown("""
| 컬럼명 | 자동 감지 키워드 | 단위 |
|---|---|---|
| `cycle` | cycle, 사이클, cyc | 정수 |
| `voltage` | voltage, volt, v, 전압 | V |
| `current` | current, amp, i, 전류 | A |
| `temperature` | temperature, temp, t, 온도 | °C |
| `time_s` | time, t, 시간 | s |

> 컬럼명이 달라도 위 키워드가 포함되면 자동 매핑됩니다.
            """)
            sample = pd.DataFrame({
                'cycle':       [1,1,1,2,2,2],
                'voltage':     [3.2,3.18,3.15,3.19,3.17,3.14],
                'current':     [1.0,1.0,-1.0,1.0,1.0,-1.0],
                'temperature': [25.1,25.3,25.5,25.2,25.4,25.6],
                'capacity':    [0.0,0.5,2.0,0.0,0.49,1.97],
                'time_s':      [0,1800,3600,0,1800,3600],
            })
            st.dataframe(sample, use_container_width=True)
            st.download_button("⬇️ 샘플 CSV 다운로드",
                               sample.to_csv(index=False),
                               "bms_sample.csv", "text/csv")

        bms_file = st.file_uploader("BMS 로그 CSV 업로드", type=['csv'])
        if bms_file:
            try:
                df_bms = pd.read_csv(bms_file)
                # 컬럼명 자동 매핑 (다양한 BMS 장비 포맷 대응)
                col_map = {}
                for col in df_bms.columns:
                    cl = col.lower()
                    if any(k in cl for k in ['cycle','cyc','사이클']):
                        col_map['cycle'] = col
                    elif any(k in cl for k in ['voltage','volt','전압']) and 'col_map' not in str(col_map.get('voltage','')):
                        col_map['voltage'] = col
                    elif any(k in cl for k in ['current','amp','전류']):
                        col_map['current'] = col
                    elif any(k in cl for k in ['temp','온도']):
                        col_map['temperature'] = col
                    elif any(k in cl for k in ['time','시간']):
                        col_map['time_s'] = col

                missing = {'cycle','voltage','current','temperature','time_s'} - set(col_map.keys())
                if missing:
                    st.error(f"필수 컬럼을 찾지 못했습니다: {missing}")
                    st.info(f"감지된 컬럼: {list(df_bms.columns)}")
                else:
                    # 표준 컬럼명으로 리네임
                    df_bms = df_bms.rename(columns={v:k for k,v in col_map.items()})
                    if col_map:
                        st.caption(f"📌 컬럼 자동 매핑: {col_map}")
                    feat_list = extract_bms_features_from_csv(df_bms, bat_type)
                    if not feat_list:
                        st.error("피처 추출 실패. 데이터를 확인해주세요.")
                    else:
                        last = feat_list[-1]
                        features_dict = {
                            'bat_type': bat_type,
                            'cycle':    last['cycle_count'],
                            'years':    last['years'],
                            'int_r':    last['internal_resistance_ohm'],
                        }
                        st.success(f"✅ {len(feat_list)}개 사이클 데이터 추출 완료 — 최신 사이클 기준 예측")

                        r_trend = [f['internal_resistance_ohm'] for f in feat_list]
                        fig_trend = go.Figure()
                        fig_trend.add_trace(go.Scatter(
                            y=r_trend, mode='lines+markers',
                            line=dict(color='#f0a500', width=2),
                            marker=dict(size=5), name='내부 저항'
                        ))
                        fig_trend.update_layout(
                            title="사이클별 내부 저항 추이 (Ω) — 노화 지표",
                            xaxis_title="사이클", yaxis_title="내부 저항 (Ω)",
                            template='plotly_dark', height=280,
                            margin=dict(l=0,r=0,t=40,b=0)
                        )
                        st.plotly_chart(fig_trend, use_container_width=True)
            except Exception as e:
                st.error(f"CSV 읽기 오류: {e}")

    # ── 수동 입력 ───────────────────────────────
    else:
        props_now = BAT_PROPS[bat_type]
        r_ref     = BAT_RESISTANCE[bat_type]
        st.caption(
            f"💡 **{bat_type} 기준값** — "
            f"설계 사이클 수명 {props_now['cycle_life']:,}회, "
            f"신품 내부저항 {r_ref['r0']}Ω, "
            f"말기 내부저항 {r_ref['r_max']}Ω (Ali et al. 2023)"
        )
        c1, c2 = st.columns(2)
        # ── 핵심 3개 입력만 받고 나머지는 자동 계산 ──────────
        # 충전시간·온도·전압강하 등을 기본값으로 고정하면
        # ML 모델이 사이클/연수를 무시하는 문제가 발생함
        # → 사이클+연수+내부저항으로 SOH를 역산한 뒤
        #   나머지 피처를 물리 모델로 자동 계산
        max_cyc      = float(props_now['cycle_life'] * 2)
        default_cyc  = min(float(cycles), max_cyc)
        cycle_ratio_now = min(default_cyc / props_now['cycle_life'], 1.3)
        auto_r = round(r_ref['r0'] + (r_ref['r_max']-r_ref['r0'])*cycle_ratio_now, 4)

        c1, c2 = st.columns(2)
        with c1:
            cycle_cnt = st.number_input(
                f"사이클 수 (설계 수명: {props_now['cycle_life']:,}회)",
                0.0, max_cyc, default_cyc, step=10.0,
                help=f"{bat_type} 설계 수명 기준 (Ali et al. 2023) | "
                     f"설계 수명 도달 시 SOH 80%"
            )
        with c2:
            # 내부저항: 사이클비 기반 자동 계산값을 기본으로 제공
            cycle_ratio_input = min(cycle_cnt / props_now['cycle_life'], 1.3)
            auto_r_input = round(
                r_ref['r0'] + (r_ref['r_max']-r_ref['r0'])*cycle_ratio_input, 4
            )
            internal_r_v = st.number_input(
                f"내부 저항 (Ω) | 신품 {r_ref['r0']} → 말기 {r_ref['r_max']}",
                float(r_ref['r0']), float(r_ref['r_max'] * 1.5),
                float(auto_r_input), step=0.005,
                help="사이클 수 기준 자동 계산값. BMS 실측값이 있으면 직접 입력"
            )

        st.caption(
            f"💡 충전시간·온도·전압강하는 입력값 기반으로 자동 계산됩니다. "
            f"사이클·연수·내부저항이 SOH 예측의 핵심 지표입니다."
        )

        # features_dict에 핵심 3개만 저장 (predict_soh_bms에서 자동 계산)
        manual_input = {
            'bat_type': bat_type,
            'cycle':    float(cycle_cnt),
            'years':    float(years),
            'int_r':    float(internal_r_v),
        }
        features_dict = manual_input  # predict_soh_bms가 직접 처리

    # ── 예측 실행 ───────────────────────────────
    if features_dict:
        st.divider()

        # 입력 요약
        with st.expander("📊 입력값 확인", expanded=False):
            fd = features_dict
            cfg = BAT_PROPS[fd['bat_type']]
            cl = cfg['cycle_life']
            cr = cfg.get('calendar_aging_rate_pct', 2.0)
            soh_theory = max(50, 100 - 20*min(fd['cycle']/cl, 1.5) - cr*fd['years'])
            feats_auto = build_bms_features(fd['bat_type'], fd['cycle'], fd['years'], fd['int_r'])
            st.markdown(f"""
| 항목 | 입력값 | 근거 |
|---|---|---|
| 배터리 종류 | {fd['bat_type']} | 선택값 |
| 사이클 수 | {fd['cycle']:.0f}회 | 사용자 입력 |
| 사용 연수 | {fd['years']:.1f}년 | 사용자 입력 |
| 내부 저항 | {fd['int_r']:.4f} Ω | 사용자 입력 |
| 이론 SOH (사이클+캘린더) | {soh_theory:.1f}% | Ali et al. (2023) |
| 자동계산 충전시간 | {feats_auto[3]:.0f} s | 물리 모델 역산 |
| 자동계산 온도 상승폭 | {feats_auto[4]:.1f} °C | 물리 모델 역산 |
| 자동계산 전압 강하 | {feats_auto[5]:.4f} V | 물리 모델 역산 |
            """)

        fd = features_dict
        soh = predict_soh_bms(
            bms_models,
            fd['bat_type'], fd['cycle'], fd['years'], fd['int_r']
        )
        render_result(soh, "BMS ML 예측 (앙상블 모델)", bat_type,
                      years, cycles, voltage, "📟 BMS 기반")

# ═══════════════════════════════════════
elif method == "✏️ SOH 직접 입력":
    st.markdown("### ✏️ SOH 직접 입력")
    st.caption("실측 용량 데이터 또는 외부 측정 장비 결과를 직접 입력합니다.")
    soh_direct = st.slider("SOH (%)", 10, 100, 80,
                           help="SOH = 현재 실제 용량 / 신품 정격 용량 × 100")
    if st.button("📊 분석 실행", type="primary"):
        render_result(soh_direct, "직접 입력 (IEC 62660-1 기준)", bat_type,
                      years, cycles, voltage, "✏️ 직접 입력")
