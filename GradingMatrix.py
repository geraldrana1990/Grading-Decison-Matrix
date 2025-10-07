# === BB360 APP (Functional vs Refurb + Reglass Labor + Grade Margins) ===
# File: GradingMatrix.py
# Version: 2025-10-06r4 (fix: Grade B uses Type='Reclaim'; reglass only for C1/C2/C2-BG)

import re, time, subprocess, hashlib
from pathlib import Path
from collections import defaultdict
import html as _html
import pandas as pd
import streamlit as st
import yaml

# -------------------- Helpers / Config --------------------
def _rerun():
    try: st.rerun()
    except AttributeError: st.experimental_rerun()

def clear_all_caches():
    for fn in (getattr(st, "cache_data", None), getattr(st, "cache_resource", None)):
        try: fn.clear()
        except Exception: pass
    for fn in (getattr(st, "experimental_memo", None), getattr(st, "experimental_singleton", None)):
        try: fn.clear()
        except Exception: pass
    try: st.session_state.clear()
    except Exception: pass

DEFAULTS = {
    "ui": {"hide_ipads_on_screen": True, "show_bucket_totals_in_summary": True},
    "pricing": {"battery_default": "BATTERY CELL", "lcd_default": "LCM GENERIC (HARD OLED)"},
    "labor": {"ceq_minutes": 2, "use_qc_labor": True}
}
def load_config():
    cfg_path = Path("config.yml")
    if cfg_path.exists():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
        except Exception:
            user_cfg = {}
    else:
        user_cfg = {}
    cfg = DEFAULTS.copy()
    for section, vals in user_cfg.items():
        if isinstance(vals, dict):
            cfg.setdefault(section, {}).update(vals)
        else:
            cfg[section] = vals
    return cfg
CFG = load_config()

def _git(cmd):
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return None
def get_git_info():
    sha = _git(["git", "rev-parse", "HEAD"])
    branch = _git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    dirty_out = _git(["git", "status", "--porcelain", "--untracked-files=no"])
    dirty = bool(dirty_out) if dirty_out is not None else None
    return sha, branch, dirty
def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

APP_FILE = Path(__file__).resolve()
GIT_SHA, GIT_BRANCH, GIT_DIRTY = get_git_info()
APP_FILE_HASH = file_hash(APP_FILE)
APP_IDENTITY = (GIT_SHA[:12] if GIT_SHA else APP_FILE_HASH[:12])
APP_CACHE_VER = f"build::{APP_IDENTITY}"
APP_NS = f"bb360::{Path(__file__).stem}::{APP_CACHE_VER}"

ALLOWED_PROFILES = {"ECOTEC GRADING TEST 1", "ECOTEC GRADING TEST 2"}

COLUMN_SYNONYMS = {
    'imei': ['imei', 'imei/meid', 'serial', 'a number', 'sn'],
    'model': ['model', 'device model'],
    'defects': ['grading summary 1', 'grading summary 2', 'failed test summary', 'defects', 'issues'],
    'battery_cycle': ['battery cycle count', 'cycle count'],
    'battery_health': ['battery health', 'battery'],
    'profile_name': ['profile name'],
    'analyst_result': ['analyst result', 'result', 'grading result']
}
MODEL_ALIASES = {
    "IPHONE SE (2020)": "IPHONE SE 2020",
    "IPHONE SEGEN2": "IPHONE SE 2020",
    "IPHONE SEGEN3": "IPHONE SE 2022",
    "IPHONE SE (2022)": "IPHONE SE 2022",
    "IPHONE 13 PRO MAX": "IPHONE 13 PRO MAX",
    "IPHONE 13PRO MAX": "IPHONE 13 PRO MAX",
    "IPHONE SE20": "IPHONE SE 2020",
    "IPHONE SE22": "IPHONE SE 2022",
}
CATEGORY_PARTS_MAPPING = {
    "C3": ["BACK COVER", "HOUSING FRAME", "BACKGLASS"],
    "C3-HF": ["HOUSING FRAME"],
    "C3-BG": ["BACKGLASS"],
    "C0": ["BACK COVER"],
    "C1": ["BACK COVER"],
    "C2-BG": ["BACKGLASS"],
    "C4": [],
    "C5": [],
    "POLISH": []
}
CATEGORY_ADHESIVES_MAPPING = {
    "C1": ["LCM ADHESIVE", "BATTERY ADHESIVE", "BACK HOUSING ADHESIVE"],
    "C0": ["LCM ADHESIVE", "BATTERY ADHESIVE", "BACK HOUSING ADHESIVE"],
    "C3": ["LCM ADHESIVE", "BATTERY ADHESIVE", "BACK HOUSING ADHESIVE"],
    "C3-HF": ["LCM ADHESIVE", "BATTERY ADHESIVE", "BACK HOUSING ADHESIVE"],
    "C2-BG": ["LCM ADHESIVE", "BACK HOUSING ADHESIVE"],
    "C3-BG": ["BACK HOUSING ADHESIVE", "HOUSING ADHESIVE"],
    "C2": ["LCM ADHESIVE"],
    "C2-C": ["LCM ADHESIVE"],
}
FRAME_BACKGLASS_MODELS = {
    "IPHONE 14", "IPHONE 14 PLUS",
    "IPHONE 15", "IPHONE 15 PLUS",
    "IPHONE 15 PRO", "IPHONE 15 PRO MAX"
}
ANODIZING_ELIGIBLE_MODELS = {
    "IPHONE SE", "IPHONE SE 2020", "IPHONE SE 2022",
    "IPHONE 11", "IPHONE XR",
    "IPHONE 12", "IPHONE 12 MINI",
    "IPHONE 13", "IPHONE 13 MINI",
    "IPHONE 14", "IPHONE 14 PLUS",
}

def normalize_model(model_str):
    if pd.isna(model_str): return ''
    model = str(model_str).upper().strip()
    model = re.sub(r'\s+', ' ', model)
    model = re.sub(r'[\(\)]', '', model)
    # Normalize SE shorthand and gens
    model = re.sub(r'\bSE\s*22\b', 'SE 2022', model)
    model = re.sub(r'\bSE\s*20\b', 'SE 2020', model)
    if 'SEGEN3' in model or 'SE 3' in model or '3RD GEN' in model: return 'IPHONE SE 2022'
    if 'SEGEN2' in model or 'SE 2' in model or '2ND GEN' in model: return 'IPHONE SE 2020'
    return model
def find_column(df: pd.DataFrame, candidates: list):
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        cand_low = cand.lower().strip()
        if cand_low in lower_map: return lower_map[cand_low]
    for col_low, col_orig in lower_map.items():
        for cand in candidates:
            if cand.lower().strip() in col_low: return col_orig
    return None
def find_columns(df: pd.DataFrame, candidates: list):
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    hits, seen = [], set()
    for cand in candidates:
        tok = cand.lower().strip()
        for col_low, col_orig in lower_map.items():
            if tok in col_low and col_orig not in seen:
                hits.append(col_orig); seen.add(col_orig)
    return hits
def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy(); df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    normalized = pd.DataFrame(index=df.index)
    for key, candidates in COLUMN_SYNONYMS.items():
        if key == 'defects':
            cols = find_columns(df, candidates)
            if cols:
                merged = (df[cols].astype(str)
                          .replace({'nan': '', 'None': ''}, regex=False)
                          .agg('|'.join, axis=1)
                          .str.replace(r'\|+', '|', regex=True)
                          .str.strip('| '))
                merged = merged.mask(merged.eq(''))
                normalized[key] = merged
            else:
                normalized[key] = pd.NA
        else:
            col = find_column(df, candidates)
            normalized[key] = df[col] if col is not None else pd.NA
    return pd.concat([df, normalized.add_prefix('_norm_')], axis=1)
def parse_failures(summary: str):
    if not summary or str(summary).lower() == 'nan': return []
    return [f.strip() for f in str(summary).split('|') if f.strip()]
def battery_status(cycle, health):
    try: cycle_num = float(cycle) if pd.notna(cycle) else None
    except: cycle_num = None
    try: health_num = float(str(health).replace('%','')) if pd.notna(health) else None
    except: health_num = None
    if cycle_num is not None and health_num is not None:
        status = 'Battery Normal' if (cycle_num < 800 and health_num > 85) else 'Battery Service'
    else:
        status = 'Battery Service'
    return status, cycle_num, health_num
_re_lcd = re.compile(r'(screen test|screen burn|pixel test)', re.I)
def keyify(s: str) -> str: return re.sub(r'[^A-Z0-9]+', '', str(s).upper())
def uph_key(name: str) -> str: return re.sub(r'[^a-z0-9]+', '', str(name).lower())

# -------------------- UI --------------------
st.set_page_config(page_title='BB360: Grade Margins (A/B/C)', layout='wide')
st.title('BB360: Grade Margins (A/B/C) — Acquisition vs Refurb Cost')

uploaded   = st.file_uploader('Upload BB360 export (CSV or Excel)', type=['xlsx','xls','csv'])
as_file    = st.file_uploader('Upload Live file (CSV or Excel with Inventory/Handset/Raw Data sheet)', type=['xlsx','xls','csv'])
parts_file = st.file_uploader('Upload Pricing + Categories Excel (with F2P, Cosmetic Category, Pricelist, UPH, Purchase Price sheets)', type=['xlsx','xls'])

if uploaded is None or parts_file is None or as_file is None:
    st.info('Upload BB360 export, Live Inventory file, and Pricing+Category file to continue.')
    st.stop()

@st.cache_data(show_spinner=False)
def load_all(uploaded, as_file, parts_file, _ns: str = APP_NS):
    df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    if as_file.name.lower().endswith('.csv'):
        as_inv = pd.read_csv(as_file); live_sheet = "(csv)"
    else:
        xls = pd.ExcelFile(as_file)
        inv_hit = hs_hit = rd_hit = None
        for s in xls.sheet_names:
            low = s.lower().strip()
            if inv_hit is None and "inventory" in low: inv_hit = s
            if hs_hit  is None and "handset"   in low: hs_hit  = s
            if rd_hit  is None and "raw data"  in low: rd_hit  = s
        target = inv_hit or hs_hit or rd_hit or xls.sheet_names[0]
        live_sheet = target
        as_inv = pd.read_excel(as_file, sheet_name=target)

    f2p_parts      = pd.read_excel(parts_file, sheet_name='F2P')
    cosmetic_cat   = pd.read_excel(parts_file, sheet_name='Cosmetic Category')
    pricelist      = pd.read_excel(parts_file, sheet_name='Pricelist')
    uph            = pd.read_excel(parts_file, sheet_name='UPH')
    purchase_price = pd.read_excel(parts_file, sheet_name='Purchase Price')

    return df_raw, as_inv, f2p_parts, cosmetic_cat, pricelist, uph, purchase_price, live_sheet

df_raw, as_inv, f2p_parts, cosmetic_cat, pricelist, uph, purchase_price, LIVE_SHEET = load_all(uploaded, as_file, parts_file, APP_NS)
st.caption(f"Live workbook sheet used: {LIVE_SHEET}")

# -------------------- Filters (Ecotec + Not Completed) --------------------
prof_col = find_column(df_raw, COLUMN_SYNONYMS['profile_name'])
if prof_col is not None:
    df_raw['_profile_upper'] = df_raw[prof_col].astype(str).str.upper().str.strip()
    df_raw = df_raw[df_raw['_profile_upper'].isin(ALLOWED_PROFILES)].copy()
    df_raw.drop(columns=['_profile_upper'], inplace=True, errors='ignore')
ar_col = find_column(df_raw, COLUMN_SYNONYMS['analyst_result'])
if ar_col is not None:
    df_raw['_analyst_lower'] = df_raw[ar_col].astype(str).str.strip().str.lower()
    df_raw = df_raw[df_raw['_analyst_lower'] != 'not completed'].copy()
    df_raw.drop(columns=['_analyst_lower'], inplace=True, errors='ignore')

# -------------------- Normalize & Merge with Live --------------------
norm = normalize_input_df(df_raw)
as_inv.columns = [str(c).strip().lower() for c in as_inv.columns]
imei_col = next((c for c in as_inv.columns if any(k in c for k in ["imei", "sn", "serial"])), None)
cat_col  = next((c for c in as_inv.columns if "category" in c), None)

def _find_live_col(substrs):
    for c in as_inv.columns:
        low = str(c).lower().strip()
        if any(s in low for s in substrs): return c
    return None

sku_col        = _find_live_col(["sku", "sku name", "sku#", "sku id", "sku description"])
live_model_col = _find_live_col(["model", "device model", "device"])
color_col      = _find_live_col(["color", "colour"])
capacity_col   = _find_live_col(["capacity", "storage", "rom"])

norm['_norm_imei'] = norm['_norm_imei'].astype(str).str.strip()
as_inv[imei_col]   = as_inv[imei_col].astype(str).str.strip()
live_cols_to_merge = [c for c in [imei_col, cat_col, sku_col, live_model_col, color_col, capacity_col] if c]

merged = norm.merge(
    as_inv[live_cols_to_merge],
    left_on='_norm_imei',
    right_on=imei_col,
    how='left',
    indicator=True
)
merged = merged[merged['_merge'] == 'both'].copy().rename(columns={cat_col:'Category'}).drop(columns=[imei_col,'_merge'], errors='ignore')
merged['Category'] = merged['Category'].astype(str).str.strip().mask(lambda s: (s.eq('')) | (s.str.lower().eq('nan')) | (s.str.lower().eq('none')), 'No Category')

def _cap_norm(x: str) -> str:
    s = str(x or '').strip()
    if not s or s.lower() in ('nan', 'none'): return ''
    m = re.search(r'(\d+)\s*(tb|gb|g|gig|gigabyte|gigabytes)?', s, re.I)
    if m:
        num = m.group(1); unit = (m.group(2) or 'GB').upper()
        unit = 'TB' if unit.startswith('T') else 'GB'
        return f"{num}{unit}"
    return s.upper()
def _capwords(s: str) -> str:
    s = str(s or '').strip()
    if not s or s.lower() in ('nan', 'none'): return ''
    return ' '.join(w.capitalize() for w in s.split())

merged['SKU'] = ''
if sku_col: merged['SKU'] = merged[sku_col].astype(str).str.strip()
fallback_model_series = merged[live_model_col] if (live_model_col and live_model_col in merged.columns) else merged['_norm_model']
cap_series = merged[capacity_col].astype(str) if (capacity_col and capacity_col in merged.columns) else ''
col_series = merged[color_col].astype(str) if (color_col and color_col in merged.columns) else ''
built = (fallback_model_series.astype(str).str.upper().str.strip()
         + (' ' + pd.Series(cap_series).apply(_cap_norm)).replace(' ', '', regex=False)
         + (' ' + pd.Series(col_series).apply(_capwords)).replace(' ', '', regex=False)).str.strip()
merged['SKU'] = merged['SKU'].mask(merged['SKU'].eq('') | merged['SKU'].str.lower().isin(['nan','none']), built)
merged['SKU'] = merged['SKU'].mask(merged['SKU'].eq('') | merged['SKU'].str.lower().isin(['nan','none']), merged['_norm_model'].astype(str).str.upper().str.strip())

# Colorless join key (model + capacity)
model_key_norm = fallback_model_series.apply(normalize_model).astype(str).str.upper().str.strip()
model_key_norm = model_key_norm.apply(lambda s: MODEL_ALIASES.get(s, s))
merged['SKU_KEY'] = (model_key_norm
                     + (' ' + pd.Series(cap_series).apply(_cap_norm)).replace(' ', '', regex=False)).str.strip()

# Cosmetic category descriptions
cosmetic_cat.columns = [str(c).strip() for c in cosmetic_cat.columns]
CAT_TO_DESC = dict(zip(cosmetic_cat['Legacy Category'], cosmetic_cat['Description']))

# -------------------- Pricelist / F2P normalization --------------------
def normalize_model_for_df(s): return s.apply(normalize_model)
f2p_parts['Model_norm']  = normalize_model_for_df(f2p_parts['iPhone Model'])
f2p_parts['Faults_norm'] = f2p_parts['Faults'].astype(str).str.lower().str.strip()
f2p_parts['Part_norm']   = f2p_parts['Part'].astype(str).str.upper().str.strip().str.replace(r'\s+', ' ', regex=True)

pricelist['Model_norm'] = normalize_model_for_df(pricelist['iPhone Model'])
pricelist['Part_norm']  = pricelist['Part'].astype(str).str.upper().str.strip().str.replace(r'\s+', ' ', regex=True)
pricelist['PRICE']      = pd.to_numeric(pricelist['PRICE'].astype(str).str.replace(r'[^0-9\.]', '', regex=True).replace('', '0'), errors="coerce").fillna(0.0)
pricelist['Type_norm']  = pricelist.get('Type', pd.Series(['']*len(pricelist))).astype(str).str.upper().str.strip()
pricelist['Part_key']   = pricelist['Part_norm'].apply(keyify)

# UPH index
uph = uph[['Type of Defect', 'Ave. Repair Time (Mins)']].dropna(subset=['Type of Defect']).copy()
uph['Defect_norm'] = uph['Type of Defect'].map(uph_key)
uph['Ave. Repair Time (Mins)'] = pd.to_numeric(uph['Ave. Repair Time (Mins)'], errors="coerce").fillna(0.0)
UPH_INDEX = dict(zip(uph['Defect_norm'], uph['Ave. Repair Time (Mins)']))
def _upm(name: str) -> float:
    return float(UPH_INDEX.get(uph_key(name), 0.0))

PRICE_INDEX = {(m, p): v for m, p, v in zip(pricelist['Model_norm'], pricelist['Part_norm'], pricelist['PRICE'])}
PL_BY_MODEL = {m: g for m, g in pricelist.groupby('Model_norm', sort=False)}

F2P_INDEX = defaultdict(list)
for m, f, p in zip(f2p_parts['Model_norm'], f2p_parts['Faults_norm'], f2p_parts['Part_norm']):
    F2P_INDEX[(m, f)].append(p)

def build_adhesive_index(pricelist_df):
    out = defaultdict(dict)
    for model, part in zip(pricelist_df['Model_norm'], pricelist_df['Part_norm']):
        up = str(part).upper()
        if "ADHESIVE" not in up: continue
        key = "LCM ADHESIVE" if "LCM" in up else ("BATTERY ADHESIVE" if "BATTERY" in up else ("BACK HOUSING ADHESIVE" if "HOUSING" in up else None))
        if key:
            prev = out[model].get(key)
            if prev is None or ("OEM" in up and "OEM" not in str(prev).upper()):
                out[model][key] = up
    return out
ADHESIVE_INDEX = build_adhesive_index(pricelist)
FRAME_BACKGLASS_MODELS = set(FRAME_BACKGLASS_MODELS)

# RE price index
RE_KEYS = {"REGLASS", "REGLASSDIGITIZER"}
PL_RE_BY_MODEL = defaultdict(dict)
for _, row in pricelist.iterrows():
    if row['Type_norm'] == 'RE' and row['Part_key'] in RE_KEYS:
        PL_RE_BY_MODEL[row['Model_norm']][row['Part_key']] = float(row['PRICE'])

# -------------------- Purchase Price --------------------
purchase_price = purchase_price.copy()
purchase_price.columns = [str(c).strip() for c in purchase_price.columns]
pp_cols = purchase_price.columns
col_map = {
    "SKU": next((c for c in pp_cols if c.strip().lower() == "sku"), "SKU"),
    "Acq": next((c for c in pp_cols if "acquisition" in c.strip().lower()), "Acquisition price"),
    "A":   next((c for c in pp_cols if c.strip().lower() in ("grade a","grade a price","a")), purchase_price.columns[2]),
    "B":   next((c for c in pp_cols if c.strip().lower() in ("grade b","grade b price","b")), purchase_price.columns[3]),
    "C":   next((c for c in pp_cols if c.strip().lower() in ("grade c","grade c price","c")), purchase_price.columns[4]),
}
purchase_price.rename(columns={
    col_map["SKU"]: "PP_SKU",
    col_map["Acq"]: "PP_Acquisition",
    col_map["A"]:   "PP_GradeA",
    col_map["B"]:   "PP_GradeB",
    col_map["C"]:   "PP_GradeC",
}, inplace=True)
for c in ["PP_Acquisition","PP_GradeA","PP_GradeB","PP_GradeC"]:
    purchase_price[c] = pd.to_numeric(purchase_price[c].astype(str).str.replace(r'[^0-9\.]','',regex=True).replace('','0'),
                                      errors='coerce').fillna(0.0)
PURCHASE_INDEX = {
    str(row["PP_SKU"]).strip().upper(): {
        "acq": float(row["PP_Acquisition"]),
        "A": float(row["PP_GradeA"]),
        "B": float(row["PP_GradeB"]),
        "C": float(row["PP_GradeC"]),
    } for _, row in purchase_price.iterrows()
}

# -------------------- Selectors --------------------
battery_type = st.selectbox("Select Battery Type", ["BATTERY CELL", "BATTERY OEM", "BATTERY OEM PULLED"], index=0).upper()
lcd_type = st.selectbox("Select LCD Type", [
    "LCM GENERIC (HARD OLED)",
    "LCM GENERIC (TFT)",
    "LCM -OEM REFURBISHED (GLASS CHANGED -GENERIC)",
    "LCM GENERIC (SOFT OLED)"
], index=0).upper()
labor_rate = st.slider("Labor Rate ($/hour)", min_value=1, max_value=35, value=5)

# -------------------- Per-row compute --------------------
def compute_row(row, labor_rate):
    analyst_result = str(row.get('_norm_analyst_result', row.get('Analyst Result',''))).strip().lower()
    if analyst_result == "not completed": return None

    failures = parse_failures(row.get('_norm_defects',''))
    batt_status, _, _ = battery_status(row.get('_norm_battery_cycle'), row.get('_norm_battery_health'))
    model_raw = normalize_model(row.get('_norm_model'))
    device_model = MODEL_ALIASES.get(model_raw, model_raw)
    cat_val = str(row.get('Category') or '').strip().upper()
    sku_key = str(row.get('SKU_KEY') or '').strip().upper()

    func_parts, src = [], {}

    # Functional F2P
    for f in [str(f).lower().strip() for f in failures]:
        for part in F2P_INDEX.get((device_model, f), []):
            key = str(part).upper().strip()
            if (device_model, key) in PRICE_INDEX and key not in func_parts:
                func_parts.append(key); src[key] = 'f2p'

    # Battery (service)
    if batt_status == "Battery Service":
        failures.append("Battery Service")
        key = battery_type
        if (device_model, key) in PRICE_INDEX and key not in func_parts:
            func_parts.append(key); src[key] = 'battery'

    # LCD
    if any(_re_lcd.search(str(f)) for f in failures) or cat_val in {"C0","C2-C"}:
        if (device_model, lcd_type) in PRICE_INDEX and lcd_type not in func_parts:
            func_parts.append(lcd_type); src[lcd_type] = 'lcd'

    func_total = sum(float(PRICE_INDEX.get((device_model, k), 0.0)) for k in func_parts)

    # Labor: tech (+CEQ if any failure)
    tech_minutes = sum(_upm(tok) for tok in failures)
    if len(failures) > 0:
        tech_minutes += float(CFG['labor'].get('ceq_minutes', 2))
    tech_labor_cost = (tech_minutes / 60.0) * float(labor_rate)

    # Refurb labor (category)
    refcat_set = {'C0', 'C1', 'C3-BG', 'C3-HF', 'C3'}
    refurb_minutes = _upm(cat_val) if cat_val in refcat_set else 0.0
    refurb_labor_cost = (refurb_minutes / 60.0) * float(labor_rate)

    # Reglass Labor & Price — only for eligible cats
    RE_ELIGIBLE_CATS = {'C1','C2','C2-BG'}
    re_applicable = cat_val in RE_ELIGIBLE_CATS
    has_mts = any("multitouchscreen" in str(f).lower().replace(" ", "").replace("-", "") for f in failures) if re_applicable else False

    re_min_candidates = []
    if re_applicable:
        if has_mts:
            re_min_candidates += [_upm("reglassdigitizer"), _upm("re-glass digitizer")]
        re_min_candidates += [_upm("reglass"), _upm("re-glass")]
    reglass_minutes = max(re_min_candidates) if re_min_candidates else 0.0
    reglass_labor_cost = (reglass_minutes / 60.0) * float(labor_rate)

    preferred_key = 'REGLASSDIGITIZER' if has_mts else 'REGLASS'
    reglass_price = float(PL_RE_BY_MODEL.get(device_model, {}).get(preferred_key, 0.0)) if re_applicable else 0.0

    # QC
    qc_min = 0.0
    if CFG['labor'].get('use_qc_labor', True):
        for key_try in ["qc process","qc inspection","qcinspection","qc","quality control","quality check"]:
            qc_min = max(qc_min, _upm(key_try))
    qc_cost = (qc_min / 60.0) * float(labor_rate)

    # Anodizing (A/B only later; C excludes in totals)
    anod_min = max(_upm("anodizing"), _upm("anodize"), _upm("anodising"), _upm("anodise"))
    anod_cats = {'C2', 'C2-C', 'C2-BG', 'C3-BG', 'C4'}
    model_is_eligible_for_anodizing = device_model in ANODIZING_ELIGIBLE_MODELS
    anod_cost = (anod_min / 60.0) * float(labor_rate) if (model_is_eligible_for_anodizing and cat_val in anod_cats and anod_min > 0) else 0.0

    # BNP side logic (A/B only later; C excludes in totals)
    fb_min = max(_upm("front buffing"), _upm("frontbuff"), _upm("front polish"), _upm("front polishing"))
    bb_min = max(_upm("back buffing"), _upm("backbuff"), _upm("back polish"), _upm("back polishing"))
    if cat_val in {'C4','C3-HF'}:        bnp_minutes = fb_min + bb_min
    elif cat_val in {'C3','C3-BG'}:      bnp_minutes = fb_min
    elif cat_val in {'C2','C2-C'}:       bnp_minutes = bb_min
    else:                                bnp_minutes = 0.0
    bnp_cost = (bnp_minutes / 60.0) * float(labor_rate) if bnp_minutes > 0 else 0.0

    # Cosmetic keywords (for A/B)
    def cosmetic_keywords_for(legacy_cat, model):
        legacy_cat = str(legacy_cat).strip().upper()
        if legacy_cat in {"C0", "C1", "C3"}:
            return ["HOUSING FRAME", "BACKGLASS"] if model in FRAME_BACKGLASS_MODELS else ["BACK COVER"]
        kws, desc = [], str(CAT_TO_DESC.get(legacy_cat, "")).upper()
        if legacy_cat != "C4":
            if "BACK COVER" in desc: kws.append("BACK COVER")
            if "BACKGLASS" in desc or "BACK GLASS" in desc: kws.append("BACKGLASS")
            if "HOUSING FRAME" in desc: kws.append("HOUSING FRAME")
        return list(dict.fromkeys(kws))

    cos_kws = cosmetic_keywords_for(cat_val, device_model)

    def select_cosmetic_total(model, keywords, type_pref):
        total = 0.0
        model_slice = PL_BY_MODEL.get(model)
        if model_slice is None: return 0.0
        for kw in keywords:
            kwu = str(kw).upper()
            if kwu == "HOUSING FRAME":
                cond = (model_slice['Part_norm'].str.contains("HOUSING", na=False) &
                        model_slice['Part_norm'].str.contains("FRAME",   na=False) &
                        ~model_slice['Part_norm'].str.contains("ADHESIVE", na=False))
            elif kwu == "BACKGLASS":
                cond = (((model_slice['Part_norm'].str.contains("BACKGLASS", na=False)) |
                        (model_slice['Part_norm'].str.contains("BACK", na=False) &
                         model_slice['Part_norm'].str.contains("GLASS", na=False))) &
                        ~model_slice['Part_norm'].str.contains("ADHESIVE", na=False))
            else:
                cond = (model_slice['Part_norm'].str.contains(kwu, na=False) &
                        ~model_slice['Part_norm'].str.contains("ADHESIVE", na=False))
            cand = model_slice[cond]
            pref = cand[cand['Type_norm'].str.upper() == str(type_pref).upper()]
            chosen = pref if not pref.empty else cand
            if not chosen.empty:
                total += float(chosen['PRICE'].iat[0])
        # adhesives included for A/B
        for ak in CATEGORY_ADHESIVES_MAPPING.get(cat_val, []):
            candidate = ADHESIVE_INDEX.get(model, {}).get(ak)
            if candidate and (model, candidate) in PRICE_INDEX:
                total += float(PRICE_INDEX[(model, candidate)])
        return total

    # Use D-COVER for Grade A, RECLAIM for Grade B
    cos_A = select_cosmetic_total(device_model, cos_kws, "D-COVER")
    cos_B = select_cosmetic_total(device_model, cos_kws, "RECLAIM")
    cos_C = 0.0  # grade C = functional only

    # Grade totals
    refurb_A = func_total + cos_A + reglass_price + reglass_labor_cost + refurb_labor_cost + anod_cost + bnp_cost + tech_labor_cost + qc_cost
    refurb_B = func_total + cos_B + reglass_price + reglass_labor_cost + refurb_labor_cost + anod_cost + bnp_cost + tech_labor_cost + qc_cost
    refurb_C = func_total + tech_labor_cost + qc_cost

    # Prices & margins
    pp = PURCHASE_INDEX.get(sku_key, {"acq":0.0,"A":0.0,"B":0.0,"C":0.0})
    acq = float(pp["acq"]); price_A = float(pp["A"]); price_B = float(pp["B"]); price_C = float(pp["C"])

    margin_A = price_A - (acq + refurb_A)
    margin_B = price_B - (acq + refurb_B)
    margin_C = price_C - (acq + refurb_C)

    best_grade, best_margin, best_refurb = max(
        [("A", margin_A, refurb_A), ("B", margin_B, refurb_B), ("C", margin_C, refurb_C)],
        key=lambda t: t[1]
    )

    return {
        'imei': row.get('_norm_imei'),
        'model': row.get('_norm_model'),
        'SKU': row.get('SKU'),
        'SKU_KEY': sku_key,
        'legacy_category': cat_val,
        'failures': "|".join(failures),

        # CSV detail
        'Functional Parts Cost': func_total,
        'Reglass Cost (Price)': reglass_price,
        'Tech Labor': tech_labor_cost,
        'Refurb Labor Cost': refurb_labor_cost,
        'Reglass Labor Cost': reglass_labor_cost,
        'QC Labor': qc_cost,
        'Anodizing Labor': anod_cost,
        'BNP Labor': bnp_cost,

        # On-screen economics
        'Acquisition Cost': acq,
        'Refurbishment Cost (Final)': best_refurb,
        'Grade A Price': price_A, 'Margin A': margin_A,
        'Grade B Price': price_B, 'Margin B': margin_B,
        'Grade C Price': price_C, 'Margin C': margin_C,
        'Final Grading': best_grade,
        'Final Margin': best_margin,

        # full grade breakdown
        'Refurb A (Total Repair)': refurb_A,
        'Refurb B (Total Repair)': refurb_B,
        'Refurb C (Total Repair)': refurb_C,
    }

rows = []
for _, r in merged.iterrows():
    out = compute_row(r, labor_rate)
    if out is None: continue
    rows.append(out)
res_df = pd.DataFrame(rows)

# -------------------- Display --------------------
if not res_df.empty:
    disp = res_df.copy()
    if CFG['ui'].get('hide_ipads_on_screen', True):
        disp = disp[~disp['model'].astype(str).str.upper().str.contains('IPAD', na=False)]

    SHOW_COLS = [
        'imei','SKU',
        'Acquisition Cost','Refurbishment Cost (Final)',
        'Grade A Price','Margin A',
        'Grade B Price','Margin B',
        'Grade C Price','Margin C',
        'Final Grading','Final Margin'
    ]
    present_cols = [c for c in SHOW_COLS if c in disp.columns]
    disp = disp[present_cols].copy()

    money_cols = ['Acquisition Cost','Refurbishment Cost (Final)',
                  'Grade A Price','Margin A','Grade B Price','Margin B','Grade C Price','Margin C','Final Margin']
    for col in money_cols:
        if col in disp.columns:
            disp[col] = pd.to_numeric(disp[col], errors='coerce').fillna(0.0)

    def _fmt_money(x: float) -> str: return f"${x:,.2f}"

    row_html = []
    for _, row in disp.iterrows():
        tds = []
        for c in present_cols:
            v = row[c]
            if c in money_cols:
                if c == 'Final Margin':
                    color = '#d6f5d6' if float(v) >= 0 else '#ffd6e7'
                    tds.append(f"<td style='white-space:nowrap;text-align:right;background:{color}'>{_fmt_money(float(v))}</td>")
                else:
                    tds.append(f"<td style='white-space:nowrap;text-align:right'>{_fmt_money(float(v))}</td>")
            else:
                tds.append(f"<td>{_html.escape(str(v))}</td>")
        row_html.append("<tr>" + "".join(tds) + "</tr>")

    header_html = "".join(f"<th>{_html.escape(c)}</th>" for c in present_cols)
    table_html = f"""
    <style>
      .bb360-wrap {{
        max-height: 72vh;
        overflow: auto;
        border: 1px solid #eee;
        border-radius: 8px;
      }}
      .bb360-table {{
        border-collapse: separate;
        border-spacing: 0;
        width: 100%;
        font-size: 14px;
      }}
      .bb360-table th, .bb360-table td {{
        padding: 8px 12px;
        border-bottom: 1px solid #f2f2f2;
        vertical-align: top;
        text-align: left;
      }}
      .bb360-table thead th {{
        position: sticky;
        top: 0;
        z-index: 3;
        background: #ffffff;
        box-shadow: 0 1px 0 0 #e5e5e5;
      }}
    </style>
    <div class="bb360-wrap">
      <table class="bb360-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(row_html)}</tbody>
      </table>
    </div>
    """

    st.subheader("Grade Margins — IMEIs in Live only")
    st.markdown(table_html, unsafe_allow_html=True)

    # CSV: keep ALL columns (old + new)
    csv_bytes = res_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Full CSV', data=csv_bytes, file_name='bb360_full_with_margins.csv', mime='text/csv')
else:
    st.info("No qualifying rows to display.")
