# === BB360 APP (Functional vs Refurb, Collapsible Cells) ===
# File: grading2.py
# Version: 2025-10-03
# - Reglass isolated (not listed under Refurb), but included in totals
# - Sticky table header while scrolling
# - Cache-safe dev helpers + cache bust on file save
# - Mapping bugfix in map_category_to_parts_fast (proper boolean mask)
# - Split labor: Tech Labor (functional/CEQ) + Refurb Labor Cost (category + anodizing + buffing)
# - Hide static parts bucket totals on screen; still present in CSV
# - Only process rows where Profile Name ∈ {Ecotec Grading Test 1, Ecotec Grading Test 2}
# - Defects merged from "Grading Summary 1" + "Grading Summary 2" (with legacy fallbacks)
# - NEW: Omit rows with Analyst Result == "Not Completed" at ingestion

import os, sys, time, re
import streamlit as st
import pandas as pd
import yaml
from collections import defaultdict
import html as _html
from pathlib import Path

# -------------------- Streamlit cross-version helpers --------------------
def _rerun():
    try:
        st.rerun()  # modern Streamlit
    except AttributeError:
        st.experimental_rerun()  # older Streamlit

def clear_all_caches():
    # New APIs
    try: st.cache_data.clear()
    except Exception: pass
    try: st.cache_resource.clear()
    except Exception: pass
    # Legacy APIs
    try: st.experimental_memo.clear()
    except Exception: pass
    try: st.experimental_singleton.clear()
    except Exception: pass
    # Session state
    try: st.session_state.clear()
    except Exception: pass

# -------------------- CONFIG / CACHE KEYS --------------------
DEFAULTS = {
    "ui": {
        "hide_ipads_on_screen": True,
        "show_bucket_totals_in_summary": True
    },
    "pricing": {
        "battery_default": "BATTERY CELL",
        "lcd_default": "LCM GENERIC (HARD OLED)"
    },
    "labor": {
        "ceq_minutes": 2,
        "use_qc_labor": True
    }
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

# Cache namespace auto-includes file mtime so every save busts cache in dev
APP_CACHE_VER = f"dev::{Path(__file__).stem}::{Path(__file__).stat().st_mtime_ns}"
APP_NS = f"bb360::{Path(__file__).stem}::{APP_CACHE_VER}"

# -------------------- FILTER: allowed profiles only --------------------
ALLOWED_PROFILES = {"ECOTEC GRADING TEST 1", "ECOTEC GRADING TEST 2"}

# -------------------- COLUMN MAPS --------------------
COLUMN_SYNONYMS = {
    'imei': ['imei', 'imei/meid', 'serial', 'a number', 'sn'],
    'model': ['model', 'device model'],
    # include grading summaries explicitly; keep legacy names as fallback
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

# -------------------- UTILITIES --------------------
def normalize_model(model_str):
    if pd.isna(model_str): return ''
    model = str(model_str).upper().strip()
    model = re.sub(r'\s+', ' ', model)
    model = re.sub(r'[\(\)]', '', model)
    if 'SEGEN3' in model or 'SE 3' in model or '3RD GEN' in model:
        return 'IPHONE SE 2022'
    if 'SEGEN2' in model or 'SE 2' in model or '2ND GEN' in model:
        return 'IPHONE SE 2020'
    return model

def find_column(df: pd.DataFrame, candidates: list):
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        cand_low = cand.lower().strip()
        if cand_low in lower_map: return lower_map[cand_low]
    for col_low, col_orig in lower_map.items():
        for cand in candidates:
            if cand.lower().strip() in col_low:
                return col_orig
    return None

def find_columns(df: pd.DataFrame, candidates: list):
    """Return all columns whose lowercase names contain any candidate token (order preserved)."""
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    hits, seen = [], set()
    for cand in candidates:
        tok = cand.lower().strip()
        for col_low, col_orig in lower_map.items():
            if tok in col_low and col_orig not in seen:
                hits.append(col_orig); seen.add(col_orig)
    return hits

def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    normalized = pd.DataFrame(index=df.index)

    for key, candidates in COLUMN_SYNONYMS.items():
        if key == 'defects':
            cols = find_columns(df, candidates)
            if cols:
                merged = (
                    df[cols]
                    .astype(str)
                    .replace({'nan': '', 'None': ''}, regex=False)
                    .agg('|'.join, axis=1)
                    .str.replace(r'\|+', '|', regex=True)
                    .str.strip('| ')
                )
                merged = merged.mask(merged.eq(''))
                normalized[key] = merged
            else:
                normalized[key] = pd.NA
        else:
            col = find_column(df, candidates)
            normalized[key] = df[col] if col is not None else pd.NA

    normalized = pd.concat([df, normalized.add_prefix('_norm_')], axis=1)
    return normalized

def parse_failures(summary: str):
    if not summary or str(summary).lower() == 'nan': return []
    return [f.strip() for f in str(summary).split('|') if f.strip()]

def battery_status(cycle, health):
    try:
        cycle_num = float(cycle) if pd.notna(cycle) else None
    except: cycle_num = None
    try:
        health_num = float(str(health).replace('%','')) if pd.notna(health) else None
    except: health_num = None
    if cycle_num is not None and health_num is not None:
        ok = cycle_num < 800 and health_num > 85
        status = 'Battery Normal' if ok else 'Battery Service'
    else:
        status = 'Battery Service'
    return status, cycle_num, health_num

_re_lcd = re.compile(r'(screen test|screen burn|pixel test)', re.I)
def is_lcd_failure(failure: str) -> bool:
    return bool(_re_lcd.search(str(failure)))

def keyify(s: str) -> str:
    # Compact alnum key (upper)
    return re.sub(r'[^A-Z0-9]+', '', str(s).upper())

# -------------------- STREAMLIT APP --------------------
st.set_page_config(page_title='BB360: Mobile Failure Quantification', layout='wide')
st.title('BB360: Mobile Failure Quantification — Functional vs Refurb')

# Sidebar debug + panic button
with st.sidebar:
    st.caption("Debug / Runtime Info")
    st.write({
        "script": str(Path(__file__).resolve()),
        "streamlit": getattr(st, "__version__", "unknown"),
        "app_ns": APP_NS,
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
    })
    if st.button("🔄 Full cache reset & rerun"):
        clear_all_caches()
        _rerun()

# -------------------- LOAD FILES (CACHED) --------------------
@st.cache_data(show_spinner=False)
def load_all(uploaded, as_file, parts_file, _ns: str = APP_NS):
    # _ns participates in cache key; unused otherwise
    df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    if as_file.name.lower().endswith('.csv'):
        as_inv = pd.read_csv(as_file)
    else:
        xls = pd.ExcelFile(as_file)
        inv_sheet = next((s for s in xls.sheet_names if s.strip().lower() == "inventory"), None)
        as_inv = pd.read_excel(as_file, sheet_name=inv_sheet)

    f2p_parts = pd.read_excel(parts_file, sheet_name='F2P')
    cosmetic_cat = pd.read_excel(parts_file, sheet_name='Cosmetic Category')
    pricelist = pd.read_excel(parts_file, sheet_name='Pricelist')
    uph = pd.read_excel(parts_file, sheet_name='UPH')
    return df_raw, as_inv, f2p_parts, cosmetic_cat, pricelist, uph

uploaded = st.file_uploader('Upload BB360 export (CSV or Excel)', type=['xlsx','xls','csv'])
as_file = st.file_uploader('Upload AS file (CSV or Excel with Inventory sheet)', type=['xlsx','xls','csv'])
parts_file = st.file_uploader('Upload Pricing + Categories Excel (with F2P, Cosmetic Category, Pricelist, UPH sheets)', type=['xlsx','xls'])

if uploaded is None or parts_file is None or as_file is None:
    st.info('Upload BB360 export, AS Inventory file, and Pricing+Category file to continue.')
    st.stop()

df_raw, as_inv, f2p_parts, cosmetic_cat, pricelist, uph = load_all(uploaded, as_file, parts_file, APP_NS)

# -------------------- FILTERS: Ecotec profiles + omit Not Completed --------------------
# Filter to Ecotec Grading Test 1/2
prof_col = find_column(df_raw, COLUMN_SYNONYMS['profile_name'])
if prof_col is not None:
    before_n = len(df_raw)
    df_raw['_profile_upper'] = df_raw[prof_col].astype(str).str.upper().str.strip()
    df_raw = df_raw[df_raw['_profile_upper'].isin(ALLOWED_PROFILES)].copy()
    df_raw.drop(columns=['_profile_upper'], inplace=True, errors='ignore')
    after_n = len(df_raw); dropped = before_n - after_n
    if dropped > 0:
        st.info(f"Filtered out {dropped} rows not in {sorted(ALLOWED_PROFILES)}.")
    if after_n == 0:
        st.warning("No rows matched Ecotec Grading Test 1/2. Check the 'Profile Name' values in your BB360 export.")
        st.stop()
else:
    st.warning("Could not find a 'Profile Name' column to apply the Ecotec filter. Proceeding without it.")

# Omit Analyst Result == Not Completed (case-insensitive)
ar_col = find_column(df_raw, COLUMN_SYNONYMS['analyst_result'])
if ar_col is not None:
    before_n = len(df_raw)
    df_raw['_analyst_lower'] = df_raw[ar_col].astype(str).str.strip().str.lower()
    df_raw = df_raw[df_raw['_analyst_lower'] != 'not completed'].copy()
    df_raw.drop(columns=['_analyst_lower'], inplace=True, errors='ignore')
    after_n = len(df_raw); dropped = before_n - after_n
    if dropped > 0:
        st.info(f"Excluded {dropped} rows with Analyst Result = Not Completed.")
    if after_n == 0:
        st.warning("All rows were 'Not Completed' after filtering. Nothing to process.")
        st.stop()
else:
    st.warning("Could not find an 'Analyst Result' column to exclude 'Not Completed' rows. Proceeding without it.")

# -------------------- NORMALIZE (ONCE) --------------------
norm = normalize_input_df(df_raw)

# Merge AS (vectorized)
as_inv.columns = [str(c).strip().lower() for c in as_inv.columns]
imei_col = next((c for c in as_inv.columns if any(k in c for k in ["imei", "sn", "serial"])), None)
cat_col = next((c for c in as_inv.columns if "category" in c), None)

norm['_norm_imei'] = norm['_norm_imei'].astype(str).str.strip()
as_inv[imei_col]   = as_inv[imei_col].astype(str).str.strip()

norm = norm.merge(as_inv[[imei_col, cat_col]], left_on='_norm_imei', right_on=imei_col, how='left')
norm = norm.rename(columns={cat_col: 'Category'})

cosmetic_cat.columns = [str(c).strip() for c in cosmetic_cat.columns]
CAT_TO_DESC = dict(zip(cosmetic_cat['Legacy Category'], cosmetic_cat['Description']))
norm['Category_Desc'] = norm['Category'].map(CAT_TO_DESC)

# Normalize pricelist + f2p (vectorized)
f2p_parts['Model_norm'] = f2p_parts['iPhone Model'].apply(normalize_model)
f2p_parts['Faults_norm'] = f2p_parts['Faults'].astype(str).str.lower().str.strip()
f2p_parts['Part_norm'] = f2p_parts['Part'].astype(str).str.upper().str.strip().str.replace(r'\s+', ' ', regex=True)

pricelist['Model_norm'] = pricelist['iPhone Model'].apply(normalize_model)
pricelist['Part_norm'] = pricelist['Part'].astype(str).str.upper().str.strip().str.replace(r'\s+', ' ', regex=True)
pricelist['PRICE'] = pricelist['PRICE'].astype(str).str.replace(r'[^0-9\.]', '', regex=True).replace('', '0').astype(float)
# Normalize 'Type' and add a compact Part_key for robust matching
pricelist['Type_norm'] = pricelist.get('Type', pd.Series(['']*len(pricelist))).astype(str).str.upper().str.strip()
pricelist['Part_key'] = pricelist['Part_norm'].apply(keyify)

# UPH index
uph = uph[['Type of Defect', 'Ave. Repair Time (Mins)']].dropna()
uph['Defect_norm'] = uph['Type of Defect'].astype(str).str.strip().str.lower().str.replace(" ", "").str.replace("_", "")
UPH_INDEX = dict(zip(uph['Defect_norm'], uph['Ave. Repair Time (Mins)']))
def uph_minutes(name: str) -> float:
    key = str(name).lower().replace(" ", "").replace("_", "")
    return float(UPH_INDEX.get(key, 0.0))

# -------------------- FAST LOOKUP INDICES --------------------
PRICE_INDEX = {(m, p): v for m, p, v in zip(pricelist['Model_norm'], pricelist['Part_norm'], pricelist['PRICE'])}
PL_BY_MODEL = {m: g for m, g in pricelist.groupby('Model_norm', sort=False)}

F2P_INDEX = defaultdict(list)
for m, f, p in zip(f2p_parts['Model_norm'], f2p_parts['Faults_norm'], f2p_parts['Part_norm']):
    F2P_INDEX[(m, f)].append(p)

def build_adhesive_index(pricelist_df):
    out = defaultdict(dict)
    for model, part in zip(pricelist_df['Model_norm'], pricelist_df['Part_norm']):
        up = str(part).upper()
        if "ADHESIVE" not in up:
            continue
        key = None
        if "LCM" in up:
            key = "LCM ADHESIVE"
        elif "BATTERY" in up:
            key = "BATTERY ADHESIVE"
        elif "HOUSING" in up:
            key = "BACK HOUSING ADHESIVE"
        if key:
            prev = out[model].get(key)
            if prev is None or ("OEM" in up and "OEM" not in str(prev).upper()):
                out[model][key] = up
    return out

ADHESIVE_INDEX = build_adhesive_index(pricelist)
FRAME_BACKGLASS_MODELS = set(FRAME_BACKGLASS_MODELS)  # ensure set

# Build RE price index by compact key ("REGLASS", "REGLASSDIGITIZER")
RE_KEYS = {"REGLASS", "REGLASSDIGITIZER"}
PL_RE_BY_MODEL = defaultdict(dict)
for _, row in pricelist.iterrows():
    if row['Type_norm'] == 'RE':
        k = row['Part_key']
        if k in RE_KEYS:
            PL_RE_BY_MODEL[row['Model_norm']][k] = float(row['PRICE'])

def map_category_to_parts_fast(legacy_cat, model):
    legacy_cat = str(legacy_cat).strip().upper()
    parts_needed = []

    part_keywords = CATEGORY_PARTS_MAPPING.get(legacy_cat, [])

    if legacy_cat in {"C0", "C1", "C3"}:
        if model in FRAME_BACKGLASS_MODELS:
            part_keywords = ["HOUSING FRAME", "BACKGLASS"]
        else:
            part_keywords = ["BACK COVER"]

    if not part_keywords:
        description = str(CAT_TO_DESC.get(legacy_cat, "")).upper()
        if "BACK COVER" in description:
            parts_needed.append("BACK COVER")
        if "BACKGLASS" in description or "BACK GLASS" in description:
            parts_needed.append("BACKGLASS")
        if "HOUSING FRAME" in description:
            parts_needed.append("HOUSING FRAME")
        if "BUFFING" in description or "POLISH" in description:
            parts_needed.append("POLISH")

    model_slice = PL_BY_MODEL.get(model)

    # If we don't have a pricelist slice for this model, just return unique of what we have
    if model_slice is None or 'Part_norm' not in model_slice.columns:
        return list(dict.fromkeys(parts_needed))

    # Resolve each keyword to the first matching pricelist item
    for k in part_keywords:
        kk = str(k).upper()
        if kk == "HOUSING FRAME":
            cond = (
                model_slice['Part_norm'].str.contains("HOUSING", na=False) &
                model_slice['Part_norm'].str.contains("FRAME",   na=False) &
                ~model_slice['Part_norm'].str.contains("ADHESIVE", na=False)
            )
            matches = model_slice[cond]
        elif kk == "BACKGLASS":
            cond = (
                (
                    model_slice['Part_norm'].str.contains("BACKGLASS", na=False) |
                    (model_slice['Part_norm'].str.contains("BACK", na=False) &
                     model_slice['Part_norm'].str.contains("GLASS", na=False))
                ) &
                ~model_slice['Part_norm'].str.contains("ADHESIVE", na=False)
            )
            matches = model_slice[cond]
        else:
            cond = model_slice['Part_norm'].str.contains(kk, na=False)
            matches = model_slice[cond]

        if not matches.empty:
            parts_needed.append(matches['Part_norm'].iat[0])

    # Add adhesives per category using the prebuilt index
    for ak in CATEGORY_ADHESIVES_MAPPING.get(legacy_cat, []):
        candidate = ADHESIVE_INDEX.get(model, {}).get(ak)
        if candidate:
            parts_needed.append(candidate)

    # De-dup while preserving order
    return list(dict.fromkeys(parts_needed))

# -------------------- SELECTORS --------------------
battery_type = st.selectbox("Select Battery Type", ["BATTERY CELL", "BATTERY OEM", "BATTERY OEM PULLED"], index=0).upper()
lcd_type = st.selectbox("Select LCD Type", [
    "LCM GENERIC (HARD OLED)",
    "LCM GENERIC (TFT)",
    "LCM -OEM REFURBISHED (GLASS CHANGED -GENERIC)",
    "LCM GENERIC (SOFT OLED)"
], index=0).upper()
labor_rate = st.slider("Labor Rate ($/hour)", min_value=1, max_value=35, value=5)

# -------------------- ROW COMPUTE --------------------
def compute_row(row, labor_rate):
    analyst_result = str(row.get('_norm_analyst_result', row.get('Analyst Result',''))).strip().lower()
    if analyst_result == "not completed":
        return None  # extra safety; most are already filtered earlier

    # inputs & normalization
    failures = parse_failures(row.get('_norm_defects',''))
    batt_status, cycle_num, health_num = battery_status(row.get('_norm_battery_cycle'), row.get('_norm_battery_health'))
    device_model = normalize_model(row.get('_norm_model'))
    device_model = MODEL_ALIASES.get(device_model, device_model)
    raw_cat = row.get('Category')
    cat_val = str(raw_cat).strip().upper() if pd.notna(raw_cat) else ""

    # buckets
    func_parts, refurb_parts = [], []
    src = {}   # part -> source

    # Functional: defects -> parts
    for f in [str(f).lower().strip() for f in failures]:
        for part in F2P_INDEX.get((device_model, f), []):
            key = str(part).upper().strip()
            if (device_model, key) in PRICE_INDEX and key not in func_parts:
                func_parts.append(key); src[key] = 'f2p'

    # Functional: battery
    if batt_status == "Battery Service":
        failures.append("Battery Service")
        key = battery_type  # already upper
        if (device_model, key) in PRICE_INDEX and key not in func_parts:
            func_parts.append(key); src[key] = 'battery'
        if key == "BATTERY CELL":
            model_slice = PL_BY_MODEL.get(device_model)
            if model_slice is not None:
                fx = model_slice[model_slice['Part_norm'].str.contains("BATTERY FLEX", na=False)]
                if not fx.empty:
                    fk = fx['Part_norm'].iat[0]
                    if (device_model, fk) in PRICE_INDEX and fk not in func_parts:
                        func_parts.append(fk); src[fk] = 'battery'

    # Functional: LCD
    if any(is_lcd_failure(f) for f in failures) or cat_val in {"C0","C2-C"}:
        if (device_model, lcd_type) in PRICE_INDEX and lcd_type not in func_parts:
            func_parts.append(lcd_type); src[lcd_type] = 'lcd'

    # Refurb: cosmetic category + adhesives
    for p in map_category_to_parts_fast(cat_val, device_model):
        k = str(p).upper().strip()
        if (device_model, k) in PRICE_INDEX and k not in refurb_parts and k not in func_parts:
            refurb_parts.append(k); src[k] = 'refurb'

    # ---------- REGLASS: price-only (do NOT list under Refurb) ----------
    reglass_cost_preferred = 0.0
    if cat_val in {"C1", "C2", "C2-BG"}:
        has_mts = any("multitouchscreen" in str(f).lower().replace(" ", "").replace("-", "") for f in failures)
        preferred_key = 'REGLASSDIGITIZER' if has_mts else 'REGLASS'
        reglass_cost_preferred = float(PL_RE_BY_MODEL.get(device_model, {}).get(preferred_key, 0.0))

    # Strip any RE items from refurb_parts (belt & suspenders)
    refurb_parts = [k for k in refurb_parts if keyify(k) not in {'REGLASS','REGLASSDIGITIZER'}]

    # ---- Build rows & subtotals ----
    func_rows, refurb_rows = [], []
    func_total = refurb_total = 0.0
    for k in func_parts:
        price = float(PRICE_INDEX.get((device_model, k), 0.0))
        func_rows.append({'Part': k, 'Source': src.get(k,'?'), 'Price': price}); func_total += price
    for k in refurb_parts:
        price = float(PRICE_INDEX.get((device_model, k), 0.0))
        refurb_rows.append({'Part': k, 'Source': src.get(k,'?'), 'Price': price}); refurb_total += price

    # ---- Reglass cost: preferred price; optional fallback from rows (should be 0 after filter) ----
    reglass_cost_from_parts = 0.0
    for r in refurb_rows:
        if keyify(r['Part']) in {'REGLASS','REGLASSDIGITIZER'}:
            reglass_cost_from_parts += float(r['Price'] or 0.0)
    reglass_cost = reglass_cost_preferred if reglass_cost_preferred > 0 else reglass_cost_from_parts

        # ---- Labor (split into Tech vs Refurb) ----
    def _upm(name: str) -> float:
        key = str(name).lower().replace(" ", "").replace("_", "")
        return float(UPH_INDEX.get(key, 0.0))

    # Tech labor: failures + CEQ if any failure
    tech_minutes = sum(_upm(tok) for tok in failures)
    if len(failures) > 0:
        tech_minutes += float(CFG['labor'].get('ceq_minutes', 2))
    tech_labor_cost = (tech_minutes / 60.0) * float(labor_rate)

    # Refurb labor: strictly category minutes for C0, C1, C3-BG, C3-HF; else 0
    refcat_set = {'C0', 'C1', 'C3-BG', 'C3-HF', 'C3'}
    refurb_minutes = _upm(cat_val) if cat_val in refcat_set else 0.0
    refurb_labor_cost = (refurb_minutes / 60.0) * float(labor_rate)

    # QC (separate)
    qc_min  = (_upm('qc process') or _upm('qcinspection') or _upm('qc')) if CFG['labor'].get('use_qc_labor', True) else 0.0
    qc_cost = (qc_min / 60.0) * float(labor_rate) if analyst_result != 'not completed' else 0.0

    # Anodizing labor (separate process, NOT included in refurb labor)
    anod_min = (_upm('anodizing') or _upm('anodize')) or 0.0
    anod_cats = {'C2', 'C2-C', 'C2-BG', 'C3-BG', 'C4'}  # categories that may involve anodizing
    if cat_val in anod_cats and anod_min > 0:
        anod_cost = (anod_min / 60.0) * float(labor_rate)
    else:
        anod_cost = 0.0

    # BNP labor (separate process, NOT included in refurb labor)
    fb_min = (_upm('front buffing') or _upm('frontbuff')) or 0.0
    bb_min = (_upm('back buffing')  or _upm('backbuff'))  or 0.0
    bnp_minutes = 0.0
    if cat_val in {'C4','C3-HF'}:        # both
        bnp_minutes = fb_min + bb_min
    elif cat_val in {'C3','C3-BG'}:      # front only
        bnp_minutes = fb_min
    elif cat_val in {'C2','C2-C'}:       # back only
        bnp_minutes = bb_min
    bnp_cost = (bnp_minutes / 60.0) * float(labor_rate) if bnp_minutes > 0 else 0.0

    # ---- Totals: include Reglass; labor = Tech + Refurb + QC + Anodizing + BNP ----
    total_parts = func_total + refurb_total + reglass_cost
    total_cost  = total_parts + tech_labor_cost + refurb_labor_cost + qc_cost + anod_cost + bnp_cost


    # ---- Collapsible cells ----
    def _mk_cell(title_text: str, rows: list, total: float) -> str:
        if not rows:
            return f"<span style='opacity:.7'>No { _html.escape(title_text.lower()) }</span>"
        items = "".join(
            f"<li>{_html.escape(r['Part'])} — ${r['Price']:,.2f} "
            f"<small style='opacity:.7'>({_html.escape(str(r['Source']))})</small></li>"
            for r in rows
        )
        show_totals = bool(CFG['ui'].get('show_bucket_totals_in_summary', True))
        summary = _html.escape(title_text) if not show_totals else f"{_html.escape(title_text)} — ${total:,.2f}"
        return f"<details class='bb360-cell'><summary>{summary}</summary><ul>{items}</ul></details>"

    functional_cell = _mk_cell("Functional", func_rows, func_total)
    refurb_cell     = _mk_cell("Refurb", refurb_rows, refurb_total)

    return {
        'imei': row.get('_norm_imei'),
        'model': row.get('_norm_model'),
        'legacy_category': cat_val,          # CSV-only fields
        'category_desc': row.get('Category_Desc'),
        'failures': "|".join(failures),
        'Functional (collapse)': functional_cell,
        'Refurb (collapse)': refurb_cell,
        'Functional Parts Cost': func_total,   # CSV only
        'Refurb Parts Cost': refurb_total,     # CSV only (no RE here)
        'Total Parts Cost': total_parts,       # CSV only (includes Reglass)
        'Reglass Cost': reglass_cost,          # Visible on-screen
        'QC Labor': qc_cost,
        'Anodizing Labor': anod_cost,          # informational (included in refurb labor)
        'BNP Labor': bnp_cost,                 # informational (included in refurb labor)
        'Tech Labor': tech_labor_cost,         # split labor
        'Refurb Labor Cost': refurb_labor_cost,# split labor
        'Total Cost': total_cost,
    }

# -------------------- RUN --------------------
rows = []
skip_count = 0
for _, r in norm.iterrows():
    out = compute_row(r, labor_rate)
    if out is None:
        skip_count += 1
        continue
    rows.append(out)

res_df = pd.DataFrame(rows)

# -------------------- DISPLAY --------------------
if not res_df.empty:
    disp = res_df.copy()
    if CFG['ui'].get('hide_ipads_on_screen', True):
        disp = disp[~disp['model'].astype(str).str.upper().str.contains('IPAD', na=False)]

    # On-screen columns: hide static part-bucket totals; show Reglass + split labor
    SHOW_COLS = [
        'imei','model',
        'Functional (collapse)','Refurb (collapse)',
        'Reglass Cost',
        'Tech Labor','Refurb Labor Cost','QC Labor','Anodizing Labor','BNP Labor',
        'Total Cost'
    ]
    FORBIDDEN = {'Functional Parts Cost','Refurb Parts Cost','Total Parts Cost'}
    # Guard against leaks
    bad = [c for c in FORBIDDEN if c in disp.columns]
    if bad:
        disp = disp.drop(columns=bad)
    present_cols = [c for c in SHOW_COLS if c in disp.columns]
    disp = disp[present_cols].copy()

    money_cols = ['Reglass Cost','Tech Labor','Refurb Labor Cost','QC Labor','Anodizing Labor','BNP Labor','Total Cost']
    for col in money_cols:
        if col in disp.columns:
            disp[col] = disp[col].astype(float)

    def _fmt_money(x: float) -> str: return f"${x:,.2f}"

    # Build HTML rows
    row_html = []
    for _, row in disp.iterrows():
        tds = []
        for c in present_cols:
            v = row[c]
            if c in money_cols:
                tds.append(f"<td style='white-space:nowrap;text-align:right'>{_fmt_money(v)}</td>")
            elif c in ['Functional (collapse)','Refurb (collapse)']:
                tds.append(f"<td>{v}</td>")
            else:
                tds.append(f"<td>{_html.escape(str(v))}</td>")
        row_html.append("<tr>" + "".join(tds) + "</tr>")

    # === Sticky header table ===
    header_html = "".join(f"<th>{_html.escape(c)}</th>" for c in present_cols)
    table_html = f"""
    <style>
      .bb360-wrap {{
        max-height: 72vh;             /* scrollable area height */
        overflow: auto;
        border: 1px solid #eee;
        border-radius: 8px;
      }}
      .bb360-table {{
        border-collapse: separate;    /* needed for sticky */
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
      /* Sticky header */
      .bb360-table thead th {{
        position: sticky;
        top: 0;
        z-index: 3;
        background: #ffffff;
        box-shadow: 0 1px 0 0 #e5e5e5;
      }}
      /* Optional: sticky first column (uncomment to enable)
      .bb360-table td:first-child, .bb360-table th:first-child {{
        position: sticky;
        left: 0;
        z-index: 2;
        background: #ffffff;
      }} */

      /* Collapsible summary styling */
      .bb360-cell summary {{ cursor: pointer; font-weight: 600; display: inline; }}
      .bb360-cell summary::-webkit-details-marker {{ display: none; }}
      .bb360-cell summary::marker {{ content: ''; }}
      .bb360-cell summary:hover {{ text-decoration: underline; }}
      .bb360-cell ul {{ margin: 8px 0 0 18px; }}
    </style>

    <div class="bb360-wrap">
      <table class="bb360-table">
        <thead><tr>{header_html}</tr></thead>
        <tbody>{''.join(row_html)}</tbody>
      </table>
    </div>
    """

    st.subheader("Final Results — Ecotec Test 1/2 only; 'Not Completed' omitted; Reglass isolated; sticky header; split labor")
    st.markdown(table_html, unsafe_allow_html=True)

    # CSV export: full dataset (including hidden fields), no HTML columns
    export_cols = [c for c in res_df.columns if c not in ['Functional (collapse)','Refurb (collapse)']]
    export_df = res_df[export_cols].copy()
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Final CSV', data=csv_bytes, file_name='bb360_functional_vs_refurb.csv', mime='text/csv')
else:
    st.info("No qualifying rows to display.")
