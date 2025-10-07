# === BB360 APP (Functional vs Refurb, Collapsible Cells) ===
# File: GradingMatrix.py
# Version: 2025-10-03r9 (SKU display)
# - Live sheet resolver accepts: "inventory", "handset", or "raw data"
# - Blank/None Category from Live becomes "No Category" (display & CSV)
# - Shows ONLY rows whose IMEI exists in Live (after Ecotec/Not-Completed filters)
# - Ecotec-only; omit "Not Completed"; Reglass isolated; matte-side anodizing eligibility
# - Sticky header; filters & sorting; Live sheet name shown
# - NEW: Display SKU (prefer Live.SKU; else Live.Model+Capacity+Color; else BB360 model)

import os, sys, time, re, subprocess, hashlib
import streamlit as st
import pandas as pd
import yaml
from collections import defaultdict
import html as _html
from pathlib import Path

# -------------------- Streamlit helpers --------------------
def _rerun():
    try: st.rerun()
    except AttributeError: st.experimental_rerun()

def clear_all_caches():
    for fn in (getattr(st, "cache_data", None), getattr(st, "cache_resource", None)):
        try: fn.clear()  # type: ignore
        except Exception: pass
    for fn in (getattr(st, "experimental_memo", None), getattr(st, "experimental_singleton", None)):
        try: fn.clear()  # type: ignore
        except Exception: pass
    try: st.session_state.clear()
    except Exception: pass

# -------------------- CONFIG --------------------
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

# --- Build/Version metadata (commit, branch, dirty) ---
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

# -------------------- FILTER: allowed profiles only --------------------
ALLOWED_PROFILES = {"ECOTEC GRADING TEST 1", "ECOTEC GRADING TEST 2"}

# -------------------- COLUMN MAPS --------------------
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
# Matte-side anodizing eligibility
ANODIZING_ELIGIBLE_MODELS = {
    "IPHONE SE", "IPHONE SE 2020", "IPHONE SE 2022",
    "IPHONE 11", "IPHONE XR",
    "IPHONE 12", "IPHONE 12 MINI",
    "IPHONE 13", "IPHONE 13 MINI",
    "IPHONE 14", "IPHONE 14 PLUS",
}

# -------------------- UTILITIES --------------------
def normalize_model(model_str):
    if pd.isna(model_str): return ''
    model = str(model_str).upper().strip()
    model = re.sub(r'\s+', ' ', model)
    model = re.sub(r'[\(\)]', '', model)
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
def is_lcd_failure(failure: str) -> bool: return bool(_re_lcd.search(str(failure)))
def keyify(s: str) -> str: return re.sub(r'[^A-Z0-9]+', '', str(s).upper())
def uph_key(name: str) -> str: return re.sub(r'[^a-z0-9]+', '', str(name).lower())

# -------------------- APP --------------------
st.set_page_config(page_title='BB360: Mobile Failure Quantification', layout='wide')
st.title('BB360: Mobile Failure Quantification — Functional vs Refurb')

with st.sidebar:
    st.caption("Build info")
    st.write({
        "commit": GIT_SHA[:12] if GIT_SHA else None,
        "branch": GIT_BRANCH,
        "dirty": GIT_DIRTY,
        "file_hash": APP_FILE_HASH[:12],
        "loaded_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script": str(APP_FILE),
    })
    st.caption("Runtime")
    st.write({
        "streamlit": getattr(st, "__version__", "unknown"),
        "app_ns": APP_NS,
        "python": sys.version.split()[0],
        "cwd": os.getcwd(),
    })
    if st.button("🔄 Full cache reset & rerun"):
        clear_all_caches()
        _rerun()

uploaded   = st.file_uploader('Upload BB360 export (CSV or Excel)', type=['xlsx','xls','csv'])
as_file    = st.file_uploader('Upload Live file (CSV or Excel with Inventory/Handset/Raw Data sheet)', type=['xlsx','xls','csv'])
parts_file = st.file_uploader('Upload Pricing + Categories Excel (with F2P, Cosmetic Category, Pricelist, UPH sheets)', type=['xlsx','xls'])

if uploaded is None or parts_file is None or as_file is None:
    st.info('Upload BB360 export, Live Inventory file, and Pricing+Category file to continue.')
    st.stop()

@st.cache_data(show_spinner=False)
def load_all(uploaded, as_file, parts_file, _ns: str = APP_NS):
    # BB360 export
    df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # LIVE inventory: accept sheet names containing "inventory", "handset", or "raw data"
    selected_live_sheet = None
    if as_file.name.lower().endswith('.csv'):
        as_inv = pd.read_csv(as_file)
        selected_live_sheet = "(csv)"
    else:
        xls = pd.ExcelFile(as_file)
        inv_hit = hs_hit = rd_hit = None
        for s in xls.sheet_names:
            low = s.lower().strip()
            if inv_hit is None and "inventory" in low: inv_hit = s
            if hs_hit  is None and "handset"   in low: hs_hit  = s
            if rd_hit  is None and "raw data"  in low: rd_hit  = s
        target = inv_hit or hs_hit or rd_hit or xls.sheet_names[0]
        selected_live_sheet = target
        as_inv = pd.read_excel(as_file, sheet_name=target)

    # Pricing workbook
    f2p_parts   = pd.read_excel(parts_file, sheet_name='F2P')
    cosmetic_cat= pd.read_excel(parts_file, sheet_name='Cosmetic Category')
    pricelist   = pd.read_excel(parts_file, sheet_name='Pricelist')
    uph         = pd.read_excel(parts_file, sheet_name='UPH')
    return df_raw, as_inv, f2p_parts, cosmetic_cat, pricelist, uph, selected_live_sheet

df_raw, as_inv, f2p_parts, cosmetic_cat, pricelist, uph, LIVE_SHEET = load_all(uploaded, as_file, parts_file, APP_NS)

with st.sidebar:
    st.caption("Live workbook sheet used")
    st.write({"sheet": LIVE_SHEET})

# -------------------- FILTERS (Ecotec + Not Completed) --------------------
prof_col = find_column(df_raw, COLUMN_SYNONYMS['profile_name'])
if prof_col is not None:
    before_n = len(df_raw)
    df_raw['_profile_upper'] = df_raw[prof_col].astype(str).str.upper().str.strip()
    df_raw = df_raw[df_raw['_profile_upper'].isin(ALLOWED_PROFILES)].copy()
    df_raw.drop(columns=['_profile_upper'], inplace=True, errors='ignore')
    after_n = len(df_raw); dropped = before_n - after_n
    if dropped > 0: st.info(f"Filtered out {dropped} rows not in {sorted(ALLOWED_PROFILES)}.")
    if after_n == 0:
        st.warning("No rows matched Ecotec Grading Test 1/2. Check 'Profile Name' values.")
        st.stop()
else:
    st.warning("Could not find a 'Profile Name' column. Proceeding without Ecotec filter.")

ar_col = find_column(df_raw, COLUMN_SYNONYMS['analyst_result'])
if ar_col is not None:
    before_n = len(df_raw)
    df_raw['_analyst_lower'] = df_raw[ar_col].astype(str).str.strip().str.lower()
    df_raw = df_raw[df_raw['_analyst_lower'] != 'not completed'].copy()
    df_raw.drop(columns=['_analyst_lower'], inplace=True, errors='ignore')
    after_n = len(df_raw); dropped = before_n - after_n
    if dropped > 0: st.info(f"Excluded {dropped} rows with Analyst Result = Not Completed.")
    if after_n == 0:
        st.warning("All rows were 'Not Completed'. Nothing to process.")
        st.stop()
else:
    st.warning("Could not find 'Analyst Result' column. Proceeding without it.")

# -------------------- NORMALIZE (ONCE) --------------------
norm = normalize_input_df(df_raw)

# -------------------- MERGE with Live + KEEP ONLY IMEIs that exist in Live -------------
as_inv.columns = [str(c).strip().lower() for c in as_inv.columns]
imei_col = next((c for c in as_inv.columns if any(k in c for k in ["imei", "sn", "serial"])), None)
cat_col  = next((c for c in as_inv.columns if "category" in c), None)

# Discover optional Live columns for SKU, color, capacity, model (display only)
def _find_live_col(substrs):
    for c in as_inv.columns:
        low = str(c).lower().strip()
        if any(s in low for s in substrs):
            return c
    return None

sku_col        = _find_live_col(["sku", "sku name", "sku#", "sku id", "sku description"])
live_model_col = _find_live_col(["model", "device model", "device"])
color_col      = _find_live_col(["color", "colour"])
capacity_col   = _find_live_col(["capacity", "storage", "rom"])

norm['_norm_imei'] = norm['_norm_imei'].astype(str).str.strip()
as_inv[imei_col]   = as_inv[imei_col].astype(str).str.strip()

# Choose which Live cols to merge
live_cols_to_merge = [c for c in [imei_col, cat_col, sku_col, live_model_col, color_col, capacity_col] if c]
merged = norm.merge(
    as_inv[live_cols_to_merge],
    left_on='_norm_imei',
    right_on=imei_col,
    how='left',
    indicator=True
)
merged = merged[merged['_merge'] == 'both'].copy()  # keep only IMEIs present in Live

# finish the merge: rename Category and drop helper cols
merged = merged.rename(columns={cat_col: 'Category'})
merged.drop(columns=[imei_col, '_merge'], inplace=True, errors='ignore')

# ---- Category fallback: show "No Category" when blank/NaN/None ----
merged['Category'] = (
    merged['Category']
    .astype(str).str.strip()
    .mask(lambda s: (s.eq('')) | (s.str.lower().eq('nan')) | (s.str.lower().eq('none')), 'No Category')
)

# --- Build a display SKU: prefer explicit SKU from Live; else Model+Capacity+Color; else BB360 model ---
def _cap_norm(x: str) -> str:
    s = str(x or '').strip()
    if not s or s.lower() in ('nan', 'none'):
        return ''
    m = re.search(r'(\d+)\s*(tb|gb|g|gig|gigabyte|gigabytes)?', s, re.I)
    if m:
        num = m.group(1)
        unit = m.group(2) or 'GB'
        unit = 'TB' if unit.lower().startswith('t') else 'GB'
        return f"{num}{unit}"
    return s.upper()

def _capwords(s: str) -> str:
    s = str(s or '').strip()
    if not s or s.lower() in ('nan', 'none'):
        return ''
    return ' '.join(w.capitalize() for w in s.split())

merged['SKU'] = ''
if sku_col:
    merged['SKU'] = merged[sku_col].astype(str).str.strip()

fallback_model_series = merged[live_model_col] if (live_model_col and live_model_col in merged.columns) else merged['_norm_model']
cap_series = merged[capacity_col].astype(str) if (capacity_col and capacity_col in merged.columns) else ''
col_series = merged[color_col].astype(str) if (color_col and color_col in merged.columns) else ''

built = (
    fallback_model_series.astype(str).str.upper().str.strip()
    + (' ' + pd.Series(cap_series).apply(_cap_norm)).replace(' ', '', regex=False)
    + (' ' + pd.Series(col_series).apply(_capwords)).replace(' ', '', regex=False)
).str.strip()

merged['SKU'] = merged['SKU'].mask(
    merged['SKU'].eq('') | merged['SKU'].str.lower().isin(['nan','none']),
    built
)
merged['SKU'] = merged['SKU'].mask(
    merged['SKU'].eq('') | merged['SKU'].str.lower().isin(['nan','none']),
    merged['_norm_model'].astype(str).str.upper().str.strip()
)

norm = merged  # proceed with filtered dataset only

# Cosmetic category descriptions
cosmetic_cat.columns = [str(c).strip() for c in cosmetic_cat.columns]
CAT_TO_DESC = dict(zip(cosmetic_cat['Legacy Category'], cosmetic_cat['Description']))
norm['Category_Desc'] = norm['Category'].map(CAT_TO_DESC).fillna('No Category')

# -------------------- Normalize pricelist + F2P --------------------
def normalize_model_for_df(s): return s.apply(normalize_model)
f2p_parts['Model_norm']  = normalize_model_for_df(f2p_parts['iPhone Model'])
f2p_parts['Faults_norm'] = f2p_parts['Faults'].astype(str).str.lower().str.strip()
f2p_parts['Part_norm']   = f2p_parts['Part'].astype(str).str.upper().str.strip().str.replace(r'\s+', ' ', regex=True)

pricelist['Model_norm'] = normalize_model_for_df(pricelist['iPhone Model'])
pricelist['Part_norm']  = pricelist['Part'].astype(str).str.upper().str.strip().str.replace(r'\s+', ' ', regex=True)
pricelist['PRICE']      = pd.to_numeric(pricelist['PRICE'].astype(str).str.replace(r'[^0-9\.]', '', regex=True).replace('', '0'), errors="coerce").fillna(0.0)
pricelist['Type_norm']  = pricelist.get('Type', pd.Series(['']*len(pricelist))).astype(str).str.upper().str.strip()
pricelist['Part_key']   = pricelist['Part_norm'].apply(keyify)

# -------------------- UPH index --------------------
uph = uph[['Type of Defect', 'Ave. Repair Time (Mins)']].dropna(subset=['Type of Defect']).copy()
uph['Defect_norm'] = uph['Type of Defect'].map(uph_key)
uph['Ave. Repair Time (Mins)'] = pd.to_numeric(uph['Ave. Repair Time (Mins)'], errors="coerce").fillna(0.0)
UPH_INDEX = dict(zip(uph['Defect_norm'], uph['Ave. Repair Time (Mins)']))

def _upm(name: str) -> float:
    return float(UPH_INDEX.get(uph_key(name), 0.0))

# -------------------- Indices --------------------
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
        key = None
        if "LCM" in up: key = "LCM ADHESIVE"
        elif "BATTERY" in up: key = "BATTERY ADHESIVE"
        elif "HOUSING" in up: key = "BACK HOUSING ADHESIVE"
        if key:
            prev = out[model].get(key)
            if prev is None or ("OEM" in up and "OEM" not in str(prev).upper()):
                out[model][key] = up
    return out
ADHESIVE_INDEX = build_adhesive_index(pricelist)
FRAME_BACKGLASS_MODELS = set(FRAME_BACKGLASS_MODELS)

# RE price index (REGLASS / REGLASS+DIGITIZER)
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
        # C4: finish-only; don't add structural parts from description
        if legacy_cat != "C4":
            if "BACK COVER" in description: parts_needed.append("BACK COVER")
            if "BACKGLASS" in description or "BACK GLASS" in description: parts_needed.append("BACKGLASS")
            if "HOUSING FRAME" in description: parts_needed.append("HOUSING FRAME")
        if "BUFFING" in description or "POLISH" in description:
            parts_needed.append("POLISH")

    model_slice = PL_BY_MODEL.get(model)
    if model_slice is None or 'Part_norm' not in model_slice.columns:
        return list(dict.fromkeys(parts_needed))

    for k in part_keywords:
        kk = str(k).upper()
        if kk == "HOUSING FRAME":
            cond = (model_slice['Part_norm'].str.contains("HOUSING", na=False) &
                    model_slice['Part_norm'].str.contains("FRAME",   na=False) &
                    ~model_slice['Part_norm'].str.contains("ADHESIVE", na=False))
            matches = model_slice[cond]
        elif kk == "BACKGLASS":
            cond = (((model_slice['Part_norm'].str.contains("BACKGLASS", na=False)) |
                    (model_slice['Part_norm'].str.contains("BACK", na=False) &
                     model_slice['Part_norm'].str.contains("GLASS", na=False))) &
                    ~model_slice['Part_norm'].str.contains("ADHESIVE", na=False))
            matches = model_slice[cond]
        else:
            matches = model_slice[model_slice['Part_norm'].str.contains(kk, na=False)]
        if not matches.empty:
            parts_needed.append(matches['Part_norm'].iat[0])

    for ak in CATEGORY_ADHESIVES_MAPPING.get(legacy_cat, []):
        candidate = ADHESIVE_INDEX.get(model, {}).get(ak)
        if candidate:
            parts_needed.append(candidate)

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
    if analyst_result == "not completed": return None

    failures = parse_failures(row.get('_norm_defects',''))
    batt_status, _, _ = battery_status(row.get('_norm_battery_cycle'), row.get('_norm_battery_health'))
    model_raw = normalize_model(row.get('_norm_model'))
    device_model = MODEL_ALIASES.get(model_raw, model_raw)
    cat_val = str(row.get('Category') or '').strip().upper()

    func_parts, refurb_parts, src = [], [], {}

    # Functional F2P
    for f in [str(f).lower().strip() for f in failures]:
        for part in F2P_INDEX.get((device_model, f), []):
            key = str(part).upper().strip()
            if (device_model, key) in PRICE_INDEX and key not in func_parts:
                func_parts.append(key); src[key] = 'f2p'

    # Battery
    if batt_status == "Battery Service":
        failures.append("Battery Service")
        key = battery_type
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

    # LCD
    if any(_re_lcd.search(str(f)) for f in failures) or cat_val in {"C0","C2-C"}:
        if (device_model, lcd_type) in PRICE_INDEX and lcd_type not in func_parts:
            func_parts.append(lcd_type); src[lcd_type] = 'lcd'

    # Refurb category + adhesives
    for p in map_category_to_parts_fast(cat_val, device_model):
        k = str(p).upper().strip()
        if (device_model, k) in PRICE_INDEX and k not in refurb_parts and k not in func_parts:
            refurb_parts.append(k); src[k] = 'refurb'

    # Reglass (price-only)
    reglass_cost_preferred = 0.0
    if cat_val in {"C1", "C2", "C2-BG"}:
        has_mts = any("multitouchscreen" in str(f).lower().replace(" ", "").replace("-", "") for f in failures)
        preferred_key = 'REGLASSDIGITIZER' if has_mts else 'REGLASS'
        reglass_cost_preferred = float(PL_RE_BY_MODEL.get(device_model, {}).get(preferred_key, 0.0))

    # strip RE from refurb list
    refurb_parts = [k for k in refurb_parts if keyify(k) not in {'REGLASS','REGLASSDIGITIZER'}]

    # Price rows
    func_rows, refurb_rows = [], []
    func_total = refurb_total = 0.0
    for k in func_parts:
        price = float(PRICE_INDEX.get((device_model, k), 0.0))
        func_rows.append({'Part': k, 'Source': src.get(k,'?'), 'Price': price}); func_total += price
    for k in refurb_parts:
        price = float(PRICE_INDEX.get((device_model, k), 0.0))
        refurb_rows.append({'Part': k, 'Source': src.get(k,'?'), 'Price': price}); refurb_total += price

    reglass_cost_from_parts = 0.0
    for r in refurb_rows:
        if keyify(r['Part']) in {'REGLASS','REGLASSDIGITIZER'}:
            reglass_cost_from_parts += float(r['Price'] or 0.0)
    reglass_cost = reglass_cost_preferred if reglass_cost_preferred > 0 else reglass_cost_from_parts

    # ---- Labor ----
    # Tech labor (failures + CEQ)
    tech_minutes = sum(_upm(tok) for tok in failures)
    if len(failures) > 0:
        tech_minutes += float(CFG['labor'].get('ceq_minutes', 2))
    tech_labor_cost = (tech_minutes / 60.0) * float(labor_rate)

    # Refurb labor (category minutes for specific categories)
    refcat_set = {'C0', 'C1', 'C3-BG', 'C3-HF', 'C3'}
    refurb_minutes = _upm(cat_val) if cat_val in refcat_set else 0.0
    refurb_labor_cost = (refurb_minutes / 60.0) * float(labor_rate)

    # NEW: Reglass labor (based on UPH "reglass" vs "reglassdigitizer", same decision rule as price)
    reglass_minutes = 0.0
    reglass_labor_cost = 0.0
    if cat_val in {"C1", "C2", "C2-BG"}:
        has_mts = any("multitouchscreen" in str(f).lower().replace(" ", "").replace("-", "") for f in failures)
        reglass_key_for_uph = "reglassdigitizer" if has_mts else "reglass"
        reglass_minutes = _upm(reglass_key_for_uph)
        reglass_labor_cost = (reglass_minutes / 60.0) * float(labor_rate)

    # QC
    qc_min = 0.0
    if CFG['labor'].get('use_qc_labor', True):
        for key_try in ["qc process","qc inspection","qcinspection","qc","quality control","quality check"]:
            qc_min = max(qc_min, _upm(key_try))
        if qc_min == 0:
            tmp = uph.copy()
            tmp["__norm"] = tmp["Type of Defect"].astype(str).map(uph_key)
            mins = pd.to_numeric(tmp["Ave. Repair Time (Mins)"], errors="coerce").fillna(0.0)
            mask = tmp["__norm"].str.contains("qc", na=False)
            if mask.any(): qc_min = float(mins[mask].max())
    qc_cost = (qc_min / 60.0) * float(labor_rate)

    # Anodizing (eligible models only; certain categories)
    anod_min = 0.0
    for key_try in ["anodizing","anodize","anodising","anodise","anodizing process"]:
        anod_min = max(anod_min, _upm(key_try))
    if anod_min == 0:
        tmp = uph.copy()
        tmp["__norm"] = tmp["Type of Defect"].astype(str).map(uph_key)
        mins = pd.to_numeric(tmp["Ave. Repair Time (Mins)"], errors="coerce").fillna(0.0)
        mask = tmp["__norm"].str.contains("anodiz", na=False)
        if mask.any(): anod_min = float(mins[mask].max())
    anod_cats = {'C2', 'C2-C', 'C2-BG', 'C3-BG', 'C4'}
    model_is_eligible_for_anodizing = device_model in ANODIZING_ELIGIBLE_MODELS
    anod_cost = (anod_min / 60.0) * float(labor_rate) if (model_is_eligible_for_anodizing and cat_val in anod_cats and anod_min > 0) else 0.0

    # BNP (front/back minutes; side-dependent categories)
    fb_min = max(_upm("front buffing"), _upm("frontbuff"), _upm("front polish"), _upm("front polishing"))
    if fb_min == 0:
        tmp = uph.copy()
        tmp["__norm"] = tmp["Type of Defect"].astype(str).map(uph_key)
        mins = pd.to_numeric(tmp["Ave. Repair Time (Mins)"], errors="coerce").fillna(0.0)
        mask = tmp["__norm"].str.contains("frontbuff", na=False)
        if mask.any(): fb_min = float(mins[mask].max())
    bb_min = max(_upm("back buffing"), _upm("backbuff"), _upm("back polish"), _upm("back polishing"))
    if bb_min == 0:
        tmp = uph.copy()
        tmp["__norm"] = tmp["Type of Defect"].astype(str).map(uph_key)
        mins = pd.to_numeric(tmp["Ave. Repair Time (Mins)"], errors="coerce").fillna(0.0)
        mask = tmp["__norm"].str.contains("backbuff", na=False)
        if mask.any(): bb_min = float(mins[mask].max())

    bnp_minutes = 0.0
    if cat_val in {'C4','C3-HF'}:        bnp_minutes = fb_min + bb_min
    elif cat_val in {'C3','C3-BG'}:      bnp_minutes = fb_min
    elif cat_val in {'C2','C2-C'}:       bnp_minutes = bb_min
    bnp_cost = (bnp_minutes / 60.0) * float(labor_rate) if bnp_minutes > 0 else 0.0

    total_parts = func_total + refurb_total + reglass_cost
    total_cost  = total_parts + tech_labor_cost + refurb_labor_cost + reglass_labor_cost + qc_cost + anod_cost + bnp_cost

    def _mk_cell(title_text: str, rows: list, total: float) -> str:
        if not rows:
            return "<span class='bb360-empty' title='No parts in this bucket'>&mdash;</span>"
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
        'model': row.get('_norm_model'),  # kept for CSV/reference
        'SKU': row.get('SKU'),            # display column
        'legacy_category': cat_val,
        'category_desc': row.get('Category_Desc'),
        'failures': "|".join(failures),
        'Functional (collapse)': functional_cell,
        'Refurb (collapse)': refurb_cell,
        'Functional Parts Cost': func_total,   # CSV only
        'Refurb Parts Cost': refurb_total,     # CSV only
        'Total Parts Cost': total_parts,       # CSV only
        'Reglass Cost': reglass_cost,          # visible
        'QC Labor': qc_cost,
        'Anodizing Labor': anod_cost,
        'BNP Labor': bnp_cost,
        'Tech Labor': tech_labor_cost,
        'Refurb Labor Cost': refurb_labor_cost,
        'Reglass Labor Cost': reglass_labor_cost,  # NEW: visible
        'Total Cost': total_cost,
    }

# -------------------- RUN (build results) --------------------
rows = []
for _, r in norm.iterrows():
    out = compute_row(r, labor_rate)
    if out is None: continue
    rows.append(out)

res_df = pd.DataFrame(rows)

# -------------------- FILTER UI + APPLY --------------------
if not res_df.empty:
    with st.sidebar:
        st.markdown("---")
        st.subheader("Filters")

        _all_categories = sorted(pd.Series(res_df.get('legacy_category', pd.Series(dtype=str))).dropna().unique().tolist())
        _min_cost = float(res_df['Total Cost'].min()) if 'Total Cost' in res_df.columns else 0.0
        _max_cost = float(res_df['Total Cost'].max()) if 'Total Cost' in res_df.columns else 0.0

        f_sku   = st.text_input("SKU contains", "")
        f_model = st.text_input("Model contains (CSV only)", "")
        f_imei  = st.text_input("IMEI contains", "")
        f_def   = st.text_input("Defects contain", "")
        f_cats  = st.multiselect("Category is", options=_all_categories, default=_all_categories)
        f_cost  = st.slider("Total Cost range", min_value=0.0, max_value=max(_max_cost, 0.0), value=( _min_cost, _max_cost ), step=0.5)

        st.markdown("---")
        st.subheader("Sort")
        sortable_cols = [
            'Reglass Cost','Tech Labor','Refurb Labor Cost','Reglass Labor Cost','QC Labor',
            'Anodizing Labor','BNP Labor','Total Cost'
        ]
        opts = [c for c in sortable_cols if c in res_df.columns]
        default_idx = opts.index('Total Cost') if 'Total Cost' in opts else (0 if opts else 0)
        s_col = st.selectbox("Sort by", options=opts, index=default_idx)
        s_asc = st.toggle("Ascending", value=False)

    filt = res_df.copy()
    if f_sku:   filt = filt[filt['SKU'].astype(str).str.contains(f_sku, case=False, na=False)]
    if f_model: filt = filt[filt['model'].astype(str).str.contains(f_model, case=False, na=False)]
    if f_imei:  filt = filt[filt['imei'].astype(str).str.contains(f_imei, case=False, na=False)]
    if f_def:   filt = filt[filt['failures'].astype(str).str.contains(f_def, case=False, na=False)]
    if f_cats:  filt = filt[filt['legacy_category'].astype(str).isin(f_cats)]
    if 'Total Cost' in filt.columns:
        lo, hi = f_cost
        filt = filt[(filt['Total Cost'] >= float(lo)) & (filt['Total Cost'] <= float(hi))]
    if s_col in filt.columns:
        filt = filt.sort_values(by=s_col, ascending=bool(s_asc), kind="mergesort")
else:
    filt = pd.DataFrame()

# -------------------- DISPLAY --------------------
if not filt.empty:
    disp = filt.copy()
    if CFG['ui'].get('hide_ipads_on_screen', True):
        disp = disp[~disp['model'].astype(str).str.upper().str.contains('IPAD', na=False)]

    SHOW_COLS = [
        'imei','SKU',   # show SKU instead of model
        'Functional (collapse)','Refurb (collapse)',
        'Reglass Cost',
        'Tech Labor','Refurb Labor Cost','Reglass Labor Cost','QC Labor','Anodizing Labor','BNP Labor',
        'Total Cost'
    ]
    FORBIDDEN = {'Functional Parts Cost','Refurb Parts Cost','Total Parts Cost'}
    bad = [c for c in FORBIDDEN if c in disp.columns]
    if bad: disp = disp.drop(columns=bad)
    present_cols = [c for c in SHOW_COLS if c in disp.columns]
    disp = disp[present_cols].copy()

    money_cols = ['Reglass Cost','Tech Labor','Refurb Labor Cost','Reglass Labor Cost','QC Labor','Anodizing Labor','BNP Labor','Total Cost']
    for col in money_cols:
        if col in disp.columns:
            disp[col] = disp[col].astype(float)

    def _fmt_money(x: float) -> str: return f"${x:,.2f}"

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
      .bb360-empty {{ opacity: .5; }}
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

    st.subheader("Final Results — IMEIs in Live only (SKU shown)")
    st.markdown(table_html, unsafe_allow_html=True)

    # CSV export: filtered rows (keep both SKU and model for traceability)
    export_cols = [c for c in filt.columns if c not in ['Functional (collapse)','Refurb (collapse)']]
    export_df = filt[export_cols].copy()
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Filtered CSV', data=csv_bytes, file_name='bb360_filtered.csv', mime='text/csv')
else:
    st.info("No qualifying rows to display after filtering.")
