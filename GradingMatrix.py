# === BB360 APP — Clean Business View (Merged) — PATCH r4 ===
# File: GradingMatrix_merged_business_view.py
# Date: 2025-10-08
#
# r4 highlights:
# • Grade B cosmetic price = ADHESIVES ONLY (reclaimed parts are $0).
# • Grade A cosmetic price = cosmetic parts (Type 'D-COVER' preferred) + adhesives.
# • Broadened cosmetic keyword matching to catch "BACK GLASS" vs "BACK COVER" labelling.
#
# Outputs per row:
# IMEI, SKU, Functional Parts Price, Refurb Price (Category Parts) Grade A, Grade B, Reglass Parts Price,
# Tech/Refurb/Reglass/QC/BNP/Anodizing labor, Total Refurbishment Cost, Acquisition Cost,
# A/B/C Selling Price & Margin, Final Selling Price, Final Margin.

import re, hashlib
from pathlib import Path
from collections import defaultdict
import html as _html
import pandas as pd
import streamlit as st
import yaml

def keyify(s: str) -> str: return re.sub(r'[^A-Z0-9]+', '', str(s).upper())
def uph_key(name: str) -> str: return re.sub(r'[^a-z0-9]+', '', str(name).lower())

def normalize_model(model_str):
    if pd.isna(model_str): return ''
    model = str(model_str).upper().strip()
    model = re.sub(r'\s+', ' ', model)
    model = re.sub(r'[\(\)]', '', model)
    model = re.sub(r'\bSE\s*22\b', 'SE 2022', model)
    model = re.sub(r'\bSE\s*20\b', 'SE 2020', model)
    if 'SEGEN3' in model or 'SE 3' in model or '3RD GEN' in model: return 'IPHONE SE 2022'
    if 'SEGEN2' in model or 'SE 2' in model or '2ND GEN' in model: return 'IPHONE SE 2020'
    return model

st.set_page_config(page_title='BB360: Clean Business View (r4)', layout='wide')
st.title('BB360: Refurb Cost & Margins — Clean Business View (r4)')

DEFAULTS = {"ui":{"hide_ipads_on_screen":True},"pricing":{"battery_default":"BATTERY CELL","lcd_default":"LCM GENERIC (HARD OLED)"},"labor":{"ceq_minutes":2,"use_qc_labor":True}}
def load_config():
    p = Path("config.yml")
    if p.exists():
        try:
            with open(p,"r",encoding="utf-8") as f: cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}
    else:
        cfg = {}
    # shallow merge
    for k,v in DEFAULTS.items():
        if k not in cfg: cfg[k]=v
        elif isinstance(v,dict) and isinstance(cfg[k],dict):
            tmp=v.copy(); tmp.update(cfg[k]); cfg[k]=tmp
    return cfg
CFG = load_config()

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
FRAME_BACKGLASS_MODELS = {"IPHONE 14","IPHONE 14 PLUS","IPHONE 15","IPHONE 15 PLUS","IPHONE 15 PRO","IPHONE 15 PRO MAX"}
ANODIZING_ELIGIBLE_MODELS = {"IPHONE SE","IPHONE SE 2020","IPHONE SE 2022","IPHONE 11","IPHONE XR","IPHONE 12","IPHONE 12 MINI","IPHONE 13","IPHONE 13 MINI","IPHONE 14","IPHONE 14 PLUS"}

uploaded   = st.file_uploader('Upload BB360 export (CSV or Excel)', type=['xlsx','xls','csv'])
as_file    = st.file_uploader('Upload Live file (CSV or Excel with Inventory/Handset/Raw Data sheet)', type=['xlsx','xls','csv'])
parts_file = st.file_uploader('Upload Pricing + Categories Excel (with F2P, Cosmetic Category, Pricelist, UPH, Purchase Price sheets)', type=['xlsx','xls'])

if not uploaded or not as_file or not parts_file:
    st.info('Upload all three files to continue.'); st.stop()

@st.cache_data(show_spinner=False)
def load_all(uploaded, as_file, parts_file):
    df_raw = pd.read_csv(uploaded) if uploaded.name.lower().endswith('.csv') else pd.read_excel(uploaded)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]
    # live
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
    # pricing
    f2p_parts      = pd.read_excel(parts_file, sheet_name='F2P')
    cosmetic_cat   = pd.read_excel(parts_file, sheet_name='Cosmetic Category')
    pricelist      = pd.read_excel(parts_file, sheet_name='Pricelist')
    uph            = pd.read_excel(parts_file, sheet_name='UPH')
    purchase_price = pd.read_excel(parts_file, sheet_name='Purchase Price')
    return df_raw, as_inv, f2p_parts, cosmetic_cat, pricelist, uph, purchase_price, live_sheet

df_raw, as_inv, f2p_parts, cosmetic_cat, pricelist, uph, purchase_price, LIVE_SHEET = load_all(uploaded, as_file, parts_file)
st.caption(f"Live workbook sheet used: {LIVE_SHEET}")

def find_column(df: pd.DataFrame, candidates: list):
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        k = cand.lower().strip()
        if k in lower_map: return lower_map[k]
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

# profile/result filter
prof_col = find_column(df_raw, COLUMN_SYNONYMS['profile_name'])
if prof_col is not None:
    df_raw['_p'] = df_raw[prof_col].astype(str).str.upper().str.strip()
    df_raw = df_raw[df_raw['_p'].isin(ALLOWED_PROFILES)].copy()
    df_raw.drop(columns=['_p'], inplace=True, errors='ignore')
ar_col = find_column(df_raw, COLUMN_SYNONYMS['analyst_result'])
if ar_col is not None:
    df_raw['_a'] = df_raw[ar_col].astype(str).str.strip().str.lower()
    df_raw = df_raw[df_raw['_a'] != 'not completed'].copy()
    df_raw.drop(columns=['_a'], inplace=True, errors='ignore')

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
                normalized[key] = merged.mask(merged.eq(''))
            else:
                normalized[key] = pd.NA
        else:
            col = find_column(df, candidates)
            normalized[key] = df[col] if col is not None else pd.NA
    return pd.concat([df, normalized.add_prefix('_norm_')], axis=1)

norm = normalize_input_df(df_raw)

# Merge with live
as_inv.columns = [str(c).strip().lower() for c in as_inv.columns]
imei_col = next((c for c in as_inv.columns if any(k in c for k in ["imei","sn","serial"])), None)
cat_col  = next((c for c in as_inv.columns if "category" in c), None)
def _find_live_col(substrs):
    for c in as_inv.columns:
        if any(s in str(c).lower() for s in substrs): return c
    return None
sku_col        = _find_live_col(["sku","sku name","sku id","sku description"])
live_model_col = _find_live_col(["model","device model","device"])
color_col      = _find_live_col(["color","colour"])
capacity_col   = _find_live_col(["capacity","storage","rom"])

norm['_norm_imei'] = norm['_norm_imei'].astype(str).str.strip()
as_inv[imei_col]   = as_inv[imei_col].astype(str).str.strip()

live_cols = [c for c in [imei_col,cat_col,sku_col,live_model_col,color_col,capacity_col] if c]
merged = norm.merge(as_inv[live_cols], left_on='_norm_imei', right_on=imei_col, how='left', indicator=True)
merged = merged[merged['_merge']=='both'].copy().rename(columns={cat_col:'Category'}).drop(columns=[imei_col,'_merge'], errors='ignore')
merged['Category'] = merged['Category'].astype(str).str.strip().replace({'':'No Category','nan':'No Category','None':'No Category'})

def _cap_norm(x: str) -> str:
    s = str(x or '').strip()
    if not s or s.lower() in ('nan','none'): return ''
    m = re.search(r'(\d+)\s*(tb|gb|g|gig|gigabyte|gigabytes)?', s, re.I)
    if m:
        num = m.group(1); unit = (m.group(2) or 'GB').upper()
        unit = 'TB' if unit.startswith('T') else 'GB'
        return f"{num}{unit}"
    return s.upper()
def _capwords(s: str) -> str:
    s = str(s or '').strip()
    if not s or s.lower() in ('nan','none'): return ''
    return ' '.join(w.capitalize() for w in s.split())

merged['SKU'] = ''
if sku_col: merged['SKU'] = merged[sku_col].astype(str).str.strip()
fallback_model_series = merged[live_model_col] if (live_model_col and live_model_col in merged.columns) else merged['_norm_model']
cap_series = merged[capacity_col].astype(str) if (capacity_col and capacity_col in merged.columns) else ''
col_series = merged[color_col].astype(str) if (color_col and color_col in merged.columns) else ''
built = (fallback_model_series.astype(str).str.upper().str.strip()
         + (' ' + pd.Series(cap_series).apply(_cap_norm)).replace(' ','',regex=False)
         + (' ' + pd.Series(col_series).apply(_capwords)).replace(' ','',regex=False)).str.strip()
merged['SKU'] = merged['SKU'].mask(merged['SKU'].eq('') | merged['SKU'].str.lower().isin(['nan','none']), built)
merged['SKU'] = merged['SKU'].mask(merged['SKU'].eq('') | merged['SKU'].str.lower().isin(['nan','none']), merged['_norm_model'].astype(str).str.upper().str.strip())

model_key_norm = fallback_model_series.apply(normalize_model).astype(str).str.upper().str.strip()
merged['SKU_KEY'] = (model_key_norm + (' ' + pd.Series(cap_series).apply(_cap_norm)).replace(' ','',regex=False)).str.strip()

# Pricing normalization
cosmetic_cat.columns = [str(c).strip() for c in cosmetic_cat.columns]
CAT_TO_DESC = dict(zip(cosmetic_cat['Legacy Category'], cosmetic_cat['Description']))

f2p_parts['Model_norm']  = f2p_parts['iPhone Model'].apply(normalize_model)
f2p_parts['Faults_norm'] = f2p_parts['Faults'].astype(str).str.lower().str.strip()
f2p_parts['Part_norm']   = f2p_parts['Part'].astype(str).str.upper().str.strip().str.replace(r'\s+',' ',regex=True)

pricelist['Model_norm'] = pricelist['iPhone Model'].apply(normalize_model)
pricelist['Part_norm']  = pricelist['Part'].astype(str).str.upper().str.strip().str.replace(r'\s+',' ',regex=True)
pricelist['PRICE']      = pd.to_numeric(pricelist['PRICE'].astype(str).str.replace(r'[^0-9\.]','',regex=True).replace('','0'), errors='coerce').fillna(0.0)
pricelist['Type_norm']  = pricelist.get('Type', pd.Series(['']*len(pricelist))).astype(str).str.upper().str.strip()
pricelist['Part_key']   = pricelist['Part_norm'].apply(keyify)

PRICE_INDEX = {(m,p):v for m,p,v in zip(pricelist['Model_norm'],pricelist['Part_norm'],pricelist['PRICE'])}
PL_BY_MODEL = {m:g for m,g in pricelist.groupby('Model_norm', sort=False)}

F2P_INDEX = defaultdict(list)
for m,f,p in zip(f2p_parts['Model_norm'], f2p_parts['Faults_norm'], f2p_parts['Part_norm']):
    F2P_INDEX[(m,f)].append(p)

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

RE_KEYS = {"REGLASS","REGLASSDIGITIZER"}
PL_RE_BY_MODEL = defaultdict(dict)
for _, row in pricelist.iterrows():
    if row['Type_norm'] == 'RE' and row['Part_key'] in RE_KEYS:
        PL_RE_BY_MODEL[row['Model_norm']][row['Part_key']] = float(row['PRICE'])

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

def adhesive_total_for(model, cat_val):
    total = 0.0
    for ak in CATEGORY_ADHESIVES_MAPPING.get(str(cat_val).upper(), []):
        part_name = ADHESIVE_INDEX.get(model, {}).get(ak)
        if part_name and (model, part_name) in PRICE_INDEX:
            total += float(PRICE_INDEX[(model, part_name)])
    return total

def cosmetic_keywords_for(legacy_cat, model):
    legacy_cat = str(legacy_cat).strip().upper()
    if legacy_cat in {"C0","C1","C3"}:
        if model in FRAME_BACKGLASS_MODELS:
            return ["BACKGLASS","HOUSING FRAME"]
        else:
            return ["BACKGLASS","BACK COVER"]
    kws, desc = [], str(CAT_TO_DESC.get(legacy_cat, "")).upper()
    if legacy_cat != "C4":
        if "BACK COVER" in desc: kws.append("BACK COVER")
        if "BACKGLASS" in desc or "BACK GLASS" in desc: kws.append("BACKGLASS")
        if "HOUSING FRAME" in desc: kws.append("HOUSING FRAME")
    seen, out = set(), []
    for k in kws:
        if k not in seen:
            out.append(k); seen.add(k)
    return out

def select_cosmetic_total_grade_A(model, cat_val):
    total = 0.0
    model_slice = PL_BY_MODEL.get(model)
    if model_slice is not None:
        for kw in cosmetic_keywords_for(cat_val, model):
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
            else: # BACK COVER
                cond = (model_slice['Part_norm'].str.contains("BACK COVER", na=False) &
                        ~model_slice['Part_norm'].str.contains("ADHESIVE", na=False))
            cand = model_slice[cond]
            pref = cand[cand['Type_norm'].str.upper() == 'D-COVER']
            chosen = pref if not pref.empty else cand
            if not chosen.empty:
                total += float(chosen['PRICE'].iat[0])
    total += adhesive_total_for(model, cat_val)
    return total

# Purchase Price
purchase_price.columns = [str(c).strip() for c in purchase_price.columns]
pp_cols = purchase_price.columns
col_map = {
    "SKU": next((c for c in pp_cols if c.strip().lower() == "sku"), "SKU"),
    "Acq": next((c for c in pp_cols if "acquisition" in c.strip().lower()), "Acquisition price"),
    "A":   next((c for c in pp_cols if c.strip().lower() in ("grade a","grade a price","a")), purchase_price.columns[2] if len(purchase_price.columns)>2 else "Grade A"),
    "B":   next((c for c in pp_cols if c.strip().lower() in ("grade b","grade b price","b")), purchase_price.columns[3] if len(purchase_price.columns)>3 else "Grade B"),
    "C":   next((c for c in pp_cols if c.strip().lower() in ("grade c","grade c price","c")), purchase_price.columns[4] if len(purchase_price.columns)>4 else "Grade C"),
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
    str(row["PP_SKU"]).strip().upper(): {"acq": float(row["PP_Acquisition"]), "A": float(row["PP_GradeA"]), "B": float(row["PP_GradeB"]), "C": float(row["PP_GradeC"])}
    for _, row in purchase_price.iterrows()
}

# User selectors
battery_type = st.selectbox("Battery Type", ["BATTERY CELL","BATTERY OEM","BATTERY OEM PULLED"], index=0).upper()
lcd_type = st.selectbox("LCD Type", ["LCM GENERIC (HARD OLED)","LCM GENERIC (TFT)","LCM -OEM REFURBISHED (GLASS CHANGED -GENERIC)","LCM GENERIC (SOFT OLED)"], index=0).upper()
labor_rate = st.slider("Labor Rate ($/hour)", 1, 35, 5)

# UPH
uph = uph[['Type of Defect', 'Ave. Repair Time (Mins)']].dropna(subset=['Type of Defect']).copy()
uph['Defect_norm'] = uph['Type of Defect'].map(uph_key)
uph['Ave. Repair Time (Mins)'] = pd.to_numeric(uph['Ave. Repair Time (Mins)'], errors='coerce').fillna(0.0)
UPH_INDEX = dict(zip(uph['Defect_norm'], uph['Ave. Repair Time (Mins)']))

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

def compute_row(row, labor_rate):
    failures = parse_failures(row.get('_norm_defects',''))
    batt_status, _, _ = battery_status(row.get('_norm_battery_cycle'), row.get('_norm_battery_health'))
    model_raw = normalize_model(row.get('_norm_model')); device_model = model_raw
    cat_val = str(row.get('Category') or '').strip().upper()
    sku_key = str(row.get('SKU_KEY') or '').strip().upper()

    # functional parts
    func_parts = []
    for f in [str(f).lower().strip() for f in failures]:
        for part in F2P_INDEX.get((device_model, f), []):
            key = str(part).upper().strip()
            if (device_model, key) in PRICE_INDEX and key not in func_parts:
                func_parts.append(key)

    if batt_status == "Battery Service":
        key = battery_type
        if (device_model, key) in PRICE_INDEX and key not in func_parts:
            func_parts.append(key)
        if key == "BATTERY CELL":
            model_slice = PL_BY_MODEL.get(device_model)
            if model_slice is not None:
                fx = model_slice[model_slice['Part_norm'].str.contains("BATTERY FLEX", na=False)]
                if not fx.empty:
                    fk = fx['Part_norm'].iat[0]
                    if (device_model, fk) in PRICE_INDEX and fk not in func_parts:
                        func_parts.append(fk)

    if any(_re_lcd.search(str(f)) for f in failures) or cat_val in {"C0","C2-C"}:
        if (device_model, lcd_type) in PRICE_INDEX and lcd_type not in func_parts:
            func_parts.append(lcd_type)

    func_total = sum(float(PRICE_INDEX.get((device_model, k), 0.0)) for k in func_parts)

    tech_minutes = sum(float(UPH_INDEX.get(uph_key(tok), 0.0)) for tok in failures)
    if failures: tech_minutes += float(CFG['labor'].get('ceq_minutes', 2))
    tech_labor_cost = (tech_minutes / 60.0) * float(labor_rate)

    refcat_set = {'C0','C1','C3-BG','C3-HF','C3'}
    refurb_minutes = float(UPH_INDEX.get(uph_key(cat_val), 0.0)) if cat_val in refcat_set else 0.0
    refurb_labor_cost = (refurb_minutes / 60.0) * float(labor_rate)

    RE_ELIGIBLE_CATS = {'C1','C2','C2-BG'}
    re_applicable = cat_val in RE_ELIGIBLE_CATS
    has_mts = any("multitouchscreen" in str(f).lower().replace(" ","").replace("-","") for f in failures) if re_applicable else False
    re_min_candidates = []
    if re_applicable:
        if has_mts:
            re_min_candidates += [float(UPH_INDEX.get(uph_key("reglassdigitizer"), 0.0)),
                                  float(UPH_INDEX.get(uph_key("re-glass digitizer"), 0.0))]
        re_min_candidates += [float(UPH_INDEX.get(uph_key("reglass"), 0.0)),
                              float(UPH_INDEX.get(uph_key("re-glass"), 0.0))]
    reglass_minutes = max(re_min_candidates) if re_min_candidates else 0.0
    reglass_labor_cost = (reglass_minutes / 60.0) * float(labor_rate)
    preferred_key = 'REGLASSDIGITIZER' if has_mts else 'REGLASS'
    reglass_price = float(PL_RE_BY_MODEL.get(device_model, {}).get(preferred_key, 0.0)) if re_applicable else 0.0

    qc_min = 0.0
    for key_try in ["qc process","qc inspection","qcinspection","qc","quality control","quality check"]:
        qc_min = max(qc_min, float(UPH_INDEX.get(uph_key(key_try), 0.0)))
    qc_cost = (qc_min / 60.0) * float(labor_rate)

    fb_min = max(float(UPH_INDEX.get(uph_key("front buffing"), 0.0)),
                 float(UPH_INDEX.get(uph_key("front polish"), 0.0)))
    bb_min = max(float(UPH_INDEX.get(uph_key("back buffing"), 0.0)),
                 float(UPH_INDEX.get(uph_key("back polish"), 0.0)))
    if cat_val in {'C4','C3-HF'}: bnp_minutes = fb_min + bb_min
    elif cat_val in {'C3','C3-BG'}: bnp_minutes = fb_min
    elif cat_val in {'C2','C2-C'}: bnp_minutes = bb_min
    else: bnp_minutes = 0.0
    bnp_cost = (bnp_minutes / 60.0) * float(labor_rate) if bnp_minutes > 0 else 0.0

    anod_min = max(float(UPH_INDEX.get(uph_key("anodizing"), 0.0)),
                   float(UPH_INDEX.get(uph_key("anodise"), 0.0)))
    anod_cats = {'C2','C2-C','C2-BG','C3-BG','C4'}
    anod_cost = (anod_min / 60.0) * float(labor_rate) if (device_model in ANODIZING_ELIGIBLE_MODELS and cat_val in anod_cats and anod_min>0) else 0.0

    cos_A = select_cosmetic_total_grade_A(device_model, cat_val)  # parts + adhesives
    cos_B = adhesive_total_for(device_model, cat_val)             # adhesives only

    refurb_A = func_total + cos_A + reglass_price + reglass_labor_cost + refurb_labor_cost + anod_cost + bnp_cost + tech_labor_cost + qc_cost
    refurb_B = func_total + cos_B + reglass_price + reglass_labor_cost + refurb_labor_cost + anod_cost + bnp_cost + tech_labor_cost + qc_cost
    refurb_C = func_total + tech_labor_cost + qc_cost

    pp = PURCHASE_INDEX.get(sku_key, {"acq":0.0,"A":0.0,"B":0.0,"C":0.0})
    acq = float(pp["acq"]); price_A = float(pp["A"]); price_B = float(pp["B"]); price_C = float(pp["C"])
    margin_A = price_A - (acq + refurb_A)
    margin_B = price_B - (acq + refurb_B)
    margin_C = price_C - (acq + refurb_C)

    best_grade, best_price, best_margin, best_refurb = max(
        [("A", price_A, margin_A, refurb_A), ("B", price_B, margin_B, refurb_B), ("C", price_C, margin_C, refurb_C)],
        key=lambda t: t[2]
    )

    return {
        'IMEI': row.get('_norm_imei'),
        'SKU': row.get('SKU'),
        'Model (CSV)': row.get('_norm_model'),
        'Legacy Category': cat_val,
        'Failures (parsed)': "|".join(failures),
        'Functional Parts Price': func_total,
        'Refurb Price (Category Parts) Grade A': cos_A,
        'Refurb Price (Category Parts) Grade B': cos_B,
        'Reglass Parts Price': reglass_price,
        'Tech Labor Cost': tech_labor_cost,
        'Refurb Labor Cost': refurb_labor_cost,
        'Reglass Labor Cost': reglass_labor_cost,
        'QC Labor Cost': qc_cost,
        'BNP Labor Cost': bnp_cost,
        'Anodizing Labor Cost': anod_cost,
        'Total Refurbishment Cost': best_refurb,
        'Acquisition Cost': acq,
        'Grade A Selling Price': price_A, 'Grade A Margin': margin_A,
        'Grade B Selling Price': price_B, 'Grade B Margin': margin_B,
        'Grade C Selling Price': price_C, 'Grade C Margin': margin_C,
        'Final Selling Price': best_price, 'Final Margin': best_margin, 'Final Grade': best_grade,
    }

rows = []
for _, r in merged.iterrows():
    rows.append(compute_row(r, labor_rate))
res_df = pd.DataFrame(rows)

if not res_df.empty:
    disp = res_df.copy()
    if CFG['ui'].get('hide_ipads_on_screen', True):
        disp = disp[~disp['Model (CSV)'].astype(str).str.upper().str.contains('IPAD', na=False)]
    SHOW_COLS = [
        'IMEI','SKU',
        'Functional Parts Price',
        'Refurb Price (Category Parts) Grade A','Refurb Price (Category Parts) Grade B',
        'Reglass Parts Price',
        'Tech Labor Cost','Refurb Labor Cost','Reglass Labor Cost','QC Labor Cost','BNP Labor Cost','Anodizing Labor Cost',
        'Total Refurbishment Cost','Acquisition Cost',
        'Grade A Selling Price','Grade A Margin',
        'Grade B Selling Price','Grade B Margin',
        'Grade C Selling Price','Grade C Margin',
        'Final Selling Price','Final Margin','Final Grade'
    ]
    present_cols = [c for c in SHOW_COLS if c in disp.columns]
    disp = disp[present_cols].copy()

    # pretty money formatting in HTML table for readability
    money_cols = [c for c in present_cols if c not in ['IMEI','SKU','Final Grade']]
    for col in money_cols:
        disp[col] = pd.to_numeric(disp[col], errors='coerce').fillna(0.0)

    def _fmt_money(x: float) -> str: return f"${x:,.2f}"

    row_html = []
    for _, row in disp.iterrows():
        tds = []
        for c in present_cols:
            v = row[c]
            if c in money_cols:
                if c in ['Final Margin']:
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
    st.subheader("Clean Business View — IMEIs present in Live (AS)")
    st.markdown(table_html, unsafe_allow_html=True)

    csv_bytes = res_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Full CSV', data=csv_bytes, file_name='bb360_business_view_full.csv', mime='text/csv')
else:
    st.info("No qualifying rows to display.")
