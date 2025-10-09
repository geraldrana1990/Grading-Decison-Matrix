# === BB360 APP — Clean Business View (Merged) — FULL r7 ===
# Date: 2025-10-09
#
# r7 highlights:
# • Secrets→Env bridge for Streamlit Cloud TOML (avoids “Invalid TOML” mismatches).
# • HTTP puller saves Live as “AS (1).xlsx” (matches auto-finder).
# • Live finder prefers AS (1).xlsx in DATA_ROOT and app dir.
# • Grade B = adhesives only; mark B as N/A for {C0,C1,C3,C3-BG,C2-BG}.
# • LCD labor adder (+45m) only for C0/C2-C when no LCD failure token exists.
# • CSV includes parts breakdown columns.
# • Soft-fail “Purchase Price” sheet (warn + zero table), plus sidebar sheet-debug.

import re, os, glob, json, time
from pathlib import Path
from collections import defaultdict
import html as _html
import pandas as pd
import streamlit as st
import yaml
import requests

# ───────────────────────────────────────────────────────────────────────────────
# Streamlit Cloud: bridge TOML Secrets → os.environ (so existing env reads work)
# ───────────────────────────────────────────────────────────────────────────────
try:
    for k in ("PARTS_PUBLIC_URL", "LIVE_PUBLIC_URL", "HTTP_TTL", "BB360_DATA_ROOT"):
        if k in st.secrets and not os.environ.get(k):
            os.environ[k] = str(st.secrets[k])
except Exception:
    pass

# ───────────────────────────────────────────────────────────────────────────────
# Paths & runtime
# ───────────────────────────────────────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
DATA_ROOT = Path(os.environ.get("BB360_DATA_ROOT", APP_DIR / "data")).resolve()
DATA_ROOT.mkdir(parents=True, exist_ok=True)

PARTS_PUBLIC_URL = os.environ.get("PARTS_PUBLIC_URL", "").strip()
LIVE_PUBLIC_URL  = os.environ.get("LIVE_PUBLIC_URL", "").strip()
HTTP_TTL         = int(os.environ.get("HTTP_TTL", "900"))

# HTTP download targets (keep names consistent with finders)
PARTS_TARGET = (DATA_ROOT / "parts" / "parts-http.auto.xlsx")   # change to .json if you serve JSON
LIVE_TARGET  = (DATA_ROOT / "AS (1).xlsx")                      # ← important: (1)

def _fresh_enough(path: Path, ttl: int) -> bool:
    try:
        return path.exists() and (time.time() - path.stat().st_mtime) < ttl
    except Exception:
        return False

def _http_download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, allow_redirects=True, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1<<20):
                if chunk: f.write(chunk)

def pull_http_public_files_if_needed():
    if PARTS_PUBLIC_URL and not _fresh_enough(PARTS_TARGET, HTTP_TTL):
        try: _http_download(PARTS_PUBLIC_URL, PARTS_TARGET)
        except Exception as e: st.warning(f"HTTP parts pull failed: {e}")
    if LIVE_PUBLIC_URL and not _fresh_enough(LIVE_TARGET, HTTP_TTL):
        try: _http_download(LIVE_PUBLIC_URL, LIVE_TARGET)
        except Exception as e: st.warning(f"HTTP live pull failed: {e}")

# ───────────────────────────────────────────────────────────────────────────────
# UI
# ───────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title='BB360: Clean Business View (r7)', layout='wide')
st.title('BB360: Refurb Cost & Margins — Clean Business View (r7)')

DEFAULTS = {
    "ui": {"hide_ipads_on_screen": True},
    "pricing": {"battery_default": "BATTERY CELL", "lcd_default": "LCM GENERIC (HARD OLED)"},
    "labor": {"ceq_minutes": 2, "use_qc_labor": True},
    "auto_sources": {
        "parts_globs": ["parts/*.json","parts/*.xlsx","data/parts/*.json","data/parts/*.xlsx","*.parts.json","*parts*.xlsx"],
        "live_globs":  ["live/*.json","live/*.xlsx","live/*.csv","data/live/*.json","data/live/*.xlsx","data/live/*.csv","*live*.json","*live*.xlsx","*live*.csv"],
    },
    "live": {"default_path": "", "require": False, "glob_paths": []}
}

def load_config():
    p = APP_DIR / "config.yml"
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f: user = yaml.safe_load(f) or {}
        except Exception:
            user = {}
    else:
        user = {}
    cfg = DEFAULTS.copy()
    # shallow merge
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            tmp = cfg[k].copy(); tmp.update(v); cfg[k] = tmp
        else:
            cfg[k] = v
    return cfg

CFG = load_config()

# ───────────────────────────────────────────────────────────────────────────────
# File discovery
# ───────────────────────────────────────────────────────────────────────────────
AUTO_SOURCES = {
    "parts_globs": CFG.get("auto_sources", {}).get("parts_globs", DEFAULTS["auto_sources"]["parts_globs"]),
    "live_globs":  CFG.get("auto_sources", {}).get("live_globs",  DEFAULTS["auto_sources"]["live_globs"]),
}
LIVE_CFG = {
    "default_path": str(CFG.get("live", {}).get("default_path", "")),
    "require": bool(CFG.get("live", {}).get("require", False)),
    "glob_paths": CFG.get("live", {}).get("glob_paths") or AUTO_SOURCES["live_globs"],
}

def _expand_globs(patterns):
    results = []
    for pat in patterns:
        for base in (APP_DIR, DATA_ROOT):
            for p in glob.glob(str((base / pat))):
                try:
                    if Path(p).is_file():
                        results.append(str(Path(p).resolve()))
                except Exception:
                    pass
    uniq = {}
    for p in results:
        try: uniq[p] = os.path.getmtime(p)
        except Exception: uniq[p] = 0
    return [k for k,_ in sorted(uniq.items(), key=lambda kv: kv[1], reverse=True)]

def _pick_parts_path():
    hits = _expand_globs(AUTO_SOURCES["parts_globs"])
    return hits[0] if hits else (str(PARTS_TARGET) if PARTS_TARGET.exists() else None)

def _find_as_xlsx():
    # prefer AS (1).xlsx
    for p in [
        APP_DIR / "AS (1).xlsx",
        DATA_ROOT / "AS (1).xlsx",
        APP_DIR / "live" / "AS (1).xlsx",
        DATA_ROOT / "live" / "AS (1).xlsx",
    ]:
        if p.exists() and p.is_file():
            return str(p.resolve())
    return None

def _pick_live_path():
    pref = _find_as_xlsx()
    if pref: return pref
    dp = LIVE_CFG.get("default_path", "").strip()
    if dp:
        p = Path(dp)
        if p.exists() and p.is_file():
            return str(p.resolve())
    hits = _expand_globs(LIVE_CFG["glob_paths"])
    if hits: return hits[0]
    return str(LIVE_TARGET) if LIVE_TARGET.exists() else None

# ───────────────────────────────────────────────────────────────────────────────
# Parsers
# ───────────────────────────────────────────────────────────────────────────────
def keyify(s: str) -> str: return re.sub(r'[^A-Z0-9]+', '', str(s).upper())
def uph_key(name: str) -> str: return re.sub(r'[^a-z0-9]+', '', str(name).lower())

def normalize_model(model_str):
    if pd.isna(model_str): return ''
    model = str(model_str).upper().strip()
    model = re.sub(r'\s+',' ', model)
    model = re.sub(r'[\(\)]','', model)
    model = re.sub(r'\bSE\s*22\b', 'SE 2022', model)
    model = re.sub(r'\bSE\s*20\b', 'SE 2020', model)
    if 'SEGEN3' in model or 'SE 3' in model or '3RD GEN' in model: return 'IPHONE SE 2022'
    if 'SEGEN2' in model or 'SE 2' in model or '2ND GEN' in model: return 'IPHONE SE 2020'
    return model

def _read_json_bundle(stream_or_path):
    close_me = None
    if isinstance(stream_or_path, (str, os.PathLike)):
        stream = open(stream_or_path, "rb"); close_me = stream
    else:
        stream = stream_or_path
    try:
        raw = json.load(stream)
    finally:
        if close_me: close_me.close()
    if not isinstance(raw, dict): raise ValueError("Parts JSON must be an object with the 5 arrays.")
    def _df(key):
        if key not in raw or not isinstance(raw[key], list):
            raise ValueError(f"Parts JSON missing array '{key}'")
        return pd.DataFrame(raw[key])
    return (_df("F2P"), _df("Cosmetic Category"), _df("Pricelist"), _df("UPH"), _df("Purchase Price"))

def _norm_sheet_map(names):
    def _n(s): return re.sub(r'[^a-z0-9]+','', str(s).lower())
    return {_n(s): s for s in names}

def _read_sheet_any(xls: pd.ExcelFile, aliases, required=True, allow_substring=True, warn_name=""):
    norm = _norm_sheet_map(xls.sheet_names)
    # exact
    for a in aliases:
        if a in xls.sheet_names:
            return pd.read_excel(xls, sheet_name=a)
    # normalized
    for a in aliases:
        key = re.sub(r'[^a-z0-9]+','', a.lower())
        if key in norm: return pd.read_excel(xls, sheet_name=norm[key])
    # substring
    if allow_substring:
        for a in aliases:
            key = re.sub(r'[^a-z0-9]+','', a.lower())
            for nk, real in norm.items():
                if key and key in nk:
                    return pd.read_excel(xls, sheet_name=real)
    if required:
        raise ValueError(f"Missing expected sheet. Looked for {aliases}; Available: {xls.sheet_names}")
    else:
        try: st.warning(f"Optional sheet '{warn_name or aliases[0]}' not found. Continuing with empty table.")
        except Exception: pass
        return pd.DataFrame()

FAIL_SOFT_PURCHASE = True

def _read_parts_bundle_any(file_obj_or_path: str | os.PathLike):
    name = str(file_obj_or_path).lower()
    if name.endswith(".json"):
        return _read_json_bundle(file_obj_or_path)
    xls = pd.ExcelFile(file_obj_or_path)
    # aliases
    F2P_ALIASES        = ["F2P","FaultsToParts","F2P Mapping","F2P_Map","F2P Sheet","Faults→Parts"]
    COSMETIC_ALIASES   = ["Cosmetic Category","Cosmetics","Cosmetic","CosmeticCategory","Cosmetic Cat","Cosmetic-Cat"]
    PRICELIST_ALIASES  = ["Pricelist","Price List","Parts Price","Part Prices","Parts Pricelist","Price_List"]
    UPH_ALIASES        = ["UPH","Labor","Repair Times","Ave. Repair Time","UPH Table","UPH Mapping"]
    PURCHASE_ALIASES   = ["Purchase Price","Acquisition","Acquisition price","Buy Price","PurchasePrice","Acq Price"]

    f2p_parts      = _read_sheet_any(xls, F2P_ALIASES,       required=True,  warn_name="F2P")
    cosmetic_cat   = _read_sheet_any(xls, COSMETIC_ALIASES,  required=True,  warn_name="Cosmetic Category")
    pricelist      = _read_sheet_any(xls, PRICELIST_ALIASES, required=True,  warn_name="Pricelist")
    uph            = _read_sheet_any(xls, UPH_ALIASES,       required=True,  warn_name="UPH")
    try:
        purchase_price = _read_sheet_any(xls, PURCHASE_ALIASES, required=not FAIL_SOFT_PURCHASE, warn_name="Purchase Price")
    except ValueError as e:
        if FAIL_SOFT_PURCHASE:
            st.warning("Purchase Price sheet not found. Created an empty table (all prices=0).")
            purchase_price = pd.DataFrame({"SKU": [], "Acquisition price": [], "Grade A": [], "Grade B": [], "Grade C": []})
        else:
            raise
    return f2p_parts, cosmetic_cat, pricelist, uph, purchase_price

def _read_live_any(path_or_file, name_hint: str = ""):
    name_low = (name_hint or str(path_or_file)).lower()
    if isinstance(path_or_file, (str, os.PathLike)):
        p = str(path_or_file)
        if p.lower().endswith(".csv"):  return pd.read_csv(p), os.path.basename(p)
        if p.lower().endswith((".xlsx",".xls")):
            xls = pd.ExcelFile(p)
            pick = None
            for s in xls.sheet_names:
                if pick is None and "inventory" in s.lower(): pick = s
            if pick is None:
                for s in xls.sheet_names:
                    if "handset" in s.lower(): pick = s; break
            if pick is None:
                for s in xls.sheet_names:
                    if "raw data" in s.lower(): pick = s; break
            if pick is None: pick = xls.sheet_names[0]
            return pd.read_excel(xls, sheet_name=pick), f"{os.path.basename(p)}[{pick}]"
        if p.lower().endswith(".json"):
            try: return pd.read_json(p, lines=True), os.path.basename(p)+"[jsonl]"
            except Exception:
                raw = json.load(open(p, "rb"))
                if isinstance(raw, list): return pd.DataFrame(raw), os.path.basename(p)+"[array]"
                if isinstance(raw, dict):
                    for k in ['Inventory','inventory','Handset','handset','Raw Data','raw data','raw_data','data','table','rows']:
                        if k in raw and isinstance(raw[k], list):
                            return pd.DataFrame(raw[k]), os.path.basename(p)+f"[{k}]"
                return pd.DataFrame([raw]), os.path.basename(p)+"[obj]"
        raise ValueError("Unsupported Live file extension.")
    else:
        # uploads
        fn = name_hint.lower()
        if fn.endswith(".csv"):  return pd.read_csv(path_or_file), "(csv upload)"
        if fn.endswith((".xlsx",".xls")):
            xls = pd.ExcelFile(path_or_file)
            pick = None
            for s in xls.sheet_names:
                if pick is None and "inventory" in s.lower(): pick = s
            if pick is None:
                for s in xls.sheet_names:
                    if "handset" in s.lower(): pick = s; break
            if pick is None:
                for s in xls.sheet_names:
                    if "raw data" in s.lower(): pick = s; break
            if pick is None: pick = xls.sheet_names[0]
            return pd.read_excel(xls, sheet_name=pick), f"(xlsx upload)[{pick}]"
        if fn.endswith(".json"):
            try: return pd.read_json(path_or_file, lines=True), "(jsonl upload)"
            except Exception:
                path_or_file.seek(0)
                raw = json.load(path_or_file)
                if isinstance(raw, list): return pd.DataFrame(raw), "(json array upload)"
                if isinstance(raw, dict):
                    for k in ['Inventory','inventory','Handset','handset','Raw Data','raw data','raw_data','data','table','rows']:
                        if k in raw and isinstance(raw[k], list):
                            return pd.DataFrame(raw[k]), f"(json upload)[{k}]"
                return pd.DataFrame([raw]), "(json upload)[obj]"
        raise ValueError("Unsupported Live upload format.")

# ───────────────────────────────────────────────────────────────────────────────
# Inputs & autoload
# ───────────────────────────────────────────────────────────────────────────────
uploaded     = st.file_uploader('Upload BB360 export (CSV / Excel / JSON)', type=['xlsx','xls','csv','json'])
live_upload  = st.file_uploader('Upload Live file (optional; CSV / Excel / JSON). Leave empty to auto/pull.', type=['xlsx','xls','csv','json'])

pull_http_public_files_if_needed()  # downloads if secrets set

@st.cache_data(show_spinner=False)
def autoload_parts_and_live(live_upload):
    # Parts
    parts_path = _pick_parts_path()
    if not parts_path:
        st.error("Cannot find Parts bundle. Place it under ./data/parts or set PARTS_PUBLIC_URL.")
        st.stop()
    f2p_parts, cosmetic_cat, pricelist, uph, purchase_price = _read_parts_bundle_any(parts_path)
    parts_label = parts_path

    # Live
    if live_upload is not None:
        as_inv, live_label = _read_live_any(live_upload, live_upload.name)
    else:
        chosen = _pick_live_path()
        if chosen:
            as_inv, live_label = _read_live_any(chosen, chosen)
        else:
            as_inv, live_label = pd.DataFrame(), "(no live found)"
    return f2p_parts, cosmetic_cat, pricelist, uph, purchase_price, as_inv, parts_label, live_label

@st.cache_data(show_spinner=False)
def load_bb360(uploaded_file):
    fn = uploaded_file.name.lower()
    if fn.endswith('.csv'):   return pd.read_csv(uploaded_file)
    if fn.endswith(('.xlsx', '.xls')): return pd.read_excel(uploaded_file)
    if fn.endswith('.json'):
        try: return pd.read_json(uploaded_file, lines=True)
        except Exception:
            uploaded_file.seek(0); raw = json.load(uploaded_file)
            return pd.DataFrame(raw if isinstance(raw, list) else [raw])
    try: return pd.read_csv(uploaded_file)
    except Exception:
        uploaded_file.seek(0); return pd.read_excel(uploaded_file)

if not uploaded:
    st.info('Upload the BB360 export to continue.'); st.stop()

df_raw = load_bb360(uploaded)
f2p_parts, cosmetic_cat, pricelist, uph, purchase_price, as_inv, PARTS_SOURCE, LIVE_SOURCE = autoload_parts_and_live(live_upload)

# Debug: inspect parts sheets on Cloud
if st.sidebar.checkbox("Show parts workbook sheets"):
    try:
        xls_dbg = pd.ExcelFile(PARTS_SOURCE)
        st.sidebar.write("Parts workbook:", PARTS_SOURCE)
        st.sidebar.write("Sheets:", xls_dbg.sheet_names)
    except Exception as e:
        st.sidebar.write("Failed to open parts workbook:", e)

st.caption(f"Parts source: {PARTS_SOURCE}")
st.caption(f"Live source:  {LIVE_SOURCE}")

if (as_inv is None or as_inv.empty) and CFG.get("live", {}).get("require", False):
    st.error("Live file is required but none was found. Set LIVE_PUBLIC_URL or upload one."); st.stop()

# ───────────────────────────────────────────────────────────────────────────────
# Normalization & filters
# ───────────────────────────────────────────────────────────────────────────────
ALLOWED_PROFILES = {"ECOTEC GRADING TEST 1", "ECOTEC GRADING TEST 2"}
COLUMN_SYNONYMS = {
    'imei': ['imei', 'imei/meid', 'serial', 'a number', 'sn'],
    'model': ['model', 'device model'],
    'defects': ['grading summary 1','grading summary 2','failed test summary','defects','issues'],
    'battery_cycle': ['battery cycle count', 'cycle count'],
    'battery_health': ['battery health', 'battery'],
    'profile_name': ['profile name'],
    'analyst_result': ['analyst result', 'result', 'grading result']
}
FRAME_BACKGLASS_MODELS = {"IPHONE 14","IPHONE 14 PLUS","IPHONE 15","IPHONE 15 PLUS","IPHONE 15 PRO","IPHONE 15 PRO MAX"}
ANODIZING_ELIGIBLE_MODELS = {"IPHONE SE","IPHONE SE 2020","IPHONE SE 2022","IPHONE 11","IPHONE XR","IPHONE 12","IPHONE 12 MINI","IPHONE 13","IPHONE 13 MINI","IPHONE 14","IPHONE 14 PLUS"}
IGNORE_B_FOR_DECISION = {'C0', 'C1', 'C3', 'C3-BG', 'C2-BG'}
LCD_CATEGORY_FALLBACK_MIN = 45.0
_re_lcd = re.compile(r'(pixel\s*test|screen\s*burn|screen\s*test)', re.I)

def find_column(df: pd.DataFrame, candidates: list):
    lower_map = {str(c).lower().strip(): c for c in df.columns}
    for cand in candidates:
        if cand.lower().strip() in lower_map: return lower_map[cand.lower().strip()]
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
                normalized[key] = merged.mask(merged.eq(''))
            else:
                normalized[key] = pd.NA
        else:
            col = find_column(df, candidates)
            normalized[key] = df[col] if col is not None else pd.NA
    return pd.concat([df, normalized.add_prefix('_norm_')], axis=1)

# Filter
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

norm = normalize_input_df(df_raw)

# Live merge (optional)
if as_inv is None or as_inv.empty:
    merged = norm.copy()
    merged['Category'] = merged.get('Category', pd.Series(['No Category'] * len(merged)))
    merged['SKU'] = merged.get('SKU', pd.Series([''] * len(merged)))
    merged['SKU_KEY'] = merged['_norm_model'].astype(str).str.upper().str.strip()
else:
    as_inv = as_inv.copy()
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

    merged['SKU'] = ''
    if sku_col: merged['SKU'] = merged[sku_col].astype(str).str.strip()
    fallback_model_series = merged[live_model_col] if (live_model_col and live_model_col in merged.columns) else merged['_norm_model']
    cap_series = merged[capacity_col].astype(str) if (capacity_col and capacity_col in merged.columns) else ''
    col_series = merged[color_col].astype(str) if (color_col and color_col in merged.columns) else ''
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
    built = (fallback_model_series.astype(str).str.upper().str.strip()
             + (' ' + pd.Series(cap_series).apply(_cap_norm)).replace(' ','',regex=False)
             + (' ' + pd.Series(col_series).apply(_capwords)).replace(' ','',regex=False)).str.strip()
    merged['SKU'] = merged['SKU'].mask(merged['SKU'].eq('') | merged['SKU'].str.lower().isin(['nan','none']), built)
    merged['SKU'] = merged['SKU'].mask(merged['SKU'].eq('') | merged['SKU'].str.lower().isin(['nan','none']),
                                       merged['_norm_model'].astype(str).str.upper().str.strip())
    model_key_norm = (fallback_model_series.apply(normalize_model).astype(str).str.upper().str.strip())
    merged['SKU_KEY'] = (model_key_norm + (' ' + pd.Series(cap_series).apply(_cap_norm)).replace(' ','',regex=False)).str.strip()

# ───────────────────────────────────────────────────────────────────────────────
# Pricing normalization
# ───────────────────────────────────────────────────────────────────────────────
cosmetic_cat = cosmetic_cat.copy()
cosmetic_cat.columns = [str(c).strip() for c in cosmetic_cat.columns]
CAT_TO_DESC = dict(zip(cosmetic_cat['Legacy Category'], cosmetic_cat['Description']))

f2p_parts = f2p_parts.copy()
f2p_parts['Model_norm']  = f2p_parts['iPhone Model'].apply(normalize_model)
f2p_parts['Faults_norm'] = f2p_parts['Faults'].astype(str).str.lower().str.strip()
f2p_parts['Part_norm']   = f2p_parts['Part'].astype(str).str.upper().str.strip().str.replace(r'\s+',' ',regex=True)

pricelist = pricelist.copy()
pricelist['Model_norm'] = pricelist['iPhone Model'].apply(normalize_model)
pricelist['Part_norm']  = pricelist['Part'].astype(str).str.upper().str.strip().str.replace(r'\s+',' ',regex=True)
pricelist['PRICE']      = pd.to_numeric(pricelist['PRICE'].astype(str).str.replace(r'[^0-9\.]','',regex=True).replace('','0'),
                                        errors='coerce').fillna(0.0)
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

def adhesive_list_for(model, cat_val):
    parts = []
    for ak in CATEGORY_ADHESIVES_MAPPING.get(str(cat_val).upper(), []):
        p = ADHESIVE_INDEX.get(model, {}).get(ak)
        if p and (model, p) in PRICE_INDEX:
            parts.append(p)
    seen, out = set(), []
    for p in parts:
        if p not in seen:
            out.append(p); seen.add(p)
    return out

def cosmetic_keywords_for(legacy_cat, model):
    legacy_cat = str(legacy_cat).strip().upper()
    if legacy_cat in {"C0","C1","C3"}:
        return ["BACKGLASS","HOUSING FRAME"] if model in FRAME_BACKGLASS_MODELS else ["BACKGLASS","BACK COVER"]
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

def select_cosmetic_breakdown_grade_A(model, cat_val):
    total = 0.0
    picked = []
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
            else:
                cond = (model_slice['Part_norm'].str.contains("BACK COVER", na=False) &
                        ~model_slice['Part_norm'].str.contains("ADHESIVE", na=False))
            cand = model_slice[cond]
            pref = cand[cand['Type_norm'].str.upper() == 'D-COVER']
            chosen = pref if not pref.empty else cand
            if not chosen.empty:
                total += float(chosen['PRICE'].iat[0])
                picked.append(str(chosen['Part_norm'].iat[0]))
    adhs = adhesive_list_for(model, cat_val)
    for a in adhs:
        total += float(PRICE_INDEX[(model, a)])
    seen, dedup = set(), []
    for p in picked:
        if p not in seen:
            dedup.append(p); seen.add(p)
    return total, dedup, adhs

# ───────────────────────────────────────────────────────────────────────────────
# Purchase Price table (soft-fail friendly)
# ───────────────────────────────────────────────────────────────────────────────
purchase_price = purchase_price.copy()
purchase_price.columns = [str(c).strip() for c in purchase_price.columns]
pp_cols = purchase_price.columns
col_map = {
    "SKU": next((c for c in pp_cols if c.strip().lower() == "sku"), pp_cols[0] if len(pp_cols) else "SKU"),
    "Acq": next((c for c in pp_cols if "acquisition" in c.strip().lower()), "Acquisition price"),
    "A":   next((c for c in pp_cols if c.strip().lower() in ("grade a","grade a price","a")),  "Grade A"),
    "B":   next((c for c in pp_cols if c.strip().lower() in ("grade b","grade b price","b")),  "Grade B"),
    "C":   next((c for c in pp_cols if c.strip().lower() in ("grade c","grade c price","c")),  "Grade C"),
}
purchase_price = purchase_price.rename(columns={
    col_map["SKU"]: "PP_SKU",
    col_map["Acq"]: "PP_Acquisition",
    col_map["A"]:   "PP_GradeA",
    col_map["B"]:   "PP_GradeB",
    col_map["C"]:   "PP_GradeC",
})
for c in ["PP_Acquisition","PP_GradeA","PP_GradeB","PP_GradeC"]:
    if c not in purchase_price.columns:
        purchase_price[c] = 0.0
    purchase_price[c] = pd.to_numeric(
        purchase_price[c].astype(str).str.replace(r'[^0-9\.]','',regex=True).replace('','0'),
        errors='coerce'
    ).fillna(0.0)

PURCHASE_INDEX = {
    str(row.get("PP_SKU","")).strip().upper(): {
        "acq": float(row.get("PP_Acquisition", 0.0)),
        "A": float(row.get("PP_GradeA", 0.0)),
        "B": float(row.get("PP_GradeB", 0.0)),
        "C": float(row.get("PP_GradeC", 0.0)),
    } for _, row in purchase_price.iterrows()
}

# ───────────────────────────────────────────────────────────────────────────────
# User selectors
# ───────────────────────────────────────────────────────────────────────────────
battery_type = st.selectbox("Battery Type", ["BATTERY CELL","BATTERY OEM","BATTERY OEM PULLED"], index=0).upper()
lcd_type = st.selectbox("LCD Type", ["LCM GENERIC (HARD OLED)","LCM GENERIC (TFT)","LCM -OEM REFURBISHED (GLASS CHANGED -GENERIC)","LCM GENERIC (SOFT OLED)"], index=0).upper()
labor_rate = st.slider("Labor Rate ($/hour)", 1, 35, 5)

# UPH index
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

# ───────────────────────────────────────────────────────────────────────────────
# Core compute
# ───────────────────────────────────────────────────────────────────────────────
def compute_row(row, labor_rate):
    analyst_result = str(row.get('_norm_analyst_result', row.get('Analyst Result',''))).strip().lower()
    if analyst_result == "not completed":
        return None

    failures = parse_failures(row.get('_norm_defects',''))
    batt_status, _, _ = battery_status(row.get('_norm_battery_cycle'), row.get('_norm_battery_health'))
    model_raw = normalize_model(row.get('_norm_model'))
    device_model = model_raw
    cat_val = str(row.get('Category') or '').strip().upper()
    sku_key = str(row.get('SKU_KEY') or '').strip().upper()

    lcd_failure_present = any(_re_lcd.search(str(f)) for f in failures)
    lcd_needed = lcd_failure_present or cat_val in {"C0", "C2-C"}

    # Functional parts gather
    func_parts, seen_fp = [], set()
    def _add_fp(p):
        p = str(p).upper().strip()
        if p and p not in seen_fp and (device_model, p) in PRICE_INDEX:
            seen_fp.add(p); func_parts.append(p)

    for f in failures:
        for part in F2P_INDEX.get((device_model, str(f).lower().strip()), []):
            _add_fp(part)
    if batt_status == "Battery Service":
        _add_fp(battery_type)
        if battery_type == "BATTERY CELL":
            model_slice = PL_BY_MODEL.get(device_model)
            if model_slice is not None:
                fx = model_slice[model_slice['Part_norm'].str.contains("BATTERY FLEX", na=False)]
                if not fx.empty: _add_fp(str(fx['Part_norm'].iat[0]).upper().strip())
    if lcd_needed:
        _add_fp(lcd_type)

    func_total = sum(float(PRICE_INDEX.get((device_model, k), 0.0)) for k in func_parts)

    # Tech labor minutes
    tech_minutes = sum(float(UPH_INDEX.get(uph_key(tok), 0.0)) for tok in failures)
    if failures: tech_minutes += float(CFG['labor'].get('ceq_minutes', 2))
    if cat_val in {"C0", "C2-C"} and not lcd_failure_present:
        tech_minutes += LCD_CATEGORY_FALLBACK_MIN
    tech_labor_cost = (tech_minutes / 60.0) * float(labor_rate)

    # Refurb labor
    refcat_set = {'C0','C1','C3-BG','C3-HF','C3'}
    refurb_minutes = float(UPH_INDEX.get(uph_key(cat_val), 0.0)) if cat_val in refcat_set else 0.0
    refurb_labor_cost = (refurb_minutes / 60.0) * float(labor_rate)

    # Re-glass
    RE_ELIGIBLE_CATS = {'C1','C2','C2-BG'}
    re_applicable = cat_val in RE_ELIGIBLE_CATS
    has_mts = any("multitouchscreen" in str(f).lower().replace(" ", "").replace("-", "") for f in failures) if re_applicable else False
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

    # QC / BNP / Anodizing
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

    # Cosmetic & adhesives
    cos_A, cosA_parts_list, cosA_adh_list = select_cosmetic_breakdown_grade_A(device_model, cat_val)
    adh_list_only = adhesive_list_for(device_model, cat_val)
    cos_B = sum(float(PRICE_INDEX.get((device_model, a), 0.0)) for a in adh_list_only)

    # Totals
    refurb_A = func_total + cos_A + reglass_price + reglass_labor_cost + refurb_labor_cost + anod_cost + bnp_cost + tech_labor_cost + qc_cost
    refurb_B = func_total + cos_B + reglass_price + reglass_labor_cost + refurb_labor_cost + anod_cost + bnp_cost + tech_labor_cost + qc_cost
    refurb_C = func_total + tech_labor_cost + qc_cost

    # Prices & margins
    pp = PURCHASE_INDEX.get(sku_key, {"acq":0.0,"A":0.0,"B":0.0,"C":0.0})
    acq = float(pp["acq"]); price_A = float(pp["A"]); price_B = float(pp["B"]); price_C = float(pp["C"])
    margin_A = price_A - (acq + refurb_A)
    margin_B = price_B - (acq + refurb_B)
    margin_C = price_C - (acq + refurb_C)

    # Decision (ignore B for specific cats)
    b_ineligible = cat_val in IGNORE_B_FOR_DECISION
    candidates = [("A", price_A, margin_A, refurb_A), ("C", price_C, margin_C, refurb_C)]
    if not b_ineligible:
        candidates.append(("B", price_B, margin_B, refurb_B))
    best_grade, best_price, best_margin, best_refurb = max(candidates, key=lambda t: t[2])

    # Final parts used
    final_parts = []
    def _extend_unique(seq):
        seen = set(final_parts)
        for x in seq:
            if x and x not in seen:
                final_parts.append(x); seen.add(x)
    _extend_unique(func_parts)
    re_part_used = preferred_key if (reglass_price > 0) else ""
    if best_grade == "A":
        _extend_unique(cosA_parts_list); _extend_unique(cosA_adh_list)
        if re_part_used: _extend_unique([re_part_used])
    elif best_grade == "B":
        _extend_unique(adh_list_only)
        if re_part_used: _extend_unique([re_part_used])

    out = {
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
        'Grade C Selling Price': price_C, 'Grade C Margin': margin_C,
        'Final Selling Price': best_price, 'Final Margin': best_margin, 'Final Grade': best_grade,
    }
    if b_ineligible:
        out['Grade B Selling Price'] = 'N/A'
        out['Grade B Margin'] = 'N/A'
    else:
        out['Grade B Selling Price'] = price_B
        out['Grade B Margin'] = margin_B

    # CSV parts breakdown
    out['Functional Parts (List)']       = "|".join(func_parts)
    out['Cosmetic Parts Grade A (List)'] = "|".join(cosA_parts_list)
    out['Adhesives (By Category)']       = "|".join(cosA_adh_list)
    out['Reglass Type Used']             = re_part_used
    out['Final Parts Used (List)']       = "|".join(final_parts)

    return out

# ───────────────────────────────────────────────────────────────────────────────
# Run
# ───────────────────────────────────────────────────────────────────────────────
rows = []
for _, r in merged.iterrows():
    out = compute_row(r, labor_rate)
    if out is None: continue
    rows.append(out)
res_df = pd.DataFrame(rows)

# ───────────────────────────────────────────────────────────────────────────────
# Display
# ───────────────────────────────────────────────────────────────────────────────
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

    money_cols = [c for c in present_cols if c not in ['IMEI','SKU','Final Grade']]
    def _fmt_money_or_na(v):
        s = str(v).strip().upper()
        if s in {"N/A","NA","NONE",""}: return "N/A"
        try: return f"${float(v):,.2f}"
        except (ValueError, TypeError): return str(v)

    row_html = []
    for _, row in disp.iterrows():
        tds = []
        for c in present_cols:
            v = row[c]
            if c in money_cols:
                if c == 'Final Margin':
                    try:
                        val = float(v); color = '#d6f5d6' if val >= 0 else '#ffd6e7'
                        tds.append(f"<td style='white-space:nowrap;text-align:right;background:{color}'>{_fmt_money_or_na(v)}</td>")
                    except (ValueError, TypeError):
                        tds.append(f"<td style='white-space:nowrap;text-align:right'>{_fmt_money_or_na(v)}</td>")
                else:
                    tds.append(f"<td style='white-space:nowrap;text-align:right'>{_fmt_money_or_na(v)}</td>")
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
    st.subheader("Clean Business View — (Live merge applied if available)")
    st.markdown(table_html, unsafe_allow_html=True)

    csv_bytes = res_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Full CSV', data=csv_bytes, file_name='bb360_business_view_full.csv', mime='text/csv')
else:
    st.info("No qualifying rows to display.")
