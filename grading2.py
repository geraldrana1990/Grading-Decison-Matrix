
# === BB360 STABLE BASELINE — LIGHTWEIGHT (Functional vs Refurb, Collapsible Cells) ===
# Version: 2025-10-02
# Changes:
#  - Restores functional part mapping (F2P + Battery + LCD) with strict per-model checks
#  - Splits parts into Functional vs Refurb groups with separate collapsible cells
#  - Columns: Functional Parts Cost, Refurb Parts Cost, Total Parts Cost, QC, Anodizing, BNP, Labor, Total
#  - No per-row expanders; iPads hidden on-screen only
import streamlit as st
import pandas as pd
import re
from collections import defaultdict
import html as _html

# -------------------- CONFIGURATION --------------------
COLUMN_SYNONYMS = {
    'imei': ['imei', 'imei/meid', 'serial', 'a number', 'sn'],
    'model': ['model', 'device model'],
    'defects': ['failed test summary', 'defects', 'issues'],
    'battery_cycle': ['battery cycle count', 'cycle count'],
    'battery_health': ['battery health', 'battery'],
    'profile_name': ['profile name'],
    'analyst_result': ['analyst result', 'result', 'grading result']
}

LCD_FAILURES = ["screen test", "screen burn", "pixel test"]

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
        # fuzzy contains
    for col_low, col_orig in lower_map.items():
        for cand in candidates:
            if cand.lower().strip() in col_low:
                return col_orig
    return None

def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() if isinstance(c, str) else c for c in df.columns]
    normalized = pd.DataFrame(index=df.index)
    for key, candidates in COLUMN_SYNONYMS.items():
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

def remove_redundant_parts(parts_list):
    if not parts_list: return []
    return list(dict.fromkeys(parts_list))

_re_lcd = re.compile(r'(screen test|screen burn|pixel test)', re.I)
def is_lcd_failure(failure: str) -> bool:
    return bool(_re_lcd.search(str(failure)))

# -------------------- STREAMLIT APP --------------------
st.set_page_config(page_title='BB360: Mobile Failure Quantification', layout='wide')
st.title('BB360: Mobile Failure Quantification — Functional vs Refurb')

# -------------------- LOAD FILES (CACHED) --------------------
@st.cache_data(show_spinner=False)
def load_all(uploaded, as_file, parts_file):
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

df_raw, as_inv, f2p_parts, cosmetic_cat, pricelist, uph = load_all(uploaded, as_file, parts_file)

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

def SAFE_ADD(part_name, device_model, bucket, source_map, source_tag):
    if not part_name: return
    key = str(part_name).upper().strip()
    if (device_model, key) in PRICE_INDEX and key not in bucket:
        bucket.append(key)
        source_map[key] = source_tag

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
            part_keywords.append("BACK COVER")
        if "BACKGLASS" in description or "BACK GLASS" in description:
            part_keywords.append("BACKGLASS")
        if "HOUSING FRAME" in description:
            part_keywords.append("HOUSING FRAME")
        if "BUFFING" in description or "POLISH" in description:
            part_keywords.append("POLISH")

    model_slice = PL_BY_MODEL.get(model)
    if model_slice is not None:
        for k in part_keywords:
            kk = k.upper()
            if kk == "HOUSING FRAME":
                cond = model_slice['Part_norm'].str.contains("HOUSING", na=False) & model_slice['Part_norm'].str.contains("FRAME", na=False) & ~model_slice['Part_norm'].str.contains("ADHESIVE", na=False)
                matches = model_slice[cond]
            elif kk == "BACKGLASS":
                cond = ((model_slice['Part_norm'].str.contains("BACKGLASS", na=False)) | (model_slice['Part_norm'].str.contains("BACK", na=False) & model_slice['Part_norm'].str.contains("GLASS", na=False))) & ~model_slice['Part_norm'].str.contains("ADHESIVE", na=False)
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
    analyst_result = str(row.get('Analyst Result', '')).strip().lower()
    if analyst_result == "not completed":
        return None

    failures = parse_failures(row.get('_norm_defects',''))
    batt_status, cycle_num, health_num = battery_status(row.get('_norm_battery_cycle'), row.get('_norm_battery_health'))
    device_model = normalize_model(row.get('_norm_model'))
    device_model = MODEL_ALIASES.get(device_model, device_model)
    raw_cat = row.get('Category')
    cat_val = str(raw_cat).strip().upper() if pd.notna(raw_cat) else ""

    # Buckets
    func_parts, refurb_parts = [], []
    src = {}   # part -> source

    # ---- Functional mapping: F2P parts from failures ----
    for f in [str(f).lower().strip() for f in failures]:
        for part in F2P_INDEX.get((device_model, f), []):
            key = str(part).upper().strip()
            if (device_model, key) in PRICE_INDEX and key not in func_parts:
                func_parts.append(key); src[key] = 'f2p'

    # ---- Functional mapping: Battery (strict per-model) ----
    if batt_status == "Battery Service":
        failures.append("Battery Service")
        key = battery_type  # already upper
        if (device_model, key) in PRICE_INDEX and key not in func_parts:
            func_parts.append(key); src[key] = 'battery'
        if key == "BATTERY CELL":
            # optional flex if exists
            model_slice = PL_BY_MODEL.get(device_model)
            if model_slice is not None:
                fx = model_slice[model_slice['Part_norm'].str.contains("BATTERY FLEX", na=False)]
                if not fx.empty:
                    fk = fx['Part_norm'].iat[0]
                    if (device_model, fk) in PRICE_INDEX and fk not in func_parts:
                        func_parts.append(fk); src[fk] = 'battery'

    # ---- Functional mapping: LCD (strict per-model) ----
    if any(is_lcd_failure(f) for f in failures) or cat_val in {"C0","C2-C"}:
        if (device_model, lcd_type) in PRICE_INDEX and lcd_type not in func_parts:
            func_parts.append(lcd_type); src[lcd_type] = 'lcd'

    # ---- Refurb mapping: Category + adhesives ----
    for p in map_category_to_parts_fast(cat_val, device_model):
        k = str(p).upper().strip()
        if (device_model, k) in PRICE_INDEX and k not in refurb_parts and k not in func_parts:
            refurb_parts.append(k); src[k] = 'refurb'

    # ---- Refurb mapping: RE-Glass ----
    if cat_val in {"C1", "C2", "C2-BG"}:
        has_mts = any("multitouchscreen" in str(f).lower().replace(" ", "").replace("-", "") for f in failures)
        target = "REGLASS+DIGITIZER" if has_mts else "REGLASS"
        pl_m = PL_BY_MODEL.get(device_model)
        if pl_m is not None and 'Type' in pl_m.columns:
            hit = pl_m[(pl_m['Part_norm'] == target) & (pl_m['Type'].astype(str).str.upper().str.strip() == "RE")]
            if not hit.empty:
                tk = hit['Part_norm'].iat[0]
                if (device_model, tk) in PRICE_INDEX and tk not in refurb_parts and tk not in func_parts:
                    refurb_parts.append(tk); src[tk] = 'refurb'

    # ---- Build rows & totals ----
    func_rows, refurb_rows = [], []
    func_total = 0.0; refurb_total = 0.0
    for k in func_parts:
        price = float(PRICE_INDEX.get((device_model, k), 0.0))
        func_rows.append({'Part': k, 'Source': src.get(k,'?'), 'Price': price})
        func_total += price
    for k in refurb_parts:
        price = float(PRICE_INDEX.get((device_model, k), 0.0))
        refurb_rows.append({'Part': k, 'Source': src.get(k,'?'), 'Price': price})
        refurb_total += price
    total_parts = func_total + refurb_total

    # ---- Labor: base + QC + anodizing + buffing ----
    qc_min = (uph_minutes('qc process') or uph_minutes('qcinspection') or uph_minutes('qc'))
    qc_cost = (qc_min / 60.0) * float(labor_rate) if analyst_result != 'not completed' else 0.0

    anod_min = (uph_minutes('anodizing') or uph_minutes('anodize'))
    anod_applicable = cat_val in {'C2','C2-C','C2-BG','C3-BG','C4'}
    anod_cost = (anod_min / 60.0) * float(labor_rate) if anod_applicable and anod_min>0 else 0.0

    fb_min = (uph_minutes('front buffing') or uph_minutes('frontbuff'))
    bb_min = (uph_minutes('back buffing') or uph_minutes('backbuff'))
    buff_min = 0.0
    if cat_val in {'C4','C3-HF'}: buff_min = fb_min + bb_min
    elif cat_val in {'C3','C3-BG'}: buff_min = fb_min
    elif cat_val in {'C2','C2-C'}: buff_min = bb_min
    bnp_cost = (buff_min / 60.0) * float(labor_rate) if buff_min>0 else 0.0

    # Base process time (CEQ +2 if there are faults)
    base_time = 0.0
    for tok in failures:
        base_time += uph_minutes(tok)
    base_time += uph_minutes(cat_val)
    if len(failures) > 0:
        base_time += 2  # CEQ
    base_labor_cost = (base_time / 60.0) * float(labor_rate)

    total_labor_cost = base_labor_cost + qc_cost + anod_cost + bnp_cost
    total_cost = total_parts + total_labor_cost

    # ---- Collapsible cells (markerless) ----
    def _mk_cell(title_text: str, rows: list, total: float) -> str:
        if not rows:
            return f"<span style='opacity:.7'>No { _html.escape(title_text.lower()) }</span>"
        items = "".join(
            f"<li>{_html.escape(r['Part'])} — ${r['Price']:,.2f} "
            f"<small style='opacity:.7'>({_html.escape(str(r['Source']))})</small></li>"
            for r in rows
        )
        return (
            "<details class='bb360-cell'>"
            f"<summary>{_html.escape(title_text)} — ${total:,.2f}</summary>"
            f"<ul>{items}</ul>"
            "</details>"
        )

    functional_cell = _mk_cell("Functional", func_rows, func_total)
    refurb_cell     = _mk_cell("Refurb", refurb_rows, refurb_total)

    return {
        'imei': row.get('_norm_imei'),
        'model': row.get('_norm_model'),
        # hidden fields for CSV completeness (not shown on screen)
        'legacy_category': cat_val,
        'category_desc': row.get('Category_Desc'),
        'failures': "|".join(failures),
        # visible columns:
        'Functional (collapse)': functional_cell,
        'Refurb (collapse)': refurb_cell,
        'Functional Parts Cost': func_total,
        'Refurb Parts Cost': refurb_total,
        'Total Parts Cost': total_parts,
        'QC Labor': qc_cost,
        'Anodizing Labor': anod_cost,
        'BNP Labor': bnp_cost,
        'Labor Cost': total_labor_cost,
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

# -------------------- DISPLAY (collapsible cells inside table) --------------------
if not res_df.empty:
    # Hide iPads on the on-screen table only; CSV remains complete
    disp = res_df[~res_df['model'].astype(str).str.upper().str.contains('IPAD', na=False)].copy()

    show_cols = [
        'imei','model',
        'Functional (collapse)','Refurb (collapse)',
        'Functional Parts Cost','Refurb Parts Cost','Total Parts Cost',
        'QC Labor','Anodizing Labor','BNP Labor',
        'Labor Cost','Total Cost'
    ]

    money_cols = ['Functional Parts Cost','Refurb Parts Cost','Total Parts Cost','QC Labor','Anodizing Labor','BNP Labor','Labor Cost','Total Cost']
    for col in money_cols:
        disp[col] = disp[col].astype(float)

    def _fmt_money(x: float) -> str: return f"${x:,.2f}"

    # Build HTML table
    row_html = []
    for _, row in disp.iterrows():
        tds = []
        for c in show_cols:
            v = row[c]
            if c in money_cols:
                tds.append(f"<td style='white-space:nowrap;text-align:right'>{_fmt_money(v)}</td>")
            elif c in ['Functional (collapse)','Refurb (collapse)']:
                tds.append(f"<td>{v}</td>")
            else:
                tds.append(f"<td>{_html.escape(str(v))}</td>")
        row_html.append("<tr>" + "".join(tds) + "</tr>")

    header_html = "".join(f"<th style='text-align:left'>{_html.escape(c)}</th>" for c in show_cols)
    table_html = f"""
    <style>
      .bb360-table {{ border-collapse: collapse; width: 100%; }}
      .bb360-table th, .bb360-table td {{ padding: 6px 10px; border-bottom: 1px solid #eee; vertical-align: top; }}
      .bb360-cell summary {{ cursor: pointer; font-weight: 600; display: inline; }}
      .bb360-cell summary::-webkit-details-marker {{ display: none; }}
      .bb360-cell summary::marker {{ content: ''; }}
      .bb360-cell summary:hover {{ text-decoration: underline; }}
      .bb360-cell ul {{ margin: 8px 0 0 18px; }}
    </style>
    <table class="bb360-table">
      <thead><tr>{header_html}</tr></thead>
      <tbody>{''.join(row_html)}</tbody>
    </table>
    """

    st.subheader("Final Results (Functional vs Refurb)")
    st.markdown(table_html, unsafe_allow_html=True)

    # CSV export: full dataset (including hidden fields), without HTML columns
    export_cols = [c for c in res_df.columns if c not in ['Functional (collapse)','Refurb (collapse)']]
    export_df = res_df[export_cols].copy()
    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
    st.download_button('Download Final CSV', data=csv_bytes, file_name='bb360_functional_vs_refurb.csv', mime='text/csv')
else:
    st.info("No qualifying rows to display.")
