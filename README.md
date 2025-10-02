# BB360: Mobile Failure Quantification (Deployable)

This repository contains a deployable Streamlit app that separates **Functional** vs **Refurb** parts, shows collapsible cells inside the main table, and splits labor (QC / Anodizing / BNP).

## Quickstart (local)

```bash
pip install -r requirements.txt
streamlit run BB360_app.py
```

Upload:
- BB360 export (CSV/XLSX)
- AS Inventory (sheet `Inventory`)
- Pricing workbook with sheets: `F2P`, `Cosmetic Category`, `Pricelist`, `UPH`

## Config

Edit `config.yml` to toggle behaviors without touching code.

```yaml
ui:
  hide_ipads_on_screen: true
pricing:
  battery_default: "BATTERY CELL"
  lcd_default: "LCM GENERIC (HARD OLED)"
labor:
  ceq_minutes: 2
  use_qc_labor: true
```

## Deploy

### Streamlit Cloud
- Push to GitHub, point the app to `BB360_app.py`.
- Use branches/tags for safe rollbacks.

### Docker
```
docker build -t bb360:latest .
docker run -p 8501:8501 bb360:latest
```

## Rollback
- Keep stable tags (e.g., `v2025-10-02-stable`).
- Or use `releases/BB360_v2025-10-02.py` as a frozen fallback.
