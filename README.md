# Satellite Tank Monitor

Estimate oil storage tank fill levels from satellite imagery using computer vision.

Two complementary methods:
1. **Ratio method** (single image): `fill = 1 - L_inner/L_outer` — inner wall shadow vs outer ground shadow. Sun angle cancels out.
2. **Time-series differential** (multiple dates): track interior brightness changes over time via Esri Wayback Machine archive.

## Quick Start

```bash
pip install -r requirements.txt
python examples/cilacap_timeseries.py
```

## Methods

See [docs/METHOD.md](docs/METHOD.md) for the geometric derivation and [docs/LESSONS.md](docs/LESSONS.md) for what failed and why.

## Case Studies

- **Cilacap, Indonesia** (Pertamina RU IV): ratio method + 10-year time-series. Found one tank emptied 2019-2022.
- **Tuban, Indonesia** (TPPI): found 3 new tanks built 2022-2024.
- **Tanjung Perak, Surabaya**: found one tank turned rust (emptied) 2022-2024.
- **Semarang** (TBBM Pengapon): one persistently dark tank.
- **Cushing, Oklahoma, USA**: massive tank farm, very stable.

## Data Sources

- Esri World Imagery (free, 0.6m)
- Esri Wayback Machine (free archive, 2014-2026)
- Google Earth tiles (free, 0.3m via tile API)

## Disclaimer

Fill estimates from free satellite imagery are approximate. The time-series method detects *changes* reliably; absolute percentages need <0.5m imagery and careful geometry.
