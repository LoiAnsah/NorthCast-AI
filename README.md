# NorthCast — Northern Ontario Winter Road Risk Dashboard

Streamlit app that maps predicted ice conditions for **38 communities and lakes** in Northern Ontario, with vehicle load assessment for winter road planning.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## What it does

- **Map** — Satellite view with colour-coded markers (load status: Safe / Borderline / Unsafe)
- **Forecast date** — One date at a time (2025–2035 in scenario mode, or today + 7 days with live weather)
- **Climate scenarios** — Optimistic, Baseline, High Emissions (SSP)
- **Vehicle / load type** — Select one weight class; ice thickness is compared to minimum required ice for that vehicle
- **Ice Condition Predictor** — Sidebar MLP (`final_mlp5_model.keras`) for custom inputs (temp, wind, snow, FDD)
- **Community search** — Find lakes/communities and view detail panel + summary table

## Data sources

| Mode | Source |
|------|--------|
| Real-time (toggle on, date within 7 days) | [Open-Meteo](https://open-meteo.com/) — temp, wind, snow, precipitation, cumulative FDD |
| Scenario / outside live window | Deterministic placeholder model (`fake_predict`) until full feature table is connected |

## Models

| Model | Use |
|-------|-----|
| `final_mlp5_model.keras` + `feature_scaler_05.pkl` | Sidebar user-input predictor (NumPy inference, no TensorFlow) |
| `final_mlp_model.pkl` | Full map model — not wired yet (needs per-lake feature CSV) |

## Technical

**Stack:** Python · Streamlit · Folium (Esri satellite tiles) · Pandas · scikit-learn · NumPy/h5py

**Sidebar MLP** (`final_mlp5_model.keras`)
- Architecture: Dense 32 → ReLU → Dropout → Dense 16 → ReLU → Dense 8 → ReLU → Dense 1 (linear)
- Inputs (5, `StandardScaler`): `sd`, `d2m`, `sqrt_cum_fdd`, `t2m_c`, `wind_speed`
- Output: ice depth in metres → converted to cm; missing inputs use training-set means
- Inference via `mlp_inference.py` (weights read from `.keras` archive, no TensorFlow runtime)

**Full map MLP** (`final_mlp_model.pkl`) — planned
- 8 features: `t2m_c`, `d2m`, `wind_speed`, `msl`, `sd`, `tcc`, `cum_fdd`, `sqrt_cum_fdd`
- Requires a per-lake, per-date feature table before replacing `fake_predict()`

**Map prediction (current)**
- **Scenario mode:** MD5-seeded `fake_predict()` — monthly state weights + SSP scenario shift + year drift
- **Real-time mode:** Open-Meteo fetch (30 parallel requests, 1 h cache, 92 days history + 7-day forecast)
  - Cumulative FDD from Oct 1: `FDD += max(0, −T)` for days where `T < 0°C`
  - Ice thickness (Stefan): `h_cm ≈ 3.4 × √FDD`
  - State from rule-based `whatif_predict()` (FDD, temp, snow, wind, precip → score 0–100)

**Derived labels**
| Step | Rule |
|------|------|
| Ice state | ≥ 30 cm → Frozen · ≥ 15 cm → Unstable · else Open |
| Risk score | `100 − min(100, ice_cm / 60 × 100)` |
| Load status | ice ≥ threshold → Safe · ≥ 85% threshold → Borderline · else Unsafe |

**Vehicle thresholds (cm):** light truck 20.3 · medium 25.4 · heavy 30.5 · 10 t 38.1 · 25 t 50.8 · 70 t 76.2 · 110 t 91.4

## Files

- `app.py` — Main dashboard
- `mlp_inference.py` — NumPy MLP loader
- `*.pkl`, `*.keras` — Trained models and scalers

## Notes

- Load thresholds are **planning indicators** for one vehicle type at a time — not convoy count, spacing, or local authority rules.
- Beyond 2030, treat outputs as climate-risk scenarios, not exact forecasts.

## License

TBD
