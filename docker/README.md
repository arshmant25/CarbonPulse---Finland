# CarbonPulse Finland — Docker Deployment Guide

## Project Structure

```
carbonpulse-finland/
│
├── docker-compose.yml          ← starts all 3 containers
├── .env.example                ← copy to .env and fill API key
├── setup.py                    ← copies model files, validates setup
│
├── streamer/
│   ├── Dockerfile
│   └── streamer.py             ← fetches Fingrid data every 15 min
│
├── api/
│   ├── Dockerfile
│   ├── main.py                 ← FastAPI: /health /current /forecast
│   └── requirements.txt
│
├── app/
│   ├── Dockerfile
│   ├── streamlit_app.py        ← Streamlit dashboard (3 pages)
│   └── requirements.txt
│
└── data/                       ← shared volume (created by setup.py)
    ├── project_data.db         ← SQLite (written by streamer, read by api+app)
    ├── gru_model.pt            ← trained GRU weights
    ├── scaler_X_15min.pkl      ← feature scaler
    ├── scaler_y_15min.pkl      ← target scaler
    └── column_config.json      ← feature column names from training
```

---

## What Each Container Does

```
┌─────────────────────────────────────────────────────┐
│                   Docker Compose                     │
│                                                      │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────┐ │
│  │   streamer   │   │     api      │   │   app    │ │
│  │              │   │              │   │          │ │
│  │ Fetches all  │   │ FastAPI      │   │Streamlit │ │
│  │ 9 Fingrid    │   │ /forecast    │   │dashboard │ │
│  │ datasets     │   │ /current     │   │3 pages   │ │
│  │ every 15 min │   │ /health      │   │          │ │
│  │              │   │              │   │reads DB  │ │
│  │ writes to    │   │ loads GRU    │   │calls API │ │
│  │ SQLite       │   │ cleans data  │   │for fcst  │ │
│  └──────┬───────┘   └──────┬───────┘   └────┬─────┘ │
│         └──────────────────┴────────────────┘        │
│                    shared /data/ volume               │
│         (project_data.db, gru_model.pt, scalers)     │
└──────────────────────────┬──────────────────────────┘
                           │
              Port 8000 (API) + Port 8501 (Dashboard)
```

---

## Step-by-Step Deployment

### Step 1 — Install Docker Desktop

Download and install Docker Desktop from https://www.docker.com/products/docker-desktop/

Verify installation:
```bash
docker --version
docker compose version
```

### Step 2 — Set Up Your API Key

```bash
# Copy the template
cp .env.example .env

# Edit .env and replace the placeholder with your real key
# The file should contain exactly:
# FINGRID_API_KEY=your_actual_key_from_data.fingrid.fi
```

### Step 3 — Copy Model Files

Your trained model files are in:
```
C:\Users\atariq25\OneDrive - University of Oulu and Oamk (1)\
  Documents\RA FCC UBICOMP\carbonpulse-finland\
  model outputs\outputs_15min\
```

Run the setup script — it finds this path automatically on Windows:
```bash
python setup.py
```

Or specify the path explicitly:
```bash
python setup.py --model-dir "C:\Users\atariq25\OneDrive - University of Oulu and Oamk (1)\Documents\RA FCC UBICOMP\carbonpulse-finland\model outputs\outputs_15min"
```

Also copy your database:
```bash
python setup.py --db-path "data/project_data.db"
```

The setup script copies these files into `data/`:
- `gru_model.pt`
- `scaler_X_15min.pkl`
- `scaler_y_15min.pkl`
- `column_config.json`
- `project_data.db`

### Step 4 — Build and Start

```bash
# Build images and start all services
docker compose up --build

# OR run in background (production mode)
docker compose up --build -d
```

First build takes 3-5 minutes (downloads Python packages). Subsequent builds are fast (cached layers).

### Step 5 — Verify Everything Is Running

```bash
# Check all containers are healthy
docker compose ps

# Expected output:
# NAME                     STATUS
# carbonpulse_streamer     running
# carbonpulse_api          running (healthy)
# carbonpulse_dashboard    running
```

Then open:
- **Dashboard:**  http://localhost:8501
- **API docs:**   http://localhost:8000/docs  (interactive Swagger UI)
- **Live forecast:** http://localhost:8000/forecast?steps=96
- **Current reading:** http://localhost:8000/current
- **Health check:** http://localhost:8000/health

---

## Daily Operations

### View logs
```bash
# Streamer logs (shows each fetch cycle)
docker compose logs -f streamer

# API logs (shows each forecast request)
docker compose logs -f api

# Dashboard logs
docker compose logs -f app
```

### Stop and restart
```bash
# Stop all
docker compose down

# Restart one service only
docker compose restart api

# Rebuild and restart after code changes
docker compose up --build -d
```

### Check the database is growing
```bash
# See how many rows are in each source table
docker compose exec api python -c "
import sqlite3
conn = sqlite3.connect('/data/project_data.db')
tables = ['co2_intensity','wind','nuclear','hydro',
          'chp_district','chp_industrial','total_production',
          'consumption_electricity','consumption_co2_intensity']
for t in tables:
    n = conn.execute(f'SELECT COUNT(*) FROM \"{t}\"').fetchone()[0]
    latest = conn.execute(f'SELECT MAX(start_time) FROM \"{t}\"').fetchone()[0]
    print(f'{t:30s}: {n:>10,} rows  latest: {latest}')
conn.close()
"
```

---

## Running on Puhti (CSC)

Puhti does not support Docker directly. Use the conda environment instead:

```bash
# On Puhti — load modules
module load miniconda

# Create environment (first time only)
conda create -n carbonpulse python=3.11
conda activate carbonpulse
pip install fastapi uvicorn torch numpy pandas scikit-learn joblib \
            streamlit plotly requests pydantic

# Copy your data files to Puhti scratch
scp -r data/ username@puhti.csc.fi:/scratch/project_xxxx/carbonpulse/

# Start the API (on Puhti interactive or batch node)
cd /scratch/project_xxxx/carbonpulse
DB_PATH=/scratch/project_xxxx/carbonpulse/data/project_data.db \
MODEL_PATH=/scratch/project_xxxx/carbonpulse/data/gru_model.pt \
SCALER_X=/scratch/project_xxxx/carbonpulse/data/scaler_X_15min.pkl \
SCALER_Y=/scratch/project_xxxx/carbonpulse/data/scaler_y_15min.pkl \
COL_CFG=/scratch/project_xxxx/carbonpulse/data/column_config.json \
uvicorn api.main:app --host 0.0.0.0 --port 8000 &

# SSH tunnel to access from your laptop
# Run this on your LOCAL machine:
ssh -L 8000:localhost:8000 username@puhti.csc.fi
# Then open http://localhost:8000/docs on your laptop
```

---

## API Endpoints Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Returns `{"status": "ok", "model_loaded": true}` |
| `/current` | GET | Latest co2_intensity reading from DB |
| `/forecast?steps=96` | GET | 24h GRU forecast (96 × 15-min steps) |
| `/forecast?steps=192` | GET | 48h forecast (max) |
| `/docs` | GET | Interactive Swagger UI |

### Example forecast response
```json
{
  "generated_at": "2026-07-15T10:00:00+00:00",
  "last_known_timestamp": "2026-07-15T09:45:00+00:00",
  "horizon_steps": 96,
  "horizon_hours": 24.0,
  "resolution_minutes": 15,
  "forecasts": [
    {
      "timestamp": "2026-07-15T10:00:00+00:00",
      "co2_intensity_gco2kwh": 32.4,
      "consumption_co2_intensity_gco2kwh": 35.1
    },
    ...
  ]
}
```

---

## How the Forecast Pipeline Works

Every call to `/forecast` runs this full pipeline:

```
1. Load last 15,000 3-min rows per source from SQLite
         ↓
2. Clean each source independently (same as training pipeline):
   - Physical bounds removal
   - Rolling z-score (window=480, threshold=4.0)
   - Forward-fill (limit=5 slots = 15 min)
         ↓
3. Aggregate to 15-min buckets (mean only, no sums)
   - MW sources: mean per bucket
   - Intensity: simple_avg + weighted_avg per bucket
         ↓
4. Feature engineering (identical to fingrid_15min_pipeline.ipynb):
   - Cyclical: hour_sin/cos, dow_sin/cos, month_sin/cos, is_weekend
   - Trend: time_index (0→1 over 2018–2026)
   - Lags: [1, 4, 96, 672] for all signal columns
   - Rolling: mean and std at windows [4, 96]
   - Physics: renewable_share, net_load, fossil_share
         ↓
5. Select exact feature columns from column_config.json
   (same 83 columns used in training)
         ↓
6. Scale with scaler_X_15min.pkl (fitted on training data)
         ↓
7. Take last SEQ_LEN=80 rows as seed window
         ↓
8. Autoregressive GRU forecast:
   For each of 96 steps:
   - Feed window → GRU → predict (co2_intensity, consumption_co2)
   - Update lag-1 features with prediction
   - Slide window forward by 1
         ↓
9. Inverse-transform with scaler_y_15min.pkl → real gCO₂/kWh
         ↓
10. Return JSON with timestamps and predictions
```

---

## Troubleshooting

**API container exits immediately:**
Check logs: `docker compose logs api`
Most common cause: model file not found. Verify `data/gru_model.pt` exists.

**Streamer gets 429 errors:**
Normal. The retry logic handles this automatically with exponential backoff.
Check logs: `docker compose logs -f streamer`

**Dashboard shows no data:**
The streamer needs to run at least one cycle first. Wait 1-2 minutes after startup.
Check: `docker compose logs streamer | grep "Cycle complete"`

**Forecast returns 503:**
The DB may not have enough recent rows. The API needs ~15,000 3-min rows per source
to cover the SEQ_LEN + lag-672 warmup. After the streamer has run for a few hours
(or after you load historical data), this resolves automatically.

**On Windows: "data directory not found":**
Run `python setup.py` first. It creates the `data/` directory and copies all files.
