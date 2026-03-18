# Research Center Quality Classification API

An end-to-end machine learning pipeline that classifies UK research
centers into **Premium**, **Standard**, or **Basic** quality tiers
using K-Means clustering, exposed via a production-ready FastAPI
REST API with full MLflow experiment tracking and model versioning.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [API Endpoints](#api-endpoints)
- [Quality Tiers](#quality-tiers)
- [Quick Start](#quick-start)
- [Environment Variables](#environment-variables)
- [MLflow Tracking](#mlflow-tracking)
- [Example Requests](#example-requests)

---

## Overview

This project solves the problem of consistently classifying research
centers by quality without manual scoring. Given five measurable
features about a research center's internal infrastructure and
surrounding healthcare environment, the system automatically assigns
it to one of three quality tiers.

### Features Used for Classification

| Feature | Description |
|---|---|
| `internalFacilitiesCount` | Number of internal labs, workstations, and testing units |
| `hospitals_10km` | Number of hospitals within 10 km |
| `pharmacies_10km` | Number of pharmacies within 10 km |
| `facilityDiversity_10km` | Diversity index (0–1) of nearby facility types |
| `facilityDensity_10km` | Spatial density of nearby healthcare facilities |

### Why These Features

Exploratory data analysis confirmed that all five features share
a strong positive correlation (Pearson r = 0.80–0.90), collectively
defining a single latent quality dimension. PCA confirmed that the
first principal component explains **89.07%** of total variance,
with near-equal loadings across all five features, validating their
joint use in K-Means clustering.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Docker Network (ml_network)              │
│                                                              │
│  ┌─────────────────────┐      ┌────────────────────────┐    │
│  │   FastAPI (api)     │      │  MLflow Tracking Server │    │
│  │   Port 8000         │─────▶│  Port 5000 (→ 5001)    │    │
│  │                     │      │                         │    │
│  │  POST /train        │      │  SQLite backend         │    │
│  │  POST /predict      │      │  Local artifact store   │    │
│  │  POST /predict/batch│      │  Model Registry         │    │
│  │  GET  /health       │      └────────────────────────┘    │
│  └─────────────────────┘                                     │
│           │                                                  │
│  ┌────────▼────────┐    ┌──────────────────────────────┐    │
│  │ api_artefacts   │    │ mlflow_data volume           │    │
│  │ volume          │    │ /mlflow/db    (SQLite)        │    │
│  │ /app/artefacts  │    │ /mlflow/artefacts (files)     │    │
│  │ ├─ scaler.pkl   │    └──────────────────────────────┘    │
│  │ ├─ kmeans_model │                                         │
│  │ └─ cluster_map  │                                         │
│  └─────────────────┘                                         │
└─────────────────────────────────────────────────────────────┘
```

### Artifact Upload Flow

```
API Container                         MLflow Container
─────────────                         ────────────────
Write file to /tmp
      │
      ▼
requests.post(
  http://mlflow:5000/
  api/2.0/mlflow/
  artifacts/upload     ──── HTTP POST ──▶  Receives file
)                                          Writes to
                                           /mlflow/artefacts
      │
      ▼
Delete /tmp file
```

> All artifact uploads route through the MLflow REST API over HTTP.

---

## Project Structure

```
📂 research-center-assignment/
│
├── 📂 app/
│   ├── __init__.py
│   ├── main.py                     ← FastAPI app factory + lifespan
│   │
│   ├── 📂 api/
│   │   ├── __init__.py
│   │   └── 📂 routes/
│   │       ├── __init__.py
│   │       ├── train.py            ← POST /train
│   │       ├── predict.py          ← POST /predict, POST /predict/batch
│   │       └── health.py           ← GET /health, /health/ready, /health/live
│   │
│   ├── 📂 core/
│   │   ├── __init__.py
│   │   ├── config.py               ← Pydantic Settings (env vars)
│   │   └── logging.py              ← Structured logging setup
│   │
│   ├── 📂 pipeline/
│   │   ├── __init__.py
│   │   ├── data_loader.py          ← CSV ingestion + structural checks
│   │   ├── data_validator.py       ← Range, type, missing value validation
│   │   ├── preprocessor.py         ← Feature selection + StandardScaler
│   │   ├── trainer.py              ← K-Means training + tier mapping
│   │   └── evaluator.py            ← Silhouette score + diagnostic plots
│   │
│   ├── 📂 schemas/
│   │   ├── __init__.py
│   │   ├── input.py                ← Pydantic request validation models
│   │   └── output.py               ← Pydantic response models
│   │
│   └── 📂 mlflow_utils/
│       ├── __init__.py
│       └── tracker.py              ← MLflow run management + HTTP uploads
│
├── 📂 data/
│   └── research_centers.csv        ← Training dataset (50 UK research centers)
│
├── EDA_and_Model.ipynb             ← EDA and baseline model
├── Dockerfile                      ← Multi-stage build (builder + runtime)
├── docker-compose.yaml             ← API + MLflow services
├── requirements.txt                ← Python dependencies
├── .env                            ← Environment variable template
├── .gitignore
├── .dockerignore
└── README.md
```

---

## ML Pipeline

Every call to `POST /train` executes the full pipeline in sequence:

```
Step 1  ── Tag MLflow run with metadata
Step 2  ── Load data from CSV
Step 3  ── Validate data quality
Step 4  ── Preprocess (StandardScaler)
Step 5  ── Train K-Means (K=3)
Step 6  ── Evaluate (silhouette score)
Step 7  ── Generate diagnostic plots
Step 8  ── Save artefacts to disk
Step 9  ── Register model in MLflow
```

### Pipeline Components

| Component | File | Responsibility |
|---|---|---|
| `DataLoader` | `pipeline/data_loader.py` | Load CSV, verify required columns exist |
| `DataValidator` | `pipeline/data_validator.py` | Missing values, range checks, dtype checks, row count |
| `Preprocessor` | `pipeline/preprocessor.py` | Feature selection, fit + apply StandardScaler |
| `Trainer` | `pipeline/trainer.py` | K-Means (k-means++, n_init=50), cluster-to-tier mapping |
| `Evaluator` | `pipeline/evaluator.py` | Silhouette score, per-cluster metrics, diagnostic plots |
| `MLflowTracker` | `mlflow_utils/tracker.py` | Log params, metrics, artifacts, register model |

### Data Validation Checks

| Check | Type | Action on Failure |
|---|---|---|
| Missing values in feature columns | Hard error | Pipeline halts |
| Out-of-range values | Hard error | Pipeline halts |
| Incorrect data types | Auto-coerce, then hard error | Warning or halt |
| Minimum row count (> K × 3) | Hard error | Pipeline halts |
| Duplicate rows | Soft warning | Duplicates removed, pipeline continues |
| Cross-column consistency | Soft warning | Warning logged, pipeline continues |

### Model Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| K (clusters) | 3 | Supported by elbow method, silhouette score, and business requirement |
| Initialisation | k-means++ | Reduces risk of poor local minima vs random init |
| n_init | 50 | Tries 50 random seeds, keeps best result |
| max_iter | 300 | Maximum iterations per run |
| random_state | 42 | Reproducible results across runs |
| algorithm | lloyd | Standard, robust on small datasets |

### Cluster-to-Tier Mapping

Clusters are mapped to tier names by ranking the mean
`internalFacilitiesCount` centroid value (original scale):

```
Highest mean internalFacilitiesCount  →  Premium
Middle  mean internalFacilitiesCount  →  Standard
Lowest  mean internalFacilitiesCount  →  Basic
```

This ranking is deterministic given fixed hyperparameters and
`random_state=42`. The mapping is saved as `cluster_to_tier.pkl`
alongside the model to ensure consistent predictions on reload.

---

## API Endpoints

| Method | Endpoint | Description | Status Codes |
|---|---|---|---|
| `GET` | `/` | API root: links to all endpoints | 200 |
| `GET` | `/health` | Full health check (model + MLflow) | 200 |
| `GET` | `/health/ready` | Readiness probe: 503 until model trained | 200, 503 |
| `GET` | `/health/live` | Liveness probe: always 200 if process running | 200 |
| `POST` | `/train/` | Run full ML pipeline + log to MLflow | 200, 422, 500 |
| `POST` | `/predict/` | Classify a single research center | 200, 422, 503 |
| `POST` | `/predict/batch` | Classify up to 1000 centers in one request | 200, 422, 503 |
| `GET` | `/docs` | Swagger UI interactive documentation | 200 |
| `GET` | `/redoc` | ReDoc API documentation | 200 |

---

## Quality Tiers

| Tier | Description | Typical Profile |
|---|---|---|
| **Premium** | Highest internal capacity and richest external healthcare access | internalFacilitiesCount ≥ 8, hospitals ≥ 3, diversity ≥ 0.80 |
| **Standard** | Moderate facilities and adequate external access | internalFacilitiesCount 4–7, hospitals 1–3, diversity 0.45–0.79 |
| **Basic** | Limited internal infrastructure and sparse external access | internalFacilitiesCount ≤ 3, hospitals 0–1, diversity ≤ 0.44 |

> Tier boundaries are data-driven — they reflect K-Means centroid
> positions on the training data, not fixed thresholds.

---

## Quick Start

### Prerequisites

- Docker ≥ 24.0
- Docker Compose ≥ 2.0
- 2 GB free disk space

---

### Option 1 — Docker Compose (Recommended)

```bash
# 1. Clone the repository
git clone <https://github.com/thedataboy001/research-center-assignment>
cd research-center-assignment

# 2. Copy the environment template
cp .env

# 3. Build and start all services
docker-compose up --build

# 4. Wait for both containers to become healthy
#    You should see in the logs:
#    mlflow_server         | Listening at: http://0.0.0.0:5000
#    research_center_api   | Application startup complete.

# 5. Trigger the training pipeline
curl -X POST http://localhost:8000/train/

# 6. Make a prediction
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "internalFacilitiesCount": 9,
    "hospitals_10km": 3,
    "pharmacies_10km": 2,
    "facilityDiversity_10km": 0.82,
    "facilityDensity_10km": 0.45
  }'
```

---

### Option 2 — Local Development

```bash
# 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment file
cp .env

# 4. Start MLflow tracking server
mlflow server \
  --host 0.0.0.0 \
  --port 5000 \
  --backend-store-uri sqlite:////mlflow/db/mlflow.db \
  --default-artifact-root ./mlflow_artefacts \
  --serve-artifacts


# 5. In a new terminal, start the FastAPI application
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 6. Train the model
curl -X POST http://localhost:8000/train/
```

---

### Stopping the Services

```bash
# Stop containers but keep volumes (artefacts and MLflow data preserved)
docker-compose down

# Stop containers AND delete all volumes (full reset)
docker-compose down -v
```

---

## Environment Variables


| Variable | Default | Description |
|---|---|---|
| `APP_NAME` | `ResearchCenterQualityAPI` | Application display name |
| `APP_VERSION` | `1.0.0` | Application version |
| `APP_ENV` | `development` | Environment (`development` / `production`) |
| `DEBUG` | `true` | Enable verbose debug logging |
| `DATA_PATH` | `data/research_centers.csv` | Path to training CSV |
| `SELECTED_FEATURES` | *(all five features)* | Comma-separated list of clustering features |
| `N_CLUSTERS` | `3` | Number of K-Means clusters |
| `N_INIT` | `50` | Number of K-Means random initialisations |
| `MAX_ITER` | `300` | Maximum K-Means iterations per run |
| `RANDOM_STATE` | `42` | Random seed for reproducibility |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | MLflow tracking server URI |
| `MLFLOW_EXPERIMENT_NAME` | `research_center_quality` | MLflow experiment name |
| `MLFLOW_MODEL_NAME` | `kmeans_quality_classifier` | MLflow Model Registry name |
| `ARTEFACT_DIR` | `artefacts` | Local directory for inference pkl files |

---

## MLflow Tracking

Every training run logs the following to MLflow:

### Parameters Logged

| Category | Parameters |
|---|---|
| **Data** | `data_row_count`, `data_col_count`, `duplicate_rows`, `validation_passed` |
| **Scaler** | `scaler_mean_<feature>`, `scaler_std_<feature>` for all 5 features |
| **Model** | `n_clusters`, `n_init`, `max_iter`, `random_state`, `init_method`, `algorithm` |
| **Mapping** | `tier_cluster_Premium`, `tier_cluster_Standard`, `tier_cluster_Basic` |

### Metrics Logged

| Metric | Description |
|---|---|
| `silhouette_score_overall` | Overall clustering quality (higher = better separation) |
| `silhouette_Premium_mean` | Mean silhouette score for Premium cluster |
| `silhouette_Standard_mean` | Mean silhouette score for Standard cluster |
| `silhouette_Basic_mean` | Mean silhouette score for Basic cluster |
| `inertia` | Within-cluster sum of squares (lower = tighter clusters) |
| `n_iter_to_converge` | Number of K-Means iterations until convergence |
| `data_missing_values_total` | Total missing values found in training data |

### Artifacts Logged

| Artifact | Path in MLflow | Description |
|---|---|---|
| `validation_report.json` | `data_quality/` | Full data validation results |
| `cluster_to_tier.json` | `model_metadata/` | Cluster number → tier name mapping |
| `scaler.pkl` | `model_artefacts/` | Fitted StandardScaler |
| `kmeans_model.pkl` | `model_artefacts/` | Trained K-Means model |
| `silhouette_plot.png` | `plots/` | Per-point silhouette analysis |
| `pca_scatter.png` | `plots/` | 2D PCA cluster visualisation |
| `centroid_heatmap.png` | `plots/` | Cluster centroid feature values |
| `research_centers_clustered.csv` | `datasets/` | Training data with assigned tiers |

### Accessing MLflow UI

```bash
# Docker Compose (mapped to host port 5001)
open http://localhost:5001

# Local development
open http://localhost:5000
```

---

## Example Requests

### Train the Model

```bash
curl -X POST http://localhost:8000/train/
```

**Response:**
```json
{
  "status": "success",
  "mlflowRunId": "811d5e35c4f649a1941d9700d1711fc6",
  "mlflowRunUrl": "http://mlflow:5000/#/experiments/1/runs/811d5e35c4f649a1941d9700d1711fc6",
  "silhouetteScore": 0.7423,
  "inertia": 24.8631,
  "clusterSizes": {
    "Premium": 16,
    "Standard": 19,
    "Basic": 15
  },
  "tierMapping": {
    "0": "Premium",
    "1": "Basic",
    "2": "Standard"
  },
  "featuresUsed": [
    "internalFacilitiesCount",
    "hospitals_10km",
    "pharmacies_10km",
    "facilityDiversity_10km",
    "facilityDensity_10km"
  ],
  "hyperparameters": {
    "n_clusters": 3,
    "n_init": 50,
    "max_iter": 300,
    "random_state": 42,
    "init_method": "k-means++",
    "algorithm": "lloyd"
  }
}
```

---

### Predict — Single Center

```bash
curl -X POST http://localhost:8000/predict/ \
  -H "Content-Type: application/json" \
  -d '{
    "internalFacilitiesCount": 9,
    "hospitals_10km": 3,
    "pharmacies_10km": 2,
    "facilityDiversity_10km": 0.82,
    "facilityDensity_10km": 0.45
  }'
```

**Response:**
```json
{
  "predictedCategory": "Premium",
  "clusterNumber": 0,
  "confidence": 0.8731,
  "modelVersion": "811d5e35c4f649a1941d9700d1711fc6",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

### Predict — Batch

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '
  {
    "centers": [
      {
        "internalFacilitiesCount": 9,
        "hospitals_10km": 3,
        "pharmacies_10km": 2,
        "facilityDiversity_10km": 0.82,
        "facilityDensity_10km": 0.45
      },
      {
        "internalFacilitiesCount": 5,
        "hospitals_10km": 2,
        "pharmacies_10km": 2,
        "facilityDiversity_10km": 0.55,
        "facilityDensity_10km": 0.30
      },
      {
        "internalFacilitiesCount": 2,
        "hospitals_10km": 0,
        "pharmacies_10km": 1,
        "facilityDiversity_10km": 0.18,
        "facilityDensity_10km": 0.09
      }
    ]
  }'
```

**Response:**
```json
{
  "predictions": [
    {
      "predictedCategory": "Premium",
      "clusterNumber": 0,
      "confidence": 0.8731,
      "modelVersion": "811d5e35c4f649a1941d9700d1711fc6",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "predictedCategory": "Standard",
      "clusterNumber": 2,
      "confidence": 0.6214,
      "modelVersion": "811d5e35c4f649a1941d9700d1711fc6",
      "timestamp": "2024-01-15T10:30:00Z"
    },
    {
      "predictedCategory": "Basic",
      "clusterNumber": 1,
      "confidence": 0.9103,
      "modelVersion": "811d5e35c4f649a1941d9700d1711fc6",
      "timestamp": "2024-01-15T10:30:00Z"
    }
  ],
  "totalCount": 3,
  "tierSummary": {
    "Premium": 1,
    "Standard": 1,
    "Basic": 1
  }
}
```

---

### Health Check

```bash
curl http://localhost:8000/health
```

**Response (model loaded):**
```json
{
  "status": "healthy",
  "appName": "ResearchCenterQualityAPI",
  "version": "1.0.0",
  "environment": "production",
  "modelLoaded": true,
  "mlflowStatus": "connected",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

**Response (before training):**
```json
{
  "status": "degraded",
  "appName": "ResearchCenterQualityAPI",
  "version": "1.0.0",
  "environment": "production",
  "modelLoaded": false,
  "mlflowStatus": "connected",
  "timestamp": "2024-01-15T10:30:00Z"
}
```
---

## Service URLs

| Service | URL | Description |
|---|---|---|
| FastAPI | http://localhost:8000 | REST API |
| Swagger UI | http://localhost:8000/docs | Interactive API docs |
| ReDoc | http://localhost:8000/redoc | API reference docs |
| MLflow UI | http://localhost:5001 | Experiment tracking dashboard |