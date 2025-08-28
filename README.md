# ML Model Deployment with TensorFlow Serving

This project demonstrates a full **MLOps workflow**: training an MNIST classifier, serving it with **TensorFlow Serving in Docker**, versioning models, exposing a **Streamlit UI** for predictions, and monitoring inference health with **Prometheus**.

---

## Features

- **Model Training & Retraining**
  - Weekly retrain pipeline (`src/pipeline.py`)
  - Accuracy gate: only promote models with `accuracy ≥ 97%`
  - Automatic versioning via `VersionManager`

- **Model Serving**
  - TensorFlow Serving in Docker
  - REST API exposed at `http://localhost:8501/v1/models/mnist:predict`
  - Multi-version serving with `models.config`

- **User Interface**
  - Streamlit app (`src/streamlit_app.py`)
  - Draw digits on a canvas → get predictions instantly
  - Select deployed model version from sidebar

- **Monitoring**
  - Prometheus scrapes TF-Serving metrics (`/monitoring/prometheus/metrics`)
  - Metrics available at `http://localhost:9090`

---

## Project Structure

```

.
├── src/
│   ├── config.py             # Global config & env vars
│   ├── streamlit_app.py      # UI for drawing & predictions
│   ├── pipeline.py           # Retraining pipeline with accuracy gate
│   ├── version_manager.py    # Handles model versioning & config
│   ├── train.py              # Training logic
│   ├── metrics.py            # Metrics recording
│   └── **init**.py
├── scripts/
│   └── create_initial_model.py # Script to bootstrap version 1
├── models/                   # Versioned models stored here
├── prometheus.yml            # Prometheus scrape config
├── Dockerfile.streamlit      # Docker configuration for streamlite 
├── docker-compose.yaml       # Multi-service deployment
└── README.md

````

---

## ⚙️ Setup & Run

### 1. Clone & Build
```bash
git clone git@github.com:hal-29/tf_model_serving.git
cd model_deployment
docker compose build
````

### 2. Start Services

```bash
docker compose up -d
```

Services:

* **Streamlit UI** → [http://localhost:8502](http://localhost:8502)
* **TensorFlow Serving** → [http://localhost:8501](http://localhost:8501)
* **Prometheus** → [http://localhost:9090](http://localhost:9090)

### 3. Create Initial Model

Run inside the container to generate version `1`:

```bash
docker compose exec streamlit-app python -m scripts.create_initial_model
```

---

## Usage (Streamlit UI)

1. Open [http://localhost:8502](http://localhost:8502).
2. Draw a digit (0–9) on the canvas.
3. Click **Predict Digit** → see prediction + probability chart.
4. Switch between deployed model versions in the sidebar.

---

## Retraining Pipeline

Trigger retraining manually:

```bash
docker compose exec streamlit-app python -m src.pipeline
```

Behavior:

* Trains a new candidate model
* Compares accuracy against baseline & threshold
* If accuracy improves:

  * Saves model under new version folder (`/models/mnist/<N>`)
  * Updates `models.config` for TensorFlow Serving
* If not → skips deployment (keeps last version active)

---

## Monitoring

Prometheus scrapes TF-Serving metrics automatically.

* Check scrape targets: [http://localhost:9090/targets](http://localhost:9090/targets)
* Query example:

  ```
  request_count
  ```