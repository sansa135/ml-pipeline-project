# End-to-End ML Pipeline

A production-style machine learning pipeline: data ingestion → preprocessing → 
feature engineering → model training → evaluation → deployment as a REST API → 
containerized with Docker.

## Problem Statement
Predict customer churn for a subscription-based business using historical 
customer data (replace with your chosen dataset/problem).

## Project Structure
```
ml-pipeline-project/
├── data/
│   ├── raw/                # Original, immutable data
│   └── processed/          # Cleaned, feature-engineered data
├── notebooks/              # EDA and experimentation (exploratory only)
├── src/
│   ├── data/                # Data loading & ingestion scripts
│   │   └── make_dataset.py
│   ├── features/            # Feature engineering
│   │   └── build_features.py
│   ├── models/               # Training, prediction, evaluation
│   │   ├── train_model.py
│   │   ├── predict_model.py
│   │   └── evaluate_model.py
│   └── api/                  # FastAPI app for serving predictions
│       ├── main.py
│       └── schemas.py
├── models/                  # Saved/serialized trained models
├── tests/                   # Unit tests
├── .github/workflows/       # CI/CD pipeline (GitHub Actions)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── config.yaml              # Central config (paths, hyperparameters)
└── README.md
```

## Setup

```bash
# Clone repo
git clone https://github.com/<your-username>/ml-pipeline-project.git
cd ml-pipeline-project

# Create virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### 1. Ingest & process data
```bash
python src/data/make_dataset.py
```

### 2. Build features
```bash
python src/features/build_features.py
```

### 3. Train model
```bash
python src/models/train_model.py
```

### 4. Evaluate model
```bash
python src/models/evaluate_model.py
```

### 5. Run API locally
```bash
uvicorn src.api.main:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### 6. Run with Docker
```bash
docker build -t ml-pipeline .
docker run -p 8000:8000 ml-pipeline
```

## Testing
```bash
pytest tests/
```

## CI/CD
GitHub Actions automatically runs tests and linting on every push 
(see `.github/workflows/ci.yml`).

## Tech Stack
- **Data/ML:** pandas, scikit-learn
- **API:** FastAPI, uvicorn
- **Containerization:** Docker
- **CI/CD:** GitHub Actions
- **Testing:** pytest

## License
MIT
