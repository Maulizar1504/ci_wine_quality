# 🍷 Wine Quality ML — CI/CD Pipeline

**Author**: Maulizar  
**GitHub**: https://github.com/Maulizar1504  
**DagsHub**: https://dagshub.com/alsyamaulizar  
**Level**: Advanced — 8-Stage CI/CD Pipeline

---

## 🔄 Pipeline Stages

| # | Stage | Description |
|---|-------|-------------|
| 1 | **Code Quality** | flake8 + black + isort |
| 2 | **Unit Tests** | pytest + coverage |
| 3 | **Preprocessing** | Run & validate preprocessing |
| 4 | **Training** | Baseline + hyperparameter tuning |
| 5 | **Evaluation** | Quality gate (F1 ≥ 0.70) |
| 6 | **Upload** | GitHub artifacts + DagsHub |
| 7 | **Docker** | Build, test & push image |
| 8 | **Summary** | Pipeline report |

## 🔐 Required GitHub Secrets

```
DAGSHUB_TOKEN   — DagsHub personal access token
                  Generate at: https://dagshub.com/user/settings/tokens
```

## 📁 Structure

```
ci_wine_quality/
├── .github/workflows/ci.yml
├── Membangun_model/
│   ├── modelling.py
│   ├── modelling_tuning.py
│   ├── wine_quality_preprocessing.py
│   └── requirements.txt
├── Monitoring dan Logging/
│   └── 7.Inference.py
├── tests/
│   └── test_preprocessing.py
├── Dockerfile
└── docker-compose.yml
```
