# 📜 Scripts Overview

This folder contains utility scripts and supporting documentation for the fraud detection project.

Scripts are designed to support:
- Reproducibility
- Automation
- Modular development

---

## 📁 Folder Purpose

The `scripts/` directory is intended for:
- Data processing helpers
- Model execution scripts
- Experiment automation
- Future pipeline integration

---

## 📌 Current Status

At this stage, core logic is implemented inside:
- `src/task1_preprocessing.py`
- `src/task2_modeling.py`
- `src/task3_model_explain.py`

This folder is intentionally kept minimal to:
- Maintain clarity
- Avoid code duplication
- Support future scaling (e.g., CLI or Airflow jobs)

---

## 🚀 Future Extensions (Optional)
- `run_pipeline.py` – end-to-end execution
- `train_model.py` – model retraining
- `evaluate_model.py` – automated evaluation

---

## ✅ Best Practices
- Keep scripts lightweight
- Import reusable logic from `src/`
- Avoid notebook-style code here
