# EquiLearn: An Explainable Machine Learning Approach for Predicting Student Academic Performance in Bangladesh with Fairness Mitigation

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-orange)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.48+-green)](https://shap.readthedocs.io/)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

A robust machine learning framework to classify student performance into low, medium, and high categories, integrated with comprehensive fairness analysis across multiple demographic groups.

---

## Features

- **Random Forest Classifier** with 300 trees and balanced class weights.
- **Preprocessing Pipeline**: Median imputation, scaling, one-hot encoding.
- **K-Means Clustering** feature to capture latent student groupings.
- **Cross-Validation**: Stratified 5-fold CV optimizing macro F1-score.
- **Permutation Importance**: Identify top predictors.
- **SHAP Explanations**: Global and class-wise feature impact.
- **Fairness Metrics**: Accuracy and recall across gender, location, school type, and academic stream.
- **Bias Mitigation**: Sample reweighting by sensitive attribute.
- **Per-Stream Analysis**: Separate models for arts, commerce, and science streams.

---

## Results

| Model Variant          | CV F1-Score | Holdout Accuracy |
|------------------------|-------------|------------------|
| Full Model             | 97.0%       | **97.6%**        |
| Without Stream Feature | 89.8%       | 90.8%            |

**Key Findings:**  
- `stu_group` is a strong class-wise explanatory feature (large SHAP values across multiple classes). Global permutation importance indicates other features (e.g., `mother_education variants`) are among the highest ranked predictors — see permutation importance table and SHAP class summaries. 
- Arts stream: 98.7% accuracy, but low recall for medium performance.  
- Science stream: 96.96% accuracy, recall imbalance across classes.

**Note**: per-stream cross-validation F1 scores are unstable (low CV F1 but high holdout accuracy) for some streams — likely due to class imbalance and small support for certain classes. See stream-specific class supports and confusion matrices in the notebook.
---

## Dataset

- **Size**: 8,612 records.  
- **Features** (17 used in modeling):
  - Demographics: age, gender, location, family_size.  
  - Socioeconomic: mother_education, father_education, parent jobs.  
  - Academic: studytime, attendance, tutoring, extra_curricular_activities, school_type, stu_group.  
- **Target**: `perf_bin` discretized into three equal quantiles.

---

## Installation & Usage

```bash
git clone https://github.com/your-username/student-performance-fairness.git
cd student-performance-fairness
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

```python
from src.model_pipeline import StudentPerformancePredictor
import pandas as pd

df = pd.read_csv('data/bd_students_per_v2.csv')
predictor = StudentPerformancePredictor()
results = predictor.train_and_evaluate(df)
fairness_report = predictor.analyze_fairness(df)
```

---

## Selected References

1. EL Habti et al. (2025), _Enhancing Student Performance Prediction in e-Learning_, IJIEIT.  
2. Lundberg & Lee (2017), _A Unified Approach to Interpreting Model Predictions_, NIPS.  
3. Arjovsky et al. (2020), _Fairness: Detection & Recourse using SHAP_, arXiv.  
4. World Bank (2021), _National Assessments of Student Learning Outcomes in Bangladesh_.  
5. Nature Sci. Reports (2025), _ML-based Academic Performance Prediction_.

---

## Contributing

1. Fork repository  
2. Create branch (`git checkout -b feature/new-idea`)  
3. Commit changes & push  
4. Open a pull request

Contributions on bias mitigation techniques, new fairness metrics, and longitudinal analyses are welcome.

---

## Future Work

- **Intersectional Fairness** analysis  
- **Causal Inference** integration  
- **Federated Learning** for multi-institution data  
- **Real-time Intervention** dashboard

---

© 2025 Your Name. Licensed under MIT.
