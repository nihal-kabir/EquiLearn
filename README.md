# An Explainable Machine Learning Approach for Predicting Student Academic Performance in Bangladesh with Fairness Mitigation

[![Python](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.6+-orange)](https://scikit-learn.org/)
[![SHAP](https://img.shields.io/badge/SHAP-0.48+-green)](https://shap.readthedocs.io/)

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
- `stu_group` is a strong class-wise explanatory feature (large SHAP values across multiple classes). Global permutation importance indicates other features (e.g., `mother_education variants`) are among the highest ranked predictors  [see permutation importance table and SHAP class summaries]. 
- Arts stream: 98.7% accuracy, but low recall for medium performance.  
- Science stream: 96.96% accuracy, recall imbalance across classes.

**Note**: per-stream cross-validation F1 scores are unstable (low CV F1 but high holdout accuracy) for some streams; likely due to class imbalance and small support for certain classes. See stream-specific class supports and confusion matrices in the notebook.

---

## Dataset

**URL** https://shorturl.at/krSqb

- **Size**: 8,612 records.  
- **Features** (17 used in modeling):
  - Demographics: age, gender, location, family_size.  
  - Socioeconomic: mother_education, father_education, parent jobs.  
  - Academic: studytime, attendance, tutoring, extra_curricular_activities, school_type, stu_group.  
- **Target**: `perf_bin` discretized into three equal quantiles.

---

## Contributing

1. Fork repository  
2. Create branch (`git checkout -b feature/new-idea`)  
3. Commit changes & push  
4. Open a pull request

Contributions on bias mitigation techniques, new fairness metrics, longitudinal analyses etc are welcome.

---

## Future Work

- **Intersectional Fairness** analysis  
- **Causal Inference** integration  
- **Federated Learning** for multi-institution data  
- **Real-time Intervention** dashboard

---

## References

[1] W. Ahmed, M. A. Wani, P. Plawiak, S. Meshoul, A. Mahmoud, and M. Hammad, “Machine learning-based academic performance prediction with explainability for enhanced decision-making in educational institutions,” Scientific Reports, vol. 15, no. 1, p. 26879, Jul. 2025. [Online]. Available: https://doi.org/10.1038/s41598-025-12353-4

[2] S. Malik, S. G. K. Patro, C. Mahanty, R. Hegde, Q. N. Naveed, A. Lasisi, A. Buradi, A. F. Emma, and N. Kraiem, “Advancing educational data mining for enhanced student performance prediction: a fusion of feature selection algorithms and classification techniques with dynamic feature ensemble evolution,” Scientific Reports, vol. 15, no. 1, p. 8738, Mar. 2025. [Online]. Available: https://doi.org/10.1038/s41598-025-92324-x

[3] Y. Wang and L. Singh, “Impact on bias mitigation algorithms to variations in inferred sensitive attribute uncertainty,” Frontiers in Artificial Intelligence, vol. 8, Art. no. 1520330, Mar. 2025. [Online]. Available: https://doi.org/10.3389/frai.2025.1520330

[4] J. Wang and Y. Yu, “Machine learning approach to student performance prediction of online learning,” PLOS ONE, vol. 20, no. 1, p. e0299018, Jan. 2025. [Online]. Available: https://doi.org/10.1371/journal.pone.0299018

[5] Z. Liu, X. Zhou, and Y. Liu, “Student dropout prediction using ensemble learning with SHAP-based explainable AI analysis,” Journal of Social Systems and Policy Analysis, vol. 2, no. 3, pp. 111–132, Aug. 2025. [Online]. Available: https://doi.org/10.62762/JSSPA.2025.321501

[6] E. Ben George, R. Senthilkumar, F. Al-Junaibi, and Z. Al-Shuaibi, “Explainable AI methods for predicting student grades and improving academic success,” Journal of Information Systems Engineering and Management, vol. 10, no. 23s, Mar. 2025. [Online]. Available: https://doi.org/10.52783/jisem.v10i23s.3680

[7] M. El Jihaoui, O. E. Abra, and K. Mansouri, “Predicting and interpreting student academic performance: A deep learning and SHAP approach,” SHS Web of Conferences, CIFEM’2024, vol. 214, Art. no. 01001, 2025. [Online]. Available: https://doi.org/10.1051/shsconf/202521401001
