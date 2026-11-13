[README.md](https://github.com/user-attachments/files/34810764/README.md)
# Hospital Readmission Risk Prediction Using MLP

**Deep Learning | Healthcare | TensorFlow/Keras | CBIM Methodology**

A Multi-Layer Perceptron (MLP) that predicts whether a diabetic patient will be **readmitted to hospital within 30 days of discharge**, built on the *Diabetes 130-US Hospitals (1999–2008)* dataset from the UCI repository.

> **Disclaimer:** This is an educational prototype trained on historical data (1999–2008). It is **not** a clinically validated decision-making system and must not be used for real patient care.

---

## 1. Project files

| File | What it is |
|---|---|
| `Trivikram_Hospital_Readmission_MLP.ipynb` | Main deliverable. The full executable notebook (CBIM, EDA, cleaning, split, MLP, evaluation, error analysis, report). |
| `Trivikram_Hospital_Readmission_MLP_Report.docx` | Short business findings report with structured error analysis, figures and recommendations. |
| `1794582628822_MLP_Hospital_Readmission_Assignment.docx` | The original assignment brief (requirements, challenges, marking scheme). |
| `hospital_readmission_mlp.keras` | The trained Keras model (about 144 KB). |
| `hospital_readmission_patient_predictions.csv` | Patient-level prediction report for the test set (19,870 rows). |
| `diabetic_data.csv` *(not included)* | Raw dataset. Download it separately (see Section 4). |

---

## 2. Problem definition

| Item | Description |
|---|---|
| Business problem | Hospitals need to find patients likely to return within 30 days so they can plan follow-up care. |
| ML task | Supervised **binary classification** |
| Target | `readmitted` converted to binary: `1` = `<30` (readmitted within 30 days), `0` = `>30` or `NO` |
| Output | Probability (0 to 1) of readmission within 30 days |
| Framework | TensorFlow / Keras Sequential API |

---

## 3. Dataset summary

| Item | Value |
|---|---|
| Source | UCI: Diabetes 130-US Hospitals for Years 1999–2008 |
| Raw size | 101,766 encounters × 50 columns |
| Unique patients | 71,518 (30,248 repeat records) |
| Original labels | NO: 54,864, >30: 35,545, <30: 11,357 |
| Class balance | 88.84% negative, 11.16% positive (**imbalanced**) |
| After cleaning | 99,343 encounters, 41 input features |
| After encoding | 114 model input columns |

---

## 4. How to run

### Requirements
- Python 3.9+ (the notebook was run with TensorFlow 2.21 and pandas 3.1.0)
- Libraries: `tensorflow`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`

```bash
pip install tensorflow pandas numpy matplotlib seaborn scikit-learn
```

### Steps
1. Download the dataset from <https://archive.ics.uci.edu/dataset/296/diabetes+130-us+hospitals+for+years+1999-2008> and unzip it to get `diabetic_data.csv`.
2. Open `Trivikram_Hospital_Readmission_MLP.ipynb` in Jupyter or Google Colab.
3. **Edit `CSV_PATH` in Part 3.** It currently points to a Windows path on the author's computer:
   ```python
   CSV_PATH = r"C:\Users\hp\Desktop\Trivikram\phase 2\Module 6 Deep Learning\diabetes+130-us+hospitals+for+years+1999-2008\diabetic_data.csv"
   ```
   Change it to where your copy of `diabetic_data.csv` is stored (for Colab, upload the file and use `"/content/diabetic_data.csv"`).
4. Run all cells from top to bottom (**Run All**). The random seed is fixed at `42`.
5. At the end the notebook writes `hospital_readmission_patient_predictions.csv` and `hospital_readmission_mlp.keras` to the working folder.

---

## 5. Notebook walkthrough (CBIM)

| Part | Content |
|---|---|
| 1 | **CBIM: business understanding**: problem, deep learning problem, objectives |
| 2–3 | Environment setup, seed, dataset loading |
| 4 | Data understanding: shape, info, statistics, missing values, duplicates, class distribution |
| 5 | Target creation, `?` → `NaN`, drop `weight`, remove duplicates, exclude expired/hospice discharges |
| 6 | Patient-grouped, stratified train/validation/test split |
| 7 | Encoding and scaling with `ColumnTransformer` |
| 8 | MLP architecture |
| 9 | Class weights, batch-size experiments (128 vs 256), final training |
| 10 | Evaluation at 0.50 and at a validation-selected threshold |
| 11 | Structured error analysis (FP / FN) |
| 12 | Patient-level prediction report and risk tiers |
| 13–14 | Business findings, recommendations, limitations, conclusion |

---

## 6. Methodology highlights

**Industry challenges addressed**

| Challenge | How it was handled |
|---|---|
| Class imbalance | Balanced class weights (class 0 = 0.564, class 1 = 4.391) and recall/F1/PR-AUC reported alongside accuracy. |
| Missing values | `?` converted to `NaN`; `weight` (96.86% missing) dropped; numeric median imputation, categorical `"Unknown"`. |
| Data leakage | `readmitted`, `encounter_id` and `patient_nbr` excluded from inputs; the preprocessor is fitted on training data only. |
| Repeat patients | `StratifiedGroupKFold` keeps every patient in one split only (zero overlap verified). |
| Generalization | Training vs validation metrics compared; early stopping used. |

**Data split**

| Split | Rows | Positive rate |
|---|---|---|
| Train | 63,578 | 11.39% |
| Validation | 15,895 | 11.39% |
| Test | 19,870 | 11.39% |

**Model architecture**

```
Input (114) → Dense(64, ReLU) → Dropout(0.30) → Dense(32, ReLU) → Dropout(0.20) → Dense(1, Sigmoid)
```

- Parameters: 9,473 (all trainable)
- Optimizer: Adam (learning rate 0.001), loss: binary crossentropy
- Callbacks: EarlyStopping (monitor `val_auc`, patience 4), ReduceLROnPlateau
- Batch-size experiment: 128 (val AUC 0.6523) vs 256 (val AUC 0.6519). **128 selected**.
- The final model stopped after **16 of a maximum 30 epochs**.

---

## 7. Results (held-out test set, 19,870 encounters)

| Metric | Threshold 0.50 | Threshold 0.536 (validation-selected) |
|---|---|---|
| Accuracy | 0.6380 | 0.7173 |
| Precision | 0.1711 | 0.1876 |
| Recall | 0.5665 | 0.4450 |
| F1-score | 0.2628 | 0.2639 |
| ROC-AUC | 0.6512 | 0.6512 |
| PR-AUC | 0.2016 | 0.2016 |

**Generalization check (final epoch):** AUC was 0.6764 on training and 0.6541 on validation, a small gap.

**Confusion matrix at threshold 0.536**

| | Predicted No | Predicted Yes |
|---|---|---|
| **Actual No** | 13,246 (TN) | 4,361 (FP) |
| **Actual Yes** | 1,256 (FN) | 1,007 (TP) |

**Risk tiers (Low ≤ 0.30, Medium 0.30–0.60, High > 0.60)**

| Tier | Patients | Actual readmissions | Readmission rate |
|---|---|---|---|
| Low | 2,536 | 115 | 4.54% |
| Medium | 15,005 | 1,606 | 10.70% |
| High | 2,329 | 542 | 23.27% |

The tiers separate risk meaningfully: the High tier is readmitted about 5× as often as the Low tier.

### Key takeaways
- The model has **modest** discriminative power (ROC-AUC ≈ 0.65). It is useful for *prioritising* follow-up, not for confident individual predictions.
- Precision is low (about 19%), so many alerts are false alarms. The model identifies roughly 45% of true readmissions at the selected threshold.
- The F1-optimised threshold (0.536) **lowers recall** compared with 0.50. If missing a readmission is costlier than a false alarm, a lower threshold is worth considering.
- Missed readmissions (FN) tend to have far fewer previous inpatient and emergency visits than caught ones (TP): previous inpatient visits average 0.19 vs 2.34.

---

## 8. Output file: `hospital_readmission_patient_predictions.csv`

| Column | Meaning |
|---|---|
| `Patient ID` | Pseudonymous label (`Patient_<patient_nbr>`). It is not unique, because a patient can have several encounters in the test set. |
| `Actual Readmission` | `Yes` / `No` (true label) |
| `Predicted Readmission` | `Yes` / `No` (at threshold 0.536) |
| `Risk Probability` | Model probability, rounded to 4 decimals |
| `Risk Tier` | `Low` / `Medium` / `High` |

---

## 9. Using the saved model

```python
from tensorflow import keras
model = keras.models.load_model("hospital_readmission_mlp.keras")
probs = model.predict(X_processed)   # X_processed must have 114 columns
```

**Important:** the model expects the **114-column preprocessed input**. The fitted `ColumnTransformer` (imputation, scaling, one-hot encoding) is **not saved** in the project files. To score new patients you must either re-run the notebook's preprocessing steps on the same training data, or save the preprocessor with `joblib.dump(preprocessor, "preprocessor.joblib")` and add it to the notebook.

---

## 10. Limitations
- Data covers US hospitals from 1999–2008 and may not reflect current practice.
- Readmissions outside the recorded data are not captured.
- Class imbalance (11% positive) limits precision, and the model is small (two hidden layers).
- No fairness, privacy, prospective validation or regulatory review has been done.
- Features should be re-checked against the exact prediction time (discharge) before any real use.

## 11. Possible improvements
- Add diagnosis codes (`diag_1`–`diag_3`), grouped into clinical categories.
- Compare against a baseline such as logistic regression or gradient boosting.
- Tune layers, dropout and learning rate. Try focal loss or resampling.
- Choose the threshold by hospital follow-up capacity or by a recall target.
- Save the preprocessing pipeline together with the model.

---

**Author:** Trivikram | Deep Learning, Module 6 (Industry Mini Use Case)
