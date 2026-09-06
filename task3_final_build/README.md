# MainCrafts AI & ML Internship – Task 3
## Model Validation, Overfitting Control & Hyperparameter Tuning

This project extends the Task 2 California Housing house-price predictor with:
- Overfitting analysis using an unrestricted Decision Tree
- 5-fold cross-validation
- GridSearchCV hyperparameter tuning
- Final model comparison against Task 2 Linear Regression and Ridge Regression
- Actual-vs-predicted and model-comparison visualizations
- Saved tuned Decision Tree model

### Dataset
California Housing Dataset — 20,640 rows, 8 input features, target `MedHouseVal`.
The CSV is included in `data/california_housing.csv`, so the notebook can run without downloading the dataset.

### Best hyperparameters
- `max_depth = 10`
- `min_samples_split = 10`

### Final test performance
| Model | RMSE | R² |
|---|---:|---:|
| Linear Regression | 0.7456 | 0.5758 |
| Ridge Regression | 0.7456 | 0.5758 |
| Tuned Decision Tree | **0.6454** | **0.6822** |

### Overfitting result
The unrestricted Decision Tree achieved approximately 1.0000 training R² but only 0.6228 test R², showing clear overfitting. After tuning, the training R² was 0.8274 and test R² was 0.6822, with a test-vs-train RMSE gap of 0.1650.

### Files
- `AI_ML_Task3_Model_Validation_Tuning.ipynb` — executed notebook with outputs
- `AI_ML_Task3_Report.pdf` — submission report
- `data/california_housing.csv` — dataset used for the executed results
- `results/` — CV, GridSearchCV, comparison, parameter and summary CSVs
- `figures/` — overfitting, model-comparison and actual-vs-predicted plots
- `models/tuned_decision_tree.joblib` — final trained model
- `run_task3.py` — reproducible Python runner
- `requirements.txt` — required packages

### Run locally
```bash
pip install -r requirements.txt
python run_task3.py
```

To view the full analysis, open the notebook and choose **Run All**.
