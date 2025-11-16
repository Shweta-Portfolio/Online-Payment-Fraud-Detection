## Online Payment Fraud Detection

This project analyzes a large online transaction dataset and builds machine learning models to detect fraudulent payments. The workflow includes exploratory data analysis (EDA), feature engineering for categorical variables, model training/evaluation (Logistic Regression, XGBoost, Random Forest), and visualization of a confusion matrix for the best-performing model.

### Dataset
- Name: Online Payments Fraud Detection (Kaggle)
- Columns (10):
  - `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, `nameDest`, `oldbalanceDest`, `newbalanceDest`, `isFraud`
- Link: `https://drive.google.com/file/d/133E0TDrfIjnhwRoGTw9OEozwBXUL38D8/view`
- Local file expected by the notebook: `new_file.csv` in the project root.

### Repository Contents
- `new.ipynb`: Main analysis notebook
- `new_file.csv`: Dataset CSV (not tracked on Git by default)

### Workflow Summary (from the notebook)
1. Load data:
   - `pd.read_csv('new_file.csv')`
   - 6,362,620 rows, 10 columns (large dataset; ensure enough memory)
2. Inspect:
   - `data.info()`, `data.describe()`
   - Data types: 3 categorical (`type`, `nameOrig`, `nameDest`), 2 integer (`step`, `isFraud`), 5 float.
3. EDA (selected):
   - Distribution of `type` via count plot
   - Average `amount` by `type`
   - Class balance for `isFraud`
   - Histogram of `step`
   - Correlation heatmap (after factorizing categoricals to numeric codes for inspection)
4. Feature Engineering:
   - One-hot encode `type` with `pd.get_dummies(drop_first=True)`
   - Drop high-cardinality identifiers `nameOrig`, `nameDest` and original `type`
5. Modeling:
   - Train/test split: 70/30 with `random_state=42`
   - Models:
     - LogisticRegression()
     - XGBClassifier()
     - RandomForestClassifier(n_estimators=7, criterion='entropy', random_state=7)
   - Metric: ROC AUC on train and validation
6. Results (ROC AUC from notebook run):
   - Logistic Regression: Train ≈ 0.9837, Val ≈ 0.9831 (convergence warning; could increase `max_iter`)
   - XGBoost: Train ≈ 1.0000, Val ≈ 0.9992 (best)
   - Random Forest: Train ≈ 1.0000, Val ≈ 0.9650
7. Evaluation:
   - Confusion matrix plotted for the XGBoost model on the test set

### Notes and Tips
- Class Imbalance: Fraud classes are typically highly imbalanced. ROC AUC is robust, but consider additional metrics (PR AUC, recall on fraud class, cost-weighted metrics) and techniques (class weights, down/up-sampling, anomaly detection).
- Feature Leakage: Dropping `nameOrig` and `nameDest` helps avoid ID leakage; further checks for leakage are recommended before productionizing.
- Thresholding: For deployment, tune the decision threshold based on business costs (false positives vs false negatives).


