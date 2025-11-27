# Diabetes 2 Analysis & Modeling

Project: Exploratory analysis, feature engineering, model training and hyperparameter tuning for a diabetes classification problem using the Pima Indians Diabetes dataset.

## Contents

- diabetes-2-analysis-model.ipynb — Jupyter notebook with full workflow:
  - Data loading & cleaning
  - EDA & visualizations (saved to `visuals/`)
  - Outlier handling (class-wise winsorization)
  - Feature engineering & scaling
  - Train/test split and SMOTE balancing
  - Model training: Logistic Regression, Random Forest
  - Hyperparameter tuning with GridSearchCV
  - Final evaluation & plots (ROC, feature importance, confusion matrix)
- visuals/ — directory created by the notebook; contains generated PNG plots

## Dataset

This project expects the diabetes CSV file used in the notebook (original code referenced `diabetes.csv`). Place your dataset at a reachable path and update the notebook cell that reads the CSV if necessary.

Typical dataset columns:

- Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome

## Requirements

Recommended Python environment (tested with Python 3.8+). Key packages:

- numpy
- pandas
- seaborn
- matplotlib
- scikit-learn
- imbalanced-learn

Install with pip:

```bash
pip install numpy pandas seaborn matplotlib scikit-learn imbalanced-learn
```

## How to run

1. Open the project folder in JupyterLab / Jupyter Notebook or VS Code.
2. Ensure the dataset path in the notebook is correct:
   - Edit the cell that does `pd.read_csv(...)` to point to your local `diabetes.csv`.
3. Run cells sequentially. The notebook creates a `visuals` folder and saves figures there.
4. For reproducible results, set the seed(s) in the notebook as required (the notebook uses `random_state=42` in several places).

## Notebook notes & tips

- Data cleaning:
  - Zeros in several columns are replaced with medians (per-class for Insulin).
  - Several zero-value columns are corrected using medians to reduce bias.
- Outliers:
  - Winsorization is applied class-wise using 1.5\*IQR bounds.
- Feature engineering:
  - Standard scaling applied to numeric features.
  - Categorical features (BMI_Category, Age_Group) are one-hot encoded.
  - New interaction / ratio features are created (GIR, BFI, BMI_Age, etc.).
- Imbalanced data:
  - SMOTE is used in the notebook to oversample the minority class for training.
- Models:
  - Logistic Regression and Random Forest are trained; class weights are computed and used.
  - GridSearchCV used to tune both models for ROC-AUC with 5-fold CV — can be time-consuming.

## Reproducibility & performance

- Hyperparameter grid search can be computationally heavy; reduce `param_grid` or CV folds to speed up during development.
- If you want to persist trained models, consider using `joblib.dump()` to save the final estimator(s) and `joblib.load()` to load them later.

## Output artifacts

- ROC curve: `visuals/ROC Curve Visualization.png`
- Confusion matrix(s): `visuals/Confusion Matrix — Final Model.png`
- Feature importance: `visuals/Random Forest Feature Importance.png`
- Distribution / EDA plots saved in `visuals/`


## Contributing

- Fork, create a branch, make changes, and open a pull request.
- Keep changes focused (e.g., add a preprocessing module, add tests, or refactor notebook into scripts).

## License

This repository is provided under the MIT License. Update license as needed.

## Contact

For issues or questions, open an issue on the GitHub repository after uploading.
