# Task 3: Linear Regression

## Objective

Implement and understand Simple and Multiple Linear Regression using Python.

## Tools

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## Dataset

A house-price dataset is included at:

```text
dataset/house_prices.csv
```

Target column: `price`

Features:

- `area`
- `bedrooms`
- `age`

## Workflow

1. Import libraries
2. Load and inspect the dataset
3. Preprocess numerical data
4. Select features and target
5. Split data into training and testing sets
6. Train Multiple Linear Regression
7. Evaluate using MAE, MSE, RMSE and R²
8. Interpret model coefficients
9. Visualize actual vs predicted values
10. Demonstrate Simple Linear Regression with a regression line

## Project Structure

```text
Task-3-Linear-Regression/
│
├── dataset/
│   └── house_prices.csv
│
├── notebooks/
│   └── linear_regression_task3.ipynb
│
├── requirements.txt
├── README.md
└── .gitignore
```

## VS code commands

Create virtual environment:

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Open:

```text
notebooks/linear_regression_task3.ipynb
```

Select the virtual-environment Python kernel and run all cells from top to bottom.

## Evaluation Metrics

- MAE — Mean Absolute Error
- MSE — Mean Squared Error
- RMSE — Root Mean Squared Error
- R² — Coefficient of Determination
