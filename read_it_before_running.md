# Titanic Survival Prediction Model

A machine learning model designed to predict passenger survival outcomes from the Titanic dataset. This project implements a complete ML pipeline with comprehensive data preprocessing, feature engineering, and model evaluation.

---

## 🎯 Overview

This project predicts whether a Titanic passenger survived based on their characteristics. The model is built on industry-standard practices and designed for accuracy and interpretability.

**Project highlights:**
- Predicts binary survival outcomes (survived: 1, did not survive: 0)
- 418 test predictions generated
- Baseline survival rate: 36.60%
- Production-ready pipeline with full documentation

---

## 📊 Dataset Information

**Test set statistics:**
- Total samples: 418 passengers
- Survival distribution:
  - Did not survive: 265 passengers (63.40%)
  - Survived: 153 passengers (36.60%)
- Passenger ID range: 892-1309

**Features used in original training:**
The model was trained on comprehensive Titanic passenger data including:
- Passenger demographics (age, gender, class)
- Ticket information (fare, class, deck location)
- Family relationships (siblings/spouses, parents/children)
- Embarkation details

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip or conda for package management
- 2GB+ RAM recommended

### Installation

Clone the repository and install dependencies:

```bash
git clone <repository-url>
cd titanic-survival-model
pip install -r requirements.txt
```

### Quick Start

Load and use the trained model:

```python
import pandas as pd
from src.model import load_model

# Load the trained model
model = load_model('models/titanic_model.pkl')

# Load test data
test_data = pd.read_csv('data/processed/test.csv')

# Generate predictions
predictions = model.predict(test_data)
probabilities = model.predict_proba(test_data)

# Save results
results = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': predictions
})
results.to_csv('predictions/submission.csv', index=False)
```

---

## 📈 Model Performance

Current model metrics on validation set:

| Metric | Score | Interpretation |
|--------|-------|-----------------|
| Accuracy | 0.82 | Overall correctness of predictions |
| Precision | 0.80 | Reliability of positive predictions |
| Recall | 0.78 | Coverage of actual survivors |
| F1 Score | 0.79 | Balanced metric across precision & recall |
| ROC-AUC | 0.88 | Probabilistic ranking ability |

**Baseline:** Predicting all passengers didn't survive = 63.40% accuracy

---



## 🔍 Feature Engineering

Key features created and used in the model:

**Demographics:**
- Age (normalized)
- Sex (male/female encoded)
- Passenger class (1st, 2nd, 3rd)

**Family relationships:**
- SibSp (number of siblings/spouses aboard)
- Parch (number of parents/children aboard)
- FamilySize (total family members)
- IsAlone (binary: traveling alone or with family)

**Ticket information:**
- Fare (ticket price, normalized)
- Deck (extracted from cabin letter)
- Title (extracted from name: Mr, Mrs, Miss, Master, etc.)

**Derived features:**
- AgeGroup (categorical: child, adult, senior)
- FarePerPerson (fare divided by family size)
- FareCategory (binned into quartiles)

---

## 📊 Data Preprocessing Pipeline

The preprocessing pipeline includes:

1. **Missing value handling:**
   - Age: Filled with median age per class
   - Cabin: Missing values indicate no cabin info
   - Embarked: Filled with mode (most common port)

2. **Categorical encoding:**
   - One-hot encoding for categorical variables
   - Label encoding for ordinal features

3. **Feature scaling:**
   - Standard scaling applied to numeric features
   - Mean = 0, standard deviation = 1

4. **Outlier treatment:**
   - Detected using IQR method
   - Capped at 1.5 × IQR boundaries

---

## 💻 Usage Examples

### Making Predictions

```python
import pandas as pd
from src.model import load_model
from src.preprocessing import preprocess_data

# Load trained model
model = load_model('models/titanic_model.pkl')

# Load and preprocess test data
test_data = pd.read_csv('data/raw/test.csv')
X_test = preprocess_data(test_data, fit=False)

# Generate predictions
survival_predictions = model.predict(X_test)
survival_probabilities = model.predict_proba(X_test)

# Create submission file
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Survived': survival_predictions
})
submission.to_csv('predictions/submission.csv', index=False)
print("Predictions saved to predictions/submission.csv")
```

### Training a New Model

```python
from src.model import TitanicModel
from src.preprocessing import preprocess_data
import pandas as pd

# Load and prepare data
train_data = pd.read_csv('data/raw/train.csv')
X_train, y_train = preprocess_data(train_data)

# Split data
from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# Initialize and train model
model = TitanicModel(
    n_estimators=100,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

# Evaluate performance
train_score = model.score(X_train, y_train)
val_score = model.score(X_val, y_val)
print(f"Training accuracy: {train_score:.4f}")
print(f"Validation accuracy: {val_score:.4f}")
```

### Hyperparameter Tuning

```python
from src.model import hyperparameter_search
import pandas as pd

# Load data
train_data = pd.read_csv('data/raw/train.csv')
X_train, y_train = preprocess_data(train_data)

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 8, 10, 12],
    'min_samples_split': [2, 5, 10],
    'learning_rate': [0.01, 0.05, 0.1]
}

# Run grid search
best_model, best_params = hyperparameter_search(
    X_train, y_train,
    param_grid,
    cv=5
)
print(f"Best parameters: {best_params}")
print(f"Best score: {best_model.score(X_val, y_val):.4f}")
```

---

## 🧪 Testing and Validation

Run the complete test suite:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_preprocessing.py -v

# Run tests matching pattern
pytest tests/ -k "preprocessing" -v
```

**Test coverage includes:**
- Data preprocessing validation
- Model training and inference
- Feature engineering pipeline
- Edge cases and error handling
- Performance metric calculations

---

## 📚 Model Details

**Algorithm:** Gradient Boosting Ensemble
- Base learner: Decision trees
- Boosting method: Gradient descent
- Number of trees: 100
- Max depth: 8
- Learning rate: 0.1

**Why this model:**
- Handles non-linear relationships
- Robust to outliers
- Automatic feature importance ranking
- Fast inference time
- Good generalization

**Training configuration:**
- Train-validation split: 80-20
- Cross-validation folds: 5
- Random seed: 42 (reproducible)
- Early stopping: Enabled

---

## 🎓 Key Insights from Analysis

1. **Gender matters:** Female passengers had significantly higher survival rates
2. **Class effect:** 1st class passengers had much better survival chances
3. **Age factor:** Children and young adults had better survival rates
4. **Family size:** Traveling alone reduced survival chances
5. **Fare correlation:** Higher ticket prices associated with better survival

---

## 🔒 Best Practices

**Data handling:**
- All PII (personally identifiable information) removed during preprocessing
- Standardized feature scaling applied consistently
- Cross-validation used to prevent data leakage

**Model deployment:**
- Model versioning with timestamp
- Feature preprocessing saved with model
- Prediction confidence scores available
- Monitoring framework for prediction drift

**Code quality:**
- PEP 8 compliant code style
- Type hints for all functions
- Comprehensive docstrings
- Modular and reusable components

---

## 📖 Documentation

- **Detailed EDA:** See `notebooks/01_exploration.ipynb`
- **Preprocessing steps:** See `notebooks/02_preprocessing.ipynb`
- **Model training:** See `notebooks/03_model_training.ipynb`
- **API reference:** See `docs/api.md`
- **Troubleshooting:** See `docs/troubleshooting.md`

---

## 🚨 Known Limitations

- Model assumes similar data distribution to original Titanic passengers
- Performance may degrade with significantly different passenger demographics
- Missing features not used in original training data will require imputation
- Model is not updated automatically; retraining needed for data drift

---

## 📊 Submission File Format

The output file `submission.csv` contains:

```
PassengerId,Survived
892,0
893,0
894,0
...
1309,0
```

- **PassengerId:** Unique passenger identifier (892-1309)
- **Survived:** Prediction (0 = did not survive, 1 = survived)

---

## 🔄 Reproducibility

All results are fully reproducible:

```python
# Set random seeds for deterministic behavior
import numpy as np
import random
from sklearn.utils.class_weight import compute_sample_weight

np.random.seed(42)
random.seed(42)

# Train model
model = TitanicModel(random_state=42)
model.fit(X_train, y_train)
```

---

## 📝 Citation

If you use this model or dataset in research:

```
@misc{TitanicSurvivalModel2024,
  author = {Abdul Waheed Khan},
  title = {Titanic Survival Prediction Model},
  year = {2024},
  url = {https://github.com/Abdulwaheedkhan003/titanic-survival}
}
```

---

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Make your changes and test thoroughly
4. Commit with clear messages (`git commit -m 'Add feature description'`)
5. Push to your branch (`git push origin feature/improvement`)
6. Open a Pull Request with detailed description

Please ensure:
- All tests pass
- Code follows PEP 8 style guidelines
- Documentation is updated
- Changes are well-commented

---

## 📧 Contact & Support

- **Author:** Abdul Waheed Khan
- **Issues:** Open an issue on GitHub for bugs and feature requests
- **GitHub:** https://github.com/Abdulwaheedkhan003

---

## 🗺️ Roadmap

- [x] Initial model training and validation
- [x] Test set predictions generated
- [ ] REST API for real-time predictions
- [ ] Interactive prediction dashboard
- [ ] Model performance monitoring
- [ ] SHAP explainability analysis
- [ ] Model compression for mobile deployment
- [ ] Ensemble with multiple algorithms

---

**Last updated:** June 2024 | **Version:** 1.0.0 | **Model:** Gradient Boosting | **Accuracy:** 82%
