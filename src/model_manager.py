import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

MODELS_DIR = "models"

def train_and_export_artifacts(df_raw: pd.DataFrame):
    """Trains all models, computes evaluation metrics, and saves binary artifacts."""
    from src.preprocessing import build_feature_pipeline, SCALE_COLS
    
    os.makedirs(MODELS_DIR, exist_ok=True)
    X, y = build_feature_pipeline(df_raw)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Fit Scaler strictly on X_train
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test_scaled[SCALE_COLS] = scaler.transform(X_test[SCALE_COLS])
    
    # Hyperparameter Tuning
    log_grid = {'C': [0.1, 1, 10], 'solver': ['liblinear', 'lbfgs']}
    log_search = GridSearchCV(LogisticRegression(max_iter=1000, random_state=42), log_grid, cv=5, scoring='f1', n_jobs=-1)
    log_search.fit(X_train_scaled, y_train)
    
    dt_grid = {'max_depth': [6, 8, 10], 'min_samples_leaf': [2, 4]}
    dt_search = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_grid, cv=5, scoring='f1', n_jobs=-1)
    dt_search.fit(X_train_scaled, y_train)
    
    rf_grid = {'n_estimators': [100, 200], 'max_depth': [10, 15]}
    rf_search = GridSearchCV(RandomForestClassifier(random_state=42), rf_grid, cv=5, scoring='f1', n_jobs=-1)
    rf_search.fit(X_train_scaled, y_train)
    
    hgb_model = HistGradientBoostingClassifier(random_state=42)
    hgb_model.fit(X_train_scaled, y_train)
    
    models = {
        'Logistic Regression': log_search.best_estimator_,
        'Decision Tree': dt_search.best_estimator_,
        'Random Forest': rf_search.best_estimator_,
        'HistGradientBoosting': hgb_model
    }
    
    # Calculate Benchmarks & Confusion Matrices
    metrics = []
    cm_dict = {}
    for name, model in models.items():
        preds = model.predict(X_test_scaled)
        metrics.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, preds),
            'Precision': precision_score(y_test, preds),
            'Recall': recall_score(y_test, preds),
            'F1-Score': f1_score(y_test, preds)
        })
        cm_dict[name] = confusion_matrix(y_test, preds).tolist()
        
    metrics_df = pd.DataFrame(metrics).sort_values(by='F1-Score', ascending=False)
    
    # Save Artifacts
    joblib.dump(models, os.path.join(MODELS_DIR, "all_models.pkl"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))
    joblib.dump(list(X.columns), os.path.join(MODELS_DIR, "feature_columns.pkl"))
    joblib.dump({'metrics': metrics_df, 'confusion_matrices': cm_dict}, os.path.join(MODELS_DIR, "model_metrics.pkl"))
    
    print("All artifacts successfully exported to models/")

def load_artifacts():
    """Loads exported models, scalers, and metric artifacts."""
    all_models = joblib.load(os.path.join(MODELS_DIR, "all_models.pkl"))
    scaler = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_columns = joblib.load(os.path.join(MODELS_DIR, "feature_columns.pkl"))
    metrics_data = joblib.load(os.path.join(MODELS_DIR, "model_metrics.pkl"))
    return all_models, scaler, feature_columns, metrics_data