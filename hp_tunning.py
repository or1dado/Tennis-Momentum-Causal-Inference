from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from utils import plot_calibration_curve, plot_overlap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler
from functools import partial
import xgboost as xgb
import pandas as pd
import numpy as np
import optuna
import pickle


def objective(trial, X, y, model_name='rf', random_state=42):
    """
    Objective function for hyperparameter optimization using Optuna.

    Parameters:
    - trial: Optuna trial object.
    - X: Training features.
    - y: Training labels.
    - model_name: Name of the model to optimize.
    - random_state: Random seed for reproducibility.

    Returns:
    - accuracy: Mean accuracy from cross-validation.
    """
    model_params = {}
    if model_name == 'lr':
        model_params['max_iter'] = 10000
        model_params['C'] = trial.suggest_float('C', 0.00001, 10.0)
        model_params['solver'] = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga'])
        model = LogisticRegression(random_state=random_state, **model_params)
    
    elif model_name == 'rf':
        model_params['n_estimators'] = trial.suggest_int('n_estimators', 10, 200)
        model_params['max_depth'] = trial.suggest_int('max_depth', 1, 200)
        model_params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 50)
        model_params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 50)
        model = RandomForestClassifier(random_state=random_state, n_jobs=-1, **model_params)
    
    elif model_name == 'gb':
        model_params['n_estimators'] = trial.suggest_int('n_estimators', 10, 100)
        model_params['max_depth'] = trial.suggest_int('max_depth', 1, 100)
        model_params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 50)
        model_params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 50)
        model = GradientBoostingClassifier(random_state=random_state, **model_params)

    elif model_name == 'xgb':
        model_params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 0, 1),
            'reg_lambda': trial.suggest_float('reg_lambda', 0, 1),
        }
        model = xgb.XGBClassifier(enable_categorical=True, eval_metric='logloss', random_state=random_state, **model_params)
    
    skf = StratifiedKFold(n_splits=5)
    scores = cross_val_score(model, X, y, cv=skf, scoring='f1') 

    return np.mean(scores)


def fine_tune_model(df, model_name, model_class, n_trials=100):
    """
    Fine-tune the specified model using Optuna for hyperparameter optimization.

    Parameters:
    - df: DataFrame containing the data.
    - model_name: Name of the model to fine-tune.
    - n_trials: Number of trials for optimization.
    - test_size: Proportion of the dataset to include in the test split.

    Returns:
    - best_params: Best hyperparameters found.
    - final_model: Model trained with the best parameters.
    """
    X = df.drop(columns=['Y'])  
    y = df['Y']

    if model_name in ['lr', 'rf', 'xgb', 'gb']:
        categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
        dummied_df = pd.get_dummies(X[categorical_cols], prefix=categorical_cols).astype(int)
        X = pd.concat([X.drop(categorical_cols, axis=1), dummied_df], axis=1)
    
    scaler = None
    if model_name == 'lr':
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y, model_name), n_trials=n_trials)

    print("Best hyperparameters: ", study.best_params)
    print("Best F1 score: ", study.best_value)

    best_params = study.best_params
    final_model = model_class(**best_params)
    final_model.fit(X, y)

    return best_params, final_model, scaler


if __name__ == "__main__":
    df = pd.read_csv('Final_Data/full_feature_df.csv', index_col=0)

    model_classes = {
        'lr': partial(LogisticRegression),
        'rf': partial(RandomForestClassifier),
        'gb': partial(GradientBoostingClassifier),
        'xgb': partial(xgb.XGBClassifier),
    }

    models_fine_tuned = {}
    try:
        for model_name, model_class in model_classes.items():
            best_params, trained_model, scaler = fine_tune_model(df, model_name, model_class, n_trials=20)
            models_fine_tuned[model_name] = {'model': trained_model, 'hyper_params': best_params, 'scaler': scaler}
    except Exception as e:
        print(f"Failed: {e}")

    with open('hp_tunning.pkl', 'wb') as f:
        pickle.dump(models_fine_tuned, f)
