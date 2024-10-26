from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from utils import plot_calibration_curve, plot_overlap
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, cv, Pool
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import pandas as pd
import numpy as np
import optuna


def estimate_e(df, model, model_name, test_size=0.2, random_state=42):
    """
    Estimate model performance and fine-tune parameters on the provided DataFrame.

    Parameters:
    - df: DataFrame containing the data.
    - model: The model to be trained.
    - model_name: The name of the model.
    - test_size: Proportion of the dataset to include in the test split.
    - random_state: Random seed for reproducibility.

    Returns:
    - y_test: True labels for the test set.
    - y_pred_proba: Predicted probabilities for the positive class.
    """
    print("Estimate Propepnsity Score:")
    categorical_cols = [col for col in df.columns if df[col].dtype == 'object']

    X = df.drop(columns=['T', 'Y'])
    y = df['T']

    if model_name in ['lr', 'rf', 'xgb']:
        dummied_df = pd.get_dummies(X[categorical_cols], drop_first=True)
        X = pd.concat([X.drop(categorical_cols, axis=1), dummied_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    if model_name == 'lr':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    if model_name == 'cat':
        model.fit(X_train, y_train, cat_features=categorical_cols)
    else:
        model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    metrics = {
        'AUC-ROC': roc_auc_score(y_test, y_pred_proba),
        'F1 Score': f1_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'Accuracy': accuracy_score(y_test, y_pred),
    }

    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    return y_test, y_pred_proba


def objective(trial, X_train, y_train, model_name='rf', random_state=42):
    """
    Objective function for hyperparameter optimization using Optuna.

    Parameters:
    - trial: Optuna trial object.
    - X_train: Training features.
    - y_train: Training labels.
    - model_name: Name of the model to optimize.
    - random_state: Random seed for reproducibility.

    Returns:
    - accuracy: Mean accuracy from cross-validation.
    """
    model_params = {}
    if model_name == 'lr':
        model_params['C'] = trial.suggest_float('C', 0.00001, 10.0)
        model_params['solver'] = trial.suggest_categorical('solver', ['liblinear', 'lbfgs', 'saga'])
        model = LogisticRegression(random_state=random_state, **model_params)
    
    elif model_name == 'rf':
        model_params['n_estimators'] = trial.suggest_int('n_estimators', 10, 200)
        model_params['max_depth'] = trial.suggest_int('max_depth', 1, 50)
        model_params['min_samples_split'] = trial.suggest_int('min_samples_split', 2, 20)
        model_params['min_samples_leaf'] = trial.suggest_int('min_samples_leaf', 1, 20)
        model = RandomForestClassifier(random_state=random_state, n_jobs=-1, **model_params)
    
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
    
    elif model_name == 'cat':
        model_params = {
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'loss_function': 'MultiClass',
            'eval_metric': 'Accuracy',
            'random_seed': random_state,
            'verbose': 0,
        }
        model = CatBoostClassifier(**model_params)
        categorical_cols = [col for col in X_train.columns if X_train[col].dtype == 'object']
        train_pool = Pool(X_train, y_train, cat_features=categorical_cols)
        
        # Use StratifiedKFold for manual cross-validation
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        accuracy_scores = []

        for train_idx, valid_idx in skf.split(X_train, y_train):
            train_data = Pool(X_train.iloc[train_idx], y_train.iloc[train_idx], cat_features=categorical_cols)
            valid_data = Pool(X_train.iloc[valid_idx], y_train.iloc[valid_idx], cat_features=categorical_cols)

            model = CatBoostClassifier(**model_params)
            model.fit(train_data, eval_set=valid_data, verbose=0)
            preds = model.predict(valid_data)
            accuracy = accuracy_score(y_train.iloc[valid_idx], preds)
            accuracy_scores.append(accuracy)

        return np.mean(accuracy_scores)

    if model_name != 'cat':
        skf = StratifiedKFold(n_splits=5)
        scores = cross_val_score(model, X_train, y_train, cv=skf, scoring='accuracy') 

    return np.mean(scores)


def fine_tune_model(df, model_name, n_trials=100, test_size=0.2):
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
    X = df.drop(columns=['T', 'Y'])
    y = df['T']

    categorical_cols = [col for col in X.columns if X[col].dtype == 'object']
    if model_name in ['lr', 'rf', 'xgb']:
        dummied_df = pd.get_dummies(X[categorical_cols], drop_first=True)
        X = pd.concat([X.drop(categorical_cols, axis=1), dummied_df], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    if model_name == 'lr':
        scaler = MinMaxScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train, model_name), n_trials=n_trials)

    print("Best hyperparameters: ", study.best_params)
    print("Best Accuracy score: ", study.best_value)

    best_params = study.best_params
    if model_name == 'lr':
        final_model = LogisticRegression(**best_params, random_state=42)
    elif model_name == 'rf':
        final_model = RandomForestClassifier(**best_params, random_state=42)
    elif model_name == 'xgb':
        final_model = xgb.XGBClassifier(**best_params, random_state=42)
    elif model_name == 'cat':
        final_model = CatBoostClassifier(**best_params, random_state=42, verbose=0)
    
    if model_name == 'cat':
        final_model.fit(X_train, y_train, cat_features=categorical_cols)
    else:
        final_model.fit(X_train, y_train)
    
    y_pred = final_model.predict(X_test)
    test_metrics = {
        'Test Accuracy': accuracy_score(y_test, y_pred),
        'Test F1 Score': f1_score(y_test, y_pred, average='weighted'),
        'Test Precision': precision_score(y_test, y_pred, average='weighted'),
        'Test Recall': recall_score(y_test, y_pred, average='weighted'),
    }

    for metric, score in test_metrics.items():
        print(f"{metric}: {score:.4f}")

    return best_params, final_model


if __name__ == "__main__":
    df = pd.read_csv('Final_Data/feature_df.csv', index_col=0)
    df['surface'] = df['surface'].astype(str)
    best_params, final_model = fine_tune_model(df, 'cat', n_trials=1)
    real_T, propensity_score = estimate_e(df, final_model, 'cat')
    plot_calibration_curve(real_T, {'CatBoost':propensity_score})
    plot_overlap(real_T, propensity_score)
    print("-" * 50)
