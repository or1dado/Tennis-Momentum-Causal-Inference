from sklearn.metrics import  f1_score, precision_score, recall_score, accuracy_score
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def plot_calibration_curve(true_t:list, t_preds_dict: dict):
    """
    Plot the calibration curves for multiple models.

    Parameters:
    true_t (list): List of true binary labels.
    t_preds_dict (dict): Dictionary where keys are model names and values are predicted probabilities.
    """
    
    plt.figure(figsize=(8, 6))  
    for model_name, t_pred in t_preds_dict.items():
        assert len(true_t) == len(t_pred), f"Length mismatch: {len(true_t)} true labels and {len(t_pred)} predictions for {model_name}"
        
        fraction_of_positives, mean_predicted_value = calibration_curve(true_t, t_pred, n_bins=15)
        plt.plot(mean_predicted_value, fraction_of_positives, marker='o', label=f'{model_name} Calibration plot')

    # Perfectly calibrated line
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly calibrated')

    # Labels and Title
    plt.xlabel('Mean predicted value')
    plt.ylabel('Fraction of positives')
    plt.title('Calibration Curve - All Data')
    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.show()



def plot_overlap(t, propensity_score, model_name=''):
    t = t.to_numpy()
    propensity_score = pd.Series(propensity_score).values
    propensity_score_df = pd.DataFrame({'t': t,  'propensity_score': propensity_score})

    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))

    sns.histplot(data=propensity_score_df, 
                 x='propensity_score', 
                 hue='t', 
                 bins=20, 
                 alpha=0.5, 
                 kde=True, 
                 stat='count', 
                 palette={0: 'red', 1: 'blue'}, 
                 element='bars'
                 )

    plt.legend(title='Group', labels=['Treated', 'Control'])
    plt.xlabel('Propensity Score')
    plt.ylabel('Number of Units')
    plt.title(f'{model_name} - Overlap of Propensity Scores')
    plt.show()