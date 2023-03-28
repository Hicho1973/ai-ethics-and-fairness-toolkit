
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from aif360.datasets import BinaryLabelDataset
from aif360.metrics import BinaryLabelDatasetMetric, ClassificationMetric
from aif360.algorithms.preprocessing import Reweighing
import numpy as np

# --- Configuration --- #
RANDOM_SEED = 42

# --- 1. Generate Synthetic Data --- #
def generate_synthetic_data(num_samples=1000):
    """Generates synthetic data for demonstration of fairness metrics."""
    np.random.seed(RANDOM_SEED)
    data = {
        'feature_1': np.random.rand(num_samples),
        'feature_2': np.random.randint(0, 10, num_samples),
        'sensitive_attr': np.random.choice([0, 1], num_samples, p=[0.6, 0.4]), # 0: majority, 1: minority
        'label': np.random.choice([0, 1], num_samples, p=[0.5, 0.5])
    }
    df = pd.DataFrame(data)
    
    # Introduce bias: minority group has lower chance of positive outcome
    df.loc[df['sensitive_attr'] == 1, 'label'] = np.random.choice([0, 1], df['sensitive_attr'].sum(), p=[0.7, 0.3])
    df.loc[df['sensitive_attr'] == 0, 'label'] = np.random.choice([0, 1], (df['sensitive_attr'] == 0).sum(), p=[0.4, 0.6])
    
    return df

# --- 2. Train a Biased Model --- #
def train_biased_model(df):
    """Trains a simple classifier on the synthetic data."""
    X = df[['feature_1', 'feature_2', 'sensitive_attr']]
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=RANDOM_SEED)
    
    model = RandomForestClassifier(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    print("\n--- Biased Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
    
    return model, X_test, y_test, y_pred

# --- 3. Evaluate Fairness Metrics --- #
def evaluate_fairness(model, X_test, y_test, y_pred, sensitive_attribute=\'sensitive_attr\'):
    """Evaluates various fairness metrics using AIF360."""
    privileged_groups = [{sensitive_attribute: 0}]
    unprivileged_groups = [{sensitive_attribute: 1}]
    
    # Create AIF360 dataset
    aif_data = BinaryLabelDataset(df=pd.concat([X_test, y_test], axis=1),
                                  label_names=['label'],
                                  protected_attribute_names=[sensitive_attribute],
                                  privileged_classes=[[0]], # 0 is privileged
                                  unprivileged_classes=[[1]]) # 1 is unprivileged
    
    # Convert predictions to AIF360 format
    aif_pred = aif_data.copy(deepcopy=True)
    aif_pred.labels = y_pred.reshape(-1, 1)
    
    # Dataset metrics
    metric_dataset = BinaryLabelDatasetMetric(aif_data, 
                                              unprivileged_groups=unprivileged_groups, 
                                              privileged_groups=privileged_groups)
    print("\n--- Dataset Fairness Metrics ---")
    print(f"Mean difference in labels (unprivileged - privileged): {metric_dataset.mean_difference():.4f}")
    
    # Classification metrics
    metric_clf = ClassificationMetric(aif_data, aif_pred,
                                      unprivileged_groups=unprivileged_groups,
                                      privileged_groups=privileged_groups)
    print("\n--- Classification Fairness Metrics ---")
    print(f"Statistical Parity Difference (SPD): {metric_clf.statistical_parity_difference():.4f}")
    print(f"Equal Opportunity Difference (EOD): {metric_clf.equal_opportunity_difference():.4f}")
    print(f"Average Odds Difference (AOD): {metric_clf.average_odds_difference():.4f}")
    print(f"Disparate Impact: {metric_clf.disparate_impact():.4f}")

# --- 4. Mitigate Bias using Reweighing --- #
def mitigate_bias_reweighing(df, sensitive_attribute=\'sensitive_attr\'):
    """Applies reweighing preprocessing technique to mitigate bias."""
    print("\n--- Mitigating Bias with Reweighing ---")
    privileged_groups = [{sensitive_attribute: 0}]
    unprivileged_groups = [{sensitive_attribute: 1}]

    aif_data = BinaryLabelDataset(df=df,
                                  label_names=['label'],
                                  protected_attribute_names=[sensitive_attribute],
                                  privileged_classes=[[0]],
                                  unprivileged_classes=[[1]])

    RW = Reweighing(unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups)
    aif_data_reweighed = RW.fit_transform(aif_data)

    # Train a new model on reweighed data
    X_reweighed = aif_data_reweighed.features
    y_reweighed = aif_data_reweighed.labels.ravel()
    sample_weights = aif_data_reweighed.instance_weights

    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X_reweighed, y_reweighed, sample_weights, test_size=0.3, random_state=RANDOM_SEED)

    model_reweighed = RandomForestClassifier(random_state=RANDOM_SEED)
    model_reweighed.fit(X_train, y_train, sample_weight=weights_train)
    y_pred_reweighed = model_reweighed.predict(X_test)

    print("\n--- Reweighed Model Performance ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_reweighed):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred_reweighed):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred_reweighed):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred_reweighed):.4f}")

    evaluate_fairness(model_reweighed, pd.DataFrame(X_test, columns=X.columns), y_test, y_pred_reweighed, sensitive_attribute)

if __name__ == "__main__":
    data_df = generate_synthetic_data()
    biased_model, X_test_biased, y_test_biased, y_pred_biased = train_biased_model(data_df)
    evaluate_fairness(biased_model, X_test_biased, y_test_biased, y_pred_biased)
    mitigate_bias_reweighing(data_df)
