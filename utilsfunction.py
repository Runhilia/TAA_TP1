from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, average_precision_score, classification_report
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

def preprocess_data(data):
    scaler = StandardScaler()
    data = data.drop(columns=['Time'])
    labels = data['Class']
    features = data.drop(columns=['Class'])
    
    
    # Normalisation
    features = scaler.fit_transform(features)
    return features, labels


def train__supervised_classic(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1] 
    return (y_pred, y_scores)

def train_supervised_imbalanced(method, model, X_train, X_test, y_train, y_test):
    X_resampled, y_resampled = method.fit_resample(X_train, y_train)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    return (y_pred, y_scores)

def print_metrics(y_test, y_pred):
    print("ROC AUC score : ", roc_auc_score(y_test, y_pred))
    print("Average precision score : ", average_precision_score(y_test, y_pred))
    print('Classifier report : \n', classification_report(y_test, y_pred))
    

def train_and_evaluate_isforest(X_train, X_test, y_train, y_test):
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso_forest.fit(X_train)
    iso_forest_scores = -iso_forest.decision_function(X_test)

    # Retourner les scores et les labels
    return (y_test, iso_forest_scores)

def train_and_evaluate_lof(X_train, X_test, y_train, y_test, n_neighbors):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05, novelty=True)
    lof.fit(X_train)
    lof_scores = lof.decision_function(X_test)

    # Retourner les scores et les labels
    return (y_test, lof_scores)




def plot_roc_pr_curves(results, model_name):
    """ Affiche les courbes ROC et Précision-Rappel pour chaque modèle """
    y_true, scores = results  # Récupération correcte des résultats

    # Vérifiez les dimensions pour s'assurer que tout est aligné
    assert len(y_true) == len(scores), "Les scores et les labels doivent avoir la même taille."

    # ROC Curve
    roc_auc = roc_auc_score(y_true, scores)
    print(f"ROC AUC ({model_name}): {roc_auc:.4f}")

    # Précision-Rappel
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    print(f"PR AUC ({model_name}): {pr_auc:.4f}")

    # Plotting
    plt.figure(figsize=(12, 6))

    # ROC Curve
    plt.subplot(1, 2, 1)
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.plot(fpr, tpr, label=f'ROC AUC: {roc_auc:.4f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({model_name})')
    plt.legend()

    # Precision-Recall Curve
    plt.subplot(1, 2, 2)
    plt.plot(recall, precision, label=f'PR AUC: {pr_auc:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve ({model_name})')
    plt.legend()

    plt.tight_layout()
    plt.show()
