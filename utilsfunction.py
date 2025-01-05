from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve, average_precision_score, classification_report, f1_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from imblearn.under_sampling import TomekLinks
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Remove the time column and normalise the data
def preprocess_data(data):
    scaler = StandardScaler()
    data = data.drop(columns=['Time'])
    labels = data['Class']
    features = data.drop(columns=['Class'])
    
    # Normalisation
    features = scaler.fit_transform(features)
    return features, labels

# Train the model with the classic supervised method
def train__supervised_classic(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1] 
    return (y_pred, y_scores)

# Train the model with the supervised method with resampling
def train_supervised_imbalanced(method, model, X_train, X_test, y_train, y_test):
    X_resampled, y_resampled = method.fit_resample(X_train, y_train)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)[:, 1]
    return (y_pred, y_scores)

# Print the metrics (ROC AUC, Average precision score, Classifier report)
def print_metrics(y_test, y_pred):
    print("ROC AUC score : ", roc_auc_score(y_test, y_pred))
    print("Average precision score : ", average_precision_score(y_test, y_pred))
    print('Classifier report : \n', classification_report(y_test, y_pred))
    

# Train and evaluate the isolation forest model
def train_and_evaluate_isforest(X_train, X_test, y_train, y_test, iso_forest):
    iso_forest.fit(X_train)
    iso_forest_scores = -iso_forest.decision_function(X_test)

    # Retourner les scores et les labels
    return (y_test, iso_forest_scores)

# Train and evaluate the local outlier factor model
def train_and_evaluate_lof(X_train, X_test, y_train, y_test, n_neighbors):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=0.05, novelty=True)
    lof.fit(X_train)
    lof_scores = lof.decision_function(X_test)

    # Retourner les scores et les labels
    return (y_test, lof_scores)



# Plot the ROC and Precision-Recall curves
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

# Evaluate the model with cross-validation
def evaluate_model_cross_validation(model, X, y, model_name, use_resampling=False, resampler=None, n_splits=5):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    roc_auc_scores = []
    pr_auc_scores = []
    
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        # Resampling if needed
        if use_resampling and resampler is not None and model_name != "Isolation Forest" and model_name != "Local Outlier Factor":
            X_train, y_train = resampler.fit_resample(X_train, y_train)
        
        # Train the model
        model.fit(X_train, y_train)
        if model_name == "Local Outlier Factor" or model_name == "Isolation Forest":
            y_scores = -model.decision_function(X_test)
        else:
            y_scores = model.predict_proba(X_test)[:, 1]
        
        # Calculate the metrics
        roc_auc = roc_auc_score(y_test, y_scores)
        precision, recall, _ = precision_recall_curve(y_test, y_scores)
        pr_auc = auc(recall, precision)
        
        roc_auc_scores.append(roc_auc)
        pr_auc_scores.append(pr_auc)
    
    return {
        "Model": model_name,
        "ROC AUC": np.mean(roc_auc_scores),
        "PR AUC": np.mean(pr_auc_scores),
        "ROC AUC Std": np.std(roc_auc_scores),
        "PR AUC Std": np.std(pr_auc_scores),
    }


###### FONCTIONS POUR KDDCUP99 ########

# Preprocess the data for the KDD99 dataset
def preprocess_data2(data):
    # One-hot encoding
    data = data.drop(columns=['duration'])
    labels = data['label']
    features = data.drop(columns=['label'])
    features = pd.get_dummies(features, columns=['protocol_type', 'service', 'flag'])
    
    # Normalisation
    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    # Labels
    encoder = LabelEncoder()
    labels = encoder.fit_transform(labels)

    return features, labels

# Train the model with the classic supervised method for multiple classes
def train__supervised_classic_multiple_classes(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    return (y_pred, y_scores)

# Train the model with the supervised method with resampling for multiple classes
def train_supervised_imbalanced_multiple_classes(method, model, X_train, X_test, y_train, y_test):
    X_resampled, y_resampled = method.fit_resample(X_train, y_train)
    model.fit(X_resampled, y_resampled)
    y_pred = model.predict(X_test)
    y_scores = model.predict_proba(X_test)
    return (y_pred, y_scores)

# Print the metrics (ROC AUC, Average precision score, Classifier report) for multiple classes
def print_metrics_multiple_classes(y_test, y_pred, y_scores):
    print("ROC AUC score : ", roc_auc_score(y_test, y_scores, multi_class='ovr'))
    print("Average precision score : ", average_precision_score(y_test, y_scores, average='weighted'))
    print('Classifier report : \n', classification_report(y_test, y_pred))

def methodological_approach(X_train, X_test, y_train, y_test):
    methods_roc = []
    methods_pr = []

    # Ravel the labels to avoid warnings
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # Approche supervisée classique : Régression logistique
    model = LogisticRegression(random_state=42, max_iter=1000)
    y_pred, y_scores = train__supervised_classic_multiple_classes(model, X_train, X_test, y_train, y_test)
    methods_roc.append(roc_auc_score(y_test, y_scores, multi_class='ovr'))
    methods_pr.append(average_precision_score(y_test, y_scores, average='weighted'))

    # Approche supervisée classique : SVM
    model = SVC(probability=True, random_state=42)
    y_pred, y_scores = train__supervised_classic_multiple_classes(model, X_train, X_test, y_train, y_test)
    methods_roc.append(roc_auc_score(y_test, y_scores, multi_class='ovr'))
    methods_pr.append(average_precision_score(y_test, y_scores, average='weighted'))

    # Undersampling : Tomek Links
    model = RandomForestClassifier(random_state=42)
    method = TomekLinks()
    y_pred, y_scores = train_supervised_imbalanced_multiple_classes(method, model, X_train, X_test, y_train, y_test)
    methods_roc.append(roc_auc_score(y_test, y_scores, multi_class='ovr'))
    methods_pr.append(average_precision_score(y_test, y_scores, average='weighted'))

    # Oversampling : SMOTE
    model = RandomForestClassifier(random_state=42)
    method = SMOTE(k_neighbors=4)
    y_pred, y_scores = train_supervised_imbalanced_multiple_classes(method, model, X_train, X_test, y_train, y_test)
    methods_roc.append(roc_auc_score(y_test, y_scores, multi_class='ovr'))
    methods_pr.append(average_precision_score(y_test, y_scores, average='weighted'))

    # Balanced Random Forest
    model = BalancedRandomForestClassifier(random_state=42)
    y_pred, y_scores = train__supervised_classic_multiple_classes(model, X_train, X_test, y_train, y_test)
    methods_roc.append(roc_auc_score(y_test, y_scores, multi_class='ovr'))
    methods_pr.append(average_precision_score(y_test, y_scores, average='weighted'))

    # Isolation Forest
    model = IsolationForest(random_state=42)
    y_true, iso_forest_scores = train_and_evaluate_isforest(X_train, X_test, y_train, y_test, model)
    methods_roc.append(roc_auc_score(y_true, iso_forest_scores))
    methods_pr.append(average_precision_score(y_true, iso_forest_scores))

    # Local Outlier Factor
    model = LocalOutlierFactor(novelty=True)
    y_true, lof_scores = train_and_evaluate_lof(X_train, X_test, y_train, y_test, 20)
    methods_roc.append(roc_auc_score(y_true, lof_scores))
    methods_pr.append(average_precision_score(y_true, lof_scores))

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.bar(["Logistic Regression", "SVM", "Random Forest (Tomek Links)", "Random Forest (SMOTE)", "Balanced Random Forest", "Isolation Forest", "Local Outlier Factor"], methods_roc, label="ROC AUC")
    plt.bar(["Logistic Regression", "SVM", "Random Forest (Tomek Links)", "Random Forest (SMOTE)", "Balanced Random Forest", "Isolation Forest", "Local Outlier Factor"], methods_pr, label="Average Precision")
    plt.xlabel("Méthode")
    plt.ylabel("Score")
    plt.title("Scores des différentes méthodes")
    plt.legend()
    plt.show()


###### FONCTIONS POUR NOVELTY DETECTION ########
def evaluate_model(y_true, scores, preds, model_name):
    roc_auc = roc_auc_score(y_true, scores)
    precision, recall, _ = precision_recall_curve(y_true, scores)
    pr_auc = auc(recall, precision)
    f1 = f1_score(y_true, preds)

    print(f"--- {model_name} ---")
    print(f"ROC AUC: {roc_auc:.4f}")
    print(f"PR AUC: {pr_auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", classification_report(y_true, preds))