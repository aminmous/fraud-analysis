import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score, roc_curve
from sklearn.utils.class_weight import compute_sample_weight

from pulearn import ElkanotoPuClassifier

def load_fraud_csv(filename=str):
    """Load fraud CSV data from a local file, excluding specified columns.

    Parameters:
    - filename (str): Name of the CSV file in the same directory.
    - cols_to_drop (list of str): Column names to exclude from features.

    Assumes:
    - One of the columns in `cols_to_drop` is the label column (e.g. 'diagnosis').
    - The label column contains binary labels: 1 = positive, 0 = negative.
    """
    # Get full path to the CSV file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    filepath = os.path.join(script_dir, filename)

    # Load CSV and drop missing values
    df = pd.read_csv(filepath)
    # df = df.drop(columns=[col for col in df.columns if col.startswith('filing_type_')])
    # df_val_train = df[df['reporting_date_year'] <= 2008]
    # df = df[[col for col in df.columns if col.startswith('sen_emb_')] + ['reporting_date', 'fraudulent']]
    df = df[[col for col in df.columns if col.startswith('mbert_full_cls')] + ['reporting_date', 'fraudulent']]
    df['reporting_date'] = pd.to_datetime(df['reporting_date'])
    df['year'] = df['reporting_date'].dt.year
    df_val_train = df[df['year'] <= 2008]

    # test_df = df[df['reporting_date_year'] == 2011]
    test_df = df[df['year'] == 2011]
    # X_test = test_df.drop('fraudulent', axis=1)
    # X_test = test_df[[col for col in test_df.columns if col.startswith('sen_emb_')]]
    X_test = test_df[[col for col in test_df.columns if col.startswith('mbert_full_cls')]]
    X_test = X_test.astype(float).to_numpy()
    y_test = test_df['fraudulent'].values

    # X = df_val_train.drop(columns=['fraudulent'])
    # X = df_val_train[[col for col in df_val_train.columns if col.startswith('sen_emb_')]]
    X = df_val_train[[col for col in df_val_train.columns if col.startswith('mbert_full_cls')]]
    X = X.astype(float).to_numpy()

    # label column is called 'fraudulent'
    y = df_val_train["fraudulent"].values

    return X, y, X_test, y_test

if __name__ == "__main__":
    np.random.seed(42)

    # filename = "df_tab.csv"
    # filename = "fraud_sen_emb.csv"
    filename = "fraud_mbert_full.csv"
    X, y, _, _, = load_fraud_csv(filename=filename)

    # Shuffle dataset
    permut = np.random.permutation(len(y))
    X = X[permut]
    y = y[permut]
    y[np.where(y == 0)[0]] = -1.0
    y[np.where(y == 1)[0]] = +1.0

    print("Loaded {} examples.".format(len(y)))
    print("{} are non fraudulent.".format(len(np.where(y == -1.0)[0])))
    print("{} are fraudulent.".format(len(np.where(y == +1.0)[0])))

    # logger.info("\nSplitting dataset into test/train sets...")
    print("\nSplitting dataset into test/train sets...")

    split = int(2 * len(y) / 3)

    # Select elements from 0 to split-1 (including both ends)
    X_train = X[:split]
    y_train = y[:split]
    X_val = X[split:]  # Select elements from index select to end.
    y_val = y[split:]

    print("Training set contains {} examples.".format(len(y_train)))
    print("{} are non fraudulent.".format(len(np.where(y_train == -1.0)[0])))
    print("{} are fraudulent.".format(len(np.where(y_train == +1.0)[0])))

    pu_f1_scores = []
    reg_f1_scores = []

    n_sacrifice_iter = range(0, len(np.where(y_train == +1.0)[0]) - 21, 100)

    print(n_sacrifice_iter)
    print(len(n_sacrifice_iter))

    for n_sacrifice in n_sacrifice_iter:
        print("PU transformation in progress...")
        print("Making {} fraudulent examples non fraudulent.".format(n_sacrifice))

        y_train_pu = np.copy(y_train)
        pos = np.where(y_train == +1.0)[0]
        np.random.shuffle(pos)
        sacrifice = pos[:n_sacrifice]
        y_train_pu[sacrifice] = -1.0
        pos = len(np.where(y_train_pu == -1.0)[0])
        unlabelled = len(np.where(y_train_pu == +1.0)[0])

        print("PU transformation applied. We now have:")
        print("{} are non fraudulent.".format(len(np.where(y_train_pu == -1.0)[0])))
        print("{} are fraudulent.".format(len(np.where(y_train_pu == +1.0)[0])))
        print("-------------------")
        print(
            (
                "Fitting PU classifier (using a random forest as an inner "
                "classifier)..."
            )
        )

        estimator = RandomForestClassifier(
            n_estimators=100,
            criterion="gini",
            bootstrap=True,
            n_jobs=1,
            random_state=42,
        )
        
        pu_estimator = ElkanotoPuClassifier(estimator)
        print(pu_estimator)

        pu_estimator.fit(X_train, y_train_pu)
        y_pred = pu_estimator.predict(X_val)
        y_val_pu = np.where(y_val == -1.0, 0.0, y_val)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_val_pu, y_pred, average='macro'
        )
        pu_f1_scores.append(f1_score)

        print("F1 Macro score: {}".format(f1_score))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))

        print("Confusion matrix:")
        cm = confusion_matrix(y_val_pu, y_pred)
        print(cm)

        print("Regular learning (w/ a random forest) in progress...")

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_train)

        estimator = RandomForestClassifier(
            n_estimators=100,
            bootstrap=True,
            n_jobs=1,
            random_state=42,
            criterion="gini",
        )
        estimator.fit(X_train, y_train_pu, sample_weight=sample_weights)
        y_pred = estimator.predict(X_val)
        precision, recall, f1_score, _ = precision_recall_fscore_support(
            y_val, y_pred, average='macro'
        )
        reg_f1_scores.append(f1_score)

        print("F1 Macro score: {}".format(f1_score))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))

        print("Confusion matrix:")
        cm = confusion_matrix(y_val, y_pred)
        print(cm)
        
    plt.title("Random forest with/without PU learning")
    plt.plot(n_sacrifice_iter, pu_f1_scores, label="PU Adapted Random Forest")
    plt.plot(n_sacrifice_iter, reg_f1_scores, label="Random Forest")
    plt.xlabel("Number of positive examples hidden in the unlabeled set")
    plt.ylabel("F1 Macro Score")
    plt.ylim(bottom=0)
    plt.legend()
    # plt.savefig("PU_plot_sen_emb_delay.png", dpi=300, bbox_inches='tight')
    plt.show()

    print("PU learning completed. Results saved to PU_plot_delay.png.")
    print("Retraining with best n_sacrifice...")

    # 1. Find best n_sacrifice (based on validation F1)
    best_index = np.argmax(pu_f1_scores)
    best_n_sacrifice = list(n_sacrifice_iter)[best_index]
    print(f"\nBest n_sacrifice (PU) = {best_n_sacrifice}, validation F1 = {pu_f1_scores[best_index]:.4f}")

    # Reload full training set for retraining
    X, y, X_test, y_test = load_fraud_csv(filename)
    permut = np.random.permutation(len(y))
    X = X[permut]
    y = y[permut]
    y[np.where(y == 0)[0]] = -1.0
    y[np.where(y == 1)[0]] = +1.0
    y_test[np.where(y_test == 0)[0]] = -1.0
    y_test[np.where(y_test == 1)[0]] = +1.0

    # Apply best PU sacrifice
    y_train_pu = np.copy(y)
    pos = np.where(y == +1.0)[0]
    np.random.shuffle(pos)
    sacrifice = pos[:best_n_sacrifice]
    y_train_pu[sacrifice] = -1.0

    # Retrain PU classifier
    pu_final = ElkanotoPuClassifier(RandomForestClassifier(
        n_estimators=100,
        bootstrap=True,
        n_jobs=1,
        random_state=42,
        criterion="gini",
    ))
    pu_final.fit(X, y_train_pu)

    # Predict on test set
    y_test_pu = np.where(y_test == -1.0, 0.0, y_test)
    y_pred_test_pu = pu_final.predict(X_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test_pu, y_pred_test_pu, average='macro'
    )
    print("\n[PU Test Set Evaluation]")
    print("F1 Macro score: {}".format(f1_score))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Classification Report:")
    print(classification_report(y_test_pu, y_pred_test_pu, digits=2))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_pu, y_pred_test_pu))

    y_scores_pu = pu_final.predict_proba(X_test)[:, 1] 
    auc_score = roc_auc_score(y_test_pu, y_scores_pu)
    print("ROC AUC Score: {:.4f}".format(auc_score))
    fpr, tpr, _ = roc_curve(y_test_pu, y_scores_pu)
    plt.plot(fpr, tpr, label=f"PU Random Forest (AUC={auc_score:.2f})")

    # Retrain regular RF
    reg_final = RandomForestClassifier(
        n_estimators=100,
        bootstrap=True,
        n_jobs=1,
        random_state=42,
        criterion="gini",
        class_weight="balanced",
    )
    reg_final.fit(X, y_train_pu, sample_weight=compute_sample_weight(class_weight="balanced", y=y))
    y_pred_test_reg = reg_final.predict(X_test)

    print("\n[Regular RF Test Set Evaluation]")
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        y_test, y_pred_test_reg, average='macro'
    )
    print("F1 Macro score: {}".format(f1_score))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Classification Report:")
    print(classification_report(y_test, y_pred_test_reg, digits=2))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred_test_reg))

    y_scores_reg = reg_final.predict_proba(X_test)[:, 1]
    auc_score_reg = roc_auc_score(y_test, y_scores_reg)
    print("ROC AUC Score: {:.4f}".format(auc_score_reg))
    fpr_reg, tpr_reg, _ = roc_curve(y_test, y_scores_reg)
    plt.plot(fpr_reg, tpr_reg, label=f"Regular Random Forest (AUC={auc_score_reg:.2f})")

    plt.plot([0,1],[0,1],'k--',alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves Comparison")
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()