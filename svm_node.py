import time
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_svm(X_train_tfidf, y_train, X_test_tfidf, y_test, params=None):
    if params is None: params = {}

    c_val = params.get('C', 1.0)
    print(f"[Node SVM] Entrenando con C={c_val}...")

    model = LinearSVC(C=c_val, random_state=42)

    start_time = time.time()
    model.fit(X_train_tfidf, y_train)
    end_time = time.time()

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()

    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model_name": f"SVM (C={c_val})",
        "accuracy": acc,
        "training_time": end_time - start_time,
        "confusion_matrix": cm,
        "report": report
    }
