import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_nb(X_train_tfidf, y_train, X_test_tfidf, y_test, params=None):
    if params is None: params = {}

    # Obtenemos alpha (suavizado), por defecto 1.0
    alpha_val = params.get('alpha', 1.0)
    print(f"[Node NB] Entrenando con Alpha={alpha_val}...")

    model = MultinomialNB(alpha=alpha_val)

    start_time = time.time()
    model.fit(X_train_tfidf, y_train)
    end_time = time.time()

    y_pred = model.predict(X_test_tfidf)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    return {
        "model_name": f"Naive Bayes (a={alpha_val})",
        "accuracy": acc,
        "training_time": end_time - start_time,
        "confusion_matrix": cm,
        "report": report
    }
