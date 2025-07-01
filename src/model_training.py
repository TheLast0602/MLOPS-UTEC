from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_and_evaluate(X_train_imputed, X_test_imputed, y_train, y_test):
    # Entrenar el modelo
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train_imputed, y_train)
    
    # Evaluar el modelo
    y_pred = model.predict(X_test_imputed)
    print(classification_report(y_test, y_pred))
    return model
