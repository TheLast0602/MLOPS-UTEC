import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

# Asegúrate de que MLflow esté apuntando a la URI correcta
mlflow.set_tracking_uri('http://localhost:5000')  # Asegúrate de que MLflow esté en ejecución

def train_and_evaluate(X_train_imputed, X_test_imputed, y_train, y_test):
    with mlflow.start_run():  # Inicia un nuevo experimento
        try:
            # Crear el modelo
            model = RandomForestClassifier(random_state=42)

            # Registrar los hiperparámetros
            mlflow.log_param("n_estimators", 200)
            mlflow.log_param("max_depth", 10)

            # Entrenamiento del modelo
            model.fit(X_train_imputed, y_train)

            # Realizar las predicciones
            y_pred = model.predict(X_test_imputed)

            # Imprimir y registrar el reporte de clasificación
            report = classification_report(y_test, y_pred)
            print(report)

            # Registrar el modelo con un nombre específico en el experimento
            mlflow.sklearn.log_model(model, "random_forest_model")  # Este nombre debe coincidir al cargarlo

            # Guardar el reporte de clasificación como un archivo de texto
            with open("classification_report.txt", "w") as f:
                f.write(report)

            # Registrar el reporte de clasificación
            mlflow.log_artifact("classification_report.txt")

            return model
        except Exception as e:
            print(f"Error en el entrenamiento: {e}")
            mlflow.log_param("error", str(e))
