import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder


mlflow.set_tracking_uri('http://localhost:5000')

# Cargar los datos
def load_data():
    train_clientes = pd.read_csv('data/train_clientes_sample.csv')
    train_requerimientos = pd.read_csv('data/train_requerimientos_sample.csv')
    oot_clientes = pd.read_csv('data/oot_clientes_sample.csv')
    oot_requerimientos = pd.read_csv('data/oot_requerimientos_sample.csv')
    return train_clientes, train_requerimientos, oot_clientes, oot_requerimientos

# Cruce de las bases de datos
def merge_datasets(train_clientes, train_requerimientos):
    return pd.merge(train_clientes, train_requerimientos, on="ID_CORRELATIVO", how="inner")

# División del dataset en train y test
def split_data(data):
    X = data.drop(columns=["ATTRITION"])
    y = data["ATTRITION"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Limpieza e imputación de datos
def clean_data(X_train, X_test):
    num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    
    # Verificar si existen las columnas numéricas y categóricas
    if not num_cols.empty:
        num_imputer = SimpleImputer(strategy="median")
        X_train_num_imputed = num_imputer.fit_transform(X_train[num_cols])
        X_test_num_imputed = num_imputer.transform(X_test[num_cols])
    else:
        X_train_num_imputed = X_train
        X_test_num_imputed = X_test

    if not cat_cols.empty:
        cat_imputer = SimpleImputer(strategy="most_frequent")
        X_train_cat_imputed = cat_imputer.fit_transform(X_train[cat_cols])
        X_test_cat_imputed = cat_imputer.transform(X_test[cat_cols])
        
        label_encoder = LabelEncoder()
        for col in cat_cols:
            all_categories = pd.concat([X_train[col], X_test[col]], axis=0)
            label_encoder.fit(all_categories)

            X_train_cat_imputed[:, cat_cols.get_loc(col)] = label_encoder.transform(X_train_cat_imputed[:, cat_cols.get_loc(col)])
            X_test_cat_imputed[:, cat_cols.get_loc(col)] = label_encoder.transform(X_test_cat_imputed[:, cat_cols.get_loc(col)])
    else:
        X_train_cat_imputed = X_train
        X_test_cat_imputed = X_test
    
    X_train_imputed = pd.DataFrame(X_train_num_imputed, columns=num_cols)
    X_train_imputed[cat_cols] = X_train_cat_imputed
    X_test_imputed = pd.DataFrame(X_test_num_imputed, columns=num_cols)
    X_test_imputed[cat_cols] = X_test_cat_imputed

    return X_train_imputed, X_test_imputed

# Entrenamiento y evaluación del modelo con MLflow
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

            # Registrar el modelo
            mlflow.sklearn.log_model(model, "random_forest_model")

            # Guardar el reporte de clasificación como un archivo de texto
            with open("classification_report.txt", "w") as f:
                f.write(report)

            # Registrar el reporte de clasificación
            mlflow.log_artifact("classification_report.txt")

            return model
        except Exception as e:
            print(f"Error en el entrenamiento: {e}")
            mlflow.log_param("error", str(e))

# Selección del mejor modelo mediante GridSearch con MLflow
def grid_search(X_train_imputed, y_train):
    
    param_grid = {
    "n_estimators": [100],  # Reducido para pruebas
    "max_depth": [10],       # Reducido para pruebas
    "min_samples_split": [2]
    }
    
    with mlflow.start_run():  # Inicia un nuevo experimento
        grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
        grid_search.fit(X_train_imputed, y_train)
        
        best_params = grid_search.best_params_
        mlflow.log_param("best_params", best_params)
        
        # Registrar el mejor modelo
        mlflow.sklearn.log_model(grid_search.best_estimator_, "best_random_forest_model")
        
        print(f"Best parameters: {best_params}")
        return grid_search.best_estimator_

# Scoreo del modelo con el dataset de aplicación
def score_model(model, oot_clientes, oot_requerimientos, imputer):
    oot_data = pd.merge(oot_clientes, oot_requerimientos, on="ID_CORRELATIVO", how="left")
    
    if 'ATTRITION' in oot_data.columns:
        oot_data = oot_data.drop(columns=["ATTRITION"])
    
    num_cols = oot_data.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = oot_data.select_dtypes(include=["object"]).columns
    
    num_imputer = SimpleImputer(strategy="median")
    oot_data_num_imputed = num_imputer.fit_transform(oot_data[num_cols])
    
    cat_imputer = SimpleImputer(strategy="most_frequent")
    oot_data_cat_imputed = cat_imputer.fit_transform(oot_data[cat_cols])
    
    label_encoder = LabelEncoder()
    for col in cat_cols:
        oot_data_cat_imputed[:, cat_cols.get_loc(col)] = label_encoder.fit_transform(oot_data_cat_imputed[:, cat_cols.get_loc(col)])
    
    oot_data_imputed = pd.DataFrame(oot_data_num_imputed, columns=num_cols)
    oot_data_imputed[cat_cols] = oot_data_cat_imputed
    
    oot_predictions = model.predict(oot_data_imputed)
    
    oot_data["predictions"] = oot_predictions
    
    return oot_data

# Función principal para ejecutar todo el proceso
def main():
    train_clientes, train_requerimientos, oot_clientes, oot_requerimientos = load_data()
    merged_data = merge_datasets(train_clientes, train_requerimientos)
    X_train, X_test, y_train, y_test = split_data(merged_data)
    X_train_imputed, X_test_imputed = clean_data(X_train, X_test)
    
    # Entrenamiento y evaluación con MLflow
    model = train_and_evaluate(X_train_imputed, X_test_imputed, y_train, y_test)
    
    # Búsqueda de mejores hiperparámetros
    best_model = grid_search(X_train_imputed, y_train)
    
    # Scorear el modelo con el dataset de prueba
    oot_data_with_predictions = score_model(best_model, oot_clientes, oot_requerimientos, SimpleImputer(strategy="median"))
    print(oot_data_with_predictions.head())
    
    # Guardar el DataFrame con las predicciones
    oot_data_with_predictions.to_csv(r'D:\my_mlops_project\output\oot_data_with_predictions.csv', index=False)

if __name__ == "__main__":
    main()
