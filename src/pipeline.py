from data_preprocessing import load_data, merge_datasets, split_data, clean_data
from model_training import train_and_evaluate
from model_deployment import app  # Esto es solo para el despliegue, si lo decides usar
import uvicorn

def main():
    # Cargar y preparar los datos
    train_clientes, train_requerimientos, oot_clientes, oot_requerimientos = load_data()
    merged_data = merge_datasets(train_clientes, train_requerimientos)
    X_train, X_test, y_train, y_test = split_data(merged_data)
    X_train_imputed, X_test_imputed = clean_data(X_train, X_test)
    
    # Entrenamiento y evaluación del modelo
    model = train_and_evaluate(X_train_imputed, X_test_imputed, y_train, y_test)
    
    # Aquí puedes agregar el despliegue del modelo si lo deseas
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    
if __name__ == "__main__":
    main()
