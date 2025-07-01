import pandas as pd
from sklearn.impute import SimpleImputer

def load_data():
    # Aquí puedes cargar los datasets
    train_clientes = pd.read_csv('data/train_clientes_sample.csv')
    train_requerimientos = pd.read_csv('data/train_requerimientos_sample.csv')
    oot_clientes = pd.read_csv('data/oot_clientes_sample.csv')
    oot_requerimientos = pd.read_csv('data/oot_requerimientos_sample.csv')
    return train_clientes, train_requerimientos, oot_clientes, oot_requerimientos

def merge_datasets(train_clientes, train_requerimientos):
    # Realizar el cruce de datos
    return pd.merge(train_clientes, train_requerimientos, on="ID_CORRELATIVO", how="inner")

def clean_data(X_train, X_test):
    # Imputar los valores faltantes
    imputer = SimpleImputer(strategy="median")
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)
    return X_train_imputed, X_test_imputed
