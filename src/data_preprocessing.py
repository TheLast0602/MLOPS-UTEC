import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

def load_data():
    # Cargar los datos
    train_clientes = pd.read_csv('data/train_clientes_sample.csv')
    train_requerimientos = pd.read_csv('data/train_requerimientos_sample.csv')
    oot_clientes = pd.read_csv('data/oot_clientes_sample.csv')
    oot_requerimientos = pd.read_csv('data/oot_requerimientos_sample.csv')
    return train_clientes, train_requerimientos, oot_clientes, oot_requerimientos

def merge_datasets(train_clientes, train_requerimientos):
    return pd.merge(train_clientes, train_requerimientos, on="ID_CORRELATIVO", how="inner")

def split_data(data):
    X = data.drop(columns=["ATTRITION"])
    y = data["ATTRITION"]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def clean_data(X_train, X_test):
    num_cols = X_train.select_dtypes(include=["float64", "int64"]).columns
    cat_cols = X_train.select_dtypes(include=["object"]).columns
    num_imputer = SimpleImputer(strategy="median")
    X_train_num_imputed = num_imputer.fit_transform(X_train[num_cols])
    X_test_num_imputed = num_imputer.transform(X_test[num_cols])
    
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X_train_cat_imputed = cat_imputer.fit_transform(X_train[cat_cols])
    X_test_cat_imputed = cat_imputer.transform(X_test[cat_cols])
    
    label_encoder = LabelEncoder()
    for col in cat_cols:
        all_categories = pd.concat([X_train[col], X_test[col]], axis=0)
        label_encoder.fit(all_categories)

        X_train_cat_imputed[:, cat_cols.get_loc(col)] = label_encoder.transform(X_train_cat_imputed[:, cat_cols.get_loc(col)])
        X_test_cat_imputed[:, cat_cols.get_loc(col)] = label_encoder.transform(X_test_cat_imputed[:, cat_cols.get_loc(col)])
    
    X_train_imputed = pd.DataFrame(X_train_num_imputed, columns=num_cols)
    X_train_imputed[cat_cols] = X_train_cat_imputed
    X_test_imputed = pd.DataFrame(X_test_num_imputed, columns=num_cols)
    X_test_imputed[cat_cols] = X_test_cat_imputed

    return X_train_imputed, X_test_imputed
