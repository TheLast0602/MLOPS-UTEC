from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def grid_search(X_train_imputed, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
    grid_search.fit(X_train_imputed, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

def grid_search(X_train_imputed, y_train):
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5]
    }
    grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)
    grid_search.fit(X_train_imputed, y_train)
    print(grid_search.best_params_)
    return grid_search.best_estimator_
