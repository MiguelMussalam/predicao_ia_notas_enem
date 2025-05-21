import predicao_XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

# Teste de melhores parametros
param_grid = {
    'max_depth': [4, 6, 8],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [500, 1000, 1500],
    'subsample': [0.6, 0.8, 1.0]
}

grid = GridSearchCV(XGBRegressor(), param_grid, cv=3, scoring='r2')
grid.fit(X_train, y_train)
print("Melhores parâmetros:", grid.best_params_)


# Teste de parametros aleatórios
param_dist = {
    'max_depth': np.arange(4, 10),
    'learning_rate': np.linspace(0.01, 0.2, 10),
   'n_estimators': [500, 1000, 1500, 2000],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}
random_search = RandomizedSearchCV(
    XGBRegressor(),
    param_distributions=param_dist,
    n_iter=20,  # Apenas 20 combinações aleatórias
    cv=3,
    scoring='r2',
    n_jobs=-1
)
random_search.fit(X_train, y_train)
print("Melhores parâmetros: ", random_search.best_params_)