import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Carregar dados
microdados = pd.read_csv(r'microdados_tratados\microdados_tratados_enem_2023.csv')
print("Dimensões originais:", microdados.shape)

# Definir variável alvo
alvo = 'NU_NOTA_CN'

# Remoção dos restantes das notas
notas_restantes = ['NU_NOTA_CH', 'NU_NOTA_MT','NU_NOTA_LC', 'NU_NOTA_REDACAO', 
                   'NU_NOTA_COMP1', 'NU_NOTA_COMP2','NU_NOTA_COMP3', 'NU_NOTA_COMP4', 
                   'NU_NOTA_COMP5']

# Separar features (X) e target (y)
X = microdados.drop(columns=[alvo] + notas_restantes)
y = microdados[alvo]

# Lista de colunas categóricas para One-Hot Encoding
categorical_cols = [
    'TP_SEXO',           # Gênero (1=Masc, 2=Fem)
    'TP_ESTADO_CIVIL',   # Estado civil (0 a 4)
    'IN_TREINEIRO',      # Treineiro (0=nao, 1=sim)
    'TP_COR_RACA',       # Raça/Cor (0 a 5)
    'TP_ESCOLA',         # Tipo de escola (1=Pública, 2=Privada)
    'TP_NACIONALIDADE',  # Nacionalidade (1=Brasileiro, etc.)
    'TP_ST_CONCLUSAO'    # Situação de conclusão (1=Já concluí, etc.)
]

# Aplicar One-Hot Encoding
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
print("Dimensões após One-Hot:", X_encoded.shape)

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Treinar modelo XGBoost
model = XGBRegressor(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train, y_train)

# Avaliar
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMSE: {mse:.2f}")
print(f"R²: {r2:.4f}")

# Gráfico de importância de features (CORRIGIDO)
feature_names = X_encoded.columns  # Nomes das features após One-Hot
importances = model.feature_importances_

# Ordenar features por importância
sorted_idx = importances.argsort()[::-1]  # Ordem decrescente
feature_names_sorted = [feature_names[i] for i in sorted_idx]
importances_sorted = importances[sorted_idx]

# Plotar apenas as top N features (ex: top 20) para melhor visualização
top_n = 20
plt.figure(figsize=(10, 8))
plt.barh(feature_names_sorted[:top_n], importances_sorted[:top_n])
plt.xlabel("Importância")
plt.title("Top 20 Features Mais Importantes (XGBoost)")
plt.gca().invert_yaxis()  # Inverter eixo Y para mostrar a mais importante no topo
plt.tight_layout()
plt.show()