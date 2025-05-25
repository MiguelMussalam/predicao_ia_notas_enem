import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from lightgbm import LGBMClassifier, plot_importance, early_stopping, log_evaluation
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
#import optuna

# Carregar os dados
microdados = pd.read_csv(r'microdados_tratados\microdados_tratados_enem_2023.csv')
print("Dimensões originais:", microdados.shape)

# Colunas a excluir
colunas_excluir = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_MT', 'NU_NOTA_LC',
                   'NU_NOTA_REDACAO', 'NU_NOTA_COMP1', 'NU_NOTA_COMP2',
                   'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5',
                   'NO_MUNICIPIO_PROVA', 'SG_UF_PROVA', 'TP_LINGUA',
                   'Q013', 'Q015', 'Q020', 'Q018']

# Definir variável alvo
alvo = 'NU_NOTA_LC'
colunas_excluir.remove(alvo)

# Separar X e y
X = microdados.drop(columns=[alvo] + colunas_excluir)
y_continuo = microdados[alvo]

# Converter a variável contínua (nota) em classes
def categorizar_nota(nota):
    if nota <= 400:
        return 0
    elif nota <= 600:
        return 1
    elif nota <= 1000:
        return 2

y = y_continuo.apply(categorizar_nota)

# Colunas categóricas
categorical_cols = [
    'TP_SEXO', 'TP_ESTADO_CIVIL', 'IN_TREINEIRO',
    'TP_COR_RACA', 'TP_ESCOLA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO'
]

# Converter para categoria
for col in categorical_cols:
    X[col] = X[col].astype('category')

print(y.value_counts())
# Aplicar SMOTE + Undersampling com Pipeline
resample_pipeline = Pipeline(steps=[
    ('smote', SMOTE(sampling_strategy={0: 400000, 2: 400000}, random_state=42)),
    ('undersample', RandomUnderSampler(sampling_strategy={1: 600000}, random_state=42))
])

X_resampled, y_resampled = resample_pipeline.fit_resample(X, y)

# Divisão dos dados
X_train, X_valid, y_train, y_valid = train_test_split(X_resampled, y_resampled, stratify=y_resampled, test_size=0.2, random_state=42)

# Calcular os pesos das classes
classes = np.unique(y_train)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
weights_dict = dict(zip(classes, class_weights))
sample_weights = y_train.map(weights_dict)
print("Pesos por classe:", weights_dict)

# Treinar modelo final com melhores parâmetros
best_model = LGBMClassifier(
    objective='multiclass',
    num_class=3,
    learning_rate=0.05,
    n_estimators=2000,
    max_depth=25,
    num_leaves=299,
    min_child_samples=11,
    subsample=0.3234957039009826,
    colsample_bytree=0.3488369807308844,
    n_jobs=-1,
    random_state=42
)

best_model.fit(
    X_train, y_train,
    sample_weight=sample_weights,
    eval_set=[(X_valid, y_valid)],
    eval_metric='multi_logloss',
    callbacks=[
        early_stopping(100),
        log_evaluation(100)
    ],
    categorical_feature=categorical_cols
)

X_test = X_valid
y_test = y_valid

# Previsão
y_pred = best_model.predict(X_test)

# Avaliação
acc = accuracy_score(y_test, y_pred)
print(f"\nAcurácia: {acc:.4f}")
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, y_pred))
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, labels=[0, 1, 2]))

# Antes do resampling
sns.countplot(x=y)
plt.title("Distribuição original das classes")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()

# Após o resampling
sns.countplot(x=y_resampled)
plt.title("Distribuição após SMOTE + Undersampling")
plt.xlabel("Classe")
plt.ylabel("Quantidade")
plt.show()

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predito')
plt.ylabel('Real')
plt.title('Matriz de Confusão')
plt.show()

top_feature = X.columns[best_model.feature_importances_.argmax()]
sns.boxplot(x=y, y=X[top_feature])
plt.title(f"Distribuição da feature mais importante: {top_feature}")
plt.xlabel("Classe")
plt.ylabel(top_feature)
plt.show()

df_full = X.copy()
df_full["classe"] = y

# Exemplo para a feature mais importante
feature = top_feature
for c in [0, 1, 2]:
    print(f"\nValores mais comuns da feature '{feature}' para a classe {c}:")
    print(df_full[df_full['classe'] == c][feature].value_counts().head())

# Criar estudo
#study = optuna.create_study(direction='maximize')
#study.optimize(objective, n_trials=30)

# Exibir melhores parâmetros
#print("\nMelhores parâmetros encontrados:")
#print(study.best_params)

#print("\nMelhor f1_weighted score:", study.best_value)



## TESTE DE PARÂMETROS COM OPTUNIA

# Função objetivo do Optuna
#def objective(trial):
#    params = {
#        'objective': 'multiclass',
#        'num_class': 3,
#        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.2),
#        'n_estimators': trial.suggest_int('n_estimators', 100, 1500),
#        'max_depth': trial.suggest_int('max_depth', 5, 25),
#        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
#        'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
#        'subsample': trial.suggest_float('subsample', 0.3, 1.0),
#        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
#        'random_state': 42,
#        'n_jobs': -1
#    }

#    model = LGBMClassifier(**params)
#    model.fit(
#        X_train, y_train,
#        sample_weight=sample_weights,
#        eval_set=[(X_valid, y_valid)],
#        eval_metric='multi_logloss',
#        callbacks=[
#        early_stopping(50),
#        log_evaluation(100)  # Loga a cada 100 rounds
#        ],
#        categorical_feature=categorical_cols,
#    )

#    y_pred = model.predict(X_valid)
#    return f1_score(y_valid, y_pred, average='weighted')