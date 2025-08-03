import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(
    page_title="Previsão de renda",
    page_icon="📉",
    layout="wide",
)

def avaliar_modelo(modelo, X_teste, y_teste):
    y_pred = modelo.predict(X_teste)
    rmse = np.sqrt(mean_squared_error(y_teste, y_pred))
    r2 = r2_score(y_teste, y_pred)
    return rmse, r2, y_pred


st.title("Previsão de Renda com Árvore de Decisão")

st.markdown("### Carregamento de Dados")
caminho_arquivo = r'C:\Users\marci\Desktop\Márcio\Programação\EBAC\02 - Phyton\5 -Desenvolvimento Modelos com Pandas e Python\Modulo 16\Meu projeto\input\previsao_de_renda.csv'
dados = pd.read_csv(caminho_arquivo)

st.markdown("### Pré-processamento dos Dados")
dados = dados.dropna().drop_duplicates().copy()
dados['log_renda'] = np.log1p(dados['renda'])

colunas_modelo = [
    'sexo', 'posse_de_veiculo', 'posse_de_imovel', 'qtd_filhos',
    'tipo_renda', 'educacao', 'estado_civil', 'tipo_residencia',
    'idade', 'tempo_emprego', 'qt_pessoas_residencia'
]

X = dados[colunas_modelo]
y = dados['log_renda']
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

cat_cols = X.select_dtypes(include=['object', 'bool']).columns.tolist()

preprocessador = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), cat_cols)],
    remainder='passthrough'
)

st.markdown("### Treinamento e Avaliação - Profundidade 5")
pipeline = Pipeline([
    ('preprocessamento', preprocessador),
    ('modelo', DecisionTreeRegressor(max_depth=5, random_state=42))
])
pipeline.fit(X_treino, y_treino)
rmse_log_5, r2_log_5, y_pred_log_5 = avaliar_modelo(pipeline, X_teste, y_teste)
rmse_real_5 = np.sqrt(mean_squared_error(np.expm1(y_teste), np.expm1(y_pred_log_5)))

col1, col2, col3 = st.columns(3)
col1.metric("RMSE (log_renda)", f"{rmse_log_5:.4f}")
col2.metric("R² (log_renda)", f"{r2_log_5:.4f}")
col3.metric("RMSE (renda real)", f"R$ {rmse_real_5:.2f}")

st.markdown("###  Modelo com Profundidade 10")
pipeline_10 = Pipeline([
    ('preprocessamento', preprocessador),
    ('modelo', DecisionTreeRegressor(max_depth=10, random_state=42))
])
pipeline_10.fit(X_treino, y_treino)
rmse_log_10, r2_log_10, _ = avaliar_modelo(pipeline_10, X_teste, y_teste)

st.write(f"**RMSE (log_renda):** {rmse_log_10:.4f}")
st.write(f"**R² (log_renda):** {r2_log_10:.4f}")

st.markdown("###  Visualização da Árvore de Decisão")
modelo = pipeline_10.named_steps['modelo']
feature_names = pipeline_10.named_steps['preprocessamento'].get_feature_names_out()
fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(modelo, feature_names=feature_names, filled=True, rounded=True, max_depth=3, fontsize=10, ax=ax)
st.pyplot(fig)

st.markdown("### Análise Visual da Renda")
opcoes = ['idade', 'tempo_emprego', 'tipo_renda', 'educacao', 'sexo']
variavel_x = st.selectbox("Selecione a variável para o eixo X:", opcoes, help="Escolha uma variável para visualizar sua relação com a renda.")
tipo_grafico = st.radio("Tipo de gráfico:", ['Dispersão (scatter)', 'Boxplot'])

fig, ax = plt.subplots(figsize=(10, 6))
if tipo_grafico == 'Dispersão (scatter)':
    sns.scatterplot(data=dados, x=variavel_x, y='renda', alpha=0.5, ax=ax)
else:
    sns.boxplot(data=dados, x=variavel_x, y='renda', ax=ax)
    ax.tick_params(axis='x', rotation=45)
ax.set_title(f"Renda vs {variavel_x}")
ax.set_xlabel(variavel_x)
ax.set_ylabel("Renda")
st.pyplot(fig)

st.markdown("---")

modelo_rf = Pipeline([
    ('preprocessamento', preprocessador),
    ('modelo', RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42))
])
modelo_rf.fit(X_treino, y_treino)
rmse_rf, r2_rf, y_pred_rf = avaliar_modelo(modelo_rf, X_teste, y_teste)
rmse_rf_real = np.sqrt(mean_squared_error(np.expm1(y_teste), np.expm1(y_pred_rf)))

st.markdown("### Importância das Variáveis (Random Forest)")

importances = modelo_rf.named_steps['modelo'].feature_importances_
feature_names = modelo_rf.named_steps['preprocessamento'].get_feature_names_out()
importancia_df = pd.Series(importances, index=feature_names).sort_values(ascending=False)

fig_imp, ax_imp = plt.subplots(figsize=(10, 6))
sns.barplot(x=importancia_df.values, y=importancia_df.index, ax=ax_imp, palette="viridis")
ax_imp.set_title("Importância das Variáveis na Previsão da Renda")
ax_imp.set_xlabel("Importância")
ax_imp.set_ylabel("Variável")
st.pyplot(fig_imp)


