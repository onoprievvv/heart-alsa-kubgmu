import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

# --- КОНФИГУРАЦИЯ ---
st.set_page_config(page_title="HEART-ALSA MVP", layout="wide")

@st.cache_resource
def load_trained_model():
    # Обучаем модель на синтетике, имитирующей логику NHANES
    np.random.seed(42)
    X_train = pd.DataFrame({
        'RIDAGEYR': np.random.randint(30, 85, 1000),
        'LBXHRW': np.random.normal(13.5, 1.5, 1000),
        'NLR': np.random.lognormal(0.7, 0.4, 1000),
        'AL_Score': np.random.randint(0, 5, 1000),
        'BMXBMI': np.random.normal(28, 5, 1000)
    })
    # Веса: RDW и AL_Score — главные предикторы
    logit = (X_train['LBXHRW'] - 13.5) * 0.8 + (X_train['AL_Score'] * 0.6) + (X_train['RIDAGEYR'] - 55) * 0.03
    y_train = (1 / (1 + np.exp(-logit)) > np.random.rand(1000)).astype(int)
    return xgb.XGBClassifier().fit(X_train, y_train)

model = load_trained_model()
explainer = shap.TreeExplainer(model)

# --- ИНТЕРФЕЙС ---
st.title("🫀 HEART-ALSA: Раннее выявление риска СН")

st.sidebar.header("📋 Ввод клинических данных")

# 1. Демография
age = st.sidebar.slider("Возраст (лет)", 18, 95, 55)

# 2. Гематология (Хаб ОАК)
st.sidebar.subheader("🔬 Гематология (ОАК)")
rdw = st.sidebar.number_input("RDW (Анизоцитоз, %)", 11.0, 20.0, 13.5)
neu = st.sidebar.number_input("Нейтрофилы (абс.)", 1.0, 15.0, 4.2)
lym = st.sidebar.number_input("Лимфоциты (абс.)", 0.5, 10.0, 2.0)
nlr = neu / (lym + 0.01)

# 3. Метаболический профиль (СЫРЫЕ ДАННЫЕ ДЛЯ AL)
st.sidebar.subheader("🍬 Метаболический профиль (AL)")
sbp = st.sidebar.slider("Систолическое АД", 90, 200, 130)
glu = st.sidebar.number_input("Глюкоза (ммоль/л)", 3.0, 15.0, 5.2)
waist = st.sidebar.number_input("Окружность талии (см)", 60, 150, 95)
bmi = st.sidebar.number_input("ИМТ (BMI)", 15.0, 50.0, 28.0)

# --- АВТОМАТИЧЕСКИЙ РАСЧЕТ AL SCORE ---
al_score = 0
if sbp >= 135: al_score += 1
if glu >= 5.6: al_score += 1
if waist >= 102: al_score += 1 # Порог для мужчин
if bmi >= 30: al_score += 1

# Подготовка данных для предсказания
input_df = pd.DataFrame({
    'RIDAGEYR': [age],
    'LBXHRW': [rdw],
    'NLR': [nlr],
    'AL_Score': [al_score],
    'BMXBMI': [bmi]
})

# --- ВЫВОД РЕЗУЛЬТАТОВ ---
prob = model.predict_proba(input_df)[0][1]
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Прогноз риска")
    color = "red" if prob > 0.5 else "green"
    st.markdown(f"<h1 style='color: {color};'>{prob:.1%}</h1>", unsafe_allow_html=True)
    st.metric("Рассчитанный AL Score", f"{al_score} / 4")
    st.caption("Индекс АН сформирован автоматически на основе АД, сахара, талии и ИМТ.")

with col2:
    st.subheader("Интерпретация вклада (SHAP)")
    shap_vals = explainer(input_df)
    # Маппинг имен для красоты
    feature_names_ru = ['Возраст', 'Гемато-Хаб (RDW)', 'Иммуно-Хаб (NLR)', 'Аллостат-Хаб (AL)', 'Метаболо-Хаб (BMI)']
    shap_vals.feature_names = feature_names_ru
    
    fig, ax = plt.subplots()
    shap.plots.bar(shap_vals[0], show=False)
    st.pyplot(fig)

st.divider()
st.subheader("Анализ по диагностическим Хабам")
c1, c2, c3 = st.columns(3)
c1.metric("Гематология (RDW)", f"{rdw}%")
c2.metric("Иммунитет (NLR)", f"{nlr:.2f}")
c3.metric("Износ (AL Index)", f"{al_score}/4")
