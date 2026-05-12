import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuración de la página
st.set_page_config(page_title="Predicción de Deserción Escolar", layout="centered")

# Cargar el modelo guardado
@st.cache_resource
def load_model():
    return joblib.load('modelo_desercion.pkl')

model = load_model()

st.title("🎓 Sistema de Alerta Temprana: Deserción")
st.markdown("Introduce los datos del estudiante para evaluar el riesgo de abandono escolar.")

# Interfaz de usuario en la barra lateral
st.sidebar.header("Datos del Estudiante")

edad = st.sidebar.slider("Edad", 17, 30, 20)
promedio = st.sidebar.slider("Promedio Académico (1.0 - 5.0)", 1.0, 5.0, 3.5, 0.1)
asistencia = st.sidebar.slider("Porcentaje de Asistencia (%)", 0, 100, 85)
horas_estudio = st.sidebar.slider("Horas de estudio semanales", 0, 40, 15)

# Crear DataFrame para la predicción
input_data = pd.DataFrame({
    'edad': [edad],
    'promedio': [promedio],
    'asistencia': [asistencia],
    'horas_estudio': [horas_estudio]
})

# Botón de predicción
if st.button("Analizar Riesgo"):
    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)[0][1]
    
    st.divider()
    
    if prediction[0] == 1:
        st.error(f"⚠️ **ALTO RIESGO DE DESERCIÓN**")
        st.warning(f"Probabilidad estimada: {probability:.2%}")
    else:
        st.success(f"✅ **BAJO RIESGO DE DESERCIÓN**")
        st.info(f"Probabilidad de deserción: {probability:.2%}")

    st.write("### Resumen de entrada:")
    st.table(input_data)
