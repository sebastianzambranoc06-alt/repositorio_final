import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

# Intentar cargar TensorFlow, si falla usar modo demo
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
    st.success("✅ TensorFlow cargado correctamente")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("⚠️ Modo demo - TensorFlow no disponible")

def modo_demo(imagen):
    """Versión demo cuando TensorFlow no está disponible"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(imagen, caption="Imagen original", use_column_width=True)
    
    with col2:
        st.markdown("""
        <div style="background: #3498db; color: white; padding: 2rem; border-radius: 10px; text-align: center;">
            <h2>HOMBRE 👨</h2>
            <h3>Confianza: 85.2%</h3>
            <p>Mujer: 0.148 | Hombre: 0.852</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Gráfico simulado
    fig, ax = plt.subplots(figsize=(8, 3))
    bars = ax.bar(['Hombre', 'Mujer'], [0.852, 0.148], color=['#3498db', '#e84393'])
    ax.set_ylim(0, 1)
    ax.bar_label(bars, fmt='%.3f', padding=3)
    st.pyplot(fig)
    
    st.info("🔧 Esta es una demostración. La funcionalidad completa con IA estará disponible en entornos compatibles.")

# Interfaz principal
st.title("🧠 Clasificador de Género IA")

if not TENSORFLOW_AVAILABLE:
    st.error("""
    ⚠️ **TensorFlow no disponible**
    - Usando modo demostración
    - Para funcionalidad completa, ejecuta localmente
    """)

uploaded_file = st.file_uploader("Sube una imagen facial", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    imagen = Image.open(uploaded_file)
    
    if TENSORFLOW_AVAILABLE:
        # Tu código original con TensorFlow aquí
        st.success("✅ Procesando con IA...")
        # ... tu código de predicción real
    else:
        modo_demo(imagen)