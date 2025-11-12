import streamlit as st
import numpy as np
from PIL import Image

st.title("🧠 Clasificador de Género IA - Modo Demo")
st.info("🔧 En mantenimiento - La funcionalidad completa estará disponible pronto")

uploaded_file = st.file_uploader("Sube una imagen facial", type=['jpg', 'jpeg', 'png'])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Imagen subida", use_column_width=True)
    st.success("✅ Imagen procesada correctamente")
    st.warning("⚡ La clasificación con IA estará disponible en la próxima actualización")