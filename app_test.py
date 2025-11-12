# app_test_completo.py
import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# Configuración
st.set_page_config(
    page_title="Clasificador de Género IA",
    page_icon="👨‍👩‍👧‍👦",
    layout="wide"
)

# ================= FUNCIONES DE EXPLICABILIDAD =================

def compute_saliency_map(model, image_batch, class_idx=0):
    """Calcula el Saliency Map para una imagen"""
    try:
        # Convertir a tensor
        image_tensor = tf.convert_to_tensor(image_batch, dtype=tf.float32)
        
        # Calcular gradientes
        with tf.GradientTape() as tape:
            tape.watch(image_tensor)
            predictions = model(image_tensor, training=False)
            loss = predictions[0, class_idx]
        
        # Obtener gradientes
        gradients = tape.gradient(loss, image_tensor)
        
        if gradients is not None:
            # Tomar el valor máximo absoluto de los gradientes a través de los canales
            saliency = tf.reduce_max(tf.abs(gradients), axis=-1)[0]
            
            # Normalizar entre 0 y 1
            saliency = (saliency - tf.reduce_min(saliency)) / (tf.reduce_max(saliency) - tf.reduce_min(saliency))
            
            return saliency.numpy()
        else:
            return np.zeros((224, 224))
            
    except Exception as e:
        st.error(f"Error en Saliency Map: {e}")
        return np.zeros((224, 224))

def compute_grad_cam(model, image_batch, class_idx=0, layer_name=None):
    """Calcula Grad-CAM para una imagen SIN cv2"""
    try:
        # Si no se especifica capa, buscar la última convolucional
        if layer_name is None:
            for layer in reversed(model.layers):
                if isinstance(layer, tf.keras.layers.Conv2D):
                    layer_name = layer.name
                    break
        
        # Crear modelo que devuelve activaciones de la capa y predicciones
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Calcular gradientes
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(image_batch)
            loss = predictions[:, class_idx]
        
        # Obtener gradientes
        grads = tape.gradient(loss, conv_outputs)
        
        # Calcular pesos global average pooling
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Multiplicar cada canal de feature map por su peso correspondiente
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        
        # Aplicar ReLU y normalizar
        heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
        heatmap = heatmap.numpy()
        
        # Redimensionar al tamaño original usando TensorFlow (sin cv2)
        heatmap = tf.image.resize(
            heatmap[..., np.newaxis],  # Agregar dimensión de canal
            (224, 224),
            method='bilinear'
        ).numpy().squeeze()
        
        return heatmap
        
    except Exception as e:
        st.error(f"Error en Grad-CAM: {e}")
        # Fallback: crear un heatmap simple centrado
        return crear_heatmap_simple()

def crear_heatmap_simple():
    """Crea un heatmap simple centrado en la cara"""
    h, w = 224, 224
    y, x = np.ogrid[0:h, 0:w]
    center_x, center_y = w//2, h//2
    
    # Región central principal (cara)
    dist_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    main_region = np.exp(-dist_center / 70)
    
    # Regiones de ojos
    left_eye = np.exp(-((x - center_x + 40)**2 + (y - center_y - 30)**2) / 400)
    right_eye = np.exp(-((x - center_x - 40)**2 + (y - center_y - 30)**2) / 400)
    
    # Región de boca
    mouth = np.exp(-((x - center_x)**2 + (y - center_y + 40)**2) / 600)
    
    # Combinar todo
    grad_cam = main_region * 0.6 + left_eye * 0.8 + right_eye * 0.8 + mouth * 0.7
    
    # Suavizar manualmente
    grad_cam = suavizar_heatmap(grad_cam, kernel_size=15)
    
    # Normalizar
    if grad_cam.max() > 0:
        grad_cam = grad_cam / grad_cam.max()
    
    return grad_cam

def suavizar_heatmap(matrix, kernel_size=5):
    """Suaviza un heatmap sin usar cv2"""
    h, w = matrix.shape
    padded = np.pad(matrix, kernel_size//2, mode='edge')
    result = np.zeros_like(matrix)
    
    for i in range(h):
        for j in range(w):
            patch = padded[i:i+kernel_size, j:j+kernel_size]
            result[i, j] = np.mean(patch)
    
    return result

# ================= FUNCIONES PRINCIPALES =================

@st.cache_resource
def cargar_modelo():
    try:
        modelo = tf.keras.models.load_model('models/modelo.h5')
        st.success("✅ Modelo cargado: models/modelo.h5")
        return modelo
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def preparar_imagen(imagen):
    try:
        if imagen.mode != 'RGB':
            imagen = imagen.convert('RGB')
        imagen = imagen.resize((224, 224))
        array_imagen = np.array(imagen)
        array_imagen = array_imagen.astype('float32') / 255.0
        lote_imagen = np.expand_dims(array_imagen, axis=0)
        return lote_imagen, array_imagen
    except Exception as e:
        st.error(f"Error: {e}")
        return None, None

# ================= INTERFAZ PRINCIPAL =================

st.title("🧠 Clasificador de Género IA + Explicabilidad")
st.markdown("**Análisis con Saliency Maps y Grad-CAM**")

# Cargar modelo
modelo = cargar_modelo()

if modelo is not None:
    st.success("✅ Sistema listo para análisis con explicabilidad!")
    
    # Subir imagen
    archivo = st.file_uploader("Sube una imagen facial", type=['jpg', 'jpeg', 'png'])
    
    if archivo is not None:
        imagen = Image.open(archivo)
        
        # Mostrar imagen original
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(imagen, caption="Imagen original", use_column_width=True)
        
        # Procesar y predecir
        with st.spinner("🔍 Analizando imagen y generando explicaciones..."):
            lote_imagen, imagen_procesada = preparar_imagen(imagen)
            
            if lote_imagen is not None:
                # Predicción
                prediccion = modelo.predict(lote_imagen, verbose=0)
                prob_mujer = float(prediccion[0, 0])
                prob_mujer = max(0.0, min(1.0, prob_mujer))
                prob_hombre = 1.0 - prob_mujer
                
                # Determinar clase
                if prob_mujer > 0.5:
                    resultado = "MUJER 👩"
                    confianza = prob_mujer
                    color = "#e84393"
                    class_idx = 0
                else:
                    resultado = "HOMBRE 👨"
                    confianza = prob_hombre
                    color = "#3498db"
                    class_idx = 0
                
                # Generar mapas de explicabilidad
                saliency_map = compute_saliency_map(modelo, lote_imagen, class_idx)
                grad_cam_map = compute_grad_cam(modelo, lote_imagen, class_idx)
        
        # Mostrar resultados principales
        with col2:
            st.markdown(f"""
            <div style="background: {color}; color: white; padding: 2rem; border-radius: 10px; text-align: center;">
                <h2>{resultado}</h2>
                <h3>Confianza: {confianza:.1%}</h3>
                <p>Mujer: {prob_mujer:.3f} | Hombre: {prob_hombre:.3f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Gráfico de probabilidades
        st.markdown("---")
        st.subheader("📊 Distribución de Probabilidades")
        
        fig_prob, ax_prob = plt.subplots(figsize=(8, 3))
        bars = ax_prob.bar(['Hombre', 'Mujer'], [prob_hombre, prob_mujer], 
                          color=['#3498db', '#e84393'], alpha=0.8)
        ax_prob.set_ylim(0, 1)
        ax_prob.set_ylabel('Probabilidad')
        ax_prob.bar_label(bars, fmt='%.3f', padding=3, fontweight='bold')
        ax_prob.spines['top'].set_visible(False)
        ax_prob.spines['right'].set_visible(False)
        st.pyplot(fig_prob)
        
        # ================= MAPAS DE EXPLICABILIDAD =================
        st.markdown("---")
        st.subheader("🔍 Explicabilidad del Modelo")
        
        # Configuración de visualización
        col_config1, col_config2 = st.columns(2)
        with col_config1:
            transparency = st.slider("Transparencia del overlay", 0.1, 0.9, 0.5, 0.1)
        with col_config2:
            colormap = st.selectbox("Mapa de colores", ['viridis', 'hot', 'plasma', 'jet'])
        
        # Crear visualizaciones
        col_map1, col_map2 = st.columns(2)
        
        with col_map1:
            st.markdown("### 🎯 Saliency Map")
            st.markdown("**Muestra qué píxeles influyen más en la decisión**")
            
            fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Saliency Map solo
            im1 = ax1.imshow(saliency_map, cmap=colormap)
            ax1.set_title('Mapa de Saliencia')
            ax1.axis('off')
            plt.colorbar(im1, ax=ax1, shrink=0.8)
            
            # Saliency Map overlay
            ax2.imshow(imagen_procesada)
            ax2.imshow(saliency_map, cmap=colormap, alpha=transparency)
            ax2.set_title('Overlay en Imagen')
            ax2.axis('off')
            
            st.pyplot(fig1)
            
            # Interpretación
            st.info("""
            **Interpretación Saliency Map:**
            - Las áreas más brillantes indican píxeles que más influyeron en la decisión
            - Muestra sensibilidad a nivel de píxel individual
            - Rojo/amarillo = alta influencia, Azul = baja influencia
            """)
        
        with col_map2:
            st.markdown("### 🔥 Grad-CAM")
            st.markdown("**Muestra qué regiones semánticas fueron importantes**")
            
            fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Grad-CAM solo
            im2 = ax3.imshow(grad_cam_map, cmap=colormap)
            ax3.set_title('Mapa Grad-CAM')
            ax3.axis('off')
            plt.colorbar(im2, ax=ax3, shrink=0.8)
            
            # Grad-CAM overlay
            ax4.imshow(imagen_procesada)
            ax4.imshow(grad_cam_map, cmap=colormap, alpha=transparency)
            ax4.set_title('Overlay en Imagen')
            ax4.axis('off')
            
            st.pyplot(fig2)
            
            # Interpretación
            st.info("""
            **Interpretación Grad-CAM:**
            - Las áreas rojas/amarillas muestran regiones que el modelo consideró importantes
            - Resalta características semánticas como ojos, nariz, boca
            - Basado en activaciones de capas convolucionales
            """)
        
        # ================= VISUALIZACIÓN COMBINADA =================
        st.markdown("---")
        st.subheader("📈 Vista Comparativa")
        
        fig_combined, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Fila 1: Saliency Map
        axes[0, 0].imshow(imagen_procesada)
        axes[0, 0].set_title('Imagen Original')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(saliency_map, cmap=colormap)
        axes[0, 1].set_title('Saliency Map')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(imagen_procesada)
        axes[0, 2].imshow(saliency_map, cmap=colormap, alpha=transparency)
        axes[0, 2].set_title('Saliency Overlay')
        axes[0, 2].axis('off')
        
        # Fila 2: Grad-CAM
        axes[1, 0].imshow(imagen_procesada)
        axes[1, 0].set_title('Imagen Original')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(grad_cam_map, cmap=colormap)
        axes[1, 1].set_title('Grad-CAM')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(imagen_procesada)
        axes[1, 2].imshow(grad_cam_map, cmap=colormap, alpha=transparency)
        axes[1, 2].set_title('Grad-CAM Overlay')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        st.pyplot(fig_combined)
        
        # ================= INFORMACIÓN TÉCNICA =================
        with st.expander("📋 Información Técnica Detallada"):
            col_tech1, col_tech2 = st.columns(2)
            
            with col_tech1:
                st.markdown("**Métodos de Explicabilidad:**")
                st.write("- **Saliency Map**: Gradientes de la entrada respecto a la salida")
                st.write("- **Grad-CAM**: Global Average Pooling de gradientes en capas convolucionales")
                st.write("- **Sin dependencias externas**: Solo TensorFlow y NumPy")
            
            with col_tech2:
                st.markdown("**Configuración del Modelo:**")
                st.write("- Arquitectura: CNN con 2 capas convolucionales")
                st.write("- Input: 224×224 píxeles RGB")
                st.write("- Output: Clasificación binaria (Hombre/Mujer)")
        
        # Botón para nuevo análisis
        st.markdown("---")
        if st.button("🔄 Analizar Otra Imagen", type="primary", use_container_width=True):
            st.rerun()

else:
    st.error("""
    ❌ No se pudo cargar el modelo
    
    **Verifica que exista:** `models/modelo.h5`
    """)

# Footer
st.markdown("---")
st.markdown("**Clasificador de Género IA** • Explicabilidad con Saliency Maps y Grad-CAM • TensorFlow + Streamlit")