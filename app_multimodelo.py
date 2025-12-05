import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prediction_models_all import ModeloPredictorRobos
import os

# ============================================================================
# CONFIGURACIÓN DE PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Predictor Multi-Tipo de Robos",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos
st.markdown("""
    <style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .tipo-robo {
        font-size: 18px;
        font-weight: bold;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# TÍTULO PRINCIPAL
# ============================================================================

st.title("🔐 Predictor Multi-Tipo de Robos")
st.markdown("Predicción de robos a casa habitación, negocios y vehículos usando redes neuronales")

# ============================================================================
# SIDEBAR - CONFIGURACIÓN
# ============================================================================

st.sidebar.header("⚙️ Configuración")

# Selector de tipo de robo
tipo_robo_opciones = {
    "🏠 Casa Habitación": "casa",
    "🏢 Negocios": "negocio",
    "🚗 Vehículos": "vehiculo"
}

tipo_robo_display = st.sidebar.radio(
    "Seleccione tipo de robo a predecir:",
    options=list(tipo_robo_opciones.keys()),
    index=0
)

tipo_robo = tipo_robo_opciones[tipo_robo_display]

# Colores según tipo de robo
colores_tipo = {
    'casa': '#FF6B6B',
    'negocio': '#4ECDC4',
    'vehiculo': '#FFE66D'
}

# Inicializar sesión para el tipo de robo actual
session_key = f'predictor_{tipo_robo}'
if session_key not in st.session_state:
    st.session_state[session_key] = {
        'predictor': None,
        'modelo_entrenado': False,
        'r2_score': None
    }

st.sidebar.divider()

# ============================================================================
# SIDEBAR - CARGA DE DATOS
# ============================================================================

st.sidebar.header("📊 Datos")

# Verificar si el modelo existe
model_path = f'modelo_robos_{tipo_robo}.pkl'
model_exists = os.path.exists(model_path)

if model_exists:
    st.sidebar.success(f"✅ Modelo pre-entrenado disponible")
else:
    st.sidebar.info("ℹ️ No hay modelo pre-entrenado. Cargue un archivo para entrenar.")

# Upload de archivo
uploaded_file = st.sidebar.file_uploader(
    "Cargue archivo Excel (opcional)",
    type=['xlsx'],
    help="Si no carga, usará los datos por defecto"
)

if uploaded_file is not None or model_exists:
    with st.spinner("Inicializando modelo..."):
        try:
            # Crear predictor
            predictor = ModeloPredictorRobos(tipo_robo)
            
            # Si hay archivo, entrenar; si no, cargar modelo pre-entrenado
            if uploaded_file is not None:
                with st.spinner("Entrenando modelo (esto puede tomar 1-2 minutos)..."):
                    r2 = predictor.entrenar_completo(file_path=uploaded_file)
                    st.sidebar.success("✅ Modelo entrenado exitosamente")
                    st.sidebar.metric("R² Score", f"{r2:.4f}")
            else:
                # Intentar cargar modelo pre-entrenado
                if predictor.cargar_modelo():
                    st.sidebar.success("✅ Modelo cargado desde memoria")
                    st.sidebar.metric("R² Score", f"{predictor.r2_score:.4f}")
                else:
                    st.sidebar.error("❌ No se pudo cargar el modelo")
                    predictor = None
            
            # Guardar en sesión
            st.session_state[session_key]['predictor'] = predictor
            st.session_state[session_key]['modelo_entrenado'] = predictor is not None
            st.session_state[session_key]['r2_score'] = predictor.r2_score if predictor else None
            
        except Exception as e:
            st.sidebar.error(f"Error: {str(e)}")
            st.session_state[session_key]['predictor'] = None
            st.session_state[session_key]['modelo_entrenado'] = False

# ============================================================================
# CONTENIDO PRINCIPAL
# ============================================================================

# Obtener predictor de la sesión
predictor = st.session_state[session_key]['predictor']
modelo_entrenado = st.session_state[session_key]['modelo_entrenado']

if predictor is not None and modelo_entrenado:
    
    # Tabs principales
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Predicción", "📊 Análisis", "🎯 Comparación", "ℹ️ Información"])
    
    # ====================================================================
    # TAB 1: PREDICCIÓN
    # ====================================================================
    
    with tab1:
        st.header(f"Predicción - {tipo_robo_display}")
        
        # Mostrar tipo de robo seleccionado
        col1, col2 = st.columns([1, 3])
        with col1:
            st.markdown(f"<div class='tipo-robo' style='background-color:{colores_tipo[tipo_robo]};color:white;'>{tipo_robo_display}</div>", unsafe_allow_html=True)
        
        st.divider()
        
        # Controles de predicción
        col1, col2 = st.columns(2)
        
        with col1:
            meses_nombre = {
                13: "Enero",
                14: "Febrero",
                15: "Marzo",
                16: "Abril",
                17: "Mayo",
                18: "Junio",
                19: "Julio",
                20: "Agosto",
                21: "Septiembre",
                22: "Octubre",
                23: "Noviembre",
                24: "Diciembre"
            }
            
            mes_seleccionado = st.selectbox(
                "Seleccione el mes a predecir:",
                options=list(meses_nombre.keys()),
                format_func=lambda x: meses_nombre[x],
                key=f"mes_{tipo_robo}"
            )
        
        with col2:
            top_n = st.slider(
                "Número de cuadrantes en el Top:",
                min_value=5,
                max_value=30,
                value=10,
                step=1,
                key=f"top_n_{tipo_robo}"
            )
        
        # Botón para predecir
        if st.button(f"🔮 Realizar Predicción - {tipo_robo_display}", use_container_width=True, type="primary", key=f"btn_{tipo_robo}"):
            with st.spinner(f"Prediciendo {tipo_robo_display.lower()} para {meses_nombre[mes_seleccionado]}..."):
                try:
                    # Realizar predicción
                    df_prediccion = predictor.predecir_top_cuadrantes(
                        mes_a_predecir=mes_seleccionado,
                        top_n=top_n
                    )
                    
                    # Guardar en sesión
                    st.session_state[f'ultima_prediccion_{tipo_robo}'] = df_prediccion
                    st.session_state[f'ultimo_mes_{tipo_robo}'] = mes_seleccionado
                    
                    # Mostrar resultados
                    st.success("✅ Predicción realizada exitosamente")
                    
                    # Estadísticas
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Máximo Predicho",
                            f"{df_prediccion['PREDICCION_ROBOS'].max():.1f}"
                        )
                    with col2:
                        st.metric(
                            "Promedio Predicho",
                            f"{df_prediccion['PREDICCION_ROBOS'].mean():.1f}"
                        )
                    with col3:
                        st.metric(
                            "Mínimo Predicho",
                            f"{df_prediccion['PREDICCION_ROBOS'].min():.1f}"
                        )
                    with col4:
                        st.metric(
                            "R² del Modelo",
                            f"{predictor.r2_score:.4f}"
                        )
                    
                    st.divider()
                    
                    # Tabla de resultados
                    st.subheader(f"🏆 Top {top_n} Cuadrantes - {meses_nombre[mes_seleccionado]}")
                    
                    df_display = df_prediccion.copy()
                    df_display['PREDICCION_ROBOS'] = df_display['PREDICCION_ROBOS'].round(2)
                    df_display['POBLACION'] = df_display['POBLACION'].round(0).astype(int)
                    df_display.index = df_display.index + 1
                    
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Gráfico de barras
                    st.subheader("📊 Visualización de Predicciones")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    
                    colores = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(df_prediccion)))
                    bars = ax.barh(
                        df_prediccion['CUADRANTE'].astype(str),
                        df_prediccion['PREDICCION_ROBOS'],
                        color=colores
                    )
                    
                    ax.set_xlabel('Número Predicho de Robos', fontsize=12, fontweight='bold')
                    ax.set_ylabel('Cuadrante', fontsize=12, fontweight='bold')
                    ax.set_title(f'Top {top_n} Cuadrantes con Mayor Riesgo ({tipo_robo_display}) - {meses_nombre[mes_seleccionado]}', 
                                fontsize=14, fontweight='bold')
                    ax.invert_yaxis()
                    
                    for bar, val in zip(bars, df_prediccion['PREDICCION_ROBOS']):
                        ax.text(val, bar.get_y() + bar.get_height()/2, 
                               f' {val:.1f}', va='center', fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                    
                    # Descargar resultados
                    csv = df_display.to_csv(index=True)
                    st.download_button(
                        label=f"📥 Descargar Resultados (CSV)",
                        data=csv,
                        file_name=f"prediccion_{tipo_robo}_{meses_nombre[mes_seleccionado].lower()}.csv",
                        mime="text/csv",
                        key=f"download_{tipo_robo}"
                    )
                    
                except Exception as e:
                    st.error(f"Error en la predicción: {str(e)}")
        
        # Mostrar última predicción si existe
        if f'ultima_prediccion_{tipo_robo}' in st.session_state:
            st.divider()
            st.subheader("📌 Última Predicción Guardada")
            df_ultima = st.session_state[f'ultima_prediccion_{tipo_robo}']
            st.write(f"Mes: {meses_nombre[st.session_state[f'ultimo_mes_{tipo_robo}']]}")
            st.dataframe(df_ultima, use_container_width=True)
    
    # ====================================================================
    # TAB 2: ANÁLISIS
    # ====================================================================
    
    with tab2:
        st.header("📊 Análisis de Datos")
        
        if f'ultima_prediccion_{tipo_robo}' in st.session_state:
            df_pred = st.session_state[f'ultima_prediccion_{tipo_robo}']
            mes_text = meses_nombre[st.session_state[f'ultimo_mes_{tipo_robo}']]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Distribución de Robos Predichos")
                fig, ax = plt.subplots(figsize=(8, 5))
                ax.hist(df_pred['PREDICCION_ROBOS'], bins=15, color='steelblue', edgecolor='black')
                ax.set_xlabel('Número de Robos')
                ax.set_ylabel('Frecuencia')
                ax.set_title(f'Distribución - {mes_text}')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.subheader("Estadísticas Descriptivas")
                stats = {
                    'Media': df_pred['PREDICCION_ROBOS'].mean(),
                    'Mediana': df_pred['PREDICCION_ROBOS'].median(),
                    'Desv. Estándar': df_pred['PREDICCION_ROBOS'].std(),
                    'Mínimo': df_pred['PREDICCION_ROBOS'].min(),
                    'Máximo': df_pred['PREDICCION_ROBOS'].max()
                }
                
                stats_df = pd.DataFrame(list(stats.items()), columns=['Métrica', 'Valor'])
                stats_df['Valor'] = stats_df['Valor'].round(2)
                
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.info("Realiza una predicción primero para ver el análisis")
    
    # ====================================================================
    # TAB 3: COMPARACIÓN CON DATOS REALES
    # ====================================================================
    
    with tab3:
        st.header("🎯 Comparación: Predicción vs Datos Reales")
        
        if f'ultima_prediccion_{tipo_robo}' in st.session_state:
            df_pred = st.session_state[f'ultima_prediccion_{tipo_robo}']
            mes_predicho = st.session_state[f'ultimo_mes_{tipo_robo}']
            
            # Mes equivalente en el ciclo de datos reales
            mes_equivalente = ((mes_predicho - 1) % 12) + 1 if mes_predicho > 12 else mes_predicho
            
            # Obtener datos reales del mes equivalente
            if predictor.df_final is not None:
                df_real = predictor.df_final[predictor.df_final['MES_N'] == mes_equivalente].copy()
                df_real = df_real[['CUADRANTE', 'ROBOS_MES_N', 'POBLACION']].copy()
                df_real.rename(columns={'ROBOS_MES_N': 'ROBOS_REALES'}, inplace=True)
                
                # Merge con predicción
                df_comparacion = df_pred.copy()
                df_comparacion = df_comparacion.merge(
                    df_real[['CUADRANTE', 'ROBOS_REALES']],
                    on='CUADRANTE',
                    how='left'
                )
                
                # Calcular error
                df_comparacion['ERROR'] = (df_comparacion['PREDICCION_ROBOS'] - df_comparacion['ROBOS_REALES']).abs()
                df_comparacion['ERROR_PORCENTAJE'] = (df_comparacion['ERROR'] / (df_comparacion['ROBOS_REALES'] + 0.001) * 100)
                
                # Mostrar estadísticas
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    mae = df_comparacion['ERROR'].mean()
                    st.metric("Error Medio (MAE)", f"{mae:.2f} robos")
                
                with col2:
                    mape = df_comparacion['ERROR_PORCENTAJE'].mean()
                    st.metric("Error Porcentual Medio (MAPE)", f"{mape:.1f}%")
                
                with col3:
                    correlacion = df_comparacion['PREDICCION_ROBOS'].corr(df_comparacion['ROBOS_REALES'])
                    st.metric("Correlación", f"{correlacion:.4f}")
                
                st.divider()
                
                # Tabla comparativa
                st.subheader("Tabla Comparativa")
                
                df_tabla = df_comparacion[[
                    'CUADRANTE', 'ROBOS_REALES', 'PREDICCION_ROBOS', 'ERROR', 'ERROR_PORCENTAJE'
                ]].copy()
                
                df_tabla['ROBOS_REALES'] = df_tabla['ROBOS_REALES'].round(2)
                df_tabla['PREDICCION_ROBOS'] = df_tabla['PREDICCION_ROBOS'].round(2)
                df_tabla['ERROR'] = df_tabla['ERROR'].round(2)
                df_tabla['ERROR_PORCENTAJE'] = df_tabla['ERROR_PORCENTAJE'].round(1)
                
                df_tabla = df_tabla.sort_values('ERROR', ascending=False)
                df_tabla.index = df_tabla.index + 1
                
                st.dataframe(df_tabla, use_container_width=True, height=400)
                
                st.divider()
                
                # Gráfico comparativo
                st.subheader("Comparación Visual")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Predicción vs Datos Reales (Top 10)**")
                    df_top10 = df_comparacion.nlargest(10, 'PREDICCION_ROBOS').sort_values('PREDICCION_ROBOS')
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    x = np.arange(len(df_top10))
                    width = 0.35
                    
                    bars1 = ax.barh(x - width/2, df_top10['ROBOS_REALES'], width, label='Datos Reales', color='#FF6B6B')
                    bars2 = ax.barh(x + width/2, df_top10['PREDICCION_ROBOS'], width, label='Predicción', color='#4ECDC4')
                    
                    ax.set_ylabel('Cuadrante', fontweight='bold')
                    ax.set_xlabel('Número de Robos', fontweight='bold')
                    ax.set_title(f'Comparación: Real vs Predicción - Top 10 Cuadrantes', fontweight='bold')
                    ax.set_yticks(x)
                    ax.set_yticklabels(df_top10['CUADRANTE'].astype(str))
                    ax.legend()
                    ax.grid(axis='x', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.write("**Distribución de Errores**")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    
                    ax.hist(df_comparacion['ERROR_PORCENTAJE'], bins=20, color='steelblue', edgecolor='black', alpha=0.7)
                    ax.axvline(df_comparacion['ERROR_PORCENTAJE'].mean(), color='red', linestyle='--', linewidth=2, label=f"Media: {df_comparacion['ERROR_PORCENTAJE'].mean():.1f}%")
                    ax.axvline(df_comparacion['ERROR_PORCENTAJE'].median(), color='orange', linestyle='--', linewidth=2, label=f"Mediana: {df_comparacion['ERROR_PORCENTAJE'].median():.1f}%")
                    
                    ax.set_xlabel('Error Porcentual (%)', fontweight='bold')
                    ax.set_ylabel('Frecuencia', fontweight='bold')
                    ax.set_title('Distribución de Errores Porcentuales', fontweight='bold')
                    ax.legend()
                    ax.grid(axis='y', alpha=0.3)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                st.divider()
                
                # Scatter plot
                st.subheader("Scatter: Predicción vs Realidad")
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                scatter = ax.scatter(
                    df_comparacion['ROBOS_REALES'],
                    df_comparacion['PREDICCION_ROBOS'],
                    c=df_comparacion['ERROR'],
                    cmap='RdYlGn_r',
                    s=100,
                    alpha=0.6,
                    edgecolors='black'
                )
                
                # Línea perfecta (y=x)
                min_val = min(df_comparacion['ROBOS_REALES'].min(), df_comparacion['PREDICCION_ROBOS'].min())
                max_val = max(df_comparacion['ROBOS_REALES'].max(), df_comparacion['PREDICCION_ROBOS'].max())
                ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
                
                ax.set_xlabel('Datos Reales', fontweight='bold', fontsize=11)
                ax.set_ylabel('Predicción', fontweight='bold', fontsize=11)
                ax.set_title('Predicción vs Datos Reales (Coloreado por Error)', fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
                
                cbar = plt.colorbar(scatter, ax=ax)
                cbar.set_label('Error Absoluto', fontweight='bold')
                
                plt.tight_layout()
                st.pyplot(fig)
                
            else:
                st.warning("No hay datos reales disponibles para comparar")
        else:
            st.info("Realiza una predicción primero para ver la comparación")
    
    # ====================================================================
    # TAB 4: INFORMACIÓN
    # ====================================================================
    
    with tab4:
        st.header("ℹ️ Información del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Desempeño del Modelo")
            st.metric("R² Score", f"{predictor.r2_score:.4f}")
            st.caption("Qué tan bien el modelo explica la variabilidad en los datos")
        
        with col2:
            st.subheader("Tipo de Robo Actual")
            st.write(f"**{tipo_robo_display}**")
            
            tipo_descripciones = {
                'casa': 'Predicción de robos a casa habitación por cuadrante',
                'negocio': 'Predicción de robos a negocios por cuadrante',
                'vehiculo': 'Predicción de robos de vehículos por cuadrante'
            }
            
            st.write(tipo_descripciones[tipo_robo])
        
        st.divider()
        
        st.subheader("Descripción del Modelo")
        st.write("""
        Este modelo utiliza **redes neuronales profundas** (Deep Learning) para predecir 
        el número de robos por cuadrante y mes.
        
        **Variables de entrada:**
        - Número de Cuadrante
        - Población
        - Robos del mes anterior (N-1)
        - Robos de dos meses atrás (N-2)
        
        **Arquitectura:**
        - Capa de entrada (4 características)
        - 2 capas ocultas (512 y 256 neuronas)
        - Dropout para regularización
        - Salida: predicción de robos
        
        **Parámetros:**
        - Optimizer: Adam (learning_rate=0.0005)
        - Loss: Mean Squared Error (MSE)
        - Epochs: 200 máx con Early Stopping
        """)
        
        st.divider()
        
        st.subheader("📋 Cómo Usar")
        st.markdown("""
        1. **Selecciona el tipo de robo** en la barra lateral
        2. **Carga un archivo Excel** (opcional, si quieres entrenar con nuevos datos)
        3. **Selecciona el mes** a predecir
        4. **Ajusta el número de cuadrantes** a mostrar
        5. **Haz clic en "Realizar Predicción"**
        6. **Analiza los resultados** en las pestañas de Análisis
        7. **Descarga los resultados** en CSV si lo necesitas
        """)

else:
    st.warning("⚠️ El modelo no está disponible")
    st.info("""
    Para comenzar:
    1. Cargue un archivo Excel en la barra lateral, O
    2. Asegúrese de que existe un modelo pre-entrenado para este tipo de robo
    """)
