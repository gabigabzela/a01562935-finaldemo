# 🔐 Predictor Multi-Tipo de Robos - Streamlit

Una aplicación web interactiva que permite predecir robos de **tres tipos diferentes** (Casa Habitación, Negocios y Vehículos) usando redes neuronales profundas con **Streamlit**.

## 🎯 Características Principales

### ✨ Multi-Tipo de Robos
- **🏠 Casa Habitación**: Predicción de robos a residencias
- **🏢 Negocios**: Predicción de robos comerciales
- **🚗 Vehículos**: Predicción de robos de automóviles

### 📈 Funcionalidades
- ✅ Selector interactivo de tipo de robo
- ✅ Predicción por mes (Enero a Diciembre del próximo ciclo)
- ✅ Tabla de Top N cuadrantes con mayor riesgo
- ✅ Visualizaciones con gráficos interactivos
- ✅ Estadísticas descriptivas y análisis
- ✅ Exportación de resultados en CSV
- ✅ Modelos separados y entrenados para cada tipo de robo

### 🎨 Interfaz de Usuario
- 3 pestañas principales: Predicción, Análisis, Información
- Colores distintivos para cada tipo de robo
- Controles intuitivos y responsivos
- Métrica R² Score visible

---

## 🚀 Cómo Ejecutar

### Instalación de Dependencias

```bash
pip install -r requirements.txt
```

### Ejecutar la Aplicación

```bash
streamlit run app_multimodelo.py
```

La aplicación se abrirá en: **http://localhost:8501**

---

## 📋 Estructura del Proyecto

```
/workspaces/A01567178-EDA/
├── app_multimodelo.py              # Aplicación Streamlit multi-tipo
├── prediction_models_all.py        # Clase ModeloPredictorRobos
├── requirements.txt                # Dependencias
├── modelo_robos_casa.pkl           # Modelo Casa (se crea automáticamente)
├── modelo_robos_negocio.pkl        # Modelo Negocios (se crea automáticamente)
├── modelo_robos_vehiculo.pkl       # Modelo Vehículos (se crea automáticamente)
└── exportados/
    ├── Robos a casa habitacion/
    │   └── cuadrantes_robos_CaHa_promedio.xlsx
    ├── Robos a negocios/
    │   └── cuadrantes_robos_negocios_promedio.xlsx
    └── Robos de vehiculos/
        └── cuadrantes_robos_vehiculos_promedio.xlsx
```

---

## 🔧 Arquitectura Técnica

### Clase Principal: `ModeloPredictorRobos`

```python
from prediction_models_all import ModeloPredictorRobos

# Crear predictor para Casa Habitación
predictor = ModeloPredictorRobos('casa')

# O para otros tipos
predictor = ModeloPredictorRobos('negocio')
predictor = ModeloPredictorRobos('vehiculo')
```

### Métodos Principales

```python
# Entrenar desde cero
predictor.entrenar_completo(file_path='ruta/archivo.xlsx')

# Cargar modelo preentrenado
predictor.cargar_modelo()

# Realizar predicción
top_cuadrantes = predictor.predecir_top_cuadrantes(
    mes_a_predecir=13,  # Enero
    top_n=10             # Top 10 cuadrantes
)

# Guardar modelo
predictor.guardar_modelo()
```

---

## 📊 Red Neuronal

```
INPUT LAYER (4 features)
    ↓
Dense 512 + ReLU + Dropout(0.3)
    ↓
Dense 256 + ReLU + Dropout(0.3)
    ↓
OUTPUT LAYER (78 neuronas - uno por cuadrante)
```

### Variables de Entrada
- Número de Cuadrante
- Población
- Robos del mes anterior (N-1)
- Robos de dos meses atrás (N-2)

### Parámetros de Entrenamiento
- **Optimizer**: Adam (lr=0.0005)
- **Loss**: Mean Squared Error (MSE)
- **Epochs**: 200 máximo
- **Early Stopping**: paciencia=15 épocas
- **Batch Size**: 2
- **Escalador**: MinMaxScaler (0-1)

---

## 💻 Cómo Usar la Aplicación

### Paso 1: Seleccionar Tipo de Robo
En la barra lateral, selecciona el tipo de robo:
- 🏠 Casa Habitación
- 🏢 Negocios
- 🚗 Vehículos

### Paso 2: Cargar Datos (Opcional)
- Si quieres entrenar con nuevos datos, sube un archivo Excel
- Si no, se usará el modelo preentrenado (más rápido)

### Paso 3: Realizar Predicción
1. Selecciona el mes a predecir (Enero a Diciembre)
2. Ajusta el número de cuadrantes a mostrar (5-30)
3. Haz clic en "🔮 Realizar Predicción"

### Paso 4: Analizar Resultados
- **Pestaña Predicción**: Tabla y gráfico con resultados
- **Pestaña Análisis**: Distribución y estadísticas
- **Pestaña Información**: Detalles del modelo

### Paso 5: Descargar Resultados
- Botón "📥 Descargar Resultados (CSV)" para exportar

---

## 📄 Formato del Archivo Excel

Cada archivo debe contener:

**Para Casa Habitación**:
```
CUADRANTE | POBLACION | PROMEDIO DE ROBOS A CASA HABITACION MES 1 | ... | MES 12
```

**Para Negocios**:
```
CUADRANTE | POBLACION | PROMEDIO DE ROBOS A NEGOCIOS MES 1 | ... | MES 12
```

**Para Vehículos**:
```
CUADRANTE | POBLACION | PROMEDIO DE ROBOS DE VEHICULOS MES 1 | ... | MES 12
```

---

## 🎯 Flujo de Datos

```
1. Usuario Selecciona Tipo de Robo
        ↓
2. Carga Archivo Excel (opcional)
        ↓
3. Datos se Preparan y Normalizan
        ↓
4. Red Neuronal se Entrena o Carga
        ↓
5. Usuario Selecciona Mes y Parámetros
        ↓
6. Modelo Realiza Predicción
        ↓
7. Resultados se Muestran en Tablas y Gráficos
        ↓
8. Usuario Descarga Resultados (opcional)
```

---

## 📊 Salida Esperada

La predicción devuelve un DataFrame con:

| CUADRANTE | PREDICCION_ROBOS | POBLACION |
|-----------|------------------|-----------|
| 1001      | 15.3            | 5000      |
| 1002      | 18.7            | 6500      |
| ...       | ...             | ...       |

---

## 🔐 Seguridad y Privacidad

- ✅ Aplicación **100% local** - no se envía datos a internet
- ✅ Modelos guardados en tu máquina
- ✅ Sin conexión con servidores externos
- ✅ Datos procesados solo en tu dispositivo

---

## 📈 Métrica de Desempeño

**R² Score**: Indica qué tan bien el modelo explica la variabilidad en los datos
- 0.8-1.0: Excelente
- 0.6-0.8: Bueno
- 0.4-0.6: Moderado
- <0.4: Pobre

---

## 🛠️ Personalización

### Cambiar Rango de Cuadrantes

En `app_multimodelo.py`, línea 180:
```python
top_n = st.slider(
    "Número de cuadrantes en el Top:",
    min_value=5,       # Cambiar aquí
    max_value=50,      # O aquí
    value=10,
    step=1
)
```

### Cambiar Meses Predichos

En `app_multimodelo.py`, línea 175:
```python
# Agregar meses 25 en adelante
meses_nombre = {
    ...
    25: "Enero (Año 2)",
    26: "Febrero (Año 2)",
}
```

### Ajustar Arquitectura del Modelo

En `prediction_models_all.py`, línea 180:
```python
modelo = Sequential([
    Dense(512, activation='relu', input_shape=(input_dim,)),  # Cambiar 512
    Dense(256, activation='relu'),                            # Cambiar 256
    Dense(output_dim, activation='linear')
])
```

---

## 📚 Documentación Adicional

Consulta estos archivos para más información:
- `README_APP.md` - Documentación técnica detallada
- `GUIA_RAPIDA.md` - Guía de inicio rápido
- `TROUBLESHOOTING.md` - Solución de problemas
- `ARQUITECTURA.md` - Diagramas técnicos

---

## 🐛 Solución de Problemas

### Error: "ModuleNotFoundError: No module named 'streamlit'"
```bash
pip install -r requirements.txt
```

### Error: "Excel file not found"
Asegúrate de que los archivos están en:
```
exportados/Robos a casa habitacion/cuadrantes_robos_CaHa_promedio.xlsx
exportados/Robos a negocios/cuadrantes_robos_negocios_promedio.xlsx
exportados/Robos de vehiculos/cuadrantes_robos_vehiculos_promedio.xlsx
```

### El entrenamiento es lento
Es normal la primera vez (1-2 minutos). Los siguientes usos cargarán el modelo preentrenado (< 1 segundo).

---

## 📞 Soporte

Si encuentras problemas:
1. Verifica que Python >= 3.8
2. Instala todas las dependencias: `pip install -r requirements.txt`
3. Revisa `TROUBLESHOOTING.md`

---

## 🎉 ¡Listo para Usar!

```bash
streamlit run app_multimodelo.py
```

Accede a **http://localhost:8501** y comienza a predecir! 🚀
