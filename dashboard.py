import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import warnings
import hashlib

try:
    import folium
    from streamlit_folium import st_folium
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

warnings.filterwarnings('ignore')

# Configurar la página
st.set_page_config(
    page_title="FICOTEC - Sistema de Análisis de Robos",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configuración de rendimiento
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.load_count = 0

# Sistema de autenticación
def check_password():
    """Retorna True si el usuario ingresó la contraseña correcta."""
    
    def password_entered():
        """Verifica si la contraseña ingresada es correcta."""
        if st.session_state["username"] in st.secrets.get("passwords", {}) and \
           hashlib.sha256(st.session_state["password"].encode()).hexdigest() == \
           st.secrets["passwords"][st.session_state["username"]]:
            st.session_state["password_correct"] = True
            st.session_state["user_name"] = st.session_state["username"]
            del st.session_state["password"]  # No guardar la contraseña
            del st.session_state["username"]
        else:
            st.session_state["password_correct"] = False

    # Primera vez o si no ha ingresado correctamente
    if "password_correct" not in st.session_state:
        # Ocultar sidebar y elementos de navegación
        st.markdown("""
            <style>
            /* Ocultar sidebar y header en login */
            [data-testid="stSidebar"] {
                display: none;
            }
            header {
                display: none !important;
            }
            
            /* Fondo oscuro general */
            .stApp {
                background: linear-gradient(135deg, #0a0e27 0%, #1a1d35 100%);
            }
            
            /* Contenedor de login */
            .login-container {
                max-width: 450px;
                margin: 8vh auto;
                padding: 3rem;
                background: linear-gradient(145deg, #1e2139 0%, #161829 100%);
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5),
                           0 0 100px rgba(79, 70, 229, 0.1);
                border: 1px solid rgba(79, 70, 229, 0.2);
            }
            
            /* Logo y título */
            .login-logo {
                text-align: center;
                font-size: 4rem;
                margin-bottom: 1rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: bold;
            }
            
            .login-title {
                text-align: center;
                color: #ffffff;
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            
            .login-subtitle {
                text-align: center;
                color: #9ca3af;
                font-size: 1rem;
                margin-bottom: 2rem;
            }
            
            /* Inputs personalizados */
            .stTextInput input {
                background-color: #0f1419 !important;
                border: 2px solid #2d3748 !important;
                border-radius: 10px !important;
                padding: 12px 16px !important;
                color: #ffffff !important;
                font-size: 1rem !important;
                transition: all 0.3s ease !important;
            }
            
            .stTextInput input:focus {
                border-color: #667eea !important;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
            }
            
            /* Botón de login */
            .stButton button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 12px 24px !important;
                font-size: 1.1rem !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            }
            
            .stButton button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
            }
            
            /* Footer del login */
            .login-footer {
                text-align: center;
                color: #6b7280;
                font-size: 0.875rem;
                margin-top: 2rem;
                padding-top: 1.5rem;
                border-top: 1px solid #2d3748;
            }
            
            /* Icono de seguridad */
            .security-badge {
                text-align: center;
                margin-top: 1.5rem;
                color: #4ade80;
                font-size: 0.875rem;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="login-container">
                <div class="login-logo">🔐</div>
                <div class="login-title">FICOTEC</div>
                <div class="login-subtitle">Sistema de Análisis de Robos</div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("👤 Usuario", key="username", placeholder="Ingresa tu usuario", label_visibility="collapsed")
            st.text_input("🔑 Contraseña", type="password", key="password", placeholder="Ingresa tu contraseña", label_visibility="collapsed")
            st.button("🚀 Iniciar Sesión", on_click=password_entered, use_container_width=True)
            
            st.markdown("""
                <div class="security-badge">
                    🔒 Conexión segura con encriptación SHA-256
                </div>
                <div class="login-footer">
                    © 2025 FICOTEC - Análisis Predictivo de Seguridad
                </div>
            """, unsafe_allow_html=True)
        
        return False
    
    # Contraseña incorrecta
    elif not st.session_state["password_correct"]:
        st.markdown("""
            <style>
            [data-testid="stSidebar"] {
                display: none;
            }
            header {
                display: none !important;
            }
            .stApp {
                background: linear-gradient(135deg, #0a0e27 0%, #1a1d35 100%);
            }
            .login-container {
                max-width: 450px;
                margin: 8vh auto;
                padding: 3rem;
                background: linear-gradient(145deg, #1e2139 0%, #161829 100%);
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5),
                           0 0 100px rgba(79, 70, 229, 0.1);
                border: 1px solid rgba(79, 70, 229, 0.2);
            }
            .login-logo {
                text-align: center;
                font-size: 4rem;
                margin-bottom: 1rem;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-weight: bold;
            }
            .login-title {
                text-align: center;
                color: #ffffff;
                font-size: 2rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }
            .login-subtitle {
                text-align: center;
                color: #9ca3af;
                font-size: 1rem;
                margin-bottom: 2rem;
            }
            .stTextInput input {
                background-color: #0f1419 !important;
                border: 2px solid #2d3748 !important;
                border-radius: 10px !important;
                padding: 12px 16px !important;
                color: #ffffff !important;
                font-size: 1rem !important;
                transition: all 0.3s ease !important;
            }
            .stTextInput input:focus {
                border-color: #667eea !important;
                box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
            }
            .stButton button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 12px 24px !important;
                font-size: 1.1rem !important;
                font-weight: 600 !important;
                transition: all 0.3s ease !important;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
            }
            .stButton button:hover {
                transform: translateY(-2px) !important;
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
            }
            .security-badge {
                text-align: center;
                margin-top: 1.5rem;
                color: #4ade80;
                font-size: 0.875rem;
            }
            .login-footer {
                text-align: center;
                color: #6b7280;
                font-size: 0.875rem;
                margin-top: 2rem;
                padding-top: 1.5rem;
                border-top: 1px solid #2d3748;
            }
            </style>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="login-container">
                <div class="login-logo">🔐</div>
                <div class="login-title">FICOTEC</div>
                <div class="login-subtitle">Sistema de Análisis de Robos</div>
            </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.text_input("👤 Usuario", key="username", placeholder="Ingresa tu usuario", label_visibility="collapsed")
            st.text_input("🔑 Contraseña", type="password", key="password", placeholder="Ingresa tu contraseña", label_visibility="collapsed")
            st.button("🚀 Iniciar Sesión", on_click=password_entered, use_container_width=True)
            st.error("❌ Usuario o contraseña incorrectos")
            
            st.markdown("""
                <div class="security-badge">
                    🔒 Conexión segura con encriptación SHA-256
                </div>
                <div class="login-footer">
                    © 2025 FICOTEC - Análisis Predictivo de Seguridad
                </div>
            """, unsafe_allow_html=True)
        
        return False
    
    # Contraseña correcta
    else:
        return True

# Verificar autenticación antes de mostrar el dashboard
if not check_password():
    st.stop()  # No continuar si no está autenticado

# Estilos personalizados - Tema Oscuro
st.markdown("""
    <style>
    /* Tema oscuro global */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar oscuro */
    [data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #fafafa;
    }
    
    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #ffffff;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 0 0 10px rgba(255, 255, 255, 0.3);
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    
    /* Metric cards */
    .metric-card {
        background-color: #1c1e26;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border: 1px solid #30363d;
    }
    
    /* Dataframes y tablas */
    [data-testid="stDataFrame"],
    .stDataFrame,
    div.stDataFrame > div {
        background-color: #161b22 !important;
    }
    
    [data-testid="stDataFrame"] div[role="grid"],
    .stDataFrame div[role="grid"] {
        background-color: #0d1117 !important;
    }
    
    [data-testid="stDataFrame"] thead,
    .stDataFrame thead {
        background-color: #161b22 !important;
    }
    
    [data-testid="stDataFrame"] th,
    .stDataFrame th,
    div[data-testid="stDataFrame"] th {
        background-color: #21262d !important;
        color: #ffffff !important;
        border-color: #30363d !important;
    }
    
    [data-testid="stDataFrame"] td,
    .stDataFrame td,
    div[data-testid="stDataFrame"] td {
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
        border-color: #30363d !important;
    }
    
    [data-testid="stDataFrame"] tr:hover,
    .stDataFrame tr:hover {
        background-color: #161b22 !important;
    }
    
    /* Selectores adicionales para tablas de Streamlit */
    .dataframe {
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
    }
    
    .dataframe thead tr th {
        background-color: #21262d !important;
        color: #ffffff !important;
    }
    
    .dataframe tbody tr td {
        background-color: #0d1117 !important;
        color: #c9d1d9 !important;
    }
    
    .dataframe tbody tr:hover {
        background-color: #161b22 !important;
    }
    
    /* Tabla interna de Streamlit */
    .element-container .stDataFrame {
        background-color: #161b22 !important;
    }
    
    [data-testid="stDataFrame"] .row-widget {
        background-color: #0d1117 !important;
    }
    
    /* Expanders */
    [data-testid="stExpander"] {
        background-color: #161b22;
        border: 1px solid #30363d;
    }
    
    /* Métricas */
    [data-testid="stMetricValue"] {
        color: #58a6ff;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #161b22;
        border-radius: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #8b949e;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        color: #58a6ff !important;
        border-bottom-color: #58a6ff !important;
    }
    
    /* Botones */
    .stButton button {
        background-color: #238636;
        color: white;
        border: none;
        border-radius: 6px;
    }
    
    .stButton button:hover {
        background-color: #2ea043;
    }
    
    /* Radio buttons */
    [data-testid="stRadio"] > div {
        background-color: #161b22;
        padding: 10px;
        border-radius: 6px;
    }
    
    /* Success/Info/Warning/Error boxes */
    .stSuccess {
        background-color: #0d1117;
        border-left: 4px solid #238636;
        color: #7ee787;
    }
    
    .stInfo {
        background-color: #0d1117;
        border-left: 4px solid #1f6feb;
        color: #58a6ff;
    }
    
    .stWarning {
        background-color: #0d1117;
        border-left: 4px solid #d29922;
        color: #f0b72f;
    }
    
    .stError {
        background-color: #0d1117;
        border-left: 4px solid #da3633;
        color: #ff7b72;
    }
    
    /* Divider */
    hr {
        border-color: #30363d;
    }
    
    /* Text color */
    p, span, label {
        color: #c9d1d9 !important;
    }
    
    /* Plotly charts dark theme */
    .js-plotly-plot {
        background-color: #161b22 !important;
    }
    
    .js-plotly-plot .plotly {
        background-color: #161b22 !important;
    }
    
    /* Select boxes y inputs */
    [data-baseweb="select"] {
        background-color: #161b22 !important;
        border-color: #30363d !important;
    }
    
    [data-baseweb="select"] > div {
        background-color: #161b22 !important;
        color: #ffffff !important;
    }
    
    input {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border-color: #30363d !important;
    }
    
    /* Selectbox y multiselect */
    .stSelectbox > div > div {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border-color: #30363d !important;
    }
    
    .stMultiSelect > div > div {
        background-color: #161b22 !important;
        border-color: #30363d !important;
    }
    
    /* Dropdown menu */
    [data-baseweb="popover"] {
        background-color: #161b22 !important;
    }
    
    [role="listbox"] {
        background-color: #161b22 !important;
    }
    
    [role="option"] {
        background-color: #161b22 !important;
        color: #c9d1d9 !important;
    }
    
    [role="option"]:hover {
        background-color: #21262d !important;
        color: #ffffff !important;
    }
    
    /* Date input */
    .stDateInput > div > div > input {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border-color: #30363d !important;
    }
    
    /* Number input */
    .stNumberInput > div > div > input {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border-color: #30363d !important;
    }
    
    /* Text input */
    .stTextInput > div > div > input {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border-color: #30363d !important;
    }
    
    /* Slider */
    .stSlider > div > div > div {
        background-color: #30363d !important;
    }
    
    .stSlider [role="slider"] {
        background-color: #58a6ff !important;
    }
    
    /* Checkbox */
    .stCheckbox {
        color: #c9d1d9 !important;
    }
    
    /* Expander header */
    .streamlit-expanderHeader {
        background-color: #161b22 !important;
        color: #ffffff !important;
        border-color: #30363d !important;
    }
    
    /* Markdown containers */
    [data-testid="stMarkdown"] {
        color: #c9d1d9;
    }
    
    /* Filtros específicos */
    .stForm {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 1rem !important;
    }
    </style>
""", unsafe_allow_html=True)

# Configuración de tema oscuro para Plotly
def apply_dark_theme(fig):
    """Aplica tema oscuro a gráficos de Plotly"""
    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#161b22',
        plot_bgcolor='#0d1117',
        font=dict(color='#c9d1d9'),
        title_font=dict(color='#ffffff'),
        legend=dict(
            bgcolor='#161b22',
            bordercolor='#30363d',
            font=dict(color='#c9d1d9')
        ),
        xaxis=dict(
            gridcolor='#30363d',
            linecolor='#30363d',
            showticklabels=True
        ),
        yaxis=dict(
            gridcolor='#30363d',
            linecolor='#30363d',
            showticklabels=True
        )
    )
    
    # Eliminar títulos de ejes si muestran "undefined" o están vacíos
    if fig.layout.xaxis.title.text in [None, '', 'undefined', 'x']:
        fig.update_xaxes(title_text='')
    if fig.layout.yaxis.title.text in [None, '', 'undefined', 'y']:
        fig.update_yaxes(title_text='')
    
    return fig

# Título principal
st.markdown("<h1 class='main-header'>Análisis de Robos - Exploratory Data Analysis</h1>", unsafe_allow_html=True)

# Sidebar para navegación
st.sidebar.title("Navegación")
page = st.sidebar.radio(
    "Selecciona una sección:",
    ["Dashboard Principal", "Análisis por Tipo", "Mapa", "Predicciones"]
)

# Cargar datos con caché mejorado
@st.cache_data(ttl=3600, show_spinner="Cargando datos principales...")
def load_data():
    """Cargar datos principales"""
    try:
        df = pd.read_csv('robos_tot_final.csv')
        return df
    except FileNotFoundError:
        st.error("Archivo 'robos_tot_final.csv' no encontrado")
        return None

@st.cache_data(ttl=3600, show_spinner="Cargando predicciones...")
def load_prediction_data(tipo_robo, mes='enero'):
    """Cargar datos de predicciones por tipo y mes"""
    mes_lower = mes.lower()
    paths = {
        'casa': f'exportados/Robos a casa habitacion/top_10_prediccion_robos_casa_habitacion_{mes_lower}.csv',
        'negocios': f'exportados/Robos a negocios/top_10_prediccion_robos_negocios_{mes_lower}.csv',
        'vehiculos': f'exportados/Robos de vehiculos/top_10_prediccion_robos_vehiculos_{mes_lower}.csv'
    }
    try:
        df = pd.read_csv(paths[tipo_robo])
        return df
    except (FileNotFoundError, KeyError):
        # Si no existe el archivo para ese mes, intentar con enero por defecto
        try:
            default_paths = {
                'casa': 'exportados/Robos a casa habitacion/top_10_prediccion_robos_casa_habitacion_enero.csv',
                'negocios': 'exportados/Robos a negocios/top_10_prediccion_robos_negocios_enero.csv',
                'vehiculos': 'exportados/Robos de vehiculos/top_10_prediccion_robos_vehiculos_enero.csv'
            }
            df = pd.read_csv(default_paths[tipo_robo])
            return df
        except (FileNotFoundError, KeyError):
            return None

@st.cache_data(ttl=3600)
def get_cuadrante_coords(df_principal):
    """Obtener mapeo de cuadrantes a coordenadas promedio"""
    if df_principal is None:
        return None
    cuadrante_coords = df_principal.groupby('CUADRANTE')[['LATITUD', 'LONGITUD']].mean().reset_index()
    return cuadrante_coords

@st.cache_data(ttl=3600)
def add_coordinates_to_predictions(df_pred, cuadrante_coords, cuadrante_col='CUADRANTE'):
    """Agregar coordenadas a predicciones basadas en cuadrante"""
    if df_pred is None or cuadrante_coords is None:
        return df_pred
    if cuadrante_col in df_pred.columns:
        df_pred = df_pred.merge(cuadrante_coords, how='left', left_on=cuadrante_col, right_on='CUADRANTE')
        return df_pred
    return df_pred

# Cargar datos principales una sola vez
df_principal = load_data()

# Calcular coordenadas una sola vez si existen datos
cuadrante_coords = get_cuadrante_coords(df_principal) if df_principal is not None else None

# Filtros de tipos de robo en sidebar
if df_principal is not None:
    st.sidebar.divider()
    st.sidebar.title(" Filtros")
    
    if 'TIPO' in df_principal.columns:
        tipos_disponibles = df_principal['TIPO'].unique().tolist()
        tipos_disponibles.insert(0, "Todos")
        
        tipo_seleccionado = st.sidebar.selectbox(
            "Tipo de robo:",
            tipos_disponibles
        )
        
        # Filtrar datos según el tipo seleccionado
        if tipo_seleccionado != "Todos":
            df_filtrado = df_principal[df_principal['TIPO'] == tipo_seleccionado]
        else:
            df_filtrado = df_principal.copy()
    else:
        df_filtrado = df_principal.copy()
    
    # Filtro de años
    if 'AÑO' in df_principal.columns:
        años_disponibles = sorted(df_principal['AÑO'].unique().tolist())
        años_seleccionados = st.sidebar.multiselect(
            "Años:",
            años_disponibles,
            default=años_disponibles
        )
        
        # Aplicar filtro de años
        if años_seleccionados:
            df_filtrado = df_filtrado[df_filtrado['AÑO'].isin(años_seleccionados)]
    
    # Filtro de distrito
    distrito_seleccionado = None
    if 'DISTRITO' in df_principal.columns:
        distritos_disponibles = sorted(df_principal['DISTRITO'].unique().tolist())
        distritos_disponibles.insert(0, "Todos")
        
        distrito_seleccionado = st.sidebar.selectbox(
            "Filtrar por Distrito:",
            distritos_disponibles
        )
        
        # Aplicar filtro de distrito si es específico
        if distrito_seleccionado != "Todos":
            df_filtrado = df_filtrado[df_filtrado['DISTRITO'] == distrito_seleccionado]
    
    st.sidebar.info(f"Registros filtrados: {len(df_filtrado):,}")
else:
    df_filtrado = None

if df_principal is None:
    st.error("No se pudieron cargar los datos. Verifica que 'robos_tot_final.csv' exista en el directorio principal.")
else:
    # ============ DASHBOARD PRINCIPAL ============
    if page == "Dashboard Principal":
        st.header("Dashboard Principal")
        
        # Calcular métricas principales
        total_robos = len(df_principal)
        robos_violentos = len(df_principal[df_principal['VIOLENCIA'].astype(str).str.upper() == 'SI']) if 'VIOLENCIA' in df_principal.columns else 0
        distritos_afectados = df_principal['DISTRITO'].nunique() if 'DISTRITO' in df_principal.columns else 0
        cuadrantes = df_principal['CUADRANTE'].nunique() if 'CUADRANTE' in df_principal.columns else 0
        
        # Mostrar métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="Total de Robos",
                value=f"{total_robos:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                label="Robos Violentos",
                value=f"{robos_violentos:,}",
                delta=None
            )
        
        with col3:
            st.metric(
                label="Distritos Afectados",
                value=f"{distritos_afectados}",
                delta=None
            )
        
        with col4:
            st.metric(
                label="Cuadrantes",
                value=f"{cuadrantes}",
                delta=None
            )
        
        st.divider()
        
        # Visión General con gráficos
        st.subheader(" Visión General")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Robos por Tipo**")
            if 'TIPO' in df_filtrado.columns:
                tipo_counts = df_filtrado['TIPO'].value_counts().head(10)
                fig_tipo = px.bar(
                    x=tipo_counts.values,
                    y=tipo_counts.index,
                    orientation='h',
                    color=tipo_counts.values,
                    color_continuous_scale='Viridis'
                )
                fig_tipo.update_layout(
                    height=350, 
                    showlegend=False, 
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                fig_tipo = apply_dark_theme(fig_tipo)
                st.plotly_chart(fig_tipo, use_container_width=True, config={'displayModeBar': False})
        
        with col2:
            st.write("**Robos: Violencia**")
            if 'VIOLENCIA' in df_filtrado.columns:
                # Limpiar datos y normalizar valores
                violencia_clean = df_filtrado['VIOLENCIA'].fillna('NO').astype(str).str.strip().str.upper()
                # Reemplazar variaciones comunes
                violencia_clean = violencia_clean.replace({
                    'S': 'SI',
                    'N': 'NO',
                    'SÍ': 'SI',
                    '1': 'SI',
                    '0': 'NO',
                    'TRUE': 'SI',
                    'FALSE': 'NO',
                    'NAN': 'NO',
                    'NONE': 'NO',
                    '': 'NO'
                })
                
                violencia_counts = violencia_clean.value_counts()
                
                # Asegurar que SI y NO existan
                si_count = violencia_counts.get('SI', 0)
                no_count = violencia_counts.get('NO', 0)
                
                # Si hay valores que no son SI/NO, agregarlos a NO
                otros_count = sum(violencia_counts[~violencia_counts.index.isin(['SI', 'NO'])])
                no_count += otros_count
                
                violencia_data = pd.DataFrame({
                    'Violencia': ['Con Violencia', 'Sin Violencia'],
                    'Cantidad': [si_count, no_count]
                })
                
                fig_violencia = px.pie(
                    violencia_data,
                    values='Cantidad',
                    names='Violencia',
                    color_discrete_map={
                        'Con Violencia': '#EF553B',
                        'Sin Violencia': '#00CC96'
                    },
                    hole=0
                )
                fig_violencia.update_layout(
                    height=350, 
                    margin=dict(l=0, r=0, t=0, b=0),
                    showlegend=True
                )
                fig_violencia.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>'
                )
                fig_violencia = apply_dark_theme(fig_violencia)
                st.plotly_chart(fig_violencia, use_container_width=True, config={'displayModeBar': False})
        
        st.divider()
        
        # ============ ANÁLISIS TEMPORAL ============
        st.subheader(" Análisis Temporal")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Evolución Anual**")
            if 'AÑO' in df_principal.columns:
                anual_data = df_principal['AÑO'].value_counts().sort_index()
                fig_anual = px.line(
                    x=anual_data.index,
                    y=anual_data.values,
                    markers=True,
                    title='Evolución Anual de Robos',
                    labels={'x': 'Año', 'y': 'Cantidad de Robos'}
                )
                fig_anual.update_traces(line=dict(color='#1f77b4', width=3), marker=dict(size=10))
                fig_anual.update_layout(height=400, hovermode='x unified')
                fig_anual = apply_dark_theme(fig_anual)
                st.plotly_chart(fig_anual, use_container_width=True)
        
        with col2:
            st.write("**Distribución Mensual (Promedio Histórico)**")
            if 'MES' in df_principal.columns:
                # Mapeo de números de mes a nombres
                mes_nombres = {
                    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
                }
                
                # Convertir MES a numérico si es necesario
                df_principal['MES_NUM'] = pd.to_numeric(df_principal['MES'], errors='coerce')
                
                mensual_data = df_principal['MES_NUM'].value_counts().sort_index()
                # Calcular promedio histórico
                promedio_mensual = mensual_data / df_principal['AÑO'].nunique() if 'AÑO' in df_principal.columns else mensual_data
                
                # Crear etiquetas de meses
                mes_labels = [mes_nombres.get(int(mes), f'Mes {mes}') for mes in promedio_mensual.index]
                
                fig_mensual = px.bar(
                    x=mes_labels,
                    y=promedio_mensual.values,
                    title='Distribución Mensual (Promedio Histórico)',
                    labels={'x': 'Mes', 'y': 'Promedio de Robos'},
                    color=promedio_mensual.values,
                    color_continuous_scale='Reds'
                )
                fig_mensual.update_layout(height=400, showlegend=False)
                fig_mensual = apply_dark_theme(fig_mensual)
                st.plotly_chart(fig_mensual, use_container_width=True)
        
        st.divider()
        
        # ============ ANÁLISIS GEOGRÁFICO ============
        st.subheader("Análisis Geográfico")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top 10 Distritos**")
            if 'DISTRITO' in df_principal.columns:
                top_10_distritos = df_principal['DISTRITO'].value_counts().head(10)
                fig_distritos = px.bar(
                    x=top_10_distritos.values,
                    y=top_10_distritos.index,
                    orientation='h',
                    title='Top 10 Distritos Afectados',
                    labels={'x': 'Cantidad de Robos', 'y': 'Distrito'},
                    color=top_10_distritos.values,
                    color_continuous_scale='Oranges'
                )
                fig_distritos.update_layout(height=400, showlegend=False)
                fig_distritos = apply_dark_theme(fig_distritos)
                st.plotly_chart(fig_distritos, use_container_width=True)
        
        with col2:
            st.write("**Top 10 Cuadrantes**")
            if 'CUADRANTE' in df_principal.columns:
                top_10_cuadrantes = df_principal['CUADRANTE'].value_counts().head(10)
                fig_cuadrantes = px.bar(
                    x=top_10_cuadrantes.values,
                    y=top_10_cuadrantes.index,
                    orientation='h',
                    title='Top 10 Cuadrantes Afectados',
                    labels={'x': 'Cantidad de Robos', 'y': 'Cuadrante'},
                    color=top_10_cuadrantes.values,
                    color_continuous_scale='Blues'
                )
                fig_cuadrantes.update_layout(height=400, showlegend=False)
                fig_cuadrantes = apply_dark_theme(fig_cuadrantes)
                st.plotly_chart(fig_cuadrantes, use_container_width=True)
        
        st.divider()
        
        # ============ ESTADÍSTICAS CLAVE ============
        st.subheader(" Estadísticas Clave")
        
        col1, col2, col3 = st.columns([1, 1.2, 1])
        
        # COLUMNA 1: Gráfico de Estación
        with col1:
            st.write("**Distribución por Estación**")
            if 'ESTACION' in df_principal.columns:
                estacion_counts = df_principal['ESTACION'].value_counts()
                fig_estacion = px.pie(
                    values=estacion_counts.values,
                    names=estacion_counts.index,
                    title='',
                    hole=0
                )
                fig_estacion.update_layout(
                    height=350, 
                    showlegend=True,
                    margin=dict(l=0, r=0, t=0, b=0),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )
                fig_estacion.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    hovertemplate='<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>'
                )
                fig_estacion = apply_dark_theme(fig_estacion)
                st.plotly_chart(fig_estacion, use_container_width=True, config={'displayModeBar': False})
        
        # COLUMNA 2: Tarjetas de Métricas
        with col2:
            if 'AÑO' in df_principal.columns:
                año_max = df_principal['AÑO'].value_counts().idxmax()
                año_count = df_principal['AÑO'].value_counts().max()
                st.markdown(f"""
                    <div style='background-color: #667eea; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                        <p style='margin: 0; font-size: 0.9rem; color: #ffffff; line-height: 1.6;'>
                            El <strong>año con mayor incidencia</strong> fue <span style='color: #fff; font-weight: bold; font-size: 1.05rem;'>{int(año_max)}</span> 
                            con <span style='color: #fff; font-weight: bold;'>{año_count:,}</span> robos.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            if 'MES' in df_principal.columns:
                mes_nombres = {
                    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
                }
                df_principal['MES_NUM'] = pd.to_numeric(df_principal['MES'], errors='coerce')
                mes_max = int(df_principal['MES_NUM'].value_counts().idxmax())
                mes_count = df_principal['MES_NUM'].value_counts().max()
                mes_nombre = mes_nombres.get(mes_max, f'Mes {mes_max}')
                st.markdown(f"""
                    <div style='background-color: #f5576c; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                        <p style='margin: 0; font-size: 0.9rem; color: #ffffff; line-height: 1.6;'>
                            El <strong>mes con mayor robo</strong> fue <span style='color: #fff; font-weight: bold; font-size: 1.05rem;'>{mes_nombre}</span> 
                            con <span style='color: #fff; font-weight: bold;'>{mes_count:,}</span> robos.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
            
            if 'VIOLENCIA' in df_principal.columns:
                total_violentos = len(df_principal[df_principal['VIOLENCIA'].astype(str).str.upper() == 'SI'])
                tasa_violencia = (total_violentos / len(df_principal) * 100)
                st.markdown(f"""
                    <div style='background-color: #ffa500; padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;'>
                        <p style='margin: 0; font-size: 0.9rem; color: #ffffff; line-height: 1.6;'>
                            La <strong>tasa de violencia general</strong> es del <span style='color: #fff; font-weight: bold; font-size: 1.05rem;'>{tasa_violencia:.2f}%</span> 
                            con <span style='color: #fff; font-weight: bold;'>{total_violentos:,}</span> robos violentos.
                        </p>
                    </div>
                """, unsafe_allow_html=True)
        
        # COLUMNA 3: Tasa de violencia por tipo de robo
        with col3:
            st.write("**Tasa de Violencia por Tipo**")
            if 'TIPO' in df_principal.columns and 'VIOLENCIA' in df_principal.columns:
                # Crear tabla de tasa de violencia por tipo
                tipos_robo = df_principal['TIPO'].unique()
                tasa_datos = []
                
                for tipo in tipos_robo:
                    subset = df_principal[df_principal['TIPO'] == tipo]
                    total = len(subset)
                    violentos = len(subset[subset['VIOLENCIA'].astype(str).str.upper() == 'SI'])
                    tasa = (violentos / total * 100) if total > 0 else 0
                    tasa_datos.append({
                        'Tipo': tipo,
                        'Total': total,
                        'Violentos': violentos,
                        'Tasa': tasa
                    })
                
                tasa_df = pd.DataFrame(tasa_datos).sort_values('Total', ascending=False)
                
                # Mostrar como texto simple
                for idx, row in tasa_df.iterrows():
                    st.write(f"**{row['Tipo']}**")
                    st.write(f"<p style='font-size: 1.5rem; font-weight: bold; margin: -0.5rem 0 0.8rem 0;'>{row['Tasa']:.1f}%</p>", unsafe_allow_html=True)
            else:
                st.info("No hay datos suficientes")
    
    # ============ ANÁLISIS POR TIPO ============
    elif page == "Análisis por Tipo":
        st.header("Análisis por Tipo de Robo")
        
        if 'TIPO' in df_principal.columns:
            tipos_robo = sorted(df_principal['TIPO'].unique().tolist())
            
            if tipos_robo:
                tabs = st.tabs([f" {tipo}" for tipo in tipos_robo])
                
                for idx, tipo in enumerate(tipos_robo):
                    with tabs[idx]:
                        st.subheader(f"Análisis: {tipo}")
                        
                        subset = df_principal[df_principal['TIPO'] == tipo]
                        
                        # Métricas
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total de Robos", f"{len(subset):,}")
                        with col2:
                            violentos = len(subset[subset['VIOLENCIA'].astype(str).str.upper() == 'SI'])
                            st.metric("Robos Violentos", f"{violentos:,}")
                        with col3:
                            tasa_violencia = (violentos / len(subset) * 100) if len(subset) > 0 else 0
                            st.metric("Tasa de Violencia", f"{tasa_violencia:.2f}%")
                        with col4:
                            distritos = subset['DISTRITO'].nunique() if 'DISTRITO' in subset.columns else 0
                            st.metric("Distritos Afectados", f"{distritos}")
                        
                        st.divider()
                        
                        # Visualizaciones
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Tendencia por Año**")
                            if 'AÑO' in subset.columns:
                                anual_data = subset['AÑO'].value_counts().sort_index()
                                fig_anual = px.line(
                                    x=anual_data.index,
                                    y=anual_data.values,
                                    markers=True,
                                    title=f'Evolución Anual - {tipo}',
                                    labels={'x': 'Año', 'y': 'Cantidad'}
                                )
                                fig_anual.update_traces(line=dict(color='#1f77b4', width=3), marker=dict(size=8))
                                fig_anual.update_layout(height=350, hovermode='x unified')
                                fig_anual = apply_dark_theme(fig_anual)
                                st.plotly_chart(fig_anual, use_container_width=True)
                        
                        with col2:
                            st.write("**Distribución por Mes**")
                            if 'MES' in subset.columns:
                                mes_nombres = {
                                    1: 'Enero', 2: 'Febrero', 3: 'Marzo', 4: 'Abril',
                                    5: 'Mayo', 6: 'Junio', 7: 'Julio', 8: 'Agosto',
                                    9: 'Septiembre', 10: 'Octubre', 11: 'Noviembre', 12: 'Diciembre'
                                }
                                subset_mes = subset.copy()
                                subset_mes['MES_NUM'] = pd.to_numeric(subset_mes['MES'], errors='coerce')
                                mes_data = subset_mes['MES_NUM'].value_counts().sort_index()
                                mes_labels = [mes_nombres.get(int(mes), f'Mes {mes}') for mes in mes_data.index]
                                
                                fig_mes = px.bar(
                                    x=mes_labels,
                                    y=mes_data.values,
                                    title=f'Distribución por Mes - {tipo}',
                                    labels={'x': 'Mes', 'y': 'Cantidad'},
                                    color=mes_data.values,
                                    color_continuous_scale='Viridis'
                                )
                                fig_mes.update_layout(height=350, showlegend=False, xaxis_tickangle=-45)
                                fig_mes = apply_dark_theme(fig_mes)
                                st.plotly_chart(fig_mes, use_container_width=True)
                        
                        st.divider()
                        
                        # Top distritos y cuadrantes
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Top 10 Distritos**")
                            if 'DISTRITO' in subset.columns:
                                top_distritos = subset['DISTRITO'].value_counts().head(10)
                                fig_dist = px.bar(
                                    x=top_distritos.values,
                                    y=top_distritos.index,
                                    orientation='h',
                                    title=f'Top 10 Distritos - {tipo}',
                                    labels={'x': 'Cantidad', 'y': 'Distrito'},
                                    color=top_distritos.values,
                                    color_continuous_scale='Oranges'
                                )
                                fig_dist.update_layout(height=350, showlegend=False)
                                fig_dist = apply_dark_theme(fig_dist)
                                st.plotly_chart(fig_dist, use_container_width=True)
                        
                        with col2:
                            st.write("**Top 10 Cuadrantes**")
                            if 'CUADRANTE' in subset.columns:
                                top_cuadrantes = subset['CUADRANTE'].value_counts().head(10)
                                fig_cuad = px.bar(
                                    x=top_cuadrantes.values,
                                    y=top_cuadrantes.index,
                                    orientation='h',
                                    title=f'Top 10 Cuadrantes - {tipo}',
                                    labels={'x': 'Cantidad', 'y': 'Cuadrante'},
                                    color=top_cuadrantes.values,
                                    color_continuous_scale='Blues'
                                )
                                fig_cuad.update_layout(height=350, showlegend=False)
                                fig_cuad = apply_dark_theme(fig_cuad)
                                st.plotly_chart(fig_cuad, use_container_width=True)
                        
                        st.divider()
                        
                        # Violencia y Estación
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write("**Violencia de Robos**")
                            if 'VIOLENCIA' in subset.columns:
                                violencia_counts = subset['VIOLENCIA'].astype(str).str.upper().value_counts()
                                violencia_data = pd.DataFrame({
                                    'Violencia': ['Con Violencia', 'Sin Violencia'],
                                    'Cantidad': [violencia_counts.get('SI', 0), violencia_counts.get('NO', 0)]
                                })
                                
                                fig_violencia = px.pie(
                                    violencia_data,
                                    values='Cantidad',
                                    names='Violencia',
                                    title=f'Violencia - {tipo}',
                                    color_discrete_map={
                                        'Con Violencia': '#EF553B',
                                        'Sin Violencia': '#00CC96'
                                    }
                                )
                                fig_violencia.update_layout(height=350)
                                fig_violencia = apply_dark_theme(fig_violencia)
                                st.plotly_chart(fig_violencia, use_container_width=True)
                        
                        with col2:
                            st.write("**Distribución por Estación**")
                            if 'ESTACION' in subset.columns:
                                estacion_counts = subset['ESTACION'].value_counts()
                                fig_estacion = px.pie(
                                    values=estacion_counts.values,
                                    names=estacion_counts.index,
                                    title=f'Estaciones - {tipo}',
                                    hole=0
                                )
                                fig_estacion.update_layout(
                                    height=350,
                                    margin=dict(l=0, r=0, t=30, b=0),
                                    paper_bgcolor='rgba(0,0,0,0)',
                                    plot_bgcolor='rgba(0,0,0,0)'
                                )
                                fig_estacion.update_traces(
                                    textposition='inside',
                                    textinfo='percent+label',
                                    hovertemplate='<b>%{label}</b><br>Cantidad: %{value}<br>Porcentaje: %{percent}<extra></extra>'
                                )
                                fig_estacion = apply_dark_theme(fig_estacion)
                                st.plotly_chart(fig_estacion, use_container_width=True, config={'displayModeBar': False})
            else:
                st.info("No hay datos de tipo de robo disponibles")
        else:
            st.info("No hay columna 'TIPO' en los datos")
    
    # ============ MAPA ============
    elif page == "Mapa":
        st.header("Visualización en Mapa")
        
        # Por defecto mostrar todos los datos (sin filtro de distrito)
        datos_mapa = df_principal.copy()
        
        st.info("Mostrando todos los robos registrados")
        
        # Estadísticas rápidas del mapa
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Robos Totales", f"{len(datos_mapa):,}")
        with col2:
            violentos = len(datos_mapa[datos_mapa['VIOLENCIA'].astype(str).str.upper() == 'SI']) if 'VIOLENCIA' in datos_mapa.columns else 0
            st.metric("Con Violencia", f"{violentos:,}")
        with col3:
            distritos = datos_mapa['DISTRITO'].nunique() if 'DISTRITO' in datos_mapa.columns else 0
            st.metric("Distritos", f"{distritos}")
        with col4:
            cuadrantes = datos_mapa['CUADRANTE'].nunique() if 'CUADRANTE' in datos_mapa.columns else 0
            st.metric("Cuadrantes", f"{cuadrantes}")
        
        st.divider()
        
        # Verificar si hay columnas de coordenadas
        lat_cols = [col for col in datos_mapa.columns if 'lat' in col.lower()]
        lon_cols = [col for col in datos_mapa.columns if 'lon' in col.lower()]
        
        if lat_cols and lon_cols:
            lat_col = lat_cols[0]
            lon_col = lon_cols[0]
            
            # Preparar datos para el mapa
            if 'DISTRITO' in datos_mapa.columns:
                map_data = datos_mapa[[lat_col, lon_col, 'DISTRITO']].dropna()
            else:
                map_data = datos_mapa[[lat_col, lon_col]].dropna()
            
            map_data = map_data.rename(columns={lat_col: 'latitude', lon_col: 'longitude'}).reset_index(drop=True)
            
            if len(map_data) > 0:
                # Limitar a 5000 puntos para mejor rendimiento
                if len(map_data) > 5000:
                    map_data = map_data.sample(n=5000, random_state=42)
                    st.info(f"Mostrando muestra de 5,000 puntos de {len(datos_mapa):,} totales para mejor rendimiento")
                
                # Crear mapa con colores por distrito si folium está disponible
                if HAS_FOLIUM and 'DISTRITO' in map_data.columns:
                    # Obtener distritos únicos y asignarles colores
                    distritos_list = sorted(map_data['DISTRITO'].unique())
                    colores = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 'darkgreen', 'cadetblue', 'darkpurple', 'pink', 'gray', 'olive', 'teal', 'salmon']
                    mapa_colores = {dist: colores[i % len(colores)] for i, dist in enumerate(distritos_list)}
                    
                    # Agregar columna de color
                    map_data['color'] = map_data['DISTRITO'].map(mapa_colores)
                    
                    # Centro del mapa (promedio de coordenadas)
                    center_lat = map_data['latitude'].mean()
                    center_lon = map_data['longitude'].mean()
                    
                    m = folium.Map(location=[center_lat, center_lon], zoom_start=11, tiles='OpenStreetMap')
                    
                    # Agregar marcadores por distrito - optimizado
                    for idx, row in map_data.iterrows():
                        folium.CircleMarker(
                            location=[row['latitude'], row['longitude']],
                            radius=3,
                            popup=f"<b>Distrito:</b> {row['DISTRITO']}<br><b>Ubicación:</b> {row['latitude']:.4f}, {row['longitude']:.4f}",
                            tooltip=f"📍 Distrito: {row['DISTRITO']}",
                            color=row['color'],
                            fill=True,
                            fillColor=row['color'],
                            fillOpacity=0.5,
                            weight=0.5
                        ).add_to(m)
                    
                    # Agregar leyenda compacta
                    legend_html = '''
                    <div style="position: fixed; 
                                bottom: 50px; right: 50px; width: 220px; height: auto; 
                                background-color: white; border:2px solid grey; z-index:9999; 
                                font-size:11px; padding: 10px; border-radius: 4px; overflow-y: auto; max-height: 400px;">
                    <p style="margin: 0 0 8px 0; font-weight: bold; font-size: 12px; border-bottom: 1px solid #ccc; padding-bottom: 5px; color: #000;">Distritos</p>
                    '''
                    for dist, color in sorted(mapa_colores.items()):
                        legend_html += f'<p style="margin: 5px 0; padding: 3px; border-radius: 3px; color: #000;"><i style="background:{color}; width: 12px; height: 12px; float: left; margin-right: 8px; border-radius: 50%; border: 1px solid #333;"></i><span style="font-size: 10px; font-weight: 500; color: #000;">{dist}</span></p>'
                    legend_html += '</div>'
                    
                    m.get_root().html.add_child(folium.Element(legend_html))
                    
                    st_folium(m, width=1200, height=500)
                    st.success(f" Puntos mostrados: {len(map_data):,}")
                else:
                    st.map(map_data)
                    st.success(f" Puntos mostrados: {len(map_data):,}")
            else:
                st.warning(" No hay datos de ubicación disponibles")
        else:
            st.info(" No se encontraron columnas de latitud y longitud en los datos")
    
    # ============ PREDICCIONES ============
    elif page == "Predicciones":
        st.header("📊 Predicciones de Robos por Mes")
        
        st.info("""
        📅 **Predicciones Mensuales 2025**
        
        Las predicciones se generan utilizando modelos de redes neuronales entrenados con datos históricos 
        de 12 meses. Los cuadrantes mostrados son los que tienen mayor riesgo de robos según los patrones identificados.
        
        Selecciona el mes para ver las predicciones específicas.
        """)
        
        # Selector de mes
        meses_disponibles = [
            'Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio',
            'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre'
        ]
        
        mes_seleccionado = st.selectbox(
            "📅 Selecciona el mes:",
            meses_disponibles,
            index=0
        )
        
        # Verificar qué archivos de predicción existen
        pred_casa_existe = Path('exportados/Robos a casa habitacion/top_10_prediccion_robos_casa_habitacion_enero.csv').exists()
        pred_negocios_existe = Path('exportados/Robos a negocios/top_10_prediccion_robos_negocios_enero.csv').exists()
        pred_vehiculos_existe = Path('exportados/Robos de vehiculos/top_10_prediccion_robos_vehiculos_enero.csv').exists()
        
        # Crear tabs solo para archivos que existen
        tabs_disponibles = []
        tabs_labels = []
        
        if pred_casa_existe:
            tabs_labels.append(" Casa Habitación")
        if pred_negocios_existe:
            tabs_labels.append(" Negocios")
        if pred_vehiculos_existe:
            tabs_labels.append(" Vehículos")
        
        if not tabs_labels:
            st.warning(" No hay archivos de predicciones disponibles. Ejecuta los notebooks de modelos para generarlos.")
            st.info("""
            Para generar predicciones:
            1. Ejecuta `modeloCasaHabitación.ipynb`
            2. Ejecuta `modeloNegocio.ipynb`
            3. Ejecuta `modeloVehiculo.ipynb`
            """)
        else:
            tabs = st.tabs(tabs_labels)
        
        tab_idx = 0
        
        # Tab 1: Casa Habitación
        if pred_casa_existe:
            with tabs[tab_idx]:
                st.subheader(" Top 10 Cuadrantes - Robos a Casa Habitación")
                
                df_predicciones = load_prediction_data('casa', mes_seleccionado)
                
                if df_predicciones is not None:
                    # Mostrar métricas clave
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        max_pred = df_predicciones['PREDICCION_ROBOS_MES_N'].max() if 'PREDICCION_ROBOS_MES_N' in df_predicciones.columns else 0
                        st.metric("Mayor Predicción", f"{max_pred:.2f} robos")
                    with col_m2:
                        avg_pred = df_predicciones['PREDICCION_ROBOS_MES_N'].mean() if 'PREDICCION_ROBOS_MES_N' in df_predicciones.columns else 0
                        st.metric("Promedio Top 10", f"{avg_pred:.2f} robos")
                    with col_m3:
                        num_cuadrantes = len(df_predicciones)
                        st.metric("Cuadrantes en Riesgo", num_cuadrantes)
                    
                    st.divider()
                    
                    col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("** Top 10 Cuadrantes con Mayor Riesgo**")
                        
                        # Formatear el dataframe para mejor presentación
                        df_display = df_predicciones.copy()
                        if 'PREDICCION_ROBOS_MES_N' in df_display.columns:
                            df_display['PREDICCION_ROBOS_MES_N'] = df_display['PREDICCION_ROBOS_MES_N'].round(2)
                        
                        # Seleccionar solo columnas relevantes (sin LATITUD y LONGITUD)
                        columnas_mostrar = ['CUADRANTE', 'PREDICCION_ROBOS_MES_N', 'DISTRITO']
                        df_display = df_display[[col for col in columnas_mostrar if col in df_display.columns]]
                        
                        # Mostrar con estilos oscuros
                        styled_html = df_display.to_html(index=False, escape=False)
                        st.markdown(f"""
                        <div style="height: 400px; overflow: auto; background-color: #0d1117; border: 1px solid #30363d; border-radius: 8px;">
                            <style>
                                table {{
                                    width: 100%;
                                    border-collapse: collapse;
                                    background-color: #0d1117;
                                    color: #c9d1d9;
                                }}
                                th {{
                                    background-color: #21262d;
                                    color: #ffffff;
                                    padding: 12px;
                                    text-align: left;
                                    border-bottom: 2px solid #30363d;
                                    position: sticky;
                                    top: 0;
                                    z-index: 10;
                                }}
                                td {{
                                    padding: 10px 12px;
                                    border-bottom: 1px solid #30363d;
                                    color: #c9d1d9;
                                }}
                                tr:hover {{
                                    background-color: #161b22;
                                }}
                            </style>
                            {styled_html}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        st.write("** Gráfico de Predicciones**")
                        
                        # Gráfico de predicciones mejorado
                        if 'CUADRANTE' in df_predicciones.columns and 'PREDICCION_ROBOS_MES_N' in df_predicciones.columns:
                            fig = px.bar(df_predicciones, 
                                        x='CUADRANTE',
                                        y='PREDICCION_ROBOS_MES_N',
                                        title="Predicción de Robos por Cuadrante",
                                        labels={'CUADRANTE': 'Cuadrante', 'PREDICCION_ROBOS_MES_N': 'Robos Predichos'},
                                        color='PREDICCION_ROBOS_MES_N',
                                        color_continuous_scale='Reds')
                            fig.update_layout(height=400, showlegend=False)
                            fig = apply_dark_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    
                    st.divider()
                    
                    # Mapa de predicciones en toda la anchura
                    st.write("** Mapa de Cuadrantes de Alto Riesgo**")
                    
                    # Usar directamente los datos de predicciones que ya tienen coordenadas
                    if 'LATITUD' in df_predicciones.columns and 'LONGITUD' in df_predicciones.columns:
                        map_pred = df_predicciones[['CUADRANTE', 'PREDICCION_ROBOS_MES_N', 'LATITUD', 'LONGITUD', 'DISTRITO']].dropna()
                        
                        if len(map_pred) > 0 and HAS_FOLIUM:
                            # Centro del mapa
                            center_lat = map_pred['LATITUD'].mean()
                            center_lon = map_pred['LONGITUD'].mean()
                            
                            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB dark_matter')
                            
                            # Normalizar valores de predicción para el gradiente de color
                            min_pred = map_pred['PREDICCION_ROBOS_MES_N'].min()
                            max_pred = map_pred['PREDICCION_ROBOS_MES_N'].max()
                            
                            # Agregar marcadores con gradiente de calor (amarillo -> naranja -> rojo)
                            for idx, row in map_pred.iterrows():
                                # Calcular intensidad normalizada (0-1)
                                intensity = (row['PREDICCION_ROBOS_MES_N'] - min_pred) / (max_pred - min_pred) if max_pred > min_pred else 0.5
                                
                                # Gradiente de calor: amarillo (bajo) -> naranja (medio) -> rojo oscuro (alto)
                                if intensity < 0.33:
                                    color = '#FFEB3B'  # Amarillo
                                    fill_color = '#FFF59D'
                                elif intensity < 0.66:
                                    color = '#FF9800'  # Naranja
                                    fill_color = '#FFB74D'
                                else:
                                    color = '#C62828'  # Rojo oscuro
                                    fill_color = '#E53935'
                                
                                popup_html = f"""
                                <div style="font-family: Arial; min-width: 180px; padding: 5px;">
                                    <h4 style="margin: 0; color: {color}; text-shadow: 1px 1px 2px black;"> Cuadrante {int(row['CUADRANTE'])}</h4>
                                    <hr style="margin: 5px 0;">
                                    <p style="margin: 3px 0;"><b> Predicción:</b> {row['PREDICCION_ROBOS_MES_N']:.2f} robos</p>
                                    <p style="margin: 3px 0;"><b> Distrito:</b> {row['DISTRITO']}</p>
                                    <p style="margin: 3px 0;"><b> Ranking:</b> #{idx+1} de {len(map_pred)}</p>
                                    <p style="margin: 3px 0;"><b> Nivel:</b> {' Alto' if intensity > 0.66 else ' Medio' if intensity > 0.33 else ' Bajo'}</p>
                                </div>
                                """
                                
                                # Tamaño proporcional a la intensidad
                                radius = 10 + (intensity * 15)
                                
                                # Crear badge de ranking
                                if idx == 0:
                                    rank_badge = "#1"
                                elif idx == 1:
                                    rank_badge = "#2"
                                elif idx == 2:
                                    rank_badge = "#3"
                                else:
                                    rank_badge = f"#{idx+1}"
                                
                                # Tooltip con HTML para saltos de línea
                                nivel_riesgo = 'ALTO' if intensity > 0.66 else 'MEDIO' if intensity > 0.33 else 'BAJO'
                                tooltip_html = f"""<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                                    <b>Casa Habitación - Ranking {rank_badge}</b><br/>
                                    • Cuadrante: {int(row['CUADRANTE'])}<br/>
                                    • Distrito: {row['DISTRITO']}<br/>
                                    • Predicción: {row['PREDICCION_ROBOS_MES_N']:.2f} robos<br/>
                                    • Nivel de riesgo: {nivel_riesgo}
                                </div>"""
                                
                                folium.CircleMarker(
                                    location=[row['LATITUD'], row['LONGITUD']],
                                    radius=radius,
                                    popup=folium.Popup(popup_html, max_width=280),
                                    tooltip=folium.Tooltip(tooltip_html),
                                    color=color,
                                    fill=True,
                                    fillColor=fill_color,
                                    fillOpacity=0.8,
                                    weight=2,
                                    opacity=0.9
                                ).add_to(m)
                            
                            st_folium(m, width=None, height=500)
                            st.success(f" Mostrando {len(map_pred)} cuadrantes de alto riesgo en el mapa")
                        else:
                            st.info(" No hay datos de ubicación disponibles o folium no está instalado")
                    else:
                        st.info(" Las predicciones no tienen coordenadas disponibles")
                else:
                    st.warning("No se encontraron datos de predicciones para Casa Habitación")
            tab_idx += 1
        
        # Tab 2: Negocios
        if pred_negocios_existe:
            with tabs[tab_idx]:
                st.subheader(" Top 10 Cuadrantes - Robos a Negocios")
                
                try:
                    df_pred_negocios = load_prediction_data('negocios', mes_seleccionado)
                    
                    if df_pred_negocios is not None:
                        # Mostrar métricas clave
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            max_pred = df_pred_negocios['PREDICCION_ROBOS_MES_N'].max() if 'PREDICCION_ROBOS_MES_N' in df_pred_negocios.columns else 0
                            st.metric("Mayor Predicción", f"{max_pred:.2f} robos")
                        with col_m2:
                            avg_pred = df_pred_negocios['PREDICCION_ROBOS_MES_N'].mean() if 'PREDICCION_ROBOS_MES_N' in df_pred_negocios.columns else 0
                            st.metric("Promedio Top 10", f"{avg_pred:.2f} robos")
                        with col_m3:
                            num_cuadrantes = len(df_pred_negocios)
                            st.metric("Cuadrantes en Riesgo", num_cuadrantes)
                        
                        st.divider()
                        
                        col1, col2 = st.columns([1, 1])
                    else:
                        st.error("No se pudieron cargar los datos de predicción")
                        col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("** Top 10 Cuadrantes con Mayor Riesgo**")
                        
                        # Formatear el dataframe
                        df_display = df_pred_negocios.copy()
                        if 'PREDICCION_ROBOS_MES_N' in df_display.columns:
                            df_display['PREDICCION_ROBOS_MES_N'] = df_display['PREDICCION_ROBOS_MES_N'].round(2)
                        
                        # Seleccionar solo columnas relevantes (sin LATITUD y LONGITUD)
                        columnas_mostrar = ['CUADRANTE', 'PREDICCION_ROBOS_MES_N', 'DISTRITO']
                        df_display = df_display[[col for col in columnas_mostrar if col in df_display.columns]]
                        
                        # Mostrar con estilos oscuros
                        styled_html = df_display.to_html(index=False, escape=False)
                        st.markdown(f"""
                        <div style="height: 400px; overflow: auto; background-color: #0d1117; border: 1px solid #30363d; border-radius: 8px;">
                            <style>
                                table {{
                                    width: 100%;
                                    border-collapse: collapse;
                                    background-color: #0d1117;
                                    color: #c9d1d9;
                                }}
                                th {{
                                    background-color: #21262d;
                                    color: #ffffff;
                                    padding: 12px;
                                    text-align: left;
                                    border-bottom: 2px solid #30363d;
                                    position: sticky;
                                    top: 0;
                                    z-index: 10;
                                }}
                                td {{
                                    padding: 10px 12px;
                                    border-bottom: 1px solid #30363d;
                                    color: #c9d1d9;
                                }}
                                tr:hover {{
                                    background-color: #161b22;
                                }}
                            </style>
                            {styled_html}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        st.write("** Gráfico de Predicciones**")
                        
                        if 'CUADRANTE' in df_pred_negocios.columns and 'PREDICCION_ROBOS_MES_N' in df_pred_negocios.columns:
                            fig = px.bar(df_pred_negocios, 
                                        x='CUADRANTE',
                                        y='PREDICCION_ROBOS_MES_N',
                                        title="Predicción de Robos por Cuadrante",
                                        labels={'CUADRANTE': 'Cuadrante', 'PREDICCION_ROBOS_MES_N': 'Robos Predichos'},
                                        color='PREDICCION_ROBOS_MES_N',
                                        color_continuous_scale='Oranges')
                            fig.update_layout(height=400, showlegend=False)
                            fig = apply_dark_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    
                    st.divider()
                    
                    # Mapa de predicciones
                    st.write("** Mapa de Cuadrantes de Alto Riesgo**")
                    
                    if 'LATITUD' in df_pred_negocios.columns and 'LONGITUD' in df_pred_negocios.columns:
                        map_pred = df_pred_negocios[['CUADRANTE', 'PREDICCION_ROBOS_MES_N', 'LATITUD', 'LONGITUD', 'DISTRITO']].dropna()
                        
                        if len(map_pred) > 0 and HAS_FOLIUM:
                            center_lat = map_pred['LATITUD'].mean()
                            center_lon = map_pred['LONGITUD'].mean()
                            
                            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB dark_matter')
                            
                            # Normalizar valores de predicción para el gradiente de color
                            min_pred = map_pred['PREDICCION_ROBOS_MES_N'].min()
                            max_pred = map_pred['PREDICCION_ROBOS_MES_N'].max()
                            
                            for idx, row in map_pred.iterrows():
                                # Calcular intensidad normalizada (0-1)
                                intensity = (row['PREDICCION_ROBOS_MES_N'] - min_pred) / (max_pred - min_pred) if max_pred > min_pred else 0.5
                                
                                # Gradiente de calor: amarillo (bajo) -> naranja (medio) -> rojo oscuro (alto)
                                if intensity < 0.33:
                                    color = '#FFEB3B'  # Amarillo
                                    fill_color = '#FFF59D'
                                elif intensity < 0.66:
                                    color = '#FF9800'  # Naranja
                                    fill_color = '#FFB74D'
                                else:
                                    color = '#C62828'  # Rojo oscuro
                                    fill_color = '#E53935'
                                
                                popup_html = f"""
                                <div style="font-family: Arial; min-width: 180px; padding: 5px;">
                                    <h4 style="margin: 0; color: {color}; text-shadow: 1px 1px 2px black;"> Cuadrante {int(row['CUADRANTE'])}</h4>
                                    <hr style="margin: 5px 0;">
                                    <p style="margin: 3px 0;"><b> Predicción:</b> {row['PREDICCION_ROBOS_MES_N']:.2f} robos</p>
                                    <p style="margin: 3px 0;"><b> Distrito:</b> {row['DISTRITO']}</p>
                                    <p style="margin: 3px 0;"><b> Ranking:</b> #{idx+1} de {len(map_pred)}</p>
                                    <p style="margin: 3px 0;"><b> Nivel:</b> {' Alto' if intensity > 0.66 else ' Medio' if intensity > 0.33 else ' Bajo'}</p>
                                </div>
                                """
                                
                                # Tamaño proporcional a la intensidad
                                radius = 10 + (intensity * 15)
                                
                                # Crear badge de ranking
                                if idx == 0:
                                    rank_badge = "#1"
                                elif idx == 1:
                                    rank_badge = "#2"
                                elif idx == 2:
                                    rank_badge = "#3"
                                else:
                                    rank_badge = f"#{idx+1}"
                                
                                # Tooltip con HTML para saltos de línea
                                nivel_riesgo = 'ALTO' if intensity > 0.66 else 'MEDIO' if intensity > 0.33 else 'BAJO'
                                tooltip_html = f"""<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                                    <b>Negocios - Ranking {rank_badge}</b><br/>
                                    • Cuadrante: {int(row['CUADRANTE'])}<br/>
                                    • Distrito: {row['DISTRITO']}<br/>
                                    • Predicción: {row['PREDICCION_ROBOS_MES_N']:.2f} robos<br/>
                                    • Nivel de riesgo: {nivel_riesgo}
                                </div>"""
                                
                                folium.CircleMarker(
                                    location=[row['LATITUD'], row['LONGITUD']],
                                    radius=radius,
                                    popup=folium.Popup(popup_html, max_width=280),
                                    tooltip=folium.Tooltip(tooltip_html),
                                    color=color,
                                    fill=True,
                                    fillColor=fill_color,
                                    fillOpacity=0.8,
                                    weight=2,
                                    opacity=0.9
                                ).add_to(m)
                            
                            st_folium(m, width=None, height=500)
                            st.success(f" Mostrando {len(map_pred)} cuadrantes de alto riesgo en el mapa")
                        else:
                            st.info(" No hay datos de ubicación disponibles")
                    else:
                        st.info(" Las predicciones no tienen coordenadas disponibles")
                except FileNotFoundError:
                    st.warning(" Archivo de predicciones no disponible")
                    st.info("Para generar predicciones de Robos a Negocios, ejecuta el notebook `modeloNegocio.ipynb`")
                except Exception as e:
                    st.error(f"Error al cargar predicciones: {str(e)}")
            tab_idx += 1
        
        # Tab 3: Vehículos
        if pred_vehiculos_existe:
            with tabs[tab_idx]:
                st.subheader(" Top 10 Cuadrantes - Robos de Vehículos")
                
                try:
                    df_pred_vehiculos = load_prediction_data('vehiculos', mes_seleccionado)
                    
                    if df_pred_vehiculos is not None:
                        # Mostrar métricas clave
                        col_m1, col_m2, col_m3 = st.columns(3)
                        with col_m1:
                            max_pred = df_pred_vehiculos['PREDICCION_ROBOS_MES_N'].max() if 'PREDICCION_ROBOS_MES_N' in df_pred_vehiculos.columns else 0
                            st.metric("Mayor Predicción", f"{max_pred:.2f} robos")
                        with col_m2:
                            avg_pred = df_pred_vehiculos['PREDICCION_ROBOS_MES_N'].mean() if 'PREDICCION_ROBOS_MES_N' in df_pred_vehiculos.columns else 0
                            st.metric("Promedio Top 10", f"{avg_pred:.2f} robos")
                        with col_m3:
                            num_cuadrantes = len(df_pred_vehiculos)
                            st.metric("Cuadrantes en Riesgo", num_cuadrantes)
                        
                        st.divider()
                        
                        col1, col2 = st.columns([1, 1])
                    else:
                        st.error("No se pudieron cargar los datos de predicción")
                        col1, col2 = st.columns([1, 1])
                    
                    with col1:
                        st.write("** Top 10 Cuadrantes con Mayor Riesgo**")
                        
                        # Formatear el dataframe
                        df_display = df_pred_vehiculos.copy()
                        if 'PREDICCION_ROBOS_MES_N' in df_display.columns:
                            df_display['PREDICCION_ROBOS_MES_N'] = df_display['PREDICCION_ROBOS_MES_N'].round(2)
                        
                        # Seleccionar solo columnas relevantes (sin LATITUD y LONGITUD)
                        columnas_mostrar = ['CUADRANTE', 'PREDICCION_ROBOS_MES_N', 'DISTRITO']
                        df_display = df_display[[col for col in columnas_mostrar if col in df_display.columns]]
                        
                        # Mostrar con estilos oscuros
                        styled_html = df_display.to_html(index=False, escape=False)
                        st.markdown(f"""
                        <div style="height: 400px; overflow: auto; background-color: #0d1117; border: 1px solid #30363d; border-radius: 8px;">
                            <style>
                                table {{
                                    width: 100%;
                                    border-collapse: collapse;
                                    background-color: #0d1117;
                                    color: #c9d1d9;
                                }}
                                th {{
                                    background-color: #21262d;
                                    color: #ffffff;
                                    padding: 12px;
                                    text-align: left;
                                    border-bottom: 2px solid #30363d;
                                    position: sticky;
                                    top: 0;
                                    z-index: 10;
                                }}
                                td {{
                                    padding: 10px 12px;
                                    border-bottom: 1px solid #30363d;
                                    color: #c9d1d9;
                                }}
                                tr:hover {{
                                    background-color: #161b22;
                                }}
                            </style>
                            {styled_html}
                        </div>
                        """, unsafe_allow_html=True)
                        
                    with col2:
                        st.write("** Gráfico de Predicciones**")
                        
                        if 'CUADRANTE' in df_pred_vehiculos.columns and 'PREDICCION_ROBOS_MES_N' in df_pred_vehiculos.columns:
                            fig = px.bar(df_pred_vehiculos, 
                                        x='CUADRANTE',
                                        y='PREDICCION_ROBOS_MES_N',
                                        title="Predicción de Robos por Cuadrante",
                                        labels={'CUADRANTE': 'Cuadrante', 'PREDICCION_ROBOS_MES_N': 'Robos Predichos'},
                                        color='PREDICCION_ROBOS_MES_N',
                                        color_continuous_scale='Blues')
                            fig.update_layout(height=400, showlegend=False)
                            fig = apply_dark_theme(fig)
                            st.plotly_chart(fig, use_container_width=True)
                    
                    
                    st.divider()
                    
                    # Mapa de predicciones
                    st.write("** Mapa de Cuadrantes de Alto Riesgo**")
                    
                    if 'LATITUD' in df_pred_vehiculos.columns and 'LONGITUD' in df_pred_vehiculos.columns:
                        map_pred = df_pred_vehiculos[['CUADRANTE', 'PREDICCION_ROBOS_MES_N', 'LATITUD', 'LONGITUD', 'DISTRITO']].dropna()
                        
                        if len(map_pred) > 0 and HAS_FOLIUM:
                            center_lat = map_pred['LATITUD'].mean()
                            center_lon = map_pred['LONGITUD'].mean()
                            
                            m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles='CartoDB dark_matter')
                            
                            # Normalizar valores de predicción para el gradiente de color
                            min_pred = map_pred['PREDICCION_ROBOS_MES_N'].min()
                            max_pred = map_pred['PREDICCION_ROBOS_MES_N'].max()
                            
                            for idx, row in map_pred.iterrows():
                                # Calcular intensidad normalizada (0-1)
                                intensity = (row['PREDICCION_ROBOS_MES_N'] - min_pred) / (max_pred - min_pred) if max_pred > min_pred else 0.5
                                
                                # Gradiente de calor: amarillo (bajo) -> naranja (medio) -> rojo oscuro (alto)
                                if intensity < 0.33:
                                    color = '#FFEB3B'  # Amarillo
                                    fill_color = '#FFF59D'
                                elif intensity < 0.66:
                                    color = '#FF9800'  # Naranja
                                    fill_color = '#FFB74D'
                                else:
                                    color = '#C62828'  # Rojo oscuro
                                    fill_color = '#E53935'
                                
                                popup_html = f"""
                                <div style="font-family: Arial; min-width: 180px; padding: 5px;">
                                    <h4 style="margin: 0; color: {color}; text-shadow: 1px 1px 2px black;"> Cuadrante {int(row['CUADRANTE'])}</h4>
                                    <hr style="margin: 5px 0;">
                                    <p style="margin: 3px 0;"><b> Predicción:</b> {row['PREDICCION_ROBOS_MES_N']:.2f} robos</p>
                                    <p style="margin: 3px 0;"><b> Distrito:</b> {row['DISTRITO']}</p>
                                    <p style="margin: 3px 0;"><b> Ranking:</b> #{idx+1} de {len(map_pred)}</p>
                                    <p style="margin: 3px 0;"><b> Nivel:</b> {' Alto' if intensity > 0.66 else ' Medio' if intensity > 0.33 else ' Bajo'}</p>
                                </div>
                                """
                                
                                # Tamaño proporcional a la intensidad
                                radius = 10 + (intensity * 15)
                                
                                # Crear badge de ranking
                                if idx == 0:
                                    rank_badge = "#1"
                                elif idx == 1:
                                    rank_badge = "#2"
                                elif idx == 2:
                                    rank_badge = "#3"
                                else:
                                    rank_badge = f"#{idx+1}"
                                
                                # Tooltip con HTML para saltos de línea
                                nivel_riesgo = 'ALTO' if intensity > 0.66 else 'MEDIO' if intensity > 0.33 else 'BAJO'
                                tooltip_html = f"""<div style="font-family: Arial, sans-serif; line-height: 1.6;">
                                    <b>Vehículos - Ranking {rank_badge}</b><br/>
                                    • Cuadrante: {int(row['CUADRANTE'])}<br/>
                                    • Distrito: {row['DISTRITO']}<br/>
                                    • Predicción: {row['PREDICCION_ROBOS_MES_N']:.2f} robos<br/>
                                    • Nivel de riesgo: {nivel_riesgo}
                                </div>"""
                                
                                folium.CircleMarker(
                                    location=[row['LATITUD'], row['LONGITUD']],
                                    radius=radius,
                                    popup=folium.Popup(popup_html, max_width=280),
                                    tooltip=folium.Tooltip(tooltip_html),
                                    color=color,
                                    fill=True,
                                    fillColor=fill_color,
                                    fillOpacity=0.8,
                                    weight=2,
                                    opacity=0.9
                                ).add_to(m)
                            
                            st_folium(m, width=None, height=500)
                            st.success(f" Mostrando {len(map_pred)} cuadrantes de alto riesgo en el mapa")
                        else:
                            st.info(" No hay datos de ubicación disponibles")
                    else:
                        st.info(" Las predicciones no tienen coordenadas disponibles")
                except FileNotFoundError:
                    st.warning(" Archivo de predicciones no disponible")
                    st.info("Para generar predicciones de Robos de Vehículos, ejecuta el notebook `modeloVehiculo.ipynb`")
                except Exception as e:
                    st.error(f"Error al cargar predicciones: {str(e)}")
        
        # Mostrar resumen de predicciones disponibles
        st.divider()
        st.subheader(" Estado de Modelos de Predicción")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            status = " Disponible" if pred_casa_existe else " No disponible"
            color = "normal" if pred_casa_existe else "off"
            st.metric(" Casa Habitación", status, delta=None)
            if not pred_casa_existe:
                st.caption("Ejecuta `modeloCasaHabitación.ipynb`")
        
        with col2:
            status = " Disponible" if pred_negocios_existe else " No disponible"
            st.metric(" Negocios", status, delta=None)
            if not pred_negocios_existe:
                st.caption("Ejecuta `modeloNegocio.ipynb`")
        
        with col3:
            status = " Disponible" if pred_vehiculos_existe else " No disponible"
            st.metric(" Vehículos", status, delta=None)
            if not pred_vehiculos_existe:
                st.caption("Ejecuta `modeloVehiculo.ipynb`")
        
        # Información adicional
        st.info("""
         **Información sobre las predicciones:**
        - Las predicciones se basan en modelos de redes neuronales entrenados con datos históricos
        - Muestran los 10 cuadrantes con mayor riesgo para cada mes de 2025
        - Los mapas permiten identificar geográficamente las zonas de alto riesgo
        - Los modelos se actualizan ejecutando el script `generar_todos_los_meses.py`
        
        💡 **Predicciones disponibles:** Enero a Diciembre 2025 (12 meses completos)
        """)

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; margin-top: 2rem; color: gray; font-size: 0.9rem;'>
    <p>Dashboard de Análisis de Robos | Exploratory Data Analysis (EDA)</p>
    <p>Datos actualizados: Robos en Chihuahua</p>
    </div>
""", unsafe_allow_html=True)
