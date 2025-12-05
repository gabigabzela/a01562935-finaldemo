"""
Script para generar predicciones de robos para todos los meses del año
Genera archivos CSV con predicciones para Casa Habitación, Negocios y Vehículos
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Mapeo de números a nombres de meses
MESES = {
    1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
    5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
    9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
}

def cargar_datos_historicos(tipo_robo):
    """
    Carga los datos históricos según el tipo de robo
    tipo_robo: 'casa', 'negocios', 'vehiculos'
    """
    paths = {
        'casa': 'exportados/Robos a casa habitacion/cuadrantes_robos_CaHa_promedio.xlsx',
        'negocios': 'exportados/Robos a negocios/cuadrantes_robos_negocios_promedio.xlsx',
        'vehiculos': 'exportados/Robos de vehiculos/cuadrantes_robos_vehiculos_promedio.xlsx'
    }
    
    file_path = paths.get(tipo_robo)
    if not file_path or not Path(file_path).exists():
        print(f"⚠️  Archivo no encontrado: {file_path}")
        return None
    
    df = pd.read_excel(file_path)
    return df

def preparar_datos(df_promedios, tipo_robo):
    """
    Prepara los datos en formato long (serie de tiempo)
    """
    # Mapeo de nombres de columnas según el tipo de robo
    columnas_robos = {
        'casa': 'PROMEDIO DE ROBOS A CASA HABITACION MES',
        'negocios': 'PROMEDIO DE ROBOS A NEGOCIOS MES',
        'vehiculos': 'PROMEDIO DE ROBOS DE VEHICULOS MES'
    }
    
    col_base = columnas_robos.get(tipo_robo)
    if not col_base:
        return None
    
    # Renombrar columnas
    column_names = {
        'CUADRANTE': 'CUADRANTE',
        'POBLACION': 'POBLACION'
    }
    
    for i in range(1, 13):
        column_names[f'{col_base} {i}'] = f'ROBOS_MES_{i}'
    
    df_promedios.rename(columns=column_names, inplace=True)
    
    # Desapilar (melt)
    df_long = df_promedios.melt(
        id_vars=['CUADRANTE', 'POBLACION'],
        value_vars=[f'ROBOS_MES_{i}' for i in range(1, 13)],
        var_name='MES_NOMBRE',
        value_name='ROBOS_MES_N'
    )
    
    # Extraer número de mes
    df_long['MES_N'] = df_long['MES_NOMBRE'].str.extract(r'(\d+)').astype(int)
    df_long.drop(columns=['MES_NOMBRE'], inplace=True)
    df_long.sort_values(by=['MES_N', 'CUADRANTE'], inplace=True)
    df_long.reset_index(drop=True, inplace=True)
    
    return df_long

def crear_features_lag(df_long):
    """
    Crea las características N-1 y N-2 (lag features)
    """
    # Crear características N-1 y N-2
    df_diciembre = df_long[df_long['MES_N'] == 12].copy()
    df_diciembre.rename(columns={'ROBOS_MES_N': 'ROBOS_DIC'}, inplace=True)
    df_diciembre = df_diciembre[['CUADRANTE', 'ROBOS_DIC']]
    
    df_noviembre = df_long[df_long['MES_N'] == 11].copy()
    df_noviembre.rename(columns={'ROBOS_MES_N': 'ROBOS_NOV'}, inplace=True)
    df_noviembre = df_noviembre[['CUADRANTE', 'ROBOS_NOV']]
    
    # Merge
    df_long = df_long.merge(df_diciembre, on='CUADRANTE', how='left')
    df_long = df_long.merge(df_noviembre, on='CUADRANTE', how='left')
    
    # Crear N-1 y N-2
    df_long.sort_values(by=['CUADRANTE', 'MES_N'], inplace=True)
    df_long['ROBOS_MES_N_MENOS_1'] = df_long.groupby('CUADRANTE')['ROBOS_MES_N'].shift(1)
    df_long['ROBOS_MES_N_MENOS_2'] = df_long.groupby('CUADRANTE')['ROBOS_MES_N'].shift(2)
    
    # Imputar enero y febrero
    mask_enero = df_long['MES_N'] == 1
    df_long.loc[mask_enero, 'ROBOS_MES_N_MENOS_1'] = df_long.loc[mask_enero, 'ROBOS_DIC']
    df_long.loc[mask_enero, 'ROBOS_MES_N_MENOS_2'] = df_long.loc[mask_enero, 'ROBOS_NOV']
    
    mask_febrero = df_long['MES_N'] == 2
    df_long.loc[mask_febrero, 'ROBOS_MES_N_MENOS_2'] = df_long.loc[mask_febrero, 'ROBOS_DIC']
    
    df_long.drop(columns=['ROBOS_DIC', 'ROBOS_NOV'], inplace=True)
    
    return df_long

def entrenar_modelo(df_final):
    """
    Entrena el modelo de red neuronal
    """
    # Preparar datos de entrenamiento
    X = df_final[['CUADRANTE', 'POBLACION', 'ROBOS_MES_N_MENOS_1', 'ROBOS_MES_N_MENOS_2']].values
    Y = df_final['ROBOS_MES_N'].values.reshape(-1, 1)
    
    # Escalar datos
    from sklearn.preprocessing import StandardScaler
    scaler_X = StandardScaler()
    scaler_Y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    Y_scaled = scaler_Y.fit_transform(Y)
    
    # Crear modelo
    modelo = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_scaled.shape[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    modelo.compile(optimizer='adam', loss='mse', metrics=['mae'])
    
    # Entrenar
    modelo.fit(X_scaled, Y_scaled, epochs=100, batch_size=32, verbose=0, validation_split=0.2)
    
    return modelo, scaler_X, scaler_Y

def predecir_mes(modelo, scaler_X, scaler_Y, df_historico, mes_a_predecir):
    """
    Genera predicción para un mes específico (mes_a_predecir = 1 a 12)
    """
    # Obtener datos del mes anterior (N-1) y dos meses antes (N-2)
    mes_n_menos_1 = 12 if mes_a_predecir == 1 else mes_a_predecir - 1
    mes_n_menos_2 = 11 if mes_a_predecir == 1 else (12 if mes_a_predecir == 2 else mes_a_predecir - 2)
    
    df_n_menos_1 = df_historico[df_historico['MES_N'] == mes_n_menos_1][['CUADRANTE', 'ROBOS_MES_N']].copy()
    df_n_menos_1.sort_values(by='CUADRANTE', inplace=True)
    
    df_n_menos_2 = df_historico[df_historico['MES_N'] == mes_n_menos_2][['CUADRANTE', 'ROBOS_MES_N']].copy()
    df_n_menos_2.sort_values(by='CUADRANTE', inplace=True)
    
    # Crear DataFrame de predicción
    df_prediccion = df_historico[df_historico['MES_N'] == 1].copy()
    df_prediccion['MES_N'] = mes_a_predecir
    df_prediccion.sort_values(by='CUADRANTE', inplace=True)
    
    df_prediccion['ROBOS_MES_N_MENOS_1'] = df_n_menos_1['ROBOS_MES_N'].values
    df_prediccion['ROBOS_MES_N_MENOS_2'] = df_n_menos_2['ROBOS_MES_N'].values
    
    # Preparar entrada
    X_pred_raw = df_prediccion[['CUADRANTE', 'POBLACION', 'ROBOS_MES_N_MENOS_1', 'ROBOS_MES_N_MENOS_2']].values
    X_pred_scaled = scaler_X.transform(X_pred_raw)
    
    # Predecir
    Y_pred_scaled = modelo.predict(X_pred_scaled, verbose=0)
    Y_pred_desescalado = scaler_Y.inverse_transform(Y_pred_scaled).flatten()
    
    # Crear resultado
    df_resultado = pd.DataFrame({
        'CUADRANTE': df_prediccion['CUADRANTE'].values,
        'PREDICCION_ROBOS_MES_N': Y_pred_desescalado,
        'DISTRITO': df_prediccion['CUADRANTE'].str[:3] if 'CUADRANTE' in df_prediccion.columns else None
    })
    
    # Obtener top 10
    top_10 = df_resultado.sort_values(by='PREDICCION_ROBOS_MES_N', ascending=False).head(10)
    
    return top_10

def agregar_coordenadas(df_pred, df_principal_path='robos_tot_final.csv'):
    """
    Agrega coordenadas de latitud y longitud basadas en el cuadrante
    """
    try:
        df_principal = pd.read_csv(df_principal_path)
        if 'CUADRANTE' in df_principal.columns and 'LATITUD' in df_principal.columns and 'LONGITUD' in df_principal.columns:
            cuadrante_coords = df_principal.groupby('CUADRANTE')[['LATITUD', 'LONGITUD']].mean().reset_index()
            df_pred = df_pred.merge(cuadrante_coords, how='left', on='CUADRANTE')
    except Exception as e:
        print(f"⚠️  No se pudieron agregar coordenadas: {e}")
    
    return df_pred

def generar_predicciones_anuales(tipo_robo):
    """
    Genera predicciones para los 12 meses del año para un tipo de robo
    """
    print(f"\n{'='*60}")
    print(f"🔮 Generando predicciones anuales para: {tipo_robo.upper()}")
    print(f"{'='*60}")
    
    # Cargar datos
    df_promedios = cargar_datos_historicos(tipo_robo)
    if df_promedios is None:
        print(f"❌ No se pudieron cargar datos para {tipo_robo}")
        return
    
    # Preparar datos
    print("📊 Preparando datos históricos...")
    df_long = preparar_datos(df_promedios, tipo_robo)
    if df_long is None:
        print(f"❌ Error al preparar datos para {tipo_robo}")
        return
    
    # Crear features
    print("🔧 Creando características lag (N-1, N-2)...")
    df_final = crear_features_lag(df_long)
    
    # Entrenar modelo
    print("🤖 Entrenando modelo de red neuronal...")
    modelo, scaler_X, scaler_Y = entrenar_modelo(df_final)
    print("✅ Modelo entrenado exitosamente")
    
    # Crear directorio de salida
    output_dirs = {
        'casa': 'exportados/Robos a casa habitacion/',
        'negocios': 'exportados/Robos a negocios/',
        'vehiculos': 'exportados/Robos de vehiculos/'
    }
    
    output_dir = Path(output_dirs[tipo_robo])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generar predicciones para cada mes
    print("\n📅 Generando predicciones mensuales...")
    for mes_num in range(1, 13):
        mes_nombre = MESES[mes_num]
        
        # Predecir
        top_10 = predecir_mes(modelo, scaler_X, scaler_Y, df_final, mes_num)
        
        # Agregar coordenadas
        top_10 = agregar_coordenadas(top_10)
        
        # Redondear predicciones
        top_10['PREDICCION_ROBOS_MES_N'] = top_10['PREDICCION_ROBOS_MES_N'].round(2)
        
        # Guardar archivo
        output_file = output_dir / f'top_10_prediccion_robos_{tipo_robo}_{"casa_habitacion" if tipo_robo == "casa" else tipo_robo}_{mes_nombre}.csv'
        
        # Ajustar nombre del archivo para casa habitación
        if tipo_robo == 'casa':
            output_file = output_dir / f'top_10_prediccion_robos_casa_habitacion_{mes_nombre}.csv'
        else:
            output_file = output_dir / f'top_10_prediccion_robos_{tipo_robo}_{mes_nombre}.csv'
        
        top_10.to_csv(output_file, index=False)
        
        print(f"  ✓ {mes_nombre.capitalize():12s} - Archivo generado: {output_file.name}")
    
    print(f"\n✅ Predicciones anuales completadas para {tipo_robo.upper()}")

def main():
    """
    Función principal
    """
    print("\n" + "="*60)
    print("🎯 GENERADOR DE PREDICCIONES ANUALES DE ROBOS")
    print("="*60)
    print("\nEste script genera predicciones para los 12 meses del año")
    print("para cada tipo de robo: Casa Habitación, Negocios y Vehículos")
    print("\n" + "="*60)
    
    tipos_robo = ['casa', 'negocios', 'vehiculos']
    
    for tipo in tipos_robo:
        try:
            generar_predicciones_anuales(tipo)
        except Exception as e:
            print(f"\n❌ Error al procesar {tipo}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)
    print("\n📂 Los archivos CSV se han guardado en:")
    print("   • exportados/Robos a casa habitacion/")
    print("   • exportados/Robos a negocios/")
    print("   • exportados/Robos de vehiculos/")
    print("\n🚀 Ahora puedes ejecutar el dashboard para ver las predicciones")
    print("   de todos los meses del año!")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
