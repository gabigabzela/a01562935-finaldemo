"""
Script simplificado para generar predicciones de todos los meses
Basado en la lógica del notebook modeloCasaHabitación.ipynb
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Mapeo de meses
MESES = {
    1: 'enero', 2: 'febrero', 3: 'marzo', 4: 'abril',
    5: 'mayo', 6: 'junio', 7: 'julio', 8: 'agosto',
    9: 'septiembre', 10: 'octubre', 11: 'noviembre', 12: 'diciembre'
}

def cargar_y_preparar_datos(file_path, prefijo_columna):
    """Carga y prepara los datos históricos"""
    print(f"  📂 Cargando: {file_path}")
    df_promedios = pd.read_excel(file_path)
    
    # Renombrar columnas
    column_names = {'CUADRANTE': 'CUADRANTE', 'POBLACION': 'POBLACION'}
    for i in range(1, 13):
        column_names[f'{prefijo_columna} {i}'] = f'ROBOS_MES_{i}'
    
    df_promedios.rename(columns=column_names, inplace=True)
    
    # Desapilar
    df_long = df_promedios.melt(
        id_vars=['CUADRANTE', 'POBLACION'],
        value_vars=[f'ROBOS_MES_{i}' for i in range(1, 13)],
        var_name='MES_NOMBRE',
        value_name='ROBOS_MES_N'
    )
    
    df_long['MES_N'] = df_long['MES_NOMBRE'].str.extract(r'(\d+)').astype(int)
    df_long.drop(columns=['MES_NOMBRE'], inplace=True)
    df_long.sort_values(by=['MES_N', 'CUADRANTE'], inplace=True)
    df_long.reset_index(drop=True, inplace=True)
    
    return df_long

def crear_features_lag(df_long):
    """Crea features N-1 y N-2"""
    print("  🔧 Creando features lag...")
    
    df_diciembre = df_long[df_long['MES_N'] == 12].copy()
    df_diciembre.rename(columns={'ROBOS_MES_N': 'ROBOS_DIC'}, inplace=True)
    df_diciembre = df_diciembre[['CUADRANTE', 'ROBOS_DIC']]
    
    df_noviembre = df_long[df_long['MES_N'] == 11].copy()
    df_noviembre.rename(columns={'ROBOS_MES_N': 'ROBOS_NOV'}, inplace=True)
    df_noviembre = df_noviembre[['CUADRANTE', 'ROBOS_NOV']]
    
    df_long['ROBOS_MES_N_MENOS_1'] = df_long.groupby('CUADRANTE')['ROBOS_MES_N'].shift(1)
    df_long['ROBOS_MES_N_MENOS_2'] = df_long.groupby('CUADRANTE')['ROBOS_MES_N'].shift(2)
    
    df_long = df_long.merge(df_diciembre, on='CUADRANTE', how='left')
    df_long.loc[df_long['MES_N'] == 1, 'ROBOS_MES_N_MENOS_1'] = df_long.loc[df_long['MES_N'] == 1, 'ROBOS_DIC']
    
    df_long = df_long.merge(df_noviembre, on='CUADRANTE', how='left')
    df_long.loc[df_long['MES_N'] == 1, 'ROBOS_MES_N_MENOS_2'] = df_long.loc[df_long['MES_N'] == 1, 'ROBOS_NOV']
    df_long.loc[df_long['MES_N'] == 2, 'ROBOS_MES_N_MENOS_2'] = df_long.loc[df_long['MES_N'] == 2, 'ROBOS_DIC']
    
    df_final = df_long.drop(columns=['ROBOS_DIC', 'ROBOS_NOV'])
    
    return df_final

def entrenar_modelo(df_final):
    """Entrena el modelo de red neuronal"""
    print("  🤖 Entrenando modelo...")
    
    df_scale = df_final[['CUADRANTE', 'POBLACION', 'ROBOS_MES_N_MENOS_1', 'ROBOS_MES_N_MENOS_2', 'ROBOS_MES_N']].copy()
    
    scaler = MinMaxScaler()
    df_scaled_values = scaler.fit_transform(df_scale)
    df_scaled = pd.DataFrame(df_scaled_values, columns=df_scale.columns)
    df_scaled['MES_N'] = df_final['MES_N'].values
    
    X_features_scaled = df_scaled[['CUADRANTE', 'POBLACION', 'ROBOS_MES_N_MENOS_1', 'ROBOS_MES_N_MENOS_2']]
    Y_output_scaled = df_scaled['ROBOS_MES_N']
    
    X_data_scaled = X_features_scaled.groupby(df_scaled['MES_N']).apply(lambda x: x.values.flatten()).tolist()
    Y_data_scaled = Y_output_scaled.groupby(df_scaled['MES_N']).apply(lambda x: x.values).tolist()
    
    X_scaled = np.array(X_data_scaled)
    Y_scaled = np.array(Y_data_scaled)
    
    X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled = train_test_split(
        X_scaled, Y_scaled, test_size=0.15, shuffle=False
    )
    
    input_dim = X_scaled.shape[1]
    output_dim = Y_scaled.shape[1]
    
    model_scaled = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(output_dim, activation='linear')
    ])
    
    model_scaled.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    
    model_scaled.fit(
        X_train_scaled, Y_train_scaled,
        epochs=200,
        batch_size=2,
        validation_data=(X_test_scaled, Y_test_scaled),
        callbacks=[early_stop],
        verbose=0
    )
    
    return model_scaled, scaler

def predecir_mes(mes_a_predecir, df_historico, modelo, escalador):
    """Predice para un mes específico (13-24 corresponde a meses 1-12)"""
    
    mes_n_menos_1 = (mes_a_predecir - 1)
    mes_n_menos_2 = (mes_a_predecir - 2)
    mes_anterior_ciclo = (mes_n_menos_1 - 1) % 12 + 1
    mes_anterior_2_ciclo = (mes_n_menos_2 - 1) % 12 + 1
    
    df_n_menos_1 = df_historico[df_historico['MES_N'] == mes_anterior_ciclo].copy()
    df_n_menos_2 = df_historico[df_historico['MES_N'] == mes_anterior_2_ciclo].copy()
    
    df_prediccion = df_historico[df_historico['MES_N'] == 1].copy()
    
    df_prediccion.sort_values(by='CUADRANTE', inplace=True)
    df_n_menos_1.sort_values(by='CUADRANTE', inplace=True)
    df_n_menos_2.sort_values(by='CUADRANTE', inplace=True)
    
    df_prediccion['ROBOS_MES_N_MENOS_1'] = df_n_menos_1['ROBOS_MES_N'].values
    df_prediccion['ROBOS_MES_N_MENOS_2'] = df_n_menos_2['ROBOS_MES_N'].values
    
    X_pred_raw = df_prediccion[['CUADRANTE', 'POBLACION', 'ROBOS_MES_N_MENOS_1', 'ROBOS_MES_N_MENOS_2']].values
    X_con_dummy = np.hstack((X_pred_raw, np.zeros((X_pred_raw.shape[0], 1))))
    X_pred_escalado_completo = escalador.transform(X_con_dummy)
    X_pred_final = X_pred_escalado_completo[:, :4].flatten().reshape(1, -1)
    
    Y_pred_scaled = modelo.predict(X_pred_final, verbose=0)
    
    Y_pred_full = np.hstack((X_pred_raw, Y_pred_scaled.T))
    Y_pred_desescalado = escalador.inverse_transform(Y_pred_full)[:, -1]
    
    df_resultado = pd.DataFrame({
        'CUADRANTE': df_prediccion['CUADRANTE'].values,
        'PREDICCION_ROBOS_MES_N': Y_pred_desescalado
    })
    
    top_10 = df_resultado.sort_values(by='PREDICCION_ROBOS_MES_N', ascending=False).head(10).round(2)
    
    return top_10

def agregar_coordenadas(df_pred):
    """Agrega coordenadas geográficas"""
    try:
        df_robos = pd.read_csv('robos_tot_final.csv')
        cuadrante_coords = df_robos.groupby('CUADRANTE')[['LATITUD', 'LONGITUD', 'DISTRITO']].agg({
            'LATITUD': 'mean',
            'LONGITUD': 'mean',
            'DISTRITO': lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
        }).reset_index()
        
        df_resultado = df_pred.merge(cuadrante_coords, on='CUADRANTE', how='left')
        df_resultado = df_resultado[['CUADRANTE', 'PREDICCION_ROBOS_MES_N', 'LATITUD', 'LONGITUD', 'DISTRITO']]
        return df_resultado
    except Exception as e:
        print(f"    ⚠️  No se pudieron agregar coordenadas: {e}")
        return df_pred

def procesar_tipo_robo(config):
    """Procesa un tipo de robo completo"""
    tipo = config['tipo']
    file_path = config['file_path']
    prefijo = config['prefijo']
    output_dir = config['output_dir']
    
    print(f"\n{'='*70}")
    print(f"🔮 GENERANDO PREDICCIONES: {tipo.upper()}")
    print(f"{'='*70}")
    
    if not Path(file_path).exists():
        print(f"  ❌ Archivo no encontrado: {file_path}")
        return False
    
    # Cargar y preparar datos
    df_long = cargar_y_preparar_datos(file_path, prefijo)
    df_final = crear_features_lag(df_long)
    
    # Entrenar modelo
    modelo, scaler = entrenar_modelo(df_final)
    print("  ✅ Modelo entrenado")
    
    # Crear directorio de salida
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generar predicciones para los 12 meses
    print(f"\n  📅 Generando predicciones mensuales:")
    for mes_num in range(1, 13):
        mes_nombre = MESES[mes_num]
        mes_prediccion = mes_num + 12  # 13-24
        
        # Predecir
        top_10 = predecir_mes(mes_prediccion, df_final, modelo, scaler)
        
        # Agregar coordenadas
        top_10 = agregar_coordenadas(top_10)
        
        # Guardar
        output_file = Path(output_dir) / f"top_10_prediccion_robos_{config['nombre_archivo']}_{mes_nombre}.csv"
        top_10.to_csv(output_file, index=False)
        
        max_pred = top_10['PREDICCION_ROBOS_MES_N'].max()
        print(f"    ✓ {mes_nombre.capitalize():12s} - Max predicción: {max_pred:5.2f} robos")
    
    print(f"\n  ✅ Completado: {tipo.upper()}")
    return True

def main():
    """Función principal"""
    print("\n" + "="*70)
    print("🎯 GENERADOR DE PREDICCIONES ANUALES")
    print("="*70)
    
    configuraciones = [
        {
            'tipo': 'Casa Habitación',
            'file_path': 'exportados/Robos a casa habitacion/cuadrantes_robos_CaHa_promedio.xlsx',
            'prefijo': 'PROMEDIO DE ROBOS A CASA HABITACION MES',
            'output_dir': 'exportados/Robos a casa habitacion/',
            'nombre_archivo': 'casa_habitacion'
        },
        {
            'tipo': 'Negocios',
            'file_path': 'exportados/Robos a negocios/cuadrantes_robos_negocios_promedio.xlsx',
            'prefijo': 'PROMEDIO DE ROBOS A NEGOCIOS MES',
            'output_dir': 'exportados/Robos a negocios/',
            'nombre_archivo': 'negocios'
        },
        {
            'tipo': 'Vehículos',
            'file_path': 'exportados/Robos de vehiculos/cuadrantes_robos_vehiculos_promedio.xlsx',
            'prefijo': 'PROMEDIO DE ROBOS DE VEHICULOS MES',
            'output_dir': 'exportados/Robos de vehiculos/',
            'nombre_archivo': 'vehiculos'
        }
    ]
    
    exitosos = 0
    fallidos = 0
    
    for config in configuraciones:
        try:
            if procesar_tipo_robo(config):
                exitosos += 1
            else:
                fallidos += 1
        except Exception as e:
            print(f"\n  ❌ Error: {str(e)}")
            fallidos += 1
    
    print("\n" + "="*70)
    print("📊 RESUMEN")
    print("="*70)
    print(f"  ✅ Exitosos: {exitosos}")
    print(f"  ❌ Fallidos:  {fallidos}")
    print(f"\n  📂 Archivos generados en exportados/")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()
