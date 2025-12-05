import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import os


class ModeloPredictorRobos:
    """
    Clase unificada para entrenar y predecir robos de tres tipos:
    - Casa Habitación
    - Negocios
    - Vehículos
    """
    
    def __init__(self, tipo_robo):
        """
        Inicializa el predictor para un tipo de robo específico.
        
        Args:
            tipo_robo (str): 'casa', 'negocio' o 'vehiculo'
        """
        self.tipo_robo = tipo_robo.lower()
        self.modelo = None
        self.escalador = None
        self.r2_score = None
        self.df_final = None
        
        # Validar tipo de robo
        if self.tipo_robo not in ['casa', 'negocio', 'vehiculo']:
            raise ValueError("tipo_robo debe ser 'casa', 'negocio' o 'vehiculo'")
        
        # Rutas de archivos según tipo de robo
        self.rutas_archivos = {
            'casa': 'exportados/Robos a casa habitacion/cuadrantes_robos_CaHa_promedio.xlsx',
            'negocio': 'exportados/Robos a negocios/cuadrantes_robos_negocios_promedio.xlsx',
            'vehiculo': 'exportados/Robos de vehiculos/cuadrantes_robos_vehiculos_promedio.xlsx'
        }
        
        self.columna_robos = {
            'casa': 'PROMEDIO DE ROBOS A CASA HABITACION MES',
            'negocio': 'PROMEDIO DE ROBOS A NEGOCIOS MES',
            'vehiculo': 'PROMEDIO DE ROBOS DE VEHICULOS MES'
        }
        
        self.ruta_modelo = f'modelo_robos_{tipo_robo}.pkl'
    
    def cargar_y_preparar_datos(self, file_path=None):
        """Carga el archivo Excel y prepara el DataFrame."""
        if file_path is None:
            file_path = self.rutas_archivos[self.tipo_robo]
        
        df_promedios = pd.read_excel(file_path)
        
        # Renombrar columnas
        column_names = {
            'CUADRANTE': 'CUADRANTE',
            'POBLACION': 'POBLACION'
        }
        for i in range(1, 13):
            col_original = f'{self.columna_robos[self.tipo_robo]} {i}'
            column_names[col_original] = f'ROBOS_MES_{i}'
        
        df_promedios.rename(columns=column_names, inplace=True)
        
        # Desapilar (Unpivot)
        df_long = df_promedios.melt(
            id_vars=['CUADRANTE', 'POBLACION'],
            value_vars=[f'ROBOS_MES_{i}' for i in range(1, 13)],
            var_name='MES_NOMBRE',
            value_name='ROBOS_MES_N'
        )
        
        # Extraer número de mes
        df_long['MES_N'] = df_long['MES_NOMBRE'].str.extract(r'(\d+)').astype(int)
        df_long.drop(columns=['MES_NOMBRE'], inplace=True)
        
        # Ordenar
        df_long.sort_values(by=['MES_N', 'CUADRANTE'], inplace=True)
        df_long.reset_index(drop=True, inplace=True)
        
        return df_long
    
    def crear_variables_desfasadas(self, df_long):
        """Crea variables desfasadas (N-1 y N-2)."""
        # Extraer promedios de Diciembre y Noviembre
        df_diciembre = df_long[df_long['MES_N'] == 12].copy()
        df_diciembre.rename(columns={'ROBOS_MES_N': 'ROBOS_DIC'}, inplace=True)
        df_diciembre = df_diciembre[['CUADRANTE', 'ROBOS_DIC']]
        
        df_noviembre = df_long[df_long['MES_N'] == 11].copy()
        df_noviembre.rename(columns={'ROBOS_MES_N': 'ROBOS_NOV'}, inplace=True)
        df_noviembre = df_noviembre[['CUADRANTE', 'ROBOS_NOV']]
        
        # Crear columnas desfasadas
        df_long['ROBOS_MES_N_MENOS_1'] = df_long.groupby('CUADRANTE')['ROBOS_MES_N'].shift(1)
        df_long['ROBOS_MES_N_MENOS_2'] = df_long.groupby('CUADRANTE')['ROBOS_MES_N'].shift(2)
        
        # Imputar Enero y Febrero
        df_long = df_long.merge(df_diciembre, on='CUADRANTE', how='left')
        df_long.loc[df_long['MES_N'] == 1, 'ROBOS_MES_N_MENOS_1'] = df_long.loc[df_long['MES_N'] == 1, 'ROBOS_DIC']
        
        df_long = df_long.merge(df_noviembre, on='CUADRANTE', how='left')
        df_long.loc[df_long['MES_N'] == 1, 'ROBOS_MES_N_MENOS_2'] = df_long.loc[df_long['MES_N'] == 1, 'ROBOS_NOV']
        df_long.loc[df_long['MES_N'] == 2, 'ROBOS_MES_N_MENOS_2'] = df_long.loc[df_long['MES_N'] == 2, 'ROBOS_DIC']
        
        df_final = df_long.drop(columns=['ROBOS_DIC', 'ROBOS_NOV'])
        
        return df_final
    
    def entrenar_modelo(self, df_final):
        """Entrena el modelo Keras."""
        # Preparar datos
        df_scale = df_final[['CUADRANTE', 'POBLACION', 'ROBOS_MES_N_MENOS_1', 'ROBOS_MES_N_MENOS_2', 'ROBOS_MES_N']].copy()
        
        # Escalador
        escalador = MinMaxScaler()
        df_scaled_values = escalador.fit_transform(df_scale)
        df_scaled = pd.DataFrame(df_scaled_values, columns=df_scale.columns)
        df_scaled['MES_N'] = df_final['MES_N'].values
        
        # Estructurar datos
        X_features_scaled = df_scaled[['CUADRANTE', 'POBLACION', 'ROBOS_MES_N_MENOS_1', 'ROBOS_MES_N_MENOS_2']]
        Y_output_scaled = df_scaled['ROBOS_MES_N']
        
        X_data_scaled = X_features_scaled.groupby(df_scaled['MES_N']).apply(lambda x: x.values.flatten()).tolist()
        Y_data_scaled = Y_output_scaled.groupby(df_scaled['MES_N']).apply(lambda x: x.values).tolist()
        
        X_scaled = np.array(X_data_scaled)
        Y_scaled = np.array(Y_data_scaled)
        
        # Dividir datos
        X_train_scaled, X_test_scaled, Y_train_scaled, Y_test_scaled = train_test_split(
            X_scaled, Y_scaled, test_size=0.15, shuffle=False
        )
        
        input_dim = X_scaled.shape[1]
        output_dim = Y_scaled.shape[1]
        
        # Construir modelo
        modelo = Sequential([
            Dense(512, activation='relu', input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dropout(0.3),
            Dense(output_dim, activation='linear')
        ])
        
        modelo.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
        early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        
        # Entrenar
        modelo.fit(
            X_train_scaled, Y_train_scaled,
            epochs=200,
            batch_size=2,
            validation_data=(X_test_scaled, Y_test_scaled),
            callbacks=[early_stop],
            verbose=0
        )
        
        # Evaluar
        Y_pred_test_scaled = modelo.predict(X_test_scaled, verbose=0)
        r2_scaled = r2_score(Y_test_scaled.flatten(), Y_pred_test_scaled.flatten())
        
        self.modelo = modelo
        self.escalador = escalador
        self.r2_score = r2_scaled
        self.df_final = df_final
        
        return modelo, escalador, r2_scaled
    
    def predecir_top_cuadrantes(self, mes_a_predecir, top_n=10):
        """Realiza predicción para un mes específico."""
        if self.modelo is None or self.escalador is None:
            raise ValueError("El modelo no ha sido entrenado. Llama a entrenar_modelo() primero.")
        
        # Meses cíclicos
        mes_n_menos_1 = (mes_a_predecir - 1)
        mes_anterior_ciclo = (mes_n_menos_1 - 1) % 12 + 1
        mes_anterior_2_ciclo = (mes_a_predecir - 2 - 1) % 12 + 1
        
        # Obtener datos
        df_n_menos_1 = self.df_final[self.df_final['MES_N'] == mes_anterior_ciclo].copy()
        df_n_menos_2 = self.df_final[self.df_final['MES_N'] == mes_anterior_2_ciclo].copy()
        
        df_prediccion = self.df_final[self.df_final['MES_N'] == 1].copy()
        
        df_prediccion.sort_values(by='CUADRANTE', inplace=True)
        df_n_menos_1.sort_values(by='CUADRANTE', inplace=True)
        df_n_menos_2.sort_values(by='CUADRANTE', inplace=True)
        
        df_prediccion['ROBOS_MES_N_MENOS_1'] = df_n_menos_1['ROBOS_MES_N'].values
        df_prediccion['ROBOS_MES_N_MENOS_2'] = df_n_menos_2['ROBOS_MES_N'].values
        
        # Escalar y predecir
        X_pred_raw = df_prediccion[['CUADRANTE', 'POBLACION', 'ROBOS_MES_N_MENOS_1', 'ROBOS_MES_N_MENOS_2']].values
        X_con_dummy = np.hstack((X_pred_raw, np.zeros((X_pred_raw.shape[0], 1))))
        X_pred_escalado_completo = self.escalador.transform(X_con_dummy)
        X_pred_final = X_pred_escalado_completo[:, :4].flatten().reshape(1, -1)
        
        Y_pred_scaled = self.modelo.predict(X_pred_final, verbose=0)
        
        # Desescalar
        Y_pred_full = np.hstack((X_pred_raw, Y_pred_scaled.T))
        Y_pred_desescalado = self.escalador.inverse_transform(Y_pred_full)[:, -1]
        
        # Crear resultado
        df_resultado = pd.DataFrame({
            'CUADRANTE': df_prediccion['CUADRANTE'].values,
            'PREDICCION_ROBOS': Y_pred_desescalado,
            'POBLACION': df_prediccion['POBLACION'].values
        })
        
        # Retornar Top N
        top_cuadrantes = df_resultado.sort_values(by='PREDICCION_ROBOS', ascending=False).head(top_n).reset_index(drop=True)
        
        return top_cuadrantes
    
    def guardar_modelo(self):
        """Guarda el modelo y escalador."""
        if self.modelo is None:
            raise ValueError("No hay modelo entrenado para guardar")
        
        with open(self.ruta_modelo, 'wb') as f:
            pickle.dump({
                'modelo': self.modelo,
                'escalador': self.escalador,
                'r2_score': self.r2_score,
                'df_final': self.df_final
            }, f)
    
    def cargar_modelo(self):
        """Carga el modelo y escalador."""
        if not os.path.exists(self.ruta_modelo):
            return False
        
        with open(self.ruta_modelo, 'rb') as f:
            data = pickle.load(f)
        
        self.modelo = data['modelo']
        self.escalador = data['escalador']
        self.r2_score = data['r2_score']
        self.df_final = data['df_final']
        
        return True
    
    def entrenar_completo(self, file_path=None):
        """Ejecuta todo el pipeline de entrenamiento."""
        df_long = self.cargar_y_preparar_datos(file_path)
        df_final = self.crear_variables_desfasadas(df_long)
        self.entrenar_modelo(df_final)
        self.guardar_modelo()
        return self.r2_score
