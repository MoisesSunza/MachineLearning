import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Crear carpeta de modelos si no existe para evitar errores
os.makedirs('models', exist_ok=True)

# ==========================================
# 1. MODELO DE CLASIFICACIÓN (Telco Churn)
# ==========================================
print("1. Entrenando modelo de Clasificación (Telco Churn)...")
try:
    df_churn = pd.read_csv('datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    # Limpiar espacios en blanco de TotalCharges y rellenar nulos
    df_churn['TotalCharges'] = pd.to_numeric(df_churn['TotalCharges'], errors='coerce').fillna(0)
    
    # Preprocesamiento
    X_churn = df_churn.drop(['customerID', 'Churn'], axis=1)
    y_churn = df_churn['Churn'].map({'Yes': 1, 'No': 0})
    
    # Convertir variables categóricas a numéricas (One-Hot Encoding)
    X_churn = pd.get_dummies(X_churn, drop_first=True)
    features_churn = X_churn.columns.tolist()
    
    scaler_churn = StandardScaler()
    X_scaled_churn = scaler_churn.fit_transform(X_churn)
    
    # Dividir en Train (70%), Validation (15%) y Test (15%)
    X_temp_c, X_test_c, y_temp_c, y_test_c = train_test_split(X_scaled_churn, y_churn, test_size=0.15, random_state=42)
    X_train_c, X_val_c, y_train_c, y_val_c = train_test_split(X_temp_c, y_temp_c, test_size=0.176, random_state=42)
    
    model_c = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    model_c.fit(X_train_c, y_train_c)
    
    # Guardar modelo e historial completo para las gráficas de MATLAB
    joblib.dump({
        'model': model_c, 'scaler': scaler_churn, 'features': features_churn,
        'X_train': X_train_c, 'y_train': y_train_c.values,
        'X_val': X_val_c, 'y_val': y_val_c.values,
        'X_test': X_test_c, 'y_test': y_test_c.values,
        'loss_curve': model_c.loss_curve_
    }, 'models/model_classification.pkl')
    print("   -> Modelo de Clasificación guardado con éxito.")
except FileNotFoundError:
    print("   -> ERROR: No se encontró 'datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv'. Asegúrate de que la carpeta datasets/ exista.")

# ==========================================
# 2. MODELO DE REGRESIÓN (Life Expectancy)
# ==========================================
print("\n2. Entrenando modelo de Regresión (Esperanza de Vida)...")
try:
    df_reg = pd.read_csv('datasets/Life Expectancy Data.csv')
    
    # Limpiar los espacios en blanco ocultos en los nombres de las columnas (ej. " BMI ")
    df_reg.columns = df_reg.columns.str.strip()
    
    # Eliminar filas donde el objetivo (Target) sea nulo
    df_reg = df_reg.dropna(subset=['Life expectancy'])
    
    features_reg = ['Adult Mortality', 'infant deaths', 'BMI', 'Polio', 'GDP', 'Schooling']
    X_reg = df_reg[features_reg]
    y_reg = df_reg['Life expectancy']
    
    # Imputar valores faltantes (NaN) con la media
    imputer_reg = SimpleImputer(strategy='mean')
    X_imputed_reg = imputer_reg.fit_transform(X_reg)
    
    scaler_reg = StandardScaler()
    X_scaled_reg = scaler_reg.fit_transform(X_imputed_reg)
    
    # Dividir en Train (70%), Validation (15%) y Test (15%)
    X_temp_r, X_test_r, y_temp_r, y_test_r = train_test_split(X_scaled_reg, y_reg, test_size=0.15, random_state=42)
    X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(X_temp_r, y_temp_r, test_size=0.176, random_state=42)
    
    model_r = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=500, random_state=42)
    model_r.fit(X_train_r, y_train_r)
    
    # Guardar modelo e historial completo
    joblib.dump({
        'model': model_r, 'scaler': scaler_reg, 'imputer': imputer_reg, 'features': features_reg,
        'X_train': X_train_r, 'y_train': y_train_r.values,
        'X_val': X_val_r, 'y_val': y_val_r.values,
        'X_test': X_test_r, 'y_test': y_test_r.values,
        'loss_curve': model_r.loss_curve_
    }, 'models/model_regression.pkl')
    print("   -> Modelo de Regresión guardado con éxito.")
except FileNotFoundError:
    print("   -> ERROR: No se encontró 'datasets/Life Expectancy Data.csv'.")

# ==========================================
# 3. MODELO SERIES DE TIEMPO (Sensores Industriales)
# ==========================================
print("\n3. Entrenando modelo de Series de Tiempo (Sensores)...")
# Generamos la secuencia temporal sintética para evitar subir archivos muy pesados a GitHub
np.random.seed(42)
n_samples = 3500
t = np.linspace(0, 100, n_samples)
sensor_00 = np.sin(t) + np.random.normal(0, 0.1, n_samples)
sensor_04 = np.cos(t) * 100 + 500 + np.random.normal(0, 10, n_samples)
sensor_10 = np.sin(t/2) * 20 + 40 + np.random.normal(0, 2, n_samples)
target_ts = sensor_00 * 2 + (sensor_04/100) + np.random.normal(0, 0.5, n_samples)

df_ts = pd.DataFrame({'sensor_00': sensor_00, 'sensor_04': sensor_04, 'sensor_10': sensor_10})
features_ts = ['sensor_00', 'sensor_04', 'sensor_10']

scaler_ts = StandardScaler()
X_scaled_ts = scaler_ts.fit_transform(df_ts)

# División temporal cronológica (shuffle=False es vital para Series de Tiempo)
X_temp_t, X_test_t, y_temp_t, y_test_t = train_test_split(X_scaled_ts, target_ts, test_size=0.15, shuffle=False)
X_train_t, X_val_t, y_train_t, y_val_t = train_test_split(X_temp_t, y_temp_t, test_size=0.176, shuffle=False)

model_t = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=200, random_state=42)
model_t.fit(X_train_t, y_train_t)

# Guardar modelo e historial completo
joblib.dump({
    'model': model_t, 'scaler': scaler_ts, 'features': features_ts,
    'X_train': X_train_t, 'y_train': y_train_t,
    'X_val': X_val_t, 'y_val': y_val_t,
    'X_test': X_test_t, 'y_test': y_test_t,
    'loss_curve': model_t.loss_curve_
}, 'models/model_timeseries.pkl')
print("   -> Modelo de Series de Tiempo guardado con éxito.")

print("\n¡Listo! Todos los modelos han sido entrenados y los archivos .pkl actualizados correctamente.")