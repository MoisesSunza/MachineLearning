import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------
# CONFIGURACIÓN INICIAL
# ---------------------------------------------------------
os.makedirs('models', exist_ok=True)

print("="*60)
print("1. ENTRENANDO RED NEURONAL: CLASIFICACIÓN (TELCO CHURN)")
print("="*60)

try:
    df_churn = pd.read_csv('datasets/WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df_churn.drop('customerID', axis=1, inplace=True)
    df_churn['TotalCharges'] = pd.to_numeric(df_churn['TotalCharges'], errors='coerce')
    df_churn['TotalCharges'].fillna(df_churn['TotalCharges'].median(), inplace=True)
    
    X_clf = df_churn.drop('Churn', axis=1)
    y_clf = df_churn['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)
    X_clf = pd.get_dummies(X_clf, drop_first=True)
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    scaler_c = StandardScaler()
    X_train_c_scaled = scaler_c.fit_transform(X_train_c)
    X_test_c_scaled = scaler_c.transform(X_test_c)
    
    mlp_clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
    mlp_clf.fit(X_train_c_scaled, y_train_c)
    
    y_pred_c = mlp_clf.predict(X_test_c_scaled)
    y_pred_proba_c = mlp_clf.predict_proba(X_test_c_scaled)[:, 1] # Necesario para la curva ROC
    print(f"Accuracy (Precisión): {accuracy_score(y_test_c, y_pred_c)*100:.2f}%")
    
    # NUEVO: Guardamos y_test, y_pred y y_pred_proba para poder graficar
    joblib.dump({
        'model': mlp_clf, 'scaler': scaler_c, 'features': list(X_clf.columns),
        'y_test': y_test_c, 'y_pred': y_pred_c, 'y_pred_proba': y_pred_proba_c
    }, 'models/model_classification.pkl')
    print("✅ Modelo de Clasificación guardado (model_classification.pkl)\n")
except Exception as e:
    print(f"❌ Error en Clasificación: {e}\n")


print("="*60)
print("2. ENTRENANDO RED NEURONAL: REGRESIÓN (LIFE EXPECTANCY)")
print("="*60)

try:
    df_life = pd.read_csv('datasets/Life Expectancy Data.csv')
    df_life.rename(columns=lambda x: x.strip(), inplace=True)
    df_life.dropna(subset=['Life expectancy'], inplace=True)
    
    X_reg = df_life.drop(['Life expectancy', 'Country', 'Year'], axis=1)
    y_reg = df_life['Life expectancy']
    X_reg = pd.get_dummies(X_reg, drop_first=True)
    
    imputer = SimpleImputer(strategy='mean')
    X_reg_imputed = imputer.fit_transform(X_reg)
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg_imputed, y_reg, test_size=0.2, random_state=42)
    scaler_r = StandardScaler()
    X_train_r_scaled = scaler_r.fit_transform(X_train_r)
    X_test_r_scaled = scaler_r.transform(X_test_r)
    
    mlp_reg = MLPRegressor(hidden_layer_sizes=(100, 50), activation='relu', solver='adam', max_iter=1000, random_state=42)
    mlp_reg.fit(X_train_r_scaled, y_train_r)
    
    y_pred_r = mlp_reg.predict(X_test_r_scaled)
    print(f"R^2 (Coef. de Determinación): {r2_score(y_test_r, y_pred_r):.4f}")
    
    # NUEVO: Guardamos y_test e y_pred para graficar la Regresión y el Histograma de Errores
    joblib.dump({
        'model': mlp_reg, 'scaler': scaler_r, 'imputer': imputer, 'features': list(X_reg.columns),
        'y_test': y_test_r, 'y_pred': y_pred_r
    }, 'models/model_regression.pkl')
    print("✅ Modelo de Regresión guardado (model_regression.pkl)\n")
except Exception as e:
    print(f"❌ Error en Regresión: {e}\n")


print("="*60)
print("3. ENTRENANDO RED NEURONAL: SERIES DE TIEMPO (SENSOR DATA)")
print("="*60)

try:
    df_sensor = pd.read_csv('datasets/sensor.csv')
    print("Limpiando datos de sensores y creando secuencias temporales...")
    
    df_sensor.drop(['Unnamed: 0', 'sensor_15'], axis=1, inplace=True, errors='ignore')
    df_sensor['timestamp'] = pd.to_datetime(df_sensor['timestamp'])
    df_sensor.sort_values('timestamp', inplace=True)
    df_sensor.ffill(inplace=True)
    df_sensor.bfill(inplace=True)
    
    df_sensor = df_sensor.iloc[::20, :].reset_index(drop=True)
    df_sensor['target'] = df_sensor['machine_status'].apply(lambda x: 0 if x == 'NORMAL' else 1)
    
    sensores_clave = ['sensor_00', 'sensor_04', 'sensor_10', 'sensor_50', 'sensor_51']
    for col in sensores_clave:
        df_sensor[f'{col}_pasado_1'] = df_sensor[col].shift(1)
        df_sensor[f'{col}_pasado_2'] = df_sensor[col].shift(2)
    
    df_sensor.dropna(inplace=True)
    
    features_ts = [c for c in df_sensor.columns if 'pasado' in c or c in sensores_clave]
    X_ts = df_sensor[features_ts]
    y_ts = df_sensor['target']
    
    X_train_ts, X_test_ts, y_train_ts, y_test_ts = train_test_split(X_ts, y_ts, test_size=0.2, shuffle=False)
    scaler_ts = StandardScaler()
    X_train_ts_scaled = scaler_ts.fit_transform(X_train_ts)
    X_test_ts_scaled = scaler_ts.transform(X_test_ts)
    
    mlp_ts = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=300, random_state=42)
    mlp_ts.fit(X_train_ts_scaled, y_train_ts)
    
    y_pred_ts = mlp_ts.predict(X_test_ts_scaled)
    print(f"Accuracy en el Tiempo: {accuracy_score(y_test_ts, y_pred_ts)*100:.2f}%")
    
    # NUEVO: Guardamos y_test e y_pred para la gráfica de Serie de Tiempo (Target vs Output)
    joblib.dump({
        'model': mlp_ts, 'scaler': scaler_ts, 'features': features_ts,
        'y_test': y_test_ts, 'y_pred': y_pred_ts
    }, 'models/model_timeseries.pkl')
    print("✅ Modelo de Series de Tiempo guardado (model_timeseries.pkl)\n")
except Exception as e:
    print(f"❌ Error en Series de Tiempo: {e}\n")

print("🎉 PROCESO FINALIZADO. Archivos actualizados.")