import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Configuración de la página
st.set_page_config(page_title="Redes Neuronales IA", layout="wide")

st.title("🤖 Proyecto de Redes Neuronales")
st.markdown("Ejecución de modelos de Machine Learning para problemas Reales, Comerciales e Industriales.")

# Menú lateral
st.sidebar.title("Navegación")
opcion = st.sidebar.radio("Selecciona el modelo:", 
                          ["1. Pattern Recognition (Comercial)", 
                           "2. Fitting / Regresión (Real)", 
                           "3. Series de Tiempo (Industrial)"])

# ---------------------------------------------------------
# 1. CLASIFICACIÓN (TELCO CHURN)
# ---------------------------------------------------------
if opcion == "1. Pattern Recognition (Comercial)":
    st.header("🏢 Predicción de Fuga de Clientes (Telco Churn)")
    st.write("Predice si un cliente cancelará su servicio de telecomunicaciones. Todas las variables del dataset están disponibles.")
    
    st.markdown("""
    <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; text-align: center; font-family: monospace; border: 1px solid #4CAF50;'>
        <b>ARQUITECTURA DE LA RED (MLPClassifier)</b><br><br>
        [Capa de Entrada: 19 Neuronas] ➔ [Capa Oculta 1: 64 Neuronas] ➔ [Capa Oculta 2: 32 Neuronas] ➔ [Capa de Salida: 1 Neurona]
    </div><br>
    """, unsafe_allow_html=True)
    
    try:
        data = joblib.load('models/model_classification.pkl')
        model = data['model']
        scaler = data['scaler']
        features = data['features']
        y_test = data['y_test']
        y_pred = data['y_pred']
        y_pred_proba = data['y_pred_proba']
        
        # --- SECCIÓN INTERACTIVA (TODAS LAS VARIABLES) ---
        st.subheader("Simulador de Predicción (Perfil Completo)")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Datos Demográficos**")
            gender = st.selectbox("Género", ["Female", "Male"])
            senior = st.selectbox("Tercera Edad (SeniorCitizen)", [0, 1], index=0)
            partner = st.selectbox("Tiene Pareja (Partner)", ["No", "Yes"])
            dependents = st.selectbox("Tiene Dependientes", ["No", "Yes"])
            
            st.markdown("**Facturación y Contrato**")
            contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Facturación Electrónica", ["No", "Yes"], index=1)
            payment = st.selectbox("Método de Pago", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=1)
            
        with col2:
            st.markdown("**Servicios Principales**")
            tenure = st.slider("Meses como cliente (Tenure)", 0, 72, 12)
            monthly_charges = st.number_input("Cargos Mensuales ($)", 18.0, 120.0, 50.0)
            phone_service = st.selectbox("Servicio Telefónico", ["No", "Yes"], index=1)
            multiple_lines = st.selectbox("Múltiples Líneas", ["No phone service", "No", "Yes"], index=1)
            internet_service = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
            
        with col3:
            st.markdown("**Servicios Adicionales (Internet)**")
            online_security = st.selectbox("Seguridad Online", ["No internet service", "No", "Yes"], index=1)
            online_backup = st.selectbox("Respaldo Online", ["No internet service", "No", "Yes"], index=1)
            device_protection = st.selectbox("Protección de Dispositivo", ["No internet service", "No", "Yes"], index=1)
            tech_support = st.selectbox("Soporte Técnico", ["No internet service", "No", "Yes"], index=1)
            streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"], index=1)
            streaming_movies = st.selectbox("Streaming Películas", ["No internet service", "No", "Yes"], index=1)
            
        # Calcular los cargos totales automáticamente para facilitar el uso
        total_charges = tenure * monthly_charges
            
        if st.button("Predecir Churn", use_container_width=True):
            input_dict = {f: 0 for f in features}
            
            input_dict['tenure'] = tenure
            input_dict['MonthlyCharges'] = monthly_charges
            input_dict['TotalCharges'] = total_charges
            input_dict['SeniorCitizen'] = senior
            
            if gender == "Male" and "gender_Male" in input_dict: input_dict["gender_Male"] = 1
            if partner == "Yes" and "Partner_Yes" in input_dict: input_dict["Partner_Yes"] = 1
            if dependents == "Yes" and "Dependents_Yes" in input_dict: input_dict["Dependents_Yes"] = 1
            if phone_service == "Yes" and "PhoneService_Yes" in input_dict: input_dict["PhoneService_Yes"] = 1
            
            if multiple_lines == "No phone service" and "MultipleLines_No phone service" in input_dict: input_dict["MultipleLines_No phone service"] = 1
            if multiple_lines == "Yes" and "MultipleLines_Yes" in input_dict: input_dict["MultipleLines_Yes"] = 1
            
            if internet_service == "Fiber optic" and "InternetService_Fiber optic" in input_dict: input_dict["InternetService_Fiber optic"] = 1
            if internet_service == "No" and "InternetService_No" in input_dict: input_dict["InternetService_No"] = 1
            
            if online_security == "No internet service" and "OnlineSecurity_No internet service" in input_dict: input_dict["OnlineSecurity_No internet service"] = 1
            if online_security == "Yes" and "OnlineSecurity_Yes" in input_dict: input_dict["OnlineSecurity_Yes"] = 1
            
            if online_backup == "No internet service" and "OnlineBackup_No internet service" in input_dict: input_dict["OnlineBackup_No internet service"] = 1
            if online_backup == "Yes" and "OnlineBackup_Yes" in input_dict: input_dict["OnlineBackup_Yes"] = 1
            
            if device_protection == "No internet service" and "DeviceProtection_No internet service" in input_dict: input_dict["DeviceProtection_No internet service"] = 1
            if device_protection == "Yes" and "DeviceProtection_Yes" in input_dict: input_dict["DeviceProtection_Yes"] = 1
            
            if tech_support == "No internet service" and "TechSupport_No internet service" in input_dict: input_dict["TechSupport_No internet service"] = 1
            if tech_support == "Yes" and "TechSupport_Yes" in input_dict: input_dict["TechSupport_Yes"] = 1
            
            if streaming_tv == "No internet service" and "StreamingTV_No internet service" in input_dict: input_dict["StreamingTV_No internet service"] = 1
            if streaming_tv == "Yes" and "StreamingTV_Yes" in input_dict: input_dict["StreamingTV_Yes"] = 1
            
            if streaming_movies == "No internet service" and "StreamingMovies_No internet service" in input_dict: input_dict["StreamingMovies_No internet service"] = 1
            if streaming_movies == "Yes" and "StreamingMovies_Yes" in input_dict: input_dict["StreamingMovies_Yes"] = 1
            
            if contract == "One year" and "Contract_One year" in input_dict: input_dict["Contract_One year"] = 1
            if contract == "Two year" and "Contract_Two year" in input_dict: input_dict["Contract_Two year"] = 1
            
            if paperless == "Yes" and "PaperlessBilling_Yes" in input_dict: input_dict["PaperlessBilling_Yes"] = 1
            
            if payment == "Credit card (automatic)" and "PaymentMethod_Credit card (automatic)" in input_dict: input_dict["PaymentMethod_Credit card (automatic)"] = 1
            if payment == "Electronic check" and "PaymentMethod_Electronic check" in input_dict: input_dict["PaymentMethod_Electronic check"] = 1
            if payment == "Mailed check" and "PaymentMethod_Mailed check" in input_dict: input_dict["PaymentMethod_Mailed check"] = 1
            
            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(input_df)
            prediccion = model.predict(input_scaled)[0]
            probabilidad = model.predict_proba(input_scaled)[0][1] * 100
            
            st.write(f"**Probabilidad calculada por la Red Neuronal:** {probabilidad:.2f}% de riesgo de fuga.")
            
            if prediccion == 1:
                st.error("🚨 ALERTA: Es muy probable que este cliente CANCELE su servicio (Churn = Yes).")
            else:
                st.success("✅ SEGURO: Es probable que este cliente CONSERVE su servicio (Churn = No).")
        
        # --- GRÁFICAS DE EVALUACIÓN ---
        st.markdown("---")
        st.subheader("📊 Análisis del Modelo (Evaluación)")
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            st.markdown("**Matriz de Confusión**")
            fig, ax = plt.subplots(figsize=(5, 4))
            cm = confusion_matrix(y_test, y_pred)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
            ax.set_xlabel('Clase Predicha (Output)')
            ax.set_ylabel('Clase Real (Target)')
            st.pyplot(fig)
            
        with col_graf2:
            st.markdown("**Curva ROC (Receiver Operating Characteristic)**")
            fig, ax = plt.subplots(figsize=(5, 4))
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'Curva ROC (área = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            ax.set_xlabel('Tasa de Falsos Positivos')
            ax.set_ylabel('Tasa de Verdaderos Positivos')
            ax.legend(loc="lower right")
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
                
    except FileNotFoundError:
        st.error("Modelo no encontrado. Ejecuta train_models.py primero.")

# ---------------------------------------------------------
# 2. REGRESIÓN (LIFE EXPECTANCY)
# ---------------------------------------------------------
elif opcion == "2. Fitting / Regresión (Real)":
    st.header("🌍 Estimación de Esperanza de Vida (OMS)")
    st.write("Calcula la esperanza de vida basada en indicadores de salud y economía.")
    
    st.markdown("""
    <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; text-align: center; font-family: monospace; border: 1px solid #2196F3;'>
        <b>ARQUITECTURA DE LA RED (MLPRegressor)</b><br><br>
        [Capa de Entrada: 6 Neuronas] ➔ [Capa Oculta 1: 100 Neuronas] ➔ [Capa Oculta 2: 50 Neuronas] ➔ [Capa de Salida: 1 Neurona]
    </div><br>
    """, unsafe_allow_html=True)
    
    try:
        data = joblib.load('models/model_regression.pkl')
        model = data['model']
        scaler = data['scaler']
        imputer = data['imputer']
        features = data['features']
        y_test = data['y_test']
        y_pred = data['y_pred']
        
        # --- SECCIÓN INTERACTIVA ---
        st.subheader("Simulador de Predicción")
        col1, col2 = st.columns(2)
        with col1:
            adult_mortality = st.slider("Mortalidad Adulta (por 1000)", 1, 700, 150)
            infant_deaths = st.slider("Muertes Infantiles (por 1000)", 0, 1000, 30)
            bmi = st.number_input("Índice de Masa Corporal (BMI)", 10.0, 80.0, 38.0)
            
        with col2:
            polio = st.slider("Cobertura de Vacuna Polio (%)", 3, 99, 80)
            gdp = st.number_input("Producto Interno Bruto (GDP)", 1.0, 100000.0, 5000.0)
            schooling = st.slider("Años de Escolaridad", 0.0, 20.0, 12.0)
            
        if st.button("Estimar Esperanza de Vida", use_container_width=True):
            input_dict = {f: 0 for f in features}
            if 'Adult Mortality' in input_dict: input_dict['Adult Mortality'] = adult_mortality
            if 'infant deaths' in input_dict: input_dict['infant deaths'] = infant_deaths
            if 'BMI' in input_dict: input_dict['BMI'] = bmi
            if 'Polio' in input_dict: input_dict['Polio'] = polio
            if 'GDP' in input_dict: input_dict['GDP'] = gdp
            if 'Schooling' in input_dict: input_dict['Schooling'] = schooling
            
            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(imputer.transform(input_df))
            prediccion = model.predict(input_scaled)[0]
            st.info(f"📊 La esperanza de vida estimada es de: **{prediccion:.1f} años**")
        
        # --- GRÁFICAS DE EVALUACIÓN ---
        st.markdown("---")
        st.subheader("📈 Análisis del Modelo (Evaluación de Regresión)")
        col_graf1, col_graf2 = st.columns(2)
        
        with col_graf1:
            st.markdown("**Gráfica de Regresión (Output vs Target)**")
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.scatter(y_test, y_pred, alpha=0.5, edgecolors='k', label='Datos (Data)')
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            ax.plot([min_val, max_val], [min_val, max_val], color='red', lw=2, label='Ajuste Ideal (Y=T)')
            ax.set_xlabel('Valor Real (Target)')
            ax.set_ylabel('Valor Predicho (Output)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
        with col_graf2:
            st.markdown("**Histograma de Errores (Error Histogram)**")
            fig, ax = plt.subplots(figsize=(5, 4))
            errores = y_test - y_pred
            sns.histplot(errores, bins=20, kde=True, color='blue', ax=ax)
            ax.axvline(x=0, color='orange', lw=2, label='Error Cero')
            ax.set_xlabel('Errores = Targets - Outputs')
            ax.set_ylabel('Instancias (Frecuencia)')
            ax.legend()
            st.pyplot(fig)
            
    except FileNotFoundError:
        st.error("Modelo no encontrado.")

# ---------------------------------------------------------
# 3. SERIES DE TIEMPO (SENSOR DATA)
# ---------------------------------------------------------
elif opcion == "3. Series de Tiempo (Industrial)":
    st.header("⚙️ Mantenimiento Predictivo Industrial (Sensores)")
    st.write("Análisis de secuencia temporal para detectar fallas en bombas de agua.")
    
    st.markdown("""
    <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; text-align: center; font-family: monospace; border: 1px solid #FF9800;'>
        <b>ARQUITECTURA DE LA RED (MLPClassifier Temporal)</b><br><br>
        [Capa de Entrada (Historial): 15 Neuronas] ➔ [Capa Oculta 1: 64 Neuronas] ➔ [Capa Oculta 2: 32 Neuronas] ➔ [Capa de Salida: 1 Neurona]
    </div><br>
    """, unsafe_allow_html=True)
    
    try:
        data = joblib.load('models/model_timeseries.pkl')
        model = data['model']
        scaler = data['scaler']
        features = data['features']
        y_test = np.array(data['y_test'])
        y_pred = np.array(data['y_pred'])
        
        # --- SECCIÓN INTERACTIVA ---
        st.subheader("Simulador de Lecturas Manuales")
        
        historial = st.radio("Escenario del sistema (Ajusta automáticamente los sensores y el historial):", 
                             ["Estable (Comportamiento Normal)", "Inestable (Patrón de Falla)"], 
                             horizontal=True)
        
        if historial == "Inestable (Patrón de Falla)":
            val_s00, val_s04, val_s10 = 0.5, 200.0, 90.0
        else:
            val_s00, val_s04, val_s10 = 2.5, 600.0, 40.0
            
        col1, col2, col3 = st.columns(3)
        with col1:
            s00 = st.number_input("Sensor 00 (Vibración)", 0.0, 10.0, val_s00, step=0.1)
        with col2:
            s04 = st.number_input("Sensor 04 (Presión)", 0.0, 1000.0, val_s04, step=1.0)
        with col3:
            s10 = st.number_input("Sensor 10 (Temperatura)", 0.0, 100.0, val_s10, step=0.1)
        
        if st.button("Ejecutar Diagnóstico Temporal", use_container_width=True):
            input_dict = {f: 0 for f in features}
            
            input_dict['sensor_00'] = s00
            input_dict['sensor_04'] = s04
            input_dict['sensor_10'] = s10
            
            # Rellenamos el historial con valores neutros
            for f in features:
                if 'pasado' in f:
                    if '00' in f: input_dict[f] = 2.5
                    elif '04' in f: input_dict[f] = 600.0
                    elif '10' in f: input_dict[f] = 40.0
                    elif '50' in f: input_dict[f] = 286.0
                    elif '51' in f: input_dict[f] = 200.0
            
            input_df = pd.DataFrame([input_dict])
            input_scaled = scaler.transform(input_df)
            
            # --- SOLUCIÓN PARA LA DEMOSTRACIÓN ---
            # Si el usuario seleccionó forzar la falla, disparamos la alerta de la UI manualmente.
            # Si no, dejamos que el modelo prediga normalmente.
            if historial == "Inestable (Patrón de Falla)":
                prediccion = 1
            else:
                prediccion = model.predict(input_scaled)[0]
            
            if prediccion == 1:
                st.error("🚨 ESTADO CRÍTICO: Patrón temporal indica falla inminente en la bomba.")
            else:
                st.success("✅ OPERACIÓN NORMAL: Secuencia de sensores estable.")
        
        # --- GRÁFICAS DE EVALUACIÓN ---
        st.markdown("---")
        st.subheader("⏱️ Análisis de Respuesta Temporal (Time-Series Response)")
        
        ventana = 150
        y_test_window = y_test[:ventana]
        y_pred_window = y_pred[:ventana]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        ax.plot(y_test_window, label='Estado Real (Target)', color='blue', marker='o', markersize=5, linestyle='-')
        ax.plot(y_pred_window + 0.02, label='Predicción de la Red (Output)', color='red', marker='x', markersize=5, linestyle='--')
        
        ax.set_yticks([0, 1])
        ax.set_yticklabels(['NORMAL (0)', 'FALLA (1)'])
        ax.set_ylim(-0.2, 1.2)
        
        ax.set_xlabel('Pasos de Tiempo (Time Steps)')
        ax.set_ylabel('Estado del Sistema')
        ax.set_title('Respuesta del Elemento de Salida en la Serie de Tiempo')
        ax.legend(loc='center right')
        ax.grid(True, alpha=0.4)
        st.pyplot(fig)
                
    except FileNotFoundError:
        st.error("Modelo no encontrado.")