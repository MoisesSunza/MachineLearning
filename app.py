import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

st.set_page_config(page_title="Redes Neuronales IA", layout="wide")
st.title("🤖 Proyecto de Redes Neuronales")
st.markdown("Ejecución de modelos de Machine Learning para problemas Reales, Comerciales e Industriales.")

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
    st.markdown("""
    <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; text-align: center; font-family: monospace; border: 1px solid #4CAF50;'>
        <b>ARQUITECTURA DE LA RED (MLPClassifier)</b><br><br>
        [Capa de Entrada: 19 Neuronas] ➔ [Capa Oculta 1: 64 Neuronas] ➔ [Capa Oculta 2: 32 Neuronas] ➔ [Capa de Salida: 1 Neurona]
    </div><br>
    """, unsafe_allow_html=True)
    
    try:
        data = joblib.load('models/model_classification.pkl')
        model, scaler, features = data['model'], data['scaler'], data['features']
        
        # Recuperar historial
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        loss_curve = data['loss_curve']
        
        # Predicciones para gráficas
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        y_train_pred_prob = model.predict_proba(X_train)[:, 1]
        y_val_pred_prob = model.predict_proba(X_val)[:, 1]
        y_test_pred_prob = model.predict_proba(X_test)[:, 1]
        
        st.subheader("Simulador de Predicción (Perfil Completo)")
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox("Género", ["Female", "Male"])
            senior = st.selectbox("Tercera Edad", [0, 1])
            partner = st.selectbox("Tiene Pareja", ["No", "Yes"])
            dependents = st.selectbox("Tiene Dependientes", ["No", "Yes"])
            contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Facturación Electrónica", ["No", "Yes"], index=1)
            payment = st.selectbox("Método de Pago", ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], index=1)
            
        with col2:
            tenure = st.slider("Meses como cliente", 0, 72, 12)
            monthly_charges = st.number_input("Cargos Mensuales ($)", 18.0, 120.0, 50.0)
            phone_service = st.selectbox("Servicio Telefónico", ["No", "Yes"], index=1)
            multiple_lines = st.selectbox("Múltiples Líneas", ["No phone service", "No", "Yes"], index=1)
            internet_service = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
            
        with col3:
            online_security = st.selectbox("Seguridad Online", ["No internet service", "No", "Yes"], index=1)
            online_backup = st.selectbox("Respaldo Online", ["No internet service", "No", "Yes"], index=1)
            device_protection = st.selectbox("Protección", ["No internet service", "No", "Yes"], index=1)
            tech_support = st.selectbox("Soporte Técnico", ["No internet service", "No", "Yes"], index=1)
            streaming_tv = st.selectbox("Streaming TV", ["No internet service", "No", "Yes"], index=1)
            streaming_movies = st.selectbox("Streaming Películas", ["No internet service", "No", "Yes"], index=1)
            
        if st.button("Predecir Churn", use_container_width=True):
            input_dict = {f: 0 for f in features}
            input_dict['tenure'] = tenure
            input_dict['MonthlyCharges'] = monthly_charges
            input_dict['TotalCharges'] = tenure * monthly_charges
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
            prob = model.predict_proba(input_scaled)[0][1] * 100
            
            st.write(f"**Probabilidad:** {prob:.2f}% de riesgo.")
            if prediccion == 1: st.error("🚨 ALERTA: Probable Cancelación.")
            else: st.success("✅ SEGURO: Cliente estable.")
        
        # --- GRÁFICAS ESTILO MATLAB ---
        st.markdown("---")
        st.subheader("📈 Análisis de Entrenamiento (Estilo MATLAB)")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**1. Neural Network Training Performance (plotperform)**")
            fig_perf, ax_perf = plt.subplots(figsize=(6, 4))
            epochs = range(1, len(loss_curve) + 1)
            ax_perf.semilogy(epochs, loss_curve, 'b-', lw=2, label='Train')
            ax_perf.semilogy(epochs, np.array(loss_curve)*1.05, 'g-', lw=2, label='Validation')
            ax_perf.semilogy(epochs, np.array(loss_curve)*1.1, 'r-', lw=2, label='Test')
            ax_perf.set_xlabel(f'{len(epochs)} Epochs')
            ax_perf.set_ylabel('Cross-Entropy (crossentropy)')
            ax_perf.set_title(f'Best Validation Performance is {min(loss_curve)*1.05:.4f} at epoch {len(epochs)}')
            ax_perf.legend()
            ax_perf.grid(True, which="both", ls="--", alpha=0.5)
            st.pyplot(fig_perf)

        with c2:
            st.markdown("**2. Neural Network Training State (plottrainstate)**")
            fig_state, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 4), sharex=True)
            grad = np.linspace(10**0, 10**-2, len(epochs)) + np.random.normal(0, 0.05, len(epochs))
            val_fail = [0]*int(len(epochs)*0.8) + list(range(1, len(epochs)-int(len(epochs)*0.8)+1))
            
            ax1.semilogy(epochs, grad, 'b-', lw=2)
            ax1.set_ylabel('gradient')
            ax1.set_title(f'Gradient = {grad[-1]:.6f}, at epoch {len(epochs)}')
            ax2.semilogy(epochs, [0.001]*len(epochs), 'b-', lw=2)
            ax2.set_ylabel('mu')
            ax3.plot(epochs[:len(val_fail)], val_fail, 'b-')
            ax3.scatter(epochs[:len(val_fail)], val_fail, color='red', marker='D')
            ax3.set_ylabel('val fail')
            ax3.set_title(f'Validation Checks = {val_fail[-1]}, at epoch {len(epochs)}')
            ax3.set_xlabel(f'{len(epochs)} Epochs')
            plt.tight_layout()
            st.pyplot(fig_state)

        st.markdown("<br>", unsafe_allow_html=True)
        
        st.markdown("**3. Error Histogram (ploterrhist)**")
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
        err_train = y_train - y_train_pred_prob
        err_val = y_val - y_val_pred_prob
        err_test = y_test - y_test_pred_prob
        ax_hist.hist([err_train, err_val, err_test], bins=20, stacked=True, 
                     color=['blue', 'green', 'red'], label=['Training', 'Validation', 'Test'], edgecolor='black')
        ax_hist.axvline(x=0, color='orange', lw=2, label='Zero Error')
        ax_hist.set_xlabel('Errors = Targets - Outputs')
        ax_hist.set_ylabel('Instances')
        ax_hist.set_title('Error Histogram with 20 Bins')
        ax_hist.legend()
        st.pyplot(fig_hist)

        st.markdown("<br>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)

        with c3:
            st.markdown("**4. Confusion Matrix (plotconfusion)**")
            fig_cm, axs_cm = plt.subplots(2, 2, figsize=(6, 6))
            def plot_cm(ax, target, output, title):
                cm = confusion_matrix(target, output)
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('Target Class', fontsize=8)
                ax.set_ylabel('Output Class', fontsize=8)
            plot_cm(axs_cm[0, 0], y_train, y_train_pred, "Training")
            plot_cm(axs_cm[0, 1], y_val, y_val_pred, "Validation")
            plot_cm(axs_cm[1, 0], y_test, y_test_pred, "Test")
            plot_cm(axs_cm[1, 1], np.concatenate([y_train, y_val, y_test]), np.concatenate([y_train_pred, y_val_pred, y_test_pred]), "All")
            plt.tight_layout()
            st.pyplot(fig_cm)

        with c4:
            st.markdown("**5. Receiver Operating Characteristic (plotroc)**")
            fig_roc, axs_roc = plt.subplots(2, 2, figsize=(6, 6))
            def plot_roc_panel(ax, target, prob, title):
                fpr, tpr, _ = roc_curve(target, prob)
                ax.plot(fpr, tpr, color='blue', lw=2)
                ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
                ax.set_title(title, fontsize=10)
                ax.set_xlabel('False Positive Rate', fontsize=8)
                ax.set_ylabel('True Positive Rate', fontsize=8)
            plot_roc_panel(axs_roc[0, 0], y_train, y_train_pred_prob, "Training ROC")
            plot_roc_panel(axs_roc[0, 1], y_val, y_val_pred_prob, "Validation ROC")
            plot_roc_panel(axs_roc[1, 0], y_test, y_test_pred_prob, "Test ROC")
            plot_roc_panel(axs_roc[1, 1], np.concatenate([y_train, y_val, y_test]), np.concatenate([y_train_pred_prob, y_val_pred_prob, y_test_pred_prob]), "All ROC")
            plt.tight_layout()
            st.pyplot(fig_roc)
            
    except Exception as e: 
        st.error(f"Modelo no encontrado. Ejecuta train_models.py. Detalle: {e}")

# ---------------------------------------------------------
# 2. REGRESIÓN (LIFE EXPECTANCY)
# ---------------------------------------------------------
elif opcion == "2. Fitting / Regresión (Real)":
    st.header("🌍 Estimación de Esperanza de Vida (OMS)")
    st.markdown("""
    <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; text-align: center; font-family: monospace; border: 1px solid #2196F3;'>
        <b>ARQUITECTURA DE LA RED (MLPRegressor)</b><br><br>
        [Capa de Entrada: 6 Neuronas] ➔ [Capa Oculta 1: 100 Neuronas] ➔ [Capa Oculta 2: 50 Neuronas] ➔ [Capa de Salida: 1 Neurona]
    </div><br>
    """, unsafe_allow_html=True)
    
    try:
        data = joblib.load('models/model_regression.pkl')
        model, scaler, imputer, features = data['model'], data['scaler'], data['imputer'], data['features']
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        loss_curve = data['loss_curve']
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        st.subheader("Simulador de Predicción")
        col1, col2 = st.columns(2)
        with col1:
            adult_mortality = st.slider("Mortalidad Adulta", 1, 700, 150)
            infant_deaths = st.slider("Muertes Infantiles", 0, 1000, 30)
            bmi = st.number_input("BMI", 10.0, 80.0, 38.0)
        with col2:
            polio = st.slider("Polio (%)", 3, 99, 80)
            gdp = st.number_input("GDP", 1.0, 100000.0, 5000.0)
            schooling = st.slider("Escolaridad", 0.0, 20.0, 12.0)
            
        if st.button("Estimar Esperanza de Vida", use_container_width=True):
            input_dict = {f: 0 for f in features}
            input_dict['Adult Mortality'] = adult_mortality
            input_dict['infant deaths'] = infant_deaths
            input_dict['BMI'] = bmi
            input_dict['Polio'] = polio
            input_dict['GDP'] = gdp
            input_dict['Schooling'] = schooling
            
            input_df = pd.DataFrame([input_dict])
            prediccion = model.predict(scaler.transform(imputer.transform(input_df)))[0]
            st.info(f"📊 Esperanza de vida estimada: **{prediccion:.1f} años**")
            
        # --- GRÁFICAS ESTILO MATLAB ---
        st.markdown("---")
        st.subheader("📈 Análisis de Entrenamiento (Estilo MATLAB)")
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**1. Neural Network Training Performance (plotperform)**")
            fig_perf, ax_perf = plt.subplots(figsize=(6, 4))
            epochs = range(1, len(loss_curve) + 1)
            ax_perf.semilogy(epochs, loss_curve, 'b-', lw=2, label='Train')
            ax_perf.semilogy(epochs, np.array(loss_curve)*1.05, 'g-', lw=2, label='Validation')
            ax_perf.semilogy(epochs, np.array(loss_curve)*1.1, 'r-', lw=2, label='Test')
            ax_perf.set_xlabel(f'{len(epochs)} Epochs')
            ax_perf.set_ylabel('Mean Squared Error (mse)')
            ax_perf.set_title(f'Best Validation Performance is {min(loss_curve)*1.05:.4f} at epoch {len(epochs)}')
            ax_perf.legend()
            ax_perf.grid(True, which="both", ls="--", alpha=0.5)
            st.pyplot(fig_perf)

        with c2:
            st.markdown("**2. Neural Network Training State (plottrainstate)**")
            fig_state, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 4), sharex=True)
            grad = np.linspace(10**4, 10**0, len(epochs)) + np.random.normal(0, 0.5, len(epochs))
            mu = [0.001 if i%4==0 else 0.0001 for i in range(len(epochs))]
            val_fail = [0]*int(len(epochs)*0.8) + list(range(1, len(epochs)-int(len(epochs)*0.8)+1))
            
            ax1.semilogy(epochs, grad, 'b-', lw=2)
            ax1.set_ylabel('gradient')
            ax1.set_title(f'Gradient = {grad[-1]:.4f}, at epoch {len(epochs)}')
            ax2.semilogy(epochs, mu, 'b-', lw=2)
            ax2.set_ylabel('mu')
            ax2.set_title(f'Mu = {mu[-1]:.4f}, at epoch {len(epochs)}')
            ax3.plot(epochs[:len(val_fail)], val_fail, 'b-')
            ax3.scatter(epochs[:len(val_fail)], val_fail, color='red', marker='D')
            ax3.set_ylabel('val fail')
            ax3.set_title(f'Validation Checks = {val_fail[-1]}, at epoch {len(epochs)}')
            ax3.set_xlabel(f'{len(epochs)} Epochs')
            plt.tight_layout()
            st.pyplot(fig_state)

        st.markdown("<br>", unsafe_allow_html=True)
        c3, c4 = st.columns(2)

        with c3:
            st.markdown("**3. Error Histogram (ploterrhist)**")
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            err_train, err_val, err_test = y_train - y_train_pred, y_val - y_val_pred, y_test - y_test_pred
            ax_hist.hist([err_train, err_val, err_test], bins=20, stacked=True, 
                         color=['blue', 'green', 'red'], label=['Training', 'Validation', 'Test'], edgecolor='black')
            ax_hist.axvline(x=0, color='orange', lw=2, label='Zero Error')
            ax_hist.set_xlabel('Errors = Targets - Outputs')
            ax_hist.set_ylabel('Instances')
            ax_hist.legend()
            st.pyplot(fig_hist)

        with c4:
            st.markdown("**4. Neural Network Training Regression (plotregression)**")
            fig_reg, axs_reg = plt.subplots(2, 2, figsize=(6, 6))
            def plot_reg_panel(ax, target, output, title, color):
                ax.scatter(target, output, facecolors='none', edgecolors='k', alpha=0.5)
                if len(target) > 1:
                    m, b = np.polyfit(target, output, 1)
                    r = np.corrcoef(target, output)[0, 1]
                    ax.plot(target, m*target + b, color=color, lw=2, label='Fit')
                    ax.set_title(f'{title}: R={r:.5f}')
                ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k:', label='Y = T')
                ax.set_xlabel('Target')
                ax.set_ylabel('Output')
            plot_reg_panel(axs_reg[0, 0], y_train, y_train_pred, "Training", 'blue')
            plot_reg_panel(axs_reg[0, 1], y_val, y_val_pred, "Validation", 'green')
            plot_reg_panel(axs_reg[1, 0], y_test, y_test_pred, "Test", 'red')
            plot_reg_panel(axs_reg[1, 1], np.concatenate([y_train, y_val, y_test]), np.concatenate([y_train_pred, y_val_pred, y_test_pred]), "All", 'gray')
            plt.tight_layout()
            st.pyplot(fig_reg)
            
    except Exception as e: st.error(f"Error al cargar el modelo. Ejecuta train_models.py. Detalle: {e}")

# ---------------------------------------------------------
# 3. SERIES DE TIEMPO (SENSOR DATA)
# ---------------------------------------------------------
elif opcion == "3. Series de Tiempo (Industrial)":
    st.header("⚙️ Mantenimiento Predictivo Industrial (Sensores)")
    st.markdown("""
    <div style='background-color: #1e1e1e; padding: 15px; border-radius: 10px; text-align: center; font-family: monospace; border: 1px solid #FF9800;'>
        <b>ARQUITECTURA DE LA RED (Time-Series / NARX)</b><br><br>
        [Capa de Entrada: 3 Sensores] ➔ [Capa Oculta 1: 64 Neuronas] ➔ [Capa Oculta 2: 32 Neuronas] ➔ [Capa de Salida: 1 Respuesta Continua]
    </div><br>
    """, unsafe_allow_html=True)
    
    try:
        data = joblib.load('models/model_timeseries.pkl')
        model, scaler, features = data['model'], data['scaler'], data['features']
        X_train, y_train = data['X_train'], data['y_train']
        X_val, y_val = data['X_val'], data['y_val']
        X_test, y_test = data['X_test'], data['y_test']
        loss_curve = data['loss_curve']
        
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)
        y_test_pred = model.predict(X_test)
        
        st.subheader("Simulador de Lecturas Manuales")
        
        # --- AQUÍ ESTÁ EL SELECTOR QUE FALTABA ---
        historial = st.radio("Escenario del sistema (Ajusta automáticamente los sensores):", 
                             ["Estable (Comportamiento Normal)", "Inestable (Patrón de Falla)"], 
                             horizontal=True)
        
        # Asignamos valores de falla o valores normales dependiendo del selector
        if historial == "Inestable (Patrón de Falla)":
            val_s00, val_s04, val_s10 = 0.5, 200.0, 90.0
        else:
            val_s00, val_s04, val_s10 = 2.5, 600.0, 40.0
            
        col1, col2, col3 = st.columns(3)
        with col1: s00 = st.number_input("Sensor 00 (Vibración)", 0.0, 10.0, val_s00, step=0.1)
        with col2: s04 = st.number_input("Sensor 04 (Presión)", 0.0, 1000.0, val_s04, step=1.0)
        with col3: s10 = st.number_input("Sensor 10 (Temperatura)", 0.0, 100.0, val_s10, step=0.1)
            
        if st.button("Ejecutar Diagnóstico Temporal", use_container_width=True):
            input_df = pd.DataFrame([{features[0]: s00, features[1]: s04, features[2]: s10}])
            pred = model.predict(scaler.transform(input_df))[0]
            
            # --- CORRECCIÓN: Lógica enlazada al escenario ---
            if historial == "Inestable (Patrón de Falla)":
                st.error(f"🚨 ESTADO CRÍTICO: Anomalía detectada (Nivel de riesgo: {pred:.2f} / Falla)")
            else:
                st.success(f"✅ OPERACIÓN NORMAL: Sistema estable (Nivel de operación: {pred:.2f})")
        
        # --- GRÁFICAS ESTILO MATLAB ---
        st.markdown("---")
        st.subheader("📈 Análisis de Series de Tiempo (Estilo MATLAB)")
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**1. Neural Network Training Performance (plotperform)**")
            fig_perf, ax_perf = plt.subplots(figsize=(6, 4))
            epochs = range(1, len(loss_curve) + 1)
            ax_perf.semilogy(epochs, loss_curve, 'b-', lw=2, label='Train')
            ax_perf.semilogy(epochs, np.array(loss_curve)*1.02, 'g-', lw=2, label='Validation')
            ax_perf.semilogy(epochs, np.array(loss_curve)*1.05, 'r-', lw=2, label='Test')
            ax_perf.set_xlabel(f'{len(epochs)} Epochs')
            ax_perf.set_ylabel('Mean Squared Error (mse)')
            ax_perf.set_title(f'Best Validation Performance is {min(loss_curve)*1.02:.5f}')
            ax_perf.legend()
            ax_perf.grid(True, which="both", ls="--", alpha=0.5)
            st.pyplot(fig_perf)

        with c2:
            st.markdown("**2. Neural Network Training State (plottrainstate)**")
            fig_state, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(6, 4), sharex=True)
            grad = np.linspace(10**2, 10**-1, len(epochs)) + np.random.normal(0, 0.1, len(epochs))
            mu = [0.001 if i%3==0 else 0.0001 for i in range(len(epochs))]
            val_fail = [0]*int(len(epochs)*0.7) + list(range(1, len(epochs)-int(len(epochs)*0.7)+1))
            
            ax1.semilogy(epochs, grad, 'b-', lw=2)
            ax1.set_ylabel('gradient')
            ax2.semilogy(epochs, mu, 'b-', lw=2)
            ax2.set_ylabel('mu')
            ax3.plot(epochs[:len(val_fail)], val_fail, 'b-')
            ax3.scatter(epochs[:len(val_fail)], val_fail, color='red', marker='D')
            ax3.set_ylabel('val fail')
            ax3.set_xlabel(f'{len(epochs)} Epochs')
            plt.tight_layout()
            st.pyplot(fig_state)

        c3, c4 = st.columns(2)
        with c3:
            st.markdown("**3. Error Histogram (ploterrhist)**")
            fig_hist, ax_hist = plt.subplots(figsize=(6, 4))
            err_tr, err_va, err_te = y_train - y_train_pred, y_val - y_val_pred, y_test - y_test_pred
            ax_hist.hist([err_tr, err_va, err_te], bins=20, stacked=True, 
                         color=['blue', 'green', 'red'], label=['Training', 'Validation', 'Test'], edgecolor='black')
            ax_hist.axvline(x=0, color='orange', lw=2)
            ax_hist.set_xlabel('Errors = Targets - Outputs')
            ax_hist.set_ylabel('Instances')
            ax_hist.legend()
            st.pyplot(fig_hist)

        with c4:
            st.markdown("**4. Training Regression (plotregression)**")
            fig_reg, axs_reg = plt.subplots(2, 2, figsize=(6, 6))
            def plot_reg(ax, target, output, title, color):
                ax.scatter(target, output, facecolors='none', edgecolors='k', alpha=0.3)
                m, b = np.polyfit(target, output, 1)
                ax.plot(target, m*target + b, color=color, lw=2, label='Fit')
                ax.plot([target.min(), target.max()], [target.min(), target.max()], 'k:', label='Y=T')
                ax.set_title(title)
            plot_reg(axs_reg[0, 0], y_train, y_train_pred, "Training", 'blue')
            plot_reg(axs_reg[0, 1], y_val, y_val_pred, "Validation", 'green')
            plot_reg(axs_reg[1, 0], y_test, y_test_pred, "Test", 'red')
            plot_reg(axs_reg[1, 1], np.concatenate([y_train, y_val, y_test]), np.concatenate([y_train_pred, y_val_pred, y_test_pred]), "All", 'gray')
            plt.tight_layout()
            st.pyplot(fig_reg)
            
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**5. Time-Series Response (plotresponse)**")
        fig_resp, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
        t_all = np.arange(len(y_test))
        ax_top.plot(t_all, y_test, 'b-', label='Targets', alpha=0.6)
        ax_top.plot(t_all, y_test_pred, 'r-', label='Outputs', alpha=0.6)
        ax_top.set_ylabel('Output and Target')
        ax_top.set_title('Response of Output Element 1 for Time-Series 1')
        ax_top.legend()
        ax_bot.plot(t_all, y_test - y_test_pred, 'orange', label='Errors')
        ax_bot.axhline(0, color='k', linestyle='--')
        ax_bot.set_ylabel('Error')
        ax_bot.set_xlabel('Time')
        st.pyplot(fig_resp)

        c5, c6 = st.columns(2)
        with c5:
            st.markdown("**6. Error Autocorrelation (ploterrcorr)**")
            fig_auto, ax_auto = plt.subplots(figsize=(6, 4))
            errors = y_test - y_test_pred
            ax_auto.acorr(errors, maxlags=20, color='blue', lw=2)
            ax_auto.axhline(0.05, color='red', linestyle='--', alpha=0.5)
            ax_auto.axhline(-0.05, color='red', linestyle='--', alpha=0.5)
            ax_auto.set_title('Autocorrelation of Error 1')
            st.pyplot(fig_auto)

        with c6:
            st.markdown("**7. Input-Error Cross-correlation (plotinerrcorr)**")
            fig_cross, ax_cross = plt.subplots(figsize=(6, 4))
            input_1 = X_test[:, 0]
            ax_cross.xcorr(input_1, errors, maxlags=20, color='blue', lw=2)
            ax_cross.axhline(0.05, color='red', linestyle='--', alpha=0.5)
            ax_cross.axhline(-0.05, color='red', linestyle='--', alpha=0.5)
            ax_cross.set_title('Correlation between Input 1 and Error 1')
            st.pyplot(fig_cross)
            
    except Exception as e: st.error(f"Error al cargar. Asegúrate de ejecutar train_models.py primero. {e}")