from utils import db_connect
engine = db_connect()

from flask import Flask, render_template, request
import pandas as pd
import pickle
import os

app = Flask(__name__)

# 1. Cargar el modelo
# IMPORTANTE: Asegúrate de guardar tu modelo XGBoost óptimo con este nombre en tu notebook
# y poner el archivo en la carpeta 'models/' de tu proyecto.
ruta_modelo = os.path.join(os.path.dirname(__file__), '../models/modelo_xgb_diabetes.pkl')

try:
    with open(ruta_modelo, 'rb') as f:
        modelo = pickle.load(f)
except FileNotFoundError:
    modelo = None
    print(f"Error: No se encontró el modelo en la ruta {ruta_modelo}.")

# Ruta principal: muestra el formulario HTML
@app.route('/', methods=['GET'])
def home():
    return render_template('index.html', resultado_prediccion=None)

# Ruta de predicción: procesa los datos y devuelve el resultado
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST' and modelo is not None:
        try:
            # Capturar los datos enviados por el usuario desde el formulario HTML
            embarazos = float(request.form['Pregnancies'])
            glucosa = float(request.form['Glucose'])
            presion = float(request.form['BloodPressure'])
            piel = float(request.form['SkinThickness'])
            insulina = float(request.form['Insulin'])
            bmi = float(request.form['BMI'])
            pedigri = float(request.form['DiabetesPedigreeFunction'])
            edad = float(request.form['Age'])
            
            # Crear el DataFrame para hacer la predicción
            # Las columnas deben llamarse EXACTAMENTE igual que cuando entrenaste el modelo
            datos_entrada = pd.DataFrame([[
                embarazos, glucosa, presion, piel, insulina, bmi, pedigri, edad
            ]], columns=[
                'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
            ])
            
            # Realizar la predicción
            prediccion = modelo.predict(datos_entrada)[0]
            
            # Interpretar el resultado (0 = Negativo, 1 = Positivo)
            if prediccion == 1:
                resultado_texto = "ALTO RIESGO de Diabetes"
                color = "red"
            else:
                resultado_texto = "BAJO RIESGO de Diabetes"
                color = "green"
                
            return render_template('index.html', resultado_prediccion=resultado_texto, color_alerta=color)
            
        except Exception as e:
            # En caso de que el usuario ponga letras en lugar de números o deje vacío
            error_msj = f"Hubo un error al procesar los datos: {str(e)}"
            return render_template('index.html', resultado_prediccion=error_msj, color_alerta="orange")
            
    return render_template('index.html', resultado_prediccion="Modelo no disponible", color_alerta="orange")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
