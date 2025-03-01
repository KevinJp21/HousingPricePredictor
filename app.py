import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib

# Cargar el modelo guardado en formato Keras
model = load_model("modelo_cnn.keras")

# Cargar los escaladores usados en el entrenamiento
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Obtener características desde el usuario
def obtener_datos_usuario():
    columnas = [
        "longitude", "latitude", "housingMedianAge", "totalRooms", "totalBedrooms", "population",
        "households", "medianIncome", "oceanProximity_<1H OCEAN", "oceanProximity_INLAND",
        "oceanProximity_ISLAND", "oceanProximity_NEAR BAY", "oceanProximity_NEAR OCEAN"
    ]
    
    print("\n Ingrese los datos para la predicción:")
    valores = []
    
    for col in columnas[:-5]:  # Datos numéricos
        val = float(input(f"{col}: "))
        valores.append(val)

    print("\n Seleccione la categoría 'oceanProximity':")
    opciones = ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"]
    for i, op in enumerate(opciones, 1):
        print(f"{i}. {op}")

    opcion = int(input("\nIngrese el número de la categoría: ")) - 1
    one_hot = [1 if i == opcion else 0 for i in range(5)]
    valores.extend(one_hot)

    return np.array(valores).reshape(1, -1)

# Normalizar los datos y hacer predicción
def predecir_precio():
    datos_usuario = obtener_datos_usuario()
    datos_usuario = scaler_X.transform(datos_usuario).reshape(1, datos_usuario.shape[1], 1)  # Normalizar y redimensionar
    
    prediccion = model.predict(datos_usuario)
    precio_predicho = scaler_y.inverse_transform(prediccion.reshape(-1, 1))  # Volver a escala original
    
    print(f"\n Precio estimado de la vivienda: ${precio_predicho[0][0]:,.2f}")

if __name__ == "__main__":
    predecir_precio()
