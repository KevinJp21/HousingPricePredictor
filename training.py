import pandas as pd
import numpy as np
import tensorflow as tf
import joblib  # Para guardar los escaladores
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

# Cargar el dataset
df = pd.read_csv("./dataset/housing.csv")

# Imprimir nombres de columnas para depuración
print("Columnas del dataset:", df.columns)

# Convertir la variable categórica 'ocean_proximity' a valores numéricos
encoder = OneHotEncoder(sparse_output=False)
ocean_encoded = encoder.fit_transform(df[['ocean_proximity']])
ocean_df = pd.DataFrame(ocean_encoded, columns=encoder.get_feature_names_out(['ocean_proximity']))

# Agregar las columnas codificadas al dataset
df = df.drop(columns=['ocean_proximity'])  # Eliminar la columna original
df = pd.concat([df, ocean_df], axis=1)  # Añadir la versión codificada

# Manejo de valores nulos en 'total_bedrooms'
df['total_bedrooms'].fillna(df['total_bedrooms'].median(), inplace=True)

# Separar variables de entrada (X) y salida (y)
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']

# Normalizar las variables de entrada
scaler_X = MinMaxScaler()
X = scaler_X.fit_transform(X)

scaler_y = MinMaxScaler()
y = scaler_y.fit_transform(y.values.reshape(-1, 1)).flatten()  # Normalizar precios entre 0 y 1

# Guardar los escaladores para futuras predicciones
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")

# Redimensionar los datos para CNN 1D (samples, timesteps, features)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir el modelo CNN 1D
model = Sequential([
    Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    MaxPooling1D(pool_size=2),
    Conv1D(filters=64, kernel_size=3, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1)  # Salida continua (predicción del precio)
])

# Compilar el modelo
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae']) ## mae = Mean Absolute Error

# Entrenar el modelo
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test))

# Evaluar el modelo
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"\nError Absoluto Medio en prueba: {test_mae:.4f}")

# Guardar el modelo para futuras predicciones
model.save("modelo_cnn.keras")
joblib.dump(scaler_X, "scaler_X.pkl")
joblib.dump(scaler_y, "scaler_y.pkl")
print("\nModelo guardado como 'modelo_cnn.keras'")