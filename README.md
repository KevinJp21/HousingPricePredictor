# Predicción de Precios de Vivienda en California

Este proyecto utiliza una red neuronal convolucional (CNN) para predecir el precio medio de las viviendas en California, disponible en [Kaggle](https://www.kaggle.com/code/shtrausslearning/bayesian-regression-house-price-prediction/input?select=housing.csv).


## Dataset
El dataset utilizado proviene de `housing.csv`, el cual contiene información sobre distintos bloques de viviendas en California. Cada fila representa una zona geográfica con múltiples propiedades y sus características asociadas.

### Columnas del Dataset
El dataset contiene las siguientes columnas:

- **longitude**: Longitud geográfica de la zona.
- **latitude**: Latitud geográfica de la zona.
- **housingMedianAge**: Edad media de las viviendas en la zona.
- **totalRooms**: Número total de habitaciones en la zona.
- **totalBedrooms**: Número total de dormitorios en la zona (se llenan valores faltantes con la mediana).
- **population**: Cantidad de personas que habitan en la zona.
- **households**: Número total de hogares en la zona.
- **medianIncome**: Ingreso medio de los residentes en la zona (en decenas de miles de dólares).
- **ocean_proximity**: Tipo de proximidad al océano, una variable categórica con las siguientes opciones:
  - `<1H OCEAN`: A menos de una hora del océano.
  - `INLAND`: Ubicación en el interior, lejos del océano.
  - `ISLAND`: En una isla.
  - `NEAR BAY`: Cerca de la bahía.
  - `NEAR OCEAN`: Cerca del océano.
- **median_house_value**: Precio medio de las viviendas en la zona (variable objetivo a predecir).

## Preprocesamiento de Datos
Dado que los modelos de machine learning requieren datos numéricos, la columna **ocean_proximity** se convierte en variables binarias mediante **One-Hot Encoding**. Esto crea nuevas columnas donde:
- Se asigna **1** si el bloque de viviendas pertenece a esa categoría.
- Se asigna **0** en caso contrario.

Ejemplo de transformación:

| ocean_proximity | oceanProximity_<1H OCEAN | oceanProximity_INLAND | oceanProximity_ISLAND | oceanProximity_NEAR BAY | oceanProximity_NEAR OCEAN |
|-----------------|-------------------------|-----------------------|------------------------|-------------------------|--------------------------|
| `<1H OCEAN`     | 1                         | 0                     | 0                      | 0                        | 0                        |
| `INLAND`        | 0                         | 1                     | 0                      | 0                        | 0                        |
| `NEAR OCEAN`    | 0                         | 0                     | 0                      | 0                        | 1                        |

## Objetivo del Proyecto
El objetivo principal de este proyecto es entrenar un modelo de aprendizaje profundo que pueda predecir con precisión el valor medio de las viviendas en función de sus características geográficas y estructurales.

## Tecnologías Utilizadas
- Python
- TensorFlow/Keras
- Pandas y NumPy
- Scikit-learn
- Joblib

## Ejecución del Proyecto
Para entrenar el modelo, ejecutar:
```bash
py training.py
```

Para realizar predicciones interactivas, ejecutar:
```bash
py app.py
```