# Proyecto de Red Neuronal con ResNet

## Descripción
Este proyecto implementa una red neuronal convolucional utilizando una capa **ResNet** para procesar imágenes a color de 150x150 píxeles. Además, se incluye un análisis de un conjunto de datos en formato CSV con 7,951 registros para predecir la edad de las personas a partir de las imágenes, utilizando técnicas de machine learning.

## Estructura del Proyecto
El proyecto se compone de las siguientes partes principales:

1. **Preprocesamiento de Datos**
   - Se cargan 16 imágenes con dimensiones de 150x150 píxeles y tres canales de color (RGB).
   - Análisis de un dataset en CSV con registros de edades, observando una distribución sesgada a la derecha.

2. **Implementación del Modelo**
   - La red neuronal se construyó con una capa **ResNet**, seguida de **Average Pooling 2D**, y una capa densa de una neurona con la función de activación `ReLU`.
   - Funciones clave para dividir los datasets, entrenar el modelo y obtener los resultados.
   - El modelo fue compilado usando:
     - **Loss:** `Mean Squared Error (MSE)`
     - **Métrica:** `Mean Absolute Error (MAE)`
     - **Optimizador:** Adam con una tasa de aprendizaje de 0.0001.

3. **Entrenamiento**
   - Se entrenó el modelo durante 3 épocas con un tiempo total de 21 minutos.
   - Se observaron mejoras significativas en las métricas de evaluación a lo largo de las épocas:
     - **Primera época:** MSE de 734.94 y MAE de 22.09 años.
     - **Segunda época:** MSE de 173.29 y MAE de 9.53 años.
     - **Tercera época:** MSE de 98 y MAE de 6.88 años.

## Resultados
El modelo logró una mejora progresiva durante el entrenamiento, alcanzando un **MAE de 6.88 años** en la última época, superando el objetivo de un MAE inferior a 8. Esto indica un buen ajuste del modelo con capacidad de generalización.

## Requisitos
Para ejecutar el proyecto, asegúrate de tener las siguientes dependencias instaladas:

```bash
pip install tensorflow numpy pandas matplotlib
