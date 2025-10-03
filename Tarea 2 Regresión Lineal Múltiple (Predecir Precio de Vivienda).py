#Tarea 2: Regresión Lineal Múltiple (Predecir Precio de Vivienda)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Datos proporcionados
datos = {'Superficie_m2': [50, 70, 65, 90, 45],
         'Num_Habitaciones': [1, 2, 2, 3, 1],
         'Distancia_Metro_km': [0.5, 1.2, 0.8, 0.2, 2.0],
         'Precio_UF': [2500, 3800, 3500, 5200, 2100]}

df = pd.DataFrame(datos)

# Variables predictoras (X) y variable objetivo (y)
X = df[['Superficie_m2', 'Num_Habitaciones', 'Distancia_Metro_km']]
y = df['Precio_UF']

# Dividir los datos en conjuntos de entrenamiento y prueba (aunque con solo 5 datos, esto es más ilustrativo que práctico)
# Usaremos un tamaño de prueba pequeño solo para la demostración
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Datos de Entrenamiento:")
print(X_train)
print("\nDatos de Prueba:")
print(X_test)

# Inicializar el modelo de Regresión Lineal
modelo = LinearRegression()

# Entrenar el modelo
modelo.fit(X_train, y_train)

# Obtener los coeficientes y el intercepto (ordenada al origen)
intercepto = modelo.intercept_
coeficientes = dict(zip(X.columns, modelo.coef_))

print(f"\nIntercepto (b0): {intercepto:.1f}")
print("Coeficientes (b1, b2, b3):")
for feature, coef in coeficientes.items():
    print(f"  {feature}: {coef:.1f}")

# Realizar predicciones en el conjunto de prueba
y_pred = modelo.predict(X_test)

print("\nPredicciones vs. Valores Reales (Conjunto de Prueba):")
resultados = pd.DataFrame({'Real': y_test.values, 'Predicción': y_pred.round(1)})
print(resultados)

# Calcular métricas de rendimiento
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n--- Métricas de Rendimiento ---")
print(f"Error Cuadrático Medio (MSE): {mse:.1f} (UF^2)")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.1f} (UF)")
print(f"Coeficiente de Determinación (R^2): {r2:.1f}")