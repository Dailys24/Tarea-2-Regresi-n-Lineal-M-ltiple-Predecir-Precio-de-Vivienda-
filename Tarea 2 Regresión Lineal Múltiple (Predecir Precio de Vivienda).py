#Tarea: Predecir Precio de Departamento en UF (Regresión Lineal Simple)

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#Datos iniciales para el modelo
datos_deptos = {
    'Superficie_m2': [50, 70, 65, 90, 45],
    'Num_Habitaciones': [1, 2, 2, 3, 1],
    'Distancia_Metro_km': [0.5, 1.2, 0.8, 0.2, 2.0],
    'Precio_UF': [2500, 3800, 3500, 5200, 2100]
}

df_propiedades = pd.DataFrame(datos_deptos)

#Variables (Características X y Objetivo Y)
X = df_propiedades[['Superficie_m2', 'Num_Habitaciones', 'Distancia_Metro_km']]
Y = df_propiedades['Precio_UF']
print("--- Data completa para el modelo (N=5) ---")
print(X)

#Entrenamiento del modelo
modelo_lineal = LinearRegression()
modelo_lineal.fit(X, Y)

#Resultados del modelo
b0_intercepto = modelo_lineal.intercept_
b_coefs = dict(zip(X.columns, modelo_lineal.coef_))

print("\n--- Ecuación de predicción (UF) ---")
print(f"Punto de partida (intercepto b0): {b0_intercepto:.1f} UF")
print("Impacto de cada característica (coeficientes):")
for feature, coef in b_coefs.items():
    print(f"{feature}: {coef:.1f} UF por unidad")

#Evaluación del modelo en el set completo
predicciones = modelo_lineal.predict(X)

print("\n--- Predicción vs realidad set completo) ---")
resultado_comparado = pd.DataFrame({
    'Precio real (UF)': Y.values,
    'Precio estimado (UF)': predicciones.round(1)
})
print(resultado_comparado)

#Cálculo de métricas clave
mse = mean_squared_error(Y, predicciones)
rmse = np.sqrt(mse)
r2 = r2_score(Y, predicciones)

print("\n--- Métrica clave de desempeño (evaluación en set de entrenamiento) ---")
print(f"Raíz del error cuadrático medio (RMSE): {rmse:.1f} UF")
print(f"En promedio, el modelo se equivoca en {rmse:.1f} UF en la predicción.")

print(f"Coeficiente de determinación (R^2): {r2:.1f} (idealmente cerca de 1.0)")
