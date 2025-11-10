"""
PruebaSingleTest.py
~~~~~~~~~~~~~~~~~~~

Script para cargar el modelo entrenado y probarlo con un único dato de test.
Muestra la entrada (imagen MNIST), la predicción de la red y el valor esperado.
"""

import os
import pickle
import numpy as np
import mnist_loader
import network

# Cargar el modelo entrenado
here = os.path.dirname(__file__)
trained_net_path = os.path.normpath(os.path.join(here, '..', 'data', 'trained_net.pkl'))

print(f"Cargando modelo desde: {trained_net_path}")
with open(trained_net_path, 'rb') as f:
    weights, biases = pickle.load(f)

# Crear una red con la misma arquitectura y cargar los parámetros
net = network.Network([784, 10, 10])
net.weights = weights
net.biases = biases
print("Modelo cargado exitosamente.\n")

# Cargar datos de test
_, _, test_data = mnist_loader.load_data_wrapper()
test_data = list(test_data)

# Seleccionar un índice (0-9999)
test_index = 6630  # Cambia este valor para probar otros ejemplos
x, y = test_data[test_index]

# Hacer predicción
output = net.feedforward(x)
prediction = np.argmax(output)  # El índice del neurón con mayor activación
confidence = output[prediction][0]  # Confianza de la predicción

print(f"Prueba del ejemplo #{test_index}")
print(f"{'='*50}")
print(f"Dígito esperado (y):        {y}")
print(f"Predicción de la red:       {prediction}")
print(f"Confianza de predicción:    {confidence:.4f}")
print(f"Predicción correcta:        {'✓ SÍ' if prediction == y else '✗ NO'}")
print(f"{'='*50}")

# Mostrar las activaciones de la capa de salida
print(f"\nActivaciones de la capa de salida (10 neuronas):")
for i in range(10):
    bar_length = int(output[i][0] * 30)
    bar = '█' * bar_length
    marker = " <--" if i == prediction else ""
    print(f"Dígito {i}: {bar} {output[i][0]:.4f}{marker}")

# Información adicional
print(f"\nInformación de la entrada:")
print(f"Forma de x (imagen):        {x.shape}")
print(f"Valor mínimo:               {x.min():.4f}")
print(f"Valor máximo:               {x.max():.4f}")
print(f"Promedio de píxeles:        {x.mean():.4f}")
