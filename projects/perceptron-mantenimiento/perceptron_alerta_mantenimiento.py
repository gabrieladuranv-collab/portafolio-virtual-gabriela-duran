"""
Perceptrón simple - Alerta de mantenimiento predictivo
Desafío: Anatomía y funcionamiento de una red neuronal artificial
"""

import numpy as np

np.random.seed(42)


class Perceptron:
    def __init__(self, n_entradas, tasa_aprendizaje=0.1, epocas=20):
        self.pesos = np.random.uniform(-0.5, 0.5, n_entradas)
        self.sesgo = np.random.uniform(-0.5, 0.5)
        self.tasa_aprendizaje = tasa_aprendizaje
        self.epocas = epocas

    def funcion_activacion(self, z):
        return 1 if z >= 0 else 0

    def predecir(self, entradas):
        z = np.dot(entradas, self.pesos) + self.sesgo
        return self.funcion_activacion(z)

    def entrenar(self, X, y):
        for epoca in range(self.epocas):
            errores_epoca = 0
            for entradas, salida_esperada in zip(X, y):
                prediccion = self.predecir(entradas)
                error = salida_esperada - prediccion
                if error != 0:
                    errores_epoca += 1
                self.pesos += self.tasa_aprendizaje * error * entradas
                self.sesgo += self.tasa_aprendizaje * error
            print(f"Época {epoca + 1:2d} | errores: {errores_epoca}")
            if errores_epoca == 0:
                print("Convergencia alcanzada.")
                break


if __name__ == "__main__":
    X = np.array([
        [0.1, 0.2],
        [0.2, 0.1],
        [0.15, 0.25],
        [0.9, 0.8],
        [0.8, 0.9],
        [0.85, 0.75],
    ])
    y = np.array([0, 0, 0, 1, 1, 1])

    modelo = Perceptron(n_entradas=2, tasa_aprendizaje=0.1, epocas=20)
    modelo.entrenar(X, y)

    nuevo_vehiculo = np.array([0.78, 0.82])
    resultado = modelo.predecir(nuevo_vehiculo)
    etiqueta = "ALERTA DE MANTENIMIENTO" if resultado == 1 else "Operación normal"
    print(f"Resultado vehículo nuevo: {etiqueta}")
