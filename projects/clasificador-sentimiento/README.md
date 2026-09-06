# Clasificador de sentimiento

## Objetivo

Construir un modelo de Deep Learning con Keras para clasificar comentarios de clientes como positivos o negativos.

## Flujo de trabajo

- Carga de datos desde Excel.
- División estratificada en entrenamiento y prueba.
- Tokenización ajustada solo con el conjunto de entrenamiento.
- Padding de secuencias.
- Modelo con Embedding, GlobalAveragePooling, Dense y Dropout.
- Evaluación mediante accuracy y loss.
- Prueba con comentarios nuevos.

## Dependencia externa

Este proyecto requiere el archivo `comentarios_clientes.xlsx` con las columnas `comentario` y `sentimiento`. Ese dataset no está incluido en este portafolio, por lo que el código queda documentado pero no se presenta un resultado de ejecución inventado.

## Ejecución

```bash
python clasificador_sentimiento.py
```
