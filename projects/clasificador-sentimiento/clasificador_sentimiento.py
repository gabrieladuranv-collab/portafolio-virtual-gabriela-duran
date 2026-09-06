"""
Clasificador de sentimiento de comentarios de clientes - Deep Learning con Keras
Desafío: Aplicaciones reales del Deep Learning

Dataset esperado: comentarios_clientes.xlsx
Columnas: comentario, sentimiento [0=negativo, 1=positivo]
"""

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, GlobalAveragePooling1D, Dense, Dropout
from sklearn.model_selection import train_test_split

# 1. Carga de datos
df = pd.read_excel("comentarios_clientes.xlsx")
comentarios = df["comentario"].astype(str).values
etiquetas = df["sentimiento"].values

# 2. Split entrenamiento / prueba
X_train_text, X_test_text, y_train, y_test = train_test_split(
    comentarios,
    etiquetas,
    test_size=0.2,
    random_state=42,
    stratify=etiquetas,
)

# 3. Preprocesamiento
VOCAB_SIZE = 1000
MAX_LEN = 12
OOV_TOKEN = "<OOV>"

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)
tokenizer.fit_on_texts(X_train_text)

X_train_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_seq = tokenizer.texts_to_sequences(X_test_text)
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

# 4. Modelo
EMBEDDING_DIM = 16
modelo = Sequential([
    Input(shape=(MAX_LEN,)),
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dropout(0.3),
    Dense(1, activation="sigmoid"),
])

modelo.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 5. Entrenamiento
historial = modelo.fit(
    X_train_pad,
    y_train,
    epochs=15,
    batch_size=16,
    validation_split=0.2,
    verbose=2,
)

# 6. Evaluación
loss, accuracy = modelo.evaluate(X_test_pad, y_test, verbose=0)
print(f"Exactitud en prueba: {accuracy:.4f}")
print(f"Pérdida en prueba: {loss:.4f}")

# 7. Predicción de ejemplos nuevos
comentarios_nuevos = [
    "El producto llegó roto y nadie responde mis mensajes",
    "Quedé muy conforme con la rapidez de la entrega",
]
seq_nuevos = tokenizer.texts_to_sequences(comentarios_nuevos)
pad_nuevos = pad_sequences(seq_nuevos, maxlen=MAX_LEN, padding="post", truncating="post")
predicciones = modelo.predict(pad_nuevos, verbose=0)

for texto, pred in zip(comentarios_nuevos, predicciones):
    etiqueta = "POSITIVO" if pred[0] >= 0.5 else "NEGATIVO"
    print(f"{texto} -> {etiqueta}")
