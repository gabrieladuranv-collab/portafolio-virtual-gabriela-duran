# =============================================================================
# Desafío - Explorando técnicas de aprendizaje no supervisado
# Autora: Gabriela Durán
# Desafío Latam
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Dataset simulado
data = {
    'Edad': [25, 30, 22, 45, 38, 26, 31, 50],
    'Cursos_aprobados': [3, 4, 2, 6, 5, 3, 4, 7],
    'Interes_formativo': ['TI', 'Administración', 'Salud', 'TI', 'Administración', 'Salud', 'TI', 'Salud'],
    'Empleabilidad': [0.8, 0.6, 0.7, 0.9, 0.5, 0.6, 0.7, 0.4]
}

df = pd.DataFrame(data)
print("Vista previa de los datos:")
print(df)

num_cols = ['Edad', 'Cursos_aprobados', 'Empleabilidad']
cat_cols = ['Interes_formativo']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(), cat_cols)
])

X_processed = preprocessor.fit_transform(df)

silhouette_scores = {}
for k in range(2, 5):
    kmeans_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels_tmp = kmeans_tmp.fit_predict(X_processed)
    score = silhouette_score(X_processed, labels_tmp)
    silhouette_scores[k] = score
    print(f"k = {k} -> Silhouette Score = {score:.4f}")

best_k = max(silhouette_scores, key=silhouette_scores.get)
print(f"\nMejor valor de k según Silhouette Score: {best_k}")

kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_processed)
df['Cluster_KMeans'] = labels_kmeans

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)
varianza_explicada = pca.explained_variance_ratio_
print(f"Varianza total explicada por las 2 componentes: {varianza_explicada.sum():.2%}")

plt.figure(figsize=(8, 5))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=labels_kmeans, palette='tab10', s=120)
plt.title(f"Clusters de K-means (k={best_k}) visualizados con PCA")
plt.xlabel(f"Componente Principal 1 ({varianza_explicada[0]:.1%} var.)")
plt.ylabel(f"Componente Principal 2 ({varianza_explicada[1]:.1%} var.)")
plt.legend(title="Cluster")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('clusters_pca.png', dpi=150)
plt.show()

print("\nSegmentación resultante:")
print(df)
print("\nPerfil promedio por cluster:")
print(df.groupby('Cluster_KMeans')[num_cols].mean())
print("\nDistribución del área de interés por cluster:")
print(pd.crosstab(df['Cluster_KMeans'], df['Interes_formativo']))
