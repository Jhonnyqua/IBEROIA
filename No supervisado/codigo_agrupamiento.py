
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Cargar dataset
df = pd.read_csv("dataset_transporte.csv")

# Preparación de datos
df_cluster = df.copy()
label_encoders = {}
categorical_cols = ['origin_station', 'destination_station', 'transport_type']
for col in categorical_cols:
    le = LabelEncoder()
    df_cluster[col] = le.fit_transform(df_cluster[col])
    label_encoders[col] = le

# Variables para clustering
features = ['hour', 'origin_station', 'destination_station', 'travel_time_min', 'transport_type']
X = df_cluster[features]

# Modelo KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
df_cluster['cluster'] = kmeans.fit_predict(X)

# Guardar resultado
df_cluster.to_csv("datos_con_clusters.csv", index=False)
