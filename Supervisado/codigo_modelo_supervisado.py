
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("dataset_supervisado.csv")

# Codificación
for col in ['origin_station', 'destination_station', 'transport_type', 'satisfaccion']:
    df[col] = LabelEncoder().fit_transform(df[col])

X = df[['hour', 'origin_station', 'destination_station', 'travel_time_min', 'transport_type']]
y = df['satisfaccion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
