
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from boruta import BorutaPy
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, roc_curve, confusion_matrix, auc, f1_score
import time

# Lectura del archivo aborto.xlsx
df = pd.read_excel(r'C:\Users\pepej\Desktop\Proyecto_Minería\Mineria_De_Datos\aborto.xlsx')

# Inspección inicial
print("Columnas disponibles en el archivo:")
print(df.columns)

# Eliminar columnas irrelevantes (como las que empiezan con 'Unnamed')
df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

# Confirmar estructura tras limpieza
print("Columnas después de la limpieza:")
print(df.columns)

# Actualiza el nombre de la columna objetivo si es diferente
target_column = 'NombreDeLaColumnaObjetivo'  # Sustituir por el nombre correcto
if target_column not in df.columns:
    raise ValueError(f"La columna objetivo '{target_column}' no se encuentra en el archivo.")

# Preparar datos (manejando valores nulos y categóricos)
df = df.dropna()  # Se eliminan valores nulos para simplificar
labelEncoder = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = labelEncoder.fit_transform(df[col])

# Separar características y variable objetivo
X = df.drop(target_column, axis=1)
y = df[target_column]

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=101)

# Usar Boruta para seleccionar características importantes
rf = RandomForestClassifier(n_estimators=100, random_state=42)
boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42, verbose=2)
boruta_selector.fit(X_train.values, y_train.values)

# Imprimir resultados de Boruta
important_features = X.columns[boruta_selector.support_]
print("\nCaracterísticas importantes según Boruta:")
print(important_features)

# Filtrar X_train y X_test con las características seleccionadas
X_train_selected = X_train[important_features]
X_test_selected = X_test[important_features]

# Definir modelos para comparación
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'DecisionTree': RandomForestClassifier(random_state=42),
}

# Entrenamiento y evaluación de modelos
for name, model in models.items():
    start_time = time.time()
    model.fit(X_train_selected, y_train)
    end_time = time.time()
    tiempo = end_time - start_time
    predictions = model.predict(X_test_selected)

    acc = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    conf_matrix = confusion_matrix(y_test, predictions)
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    print(f'\nModelo: {name}')
    print(f'Accuracy: {acc:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | AUC: {roc_auc:.4f}')
    print('Matriz de Confusión:')
    print(conf_matrix)

    # Graficar curva ROC
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

# Gráfica comparativa de curvas ROC
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('Falsos Positivos')
plt.ylabel('Verdaderos Positivos')
plt.title('Curva ROC Comparativa')
plt.legend()
plt.show()
