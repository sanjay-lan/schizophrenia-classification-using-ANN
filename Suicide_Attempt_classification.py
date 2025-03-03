
!pip install tensorflow

import tensorflow as tf
print("Tensorflow version " + tf.__version__)

# import some basic libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('/content/drive/MyDrive/schizophrenia classification/schizophrenia_dataset.csv')
dataset.drop(columns=['Disease_Duration'], inplace=True)
dataset.head()

# to convert all values to 0 or 1 (binary) 

dataset.Medication_Adherence = (dataset.Medication_Adherence == 0).astype(int)
dataset.Education_Level = (dataset.Education_Level < 3).astype(int)
dataset.Stress_Factors = (dataset.Stress_Factors == 2).astype(int)
dataset.Negative_Symptom_Score = (dataset.Negative_Symptom_Score >= 50).astype(int)
dataset.Positive_Symptom_Score = (dataset.Positive_Symptom_Score >= 50).astype(int)
dataset.GAF_Score = (dataset.GAF_Score <= 30).astype(int)
# dataset.Disease_Duration = (dataset.Disease_Duration > 20).astype(int)
dataset.Hospitalization = (dataset.Hospitalization >= 5).astype(int)
dataset.Marital_Status = (dataset.Marital_Status >= 2).astype(int)
dataset.Income_Level = (dataset.Income_Level < 1).astype(int)
dataset.Age = (dataset.Age <= 30).astype(int)
dataset.Social_Support = (dataset.Social_Support == 0).astype(int)
dataset.Occupation = (dataset.Occupation <=0).astype(int)


# To balance Suicide_Attempt column
df_0 = dataset[dataset['Suicide_Attempt']==0]
df_1= dataset[dataset['Suicide_Attempt']!=0]
df_0_down = df_0.sample(n=len(df_1),random_state=42)
df_balanced = pd.concat([df_0_down, df_1])
dataset = df_balanced.sample(frac=1).reset_index(drop=True)


# For Suicide_Attempt classification divide the dataset into independent and dependent features
X = dataset.drop(columns=['Patient_ID', 'Age', 'Gender','Education_Level', 'Suicide_Attempt', 'Hospitalization', 'Social_Support','Suicide_Attempt', 'Medication_Adherence', ])
y = dataset['Suicide_Attempt']

# all the column names are here
# 'Age', 'Gender', 'Education_Level', 'Marital_Status', 'Income_Level', 'Occupation', 'Live_Area', 'Family_History', 'Substance_Use', 'Positive_Symptom_Score',	'Negative_Symptom_Score',	'GAF_Score', 'Suicide_Attempt','Social_Support', 'Stress_Factors', 'Medication_Adherence', 'Disease_Duration', 'Hospitalization', 'Diagnosis'

X.head()

# splitting the dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train.shape

# ANN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import ReLU, LeakyReLU, PReLU, ELU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

#  ANN initialization
classifier = Sequential([
    Dense(11, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(5, activation='relu'),
    Dropout(0.2),
    Dense(2, activation='relu'),
    Dense(1, activation='sigmoid')

])


# compilation
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# early stopping for validation loss
import tensorflow as tf
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    min_delta=0.01,
    patience=10,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)

# training
model_history = classifier.fit(X_train, y_train, validation_split=0.33, batch_size=10, epochs=100, callbacks=early_stopping)


# visualization section
# roc curve generation
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score

# model prediction probability between 0 and 1
y_probs = classifier.predict(X_test).ravel()

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_test, y_probs)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='grey', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# converting the probability to prediction 0 or 1
y_probs = (y_probs >= 0.5).astype(int)

df = pd.DataFrame({'Actual': y_test[:100], 'Predicted': y_probs[:100]})

# actual values and model predictions save to CSV file
df.to_csv('predictions.csv', index=False)

#  visualization
#tpr fpr is calculated to plot them in a bar graph
fpr, tpr, thresholds = roc_curve(y_test, y_probs)

# Selecting a subset of points to show in the graph
indices = np.linspace(0, len(fpr) - 1, num=3, dtype=int)  
selected_fpr = fpr[indices]
selected_tpr = tpr[indices]
selected_thresholds = thresholds[indices]

# Bar plot for TPR & FPR
x_labels = [f'{t:.2f}' for t in selected_thresholds]  # Convert thresholds to labels
x = np.arange(len(x_labels))  # Bar positions

plt.figure(figsize=(10, 5))
plt.bar(x - 0.2, selected_tpr, 0.4, label="True Positive Rate (TPR)", color='blue')
plt.bar(x + 0.2, selected_fpr, 0.4, label="False Positive Rate (FPR)", color='red')
plt.xticks(x, x_labels, rotation=45)
plt.xlabel("Thresholds")
plt.ylabel("Rate")
plt.title("TPR and FPR at Different Thresholds")
plt.legend()
plt.show()

# computing confusion matix and plot them in bar graph
from sklearn.metrics import confusion_matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_probs).ravel()

# Bar plot for TP, TN, FP, FN
labels = ["True Positive (TP)", "True Negative (TN)", "False Positive (FP)", "False Negative (FN)"]
values = [tp, tn, fp, fn]
colors = ["blue", "green", "red", "orange"]

plt.figure(figsize=(8, 5))
plt.bar(labels, values, color=colors)
plt.ylabel("Count")
plt.title("Confusion Matrix Components")
plt.show()


# calculating accuracy precision recall and f1 percentage of the classification and plot them in bar graph
accuracy = accuracy_score(y_test, y_probs) * 100
precision = precision_score(y_test, y_probs) * 100
recall = recall_score(y_test, y_probs) * 100
f1 = f1_score(y_test, y_probs) * 100

# Bar Plot for Metrics with Percentage
metrics = ["Accuracy", "Precision", "Recall", "F1-score"]
values = [accuracy, precision, recall, f1]
colors = ["blue", "green", "red", "orange"]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, values, color=colors)
plt.ylim(0, 100)  # Percentage scale
plt.ylabel("Score (%)")
plt.title("Performance Metrics of ANN Model")

# Add percentage labels on top of bars
for bar in bars:
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 5, f'{bar.get_height():.1f}%', ha='center', color='white', fontsize=12, fontweight='bold')

plt.show()

