import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras import callbacks
import matplotlib.pyplot as plt

# Ensure reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Step 1: Load and preprocess data
# Load your dataset here (update the file path accordingly)
data = pd.read_csv(r"water-quality\data\processed\water_potability_imputed.csv")
# Placeholder: Make sure 'data' exists
# Drop the redundant 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in data.columns:
    data = data.drop(columns=['Unnamed: 0'])

# Separate features and target variable
X = data.drop(columns=['Potability'])
y = data['Potability']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Handle class imbalance by calculating class weights
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights))

# Step 2: Build the neural network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),  # Regularization
    Dense(32, activation='relu'),
    Dropout(0.3),  # Regularization
    Dense(1, activation='sigmoid')  # Binary classification output
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss=BinaryCrossentropy(),
              metrics=['accuracy', AUC(name='auc'), Precision(name='precision'), Recall(name='recall')])

# Step 3: Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1,
                    class_weight=class_weights_dict, callbacks= [
        callbacks.EarlyStopping(
            monitor = 'val_loss',
            patience = 30,
            verbose = 1
        ),
        callbacks.ModelCheckpoint(
            filepath = r"water-quality\models\neural\checkpoints\model.keras",
            save_weights_only = False,
            monitor = 'loss',
            mode = 'min',
            save_best_only = True
        )
    ])

# Step 4: Evaluate the model
test_loss, test_accuracy, test_auc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")
print(f"Test Precision: {test_precision:.4f}")
print(f"Test Recall: {test_recall:.4f}")

# Step 5: Visualize training history
def plot_training_history(history):
    metrics = ['loss', 'accuracy', 'auc', 'precision', 'recall']
    for metric in metrics:
        plt.figure()
        plt.plot(history.history[metric], label=f'Training {metric.capitalize()}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric.capitalize()}')
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.show()

plot_training_history(history)
