#!/usr/bin/env python
# coding: utf-8

# Importing Libraries

# In[73]:


import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models, Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import time


# Load and Preprocess Data

# In[74]:


#load the data
train_pix = pd.read_csv('trainingpix.csv', header=None)
train_labels = pd.read_csv('traininglabels.csv', header=None)
test_pix = pd.read_csv('testingpix.csv', header=None)
test_labels = pd.read_csv('testinglabels.csv', header=None)

#convert to numpy arrays
train_pix = train_pix.values
train_labels = train_labels.values
test_pix = test_pix.values
test_labels = test_labels.values

#normalize pixel values from [-1, 1] to [0, 1]
train_pix = (train_pix + 1) / 2
test_pix = (test_pix + 1) / 2

#reshape for CNN input (15x15 images with 1 channel - grayscale images)
train_pix = train_pix.reshape(-1, 15, 15, 1)
test_pix = test_pix.reshape(-1, 15, 15, 1)


# CNN for Predicting the x-coordinate

# In[75]:


#extract x-coordinate labels
train_labels_x = train_labels[:, 0]
test_labels_x = test_labels[:, 0]

#normalize x labels between 0 and 1
scaler_x = MinMaxScaler(feature_range=(0, 1))
train_labels_x = scaler_x.fit_transform(train_labels_x.reshape(-1, 1))
test_labels_x = scaler_x.transform(test_labels_x.reshape(-1, 1))

#split into training and validation sets
train_pix_x, val_pix_x, train_labels_x, val_labels_x = train_test_split(
    train_pix, train_labels_x, test_size=0.2, random_state=42
)

#build the CNN model for x-coordinate
model_x = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(15, 15, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output is a single value (x-coordinate)
])

#compile the model with MSE (Mean Squared Error)
model_x.compile(optimizer='adam', loss='mse', metrics=['mae'])

#start the timer
start_time = time.time()

#train the model
history_x = model_x.fit(
    train_pix_x, train_labels_x, epochs=20, batch_size=32,
    validation_data=(val_pix_x, val_labels_x)
)

#evaluate the model on the test set
test_loss, test_mae = model_x.evaluate(test_pix, test_labels_x)

#print test loss and MAE
print(f"Test Loss (MSE): {test_loss}")
print(f"Test MAE (Mean Absolute Error): {test_mae}")

#predict x-coordinate on the test set
predictions_x = model_x.predict(test_pix)

#inverse transform the predictions back to the original scale
predicted_x = scaler_x.inverse_transform(predictions_x)
true_x = scaler_x.inverse_transform(test_labels_x)

#calculate accuracy using a tolerance
tolerance = 0.5
accuracy = np.mean(np.abs(predicted_x - true_x) <= tolerance)
print(f"Test Set Accuracy (within {tolerance} units): {accuracy * 100:.2f}%")

#calculate total run time
total_run_time = time.time() - start_time
print(f"Total model run time: {total_run_time:.2f} seconds")

#save predicted_x to CSV
predicted_x_df = pd.DataFrame(predicted_x)
predicted_x_df.to_csv('predicted_x.csv', index=False, header=False)


# In[81]:


#plotting Loss and MAE
plt.figure(figsize=(12, 5))

#plot training & validation loss values
plt.subplot(1, 2, 1)
plt.plot(history_x.history['loss'], label='Training Loss')
plt.plot(history_x.history['val_loss'], label='Validation Loss')
plt.title('Model Loss (x)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

#plot training & validation MAE values
plt.subplot(1, 2, 2)
plt.plot(history_x.history['mae'], label='Training MAE')
plt.plot(history_x.history['val_mae'], label='Validation MAE')
plt.title('Model Mean Absolute Error (MAE) (x)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.show()


# CNN for Predicting the xyz-coordinate

# In[77]:


#normalize the labels (x, y, z) between 0 and 1 separately
scaler_x = MinMaxScaler(feature_range=(0, 1))
scaler_y = MinMaxScaler(feature_range=(0, 1))
scaler_z = MinMaxScaler(feature_range=(0, 1))

train_labels_x = scaler_x.fit_transform(train_labels[:, 0].reshape(-1, 1))
train_labels_y = scaler_y.fit_transform(train_labels[:, 1].reshape(-1, 1))
train_labels_z = scaler_z.fit_transform(train_labels[:, 2].reshape(-1, 1))

test_labels_x = scaler_x.transform(test_labels[:, 0].reshape(-1, 1))
test_labels_y = scaler_y.transform(test_labels[:, 1].reshape(-1, 1))
test_labels_z = scaler_z.transform(test_labels[:, 2].reshape(-1, 1))

#split into training and validation sets (inputs and labels)
train_pix, val_pix, train_labels_x, val_labels_x = train_test_split(
    train_pix, train_labels_x, test_size=0.2, random_state=42
)
train_labels_y, val_labels_y = train_test_split(train_labels_y, test_size=0.2, random_state=42)
train_labels_z, val_labels_z = train_test_split(train_labels_z, test_size=0.2, random_state=42)

#define inputs
input_layer = Input(shape=(15, 15, 1))

#first branch for x-coordinate
x_branch = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
x_branch = layers.MaxPooling2D((2, 2))(x_branch)
x_branch = layers.Conv2D(64, (3, 3), activation='relu')(x_branch)
x_branch = layers.Flatten()(x_branch)
x_output = layers.Dense(64, activation='relu')(x_branch)
x_output = layers.Dense(1, name='x_output')(x_output)

#second branch for y-coordinate
y_branch = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
y_branch = layers.MaxPooling2D((2, 2))(y_branch)
y_branch = layers.Conv2D(64, (3, 3), activation='relu')(y_branch)
y_branch = layers.Flatten()(y_branch)
y_output = layers.Dense(64, activation='relu')(y_branch)
y_output = layers.Dense(1, name='y_output')(y_output)

#third branch for z-coordinate
z_branch = layers.Conv2D(32, (3, 3), activation='relu')(input_layer)
z_branch = layers.MaxPooling2D((2, 2))(z_branch)
z_branch = layers.Conv2D(64, (3, 3), activation='relu')(z_branch)
z_branch = layers.Flatten()(z_branch)
z_output = layers.Dense(64, activation='relu')(z_branch)
z_output = layers.Dense(1, name='z_output')(z_output)

#create the model with three separate outputs
model = models.Model(inputs=input_layer, outputs=[x_output, y_output, z_output])

#compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

#start the timer
start_time = time.time()

#train the model (for x, y, and z simultaneously)
history = model.fit(
    train_pix,
    {
        'x_output': train_labels_x,  # x labels
        'y_output': train_labels_y,  # y labels
        'z_output': train_labels_z   # z labels
    },
    epochs=20, batch_size=32,
    validation_data=(
        val_pix,
        {
            'x_output': val_labels_x,  # validation x labels
            'y_output': val_labels_y,  # validation y labels
            'z_output': val_labels_z   # validation z labels
        }
    )
)

#evaluate the model on the test set
test_loss = model.evaluate(test_pix, {'x_output': test_labels_x, 'y_output': test_labels_y, 'z_output': test_labels_z})

#predict (x, y, z) on the test set
predictions = model.predict(test_pix)

#inverse transform the predictions back to the original scale
predicted_x = scaler_x.inverse_transform(predictions[0])
predicted_y = scaler_y.inverse_transform(predictions[1])
predicted_z = scaler_z.inverse_transform(predictions[2])

#combine predicted (x, y, z)
predicted_xyz = np.hstack([predicted_x, predicted_y, predicted_z])

#calculate total run time
total_run_time = time.time() - start_time
print(f"Total model run time: {total_run_time:.2f} seconds")

#save predictions to CSV without the column names
predicted_xyz_df = pd.DataFrame(predicted_xyz)
predicted_xyz_df.to_csv('predicted_xyz.csv', index=False, header=False)

print("Predicted (x, y, z) coordinates saved to 'predicted_xyz.csv'")


# In[78]:


#plotting merged Loss and MAE for x, y, z outputs
plt.figure(figsize=(12, 5))

#plot Training & Validation Loss (merged for x, y, z outputs)
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss (merged)')
plt.plot(history.history['val_loss'], label='Validation Loss (merged)')
plt.title('Model Loss (Merged x, y, z)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

#calculate the average MAE for x, y, z during training and validation
train_mae_avg = np.mean([history.history['x_output_mae'], history.history['y_output_mae'], history.history['z_output_mae']], axis=0)
val_mae_avg = np.mean([history.history['val_x_output_mae'], history.history['val_y_output_mae'], history.history['val_z_output_mae']], axis=0)

#plot Training & Validation MAE (merged for x, y, z outputs)
plt.subplot(1, 2, 2)
plt.plot(train_mae_avg, label='Training MAE (merged)')
plt.plot(val_mae_avg, label='Validation MAE (merged)')
plt.title('Model Mean Absolute Error (MAE) (Merged x, y, z)')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()

plt.tight_layout()
plt.show()


# In[80]:


#load the true values CSV and predicted values CSV (no column names)
true_values = pd.read_csv('testinglabels.csv', header=None)  # CSV with true values (x, y, z)
predicted_values = pd.read_csv('predicted_xyz.csv', header=None)  # CSV with predicted values

#convert the dataframes to numpy arrays for comparison
true_values = true_values.values
predicted_values = predicted_values.values

#tolerance value
tolerance = 0.5

#tolerance-based accuracy for each coordinate (x, y, z)
x_accuracy = np.mean(np.abs(predicted_values[:, 0] - true_values[:, 0]) <= tolerance)
y_accuracy = np.mean(np.abs(predicted_values[:, 1] - true_values[:, 1]) <= tolerance)
z_accuracy = np.mean(np.abs(predicted_values[:, 2] - true_values[:, 2]) <= tolerance)

#overall accuracy as the average of x, y, and z accuracy
overall_accuracy = (x_accuracy + y_accuracy + z_accuracy) / 3

#accuracy results
print(f"Accuracy for x-coordinate (within {tolerance} tolerance): {x_accuracy * 100:.2f}%")
print(f"Accuracy for y-coordinate (within {tolerance} tolerance): {y_accuracy * 100:.2f}%")
print(f"Accuracy for z-coordinate (within {tolerance} tolerance): {z_accuracy * 100:.2f}%")
print(f"Overall accuracy (average of x, y, z): {overall_accuracy * 100:.2f}%")

