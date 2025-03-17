import numpy as np
import pandas as pd
from itertools import product
import tensorflow as tf
from time import time
import multiprocessing
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Concatenate, Lambda
from keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import os

os.environ["OMP_NUM_THREADS"] = "1"
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)
ncores=27
list_batch_size = 27

df = pd.read_csv("Complete_BF_CPP_XY_Data.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
X = df.iloc[:, 1:27]
Y = df.iloc[:, 27:]
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)
Y_scaler = StandardScaler()
Y_scaled = Y_scaler.fit_transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y_scaled, test_size=0.2, random_state=42)

def create_model(layer1_neurons, learning_rate):
    # Input layer
    input_layer = Input(shape=(26,), name="Input_Layer")
    
    # First network (Network 1)
    x = Dense(layer1_neurons[0], activation='relu', name="Dense_Layer_1")(input_layer)
    for i, neurons in enumerate(layer1_neurons[1:], start=2):
        x = Dense(neurons, activation='relu', name=f"Dense_Layer_{i}")(x)

    output_1 = Dense(23, activation='linear', name="Output_Layer_2")(x)
    
    # Define the model with three outputs
    model = Model(inputs=input_layer, outputs=output_1, name="Model_2")
    
    # Compile the model with specified losses for each output
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    
    return model

def evaluate_model(layer1_neurons, learning_rate):
    model = create_model(layer1_neurons, learning_rate)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.92, patience=10, min_lr=0, verbose=1, mode='min')
    model.fit(
        X_train,
        Y_train,  # Targets as a list
        epochs=100,
        batch_size=256,
        verbose=0,
        validation_split=0.15,
        callbacks=[reduce_lr]
    )
    Y_pred = model.predict(X_test, verbose=0)

    # Inverse transform predictions and ground truths for R2 scoring
    y_pred = Y_scaler.inverse_transform(Y_pred)
    y_pred = Y_scaler.inverse_transform(Y_pred)

    y_test = Y_scaler.inverse_transform(Y_test)
    y_test = Y_scaler.inverse_transform(Y_test)
    
    r2 = r2_score(y_test, y_pred)
    return r2, layer1_neurons, learning_rate

def evaluate_hyperparameters(config):
    layer1_neurons, learning_rate = config
    return evaluate_model(layer1_neurons, learning_rate)

def run_grid_search():  # Number of configurations per batch
    total_combinations = len(hyperparameter_combinations)
    num_batches = (total_combinations + list_batch_size - 1) // list_batch_size  # Calculate the total number of batches

    # Create or overwrite the CSV to store results incrementally
    results_df = pd.DataFrame(columns=["Average R2 Score", "Layer1 Neurons", "Learning Rate"])
    results_df.to_csv("mod2_grid_search_results.csv", index=False)

    # Loop over each batch
    for batch_index in range(num_batches):
        # Select the next batch of combinations
        start_index = batch_index * list_batch_size
        end_index = min(start_index + list_batch_size, total_combinations)
        batch_combinations = hyperparameter_combinations[start_index:end_index]

        # Process the batch with multiprocessing
        with multiprocessing.Pool(processes=ncores) as pool:
            batch_results = pool.map(evaluate_hyperparameters, batch_combinations)

        # Append the batch results to the CSV
        batch_df = pd.DataFrame(batch_results, columns=["R2 Score", "Layer1 Neurons", "Learning Rate"])
        batch_df.to_csv("mod2_grid_search_results.csv", mode='a', header=False, index=False)

        # Print the completion percentage
        completion_percentage = ((batch_index + 1) / num_batches) * 100
        print(f"Completed {completion_percentage:.2f}% of hyperparameter tuning")

    # Find the best configuration after all batches are processed
    all_results = pd.read_csv("mod2_grid_search_results.csv")
    best_result = all_results.loc[all_results["Average R2 Score"].idxmax()]
    print(f"Best configuration: {best_result}")

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    layer1_neurons_options = [
    [700]*3,  [700]*4,  [700]*5,
    [850]*3,  [850]*4,  [850]*5,
    [1000]*3, [1000]*4, [1000]*5,
    ]
    learning_rate_options = [7e-3, 4e-4, 2.5e-4]
    hyperparameter_combinations = list(product(layer1_neurons_options, learning_rate_options))

    best_result = run_grid_search()
    print("Best Parameters:", best_result)