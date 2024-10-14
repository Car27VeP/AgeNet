
import pandas as pd

import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam

path = '/datasets/faces/final_files/'
df_ages = '/datasets/faces/labels.csv'

def load_train(path):
    
    """
    Carga la parte de entrenamiento del conjunto de datos desde la ruta.
    """
    
    # coloca tu código aquí
    
    train_gen = ImageDataGenerator(
        validation_split = 0.25,
        rescale = 1./255.
    )
    
    train_gen_flow = train_gen.flow_from_dataframe(
        dataframe = df_ages,
        directory = path,
        x_col = 'file_name',
        y_col = 'real_age',
        target_size = (150,150),
        batch_size = 16,
        class_mode = 'raw',
        subset = 'training',
        seed = 42
    )

    return train_gen_flow

def load_test(path):
    
    """
    Carga la parte de validación/prueba del conjunto de datos desde la ruta
    """
    
    #  coloca tu código aquí
    
    test_gen = ImageDataGenerator(
        validation_split = 0.25,
        rescale = 1./255.
    )
    
    test_gen_flow = test_gen.flow_from_dataframe(
        dataframe = df_ages,
        directory = path,
        x_col = 'file_name',
        y_col = 'real_age',
        target_size = (150,150),
        batch_size = 16,
        class_mode = 'raw',
        subset = 'validation',
        seed = 42
    )

    return test_gen_flow

def create_model(input_shape):
    
    """
    Define el modelo
    """
    
    #  coloca tu código aquí
    
    backbone = ResNet50(
        input_shape = input_shape,
        weights = 'imagenet',
        include_top = False
    )
    
    # Creación del modelo.
    
    model = Sequential()
    
    # Capa ResNet.
    
    model.add(backbone)
    
    # Capa Average Pooling
    
    model.add(GlobalAveragePooling2D())
    
    # Capa Dense con 1 nodo y función de activación relu.
    model.add(Dense(units = 1, activation = 'relu'))

    
    # Optmizador 
    optimizer = Adam(learning_rate = 0.0001)
    
    # Método compile
    model.compile(
        loss = 'mse',
        optimizer = optimizer,
        metrics = ['mae']
    )
    
    return model

def train_model(model, train_data, test_data, batch_size = None, epochs = 3,
                steps_per_epoch = None, validation_steps = None):

    """
    Entrena el modelo dados los parámetros
    """
    
    #  coloca tu código aquí
    
    model.fit(
        train_data,
        batch_size = batch_size,
        epochs = epochs,
        steps_per_epoch = steps_per_epoch,
        validation_steps = validation_steps,
        verbose = 2,
        validation_data = test_data
    )
    

    return model