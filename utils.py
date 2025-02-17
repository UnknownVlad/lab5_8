import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau

def load_data(filepath):
    """Загрузка данных из файла."""
    df = pd.read_csv(filepath)
    return df

def scale_data(df):
    """Масштабирование данных."""
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[['Page.Loads', 'Unique.Visits', 'First.Time.Visits', 'Returning.Visits']])
    return scaled_data, scaler

def split_data(X, y, test_size=0.2):
    """Разделение данных на тренировочную и тестовую выборки."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
    return X_train, X_test, y_train, y_test

def visualize_loss(history):
    """График потерь модели."""
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('График потерь')
    plt.xlabel('Эпохи')
    plt.ylabel('Потери')
    plt.legend()
    plt.show()

def visualize_predictions(y_test, y_pred):
    """График сравнения реальных значений и предсказаний."""
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label='Реальные значения')
    plt.plot(y_pred, label='Предсказания модели')
    plt.title('Сравнение реальных значений и предсказаний')
    plt.xlabel('Дата')
    plt.ylabel('Unique.Visits')
    plt.legend()
    plt.show()

def mape(y_true, y_pred):
    """Метрика MAPE."""
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def create_lr_scheduler():
    """Создание колбэка для изменения скорости обучения."""
    return ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
