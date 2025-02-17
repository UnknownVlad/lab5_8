import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from utils import load_data, scale_data, split_data, visualize_loss, visualize_predictions, mape, create_lr_scheduler

def build_model(input_shape):
    """Создание модели LSTM."""
    model = Sequential()

    # Первый LSTM слой
    model.add(LSTM(units=64, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Второй LSTM слой
    model.add(LSTM(units=32))
    model.add(Dropout(0.2))

    # Слой нормализации
    model.add(BatchNormalization())

    # Выходной слой
    model.add(Dense(1))

    # Компиляция модели
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=50, batch_size=32):
    """Обучение модели."""
    # Подготовка данных для подачи в модель LSTM
    X_train_reshaped = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    # Создание колбэка для изменения скорости обучения
    lr_scheduler = create_lr_scheduler()

    # Обучение модели
    history = model.fit(X_train_reshaped, y_train, epochs=epochs, batch_size=batch_size,
                        validation_data=(X_test_reshaped, y_test), callbacks=[lr_scheduler])

    return model, history

def evaluate_model(model, X_test, y_test):
    """Оценка модели на тестовых данных."""
    # Прогнозирование
    X_test_reshaped = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    y_pred = model.predict(X_test_reshaped)

    # Преобразуем y_pred в одномерный массив
    y_pred = y_pred.flatten()

    # Визуализация результатов
    visualize_predictions(y_test, y_pred)

    # Вычисление MAPE
    error = mape(y_test, y_pred)
    print(f'MAPE: {error:.2f}%')

    return error


def main():
    # Загрузка данных
    df = load_data('processed_data.csv')

    # Масштабируем данные
    scaled_data, scaler = scale_data(df)

    # Разделяем данные на признаки и целевую переменную
    X = scaled_data
    y = df['Unique.Visits']

    # Разделение на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Строим модель
    model = build_model(input_shape=(X_train.shape[1], 1))

    # Обучаем модель
    model, history = train_model(model, X_train, y_train, X_test, y_test)

    # Визуализируем процесс обучения
    visualize_loss(history)

    # Оценка модели
    evaluate_model(model, X_test, y_test)

if __name__ == '__main__':
    main()
