import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """Загрузка данных из файла."""
    df = pd.read_csv(filepath)

    # Преобразование строковых чисел с запятыми в числовые значения
    cols_to_convert = ['Page.Loads', 'Unique.Visits', 'First.Time.Visits', 'Returning.Visits']
    for col in cols_to_convert:
        df[col] = df[col].str.replace(',', '').astype(float)

    # Преобразование столбца Date в datetime
    df['Date'] = pd.to_datetime(df['Date'])

    # Удаляем ненужную колонку Row
    df = df.drop(columns=['Row'])

    return df


def check_missing_values(df):
    """Проверка на пропущенные значения."""
    return df.isnull().sum()


def remove_outliers(df, column, threshold=3):
    """Удаление выбросов по стандартным отклонениям."""
    mean = df[column].mean()
    std = df[column].std()
    return df[df[column] < mean + threshold * std]


def visualize_missing_data(df):
    """График пропущенных значений."""
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.show()


def plot_correlation(df):
    """Построение корреляционной матрицы для числовых данных."""
    # Выбираем только числовые колонки
    numeric_df = df.select_dtypes(include=np.number)

    # Строим корреляционную матрицу
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    plt.show()


def plot_feature_distribution(df):
    """График распределения признаков."""
    numeric_columns = df.select_dtypes(include=np.number).columns
    for col in numeric_columns:
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True, bins=30)
        plt.title(f'Distribution of {col}')
        plt.show()


def plot_scatter(df, x_col, y_col):
    """График разброса между признаками и целевой переменной."""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col)
    plt.title(f'{x_col} vs {y_col}')
    plt.show()


def preprocess_data(df):
    """Основная обработка данных: удаление выбросов, масштабирование."""
    # Визуализируем распределение признаков до обработки
    plot_feature_distribution(df)

    # Удаляем выбросы для всех числовых колонок
    for col in df.select_dtypes(include=np.number).columns:
        df = remove_outliers(df, col)

    # Масштабирование данных
    scaler = StandardScaler()
    df_scaled = df.copy()
    for col in df.select_dtypes(include=np.number).columns:
        df_scaled[col] = scaler.fit_transform(df[col].values.reshape(-1, 1))

    # Визуализируем распределение признаков после обработки
    plot_feature_distribution(df_scaled)

    return df_scaled


def save_processed_data(df, output_file_path):
    """Сохранение предобработанных данных в новый файл."""
    df.to_csv(output_file_path, index=False)
    print(f"Предобработанные данные сохранены в {output_file_path}")


def split_data(df, target_column, test_size=0.2):
    """Разделение данных на тренировочные и тестовые."""
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return train_test_split(X, y, test_size=test_size, random_state=42)


def plot_target_distribution(df, target_column):
    """График распределения целевой переменной."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[target_column], kde=True, bins=30)
    plt.title(f'Distribution of {target_column}')
    plt.show()


# Основная логика
if __name__ == '__main__':
    # Загружаем данные
    df = load_data('daily-website-visitors.csv')

    # Проверяем на пропущенные значения
    missing_values = check_missing_values(df)
    print(missing_values)

    # Визуализируем пропущенные данные
    visualize_missing_data(df)

    # Строим корреляционную матрицу
    plot_correlation(df)

    # Визуализируем распределение целевой переменной
    plot_target_distribution(df, target_column='Unique.Visits')

    # Предобработка данных (удаление выбросов и масштабирование)
    df_scaled = preprocess_data(df)

    # Визуализируем связь между признаками
    plot_scatter(df, x_col='Page.Loads', y_col='Unique.Visits')

    # Сохранение предобработанных данных
    save_processed_data(df_scaled, 'processed_data.csv')

    # Разделяем данные на тренировочную и тестовую выборки
    X_train, X_test, y_train, y_test = split_data(df_scaled, target_column='Unique.Visits')
