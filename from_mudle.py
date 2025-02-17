import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout, RepeatVector
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

df = pd.read_csv('/content/electricity_demand_forecasting.csv') # загрузка набора данных, используя библиотеку pandas
df # вывод набора данных

# Переведём признак date из string в datetime
df['date'] = pd.to_datetime(df['date'])

# Выведем статистику по числовым признакам
print("\nСтатистика по числовым данным:")
df.describe().transpose()

# Проверим наличие пустых значений в данных
print("Проверка на пустые значения:")
print(df.isnull().sum())

# Заполнение пропущенных значений с помощью линейной интерполяции в столбце 'rainfall' и 'solar_exposure'
df['rainfall'] = df['rainfall'].interpolate(method='linear')
df['solar_exposure'] = df['solar_exposure'].interpolate(method='linear')

# Проверка типов данных
print(df.dtypes)

# Для столбцов school_day и holiday, которые являются булевыми, нужно привести их в числовой формат:
# Заменим строковые значения на булевы
df['school_day'] = df['school_day'].replace({'Y': 1, 'N': 0}).astype(int)
df['holiday'] = df['holiday'].replace({'Y': 1, 'N': 0}).astype(int)

# График спроса на электроэнергию (demand) по времени:
plt.figure(figsize=(12,6))
plt.plot(df['date'], df['demand'], color='blue')
plt.title('Спрос на электроэнергию по времени')
plt.xlabel('Дата')
plt.ylabel('Спрос (MWh)')
plt.grid(True)
plt.show()

# Добавляем дополнительные признаки времени
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
df['quarter'] = df['date'].dt.quarter

# Создание статистик
df['rolling_mean'] = df['demand'].rolling(window=30).mean()
df['rolling_std'] = df['demand'].rolling(window=30).std()
df['rolling_min'] = df['demand'].rolling(window=30).min()
df['rolling_max'] = df['demand'].rolling(window=30).max()
df.dropna(inplace=True)

# Добавляем синусоидальные и косинусоидальные признаки
timestamp_s = df['date'].astype('int64') // 10**9  # Преобразуем дату в секунды
day = 24 * 60 * 60  # Количество секунд в дне
month = 30.44 * day # Количество секунд в месяце (среднее, с учетом 30.44 дня в месяце)
year = 365.25 * day  # Количество секунд в году (учитываем високосные годы)
df['Month sin'] = np.sin(timestamp_s * (2 * np.pi / month))
df['Month cos'] = np.cos(timestamp_s * (2 * np.pi / month))
df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

# Построим корреляционную матрицу между непрерывными числовыми признаками:

corr = df[['demand', 'RRP', 'demand_pos_RRP', 'RRP_positive', 'demand_neg_RRP',
           'RRP_negative', 'frac_at_neg_RRP', 'min_temperature', 'max_temperature',
           'solar_exposure', 'rainfall', 'Month sin', 'Month cos', 'Year sin', 'Year cos']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Корреляционная матрица')
plt.show()

# Удаление высоко коррелирующих признаков для уменьшения мультиколлинеарности.
# demand_pos_RRP фактически представляет собой demand, ограниченный положительным RRP.
# RRP и RRP_positive — практически идентичные данные. Можно оставить только один из этих признаков, скорее всего RRP_positive.
# demand_neg_RRP описывает спрос на энергию при отрицательном RRP, а frac_at_neg_RRP — долю времени, когда RRP был отрицательным. Они связаны почти один к одному, так что можно оставить только один из этих признаков.

df = df.drop(columns=['demand_pos_RRP', 'RRP_positive', 'frac_at_neg_RRP'])

# Создаем скейлер для нормализации
scaler = MinMaxScaler()

# Определим столбцы, которые нужно исключить из нормализации
exclude_cols = ['school_day', 'holiday', 'date', 'demand', 'day_of_week', 'month', 'is_weekend', 'quarter']

# Выбираем только числовые столбцы, которые нуждаются в нормализации
cols_to_normalize = [col for col in df.columns if col not in exclude_cols]

# Нормализуем только выбранные столбцы
df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])

# Создаем отдельный scaler для нормализации целевой переменной. Мы будем использовать его позже для денормализации в процессе тестирования и визуализации
target_scaler = MinMaxScaler(feature_range=(0, 1))

# Нормализуем целевую переменную
df['demand'] = target_scaler.fit_transform(df[['demand']])

# Разделение на обучающую, валидационную и тестовую выборки (80%/10%/10%)
n = len(df)
train_df = df[0:int(n*0.8)]
val_df = df[int(n*0.8):int(n*0.9)]
test_df = df[int(n*0.9):]


# График с разделёнными наборами данных:
plt.figure(figsize=(12,6))
plt.plot(train_df['date'], train_df['demand'], label='Train Set', color='blue')
plt.plot(val_df['date'], val_df['demand'], label='Val Set', color='green')
plt.plot(test_df['date'], test_df['demand'], label='Test Set',color='red')
plt.title('Спрос на электроэнергию по времени')
plt.xlabel('Дата')
plt.ylabel('Спрос (MWh)')
plt.grid(True)
plt.legend()
plt.show()

# Функция для создания окон
def create_sequences(df, window_size, step_size):
    X, y = [], []
    for i in range(window_size, len(df) - window_size + 1, step_size):
        X.append(df.iloc[i-window_size:i].drop(columns=['date']).values)
        y.append(df.iloc[i:i + window_size]['demand'].values)
    return np.array(X), np.array(y)

# Укажем параметр скользящего окна. Получается, мы будем на основе 30 дневных наблюдений предсказывать целевую переменную на следующие 30 дней.
window_size = 30

# Сделаем для тренировочных и валидационных данных временные последовательности длинной 30 каждые 15 шагов (с перекрытием).
train_val_step = 15

# Создаем окна для всех признаков
X_train, y_train = create_sequences(train_df, window_size, step_size=train_val_step)
X_val, y_val = create_sequences(val_df, window_size, step_size=train_val_step)
X_test, y_test = create_sequences(test_df, window_size, step_size=window_size)

# Убедимся, что формы данных правильные
print("Размерность обучающей выборки: " + str(X_train.shape) + str(y_train.shape))
print("Размерность валидационной выборки: " + str(X_val.shape) + str(y_val.shape))
print("Размерность валидационной выборки: " + str(X_test.shape) + str(y_test.shape))

# Выведем пример с окном из тестового набора (возьмём первую временную последовательность)
sample_idx = 0
plt.figure(figsize=(12, 6))
plt.plot(np.arange(0, window_size), X_test[sample_idx, :, 0], label='Исходные данные (X)', color='blue', marker='o')
plt.plot(np.arange(0, window_size) + window_size, y_test[sample_idx], label='Целевые значения (y)', color='red', linestyle='--', marker='x')
plt.title('Пример временной последовательности: исходные данные и целевые предсказания на тестовом наборе')
plt.xlabel('Дата')
plt.ylabel('Спрос')
plt.legend()
plt.show()

# # Функция для создания окон
# def create_sequences(df, window_size, forecast_len, step_size):
#     X, y = [], []
#     for i in range(window_size, len(df) - window_size + 1, step_size):
#         X.append(df.iloc[i-window_size:i].drop(columns=['date']).values)
#         y.append(df.iloc[i:i + forecast_len]['demand'].values)
#     return np.array(X), np.array(y)

# # Укажем параметр скользящего окна (на основе скольки дней будем делать прогноз в будущее).
# window_size = 30

# # Укажем горизонт прогнозирования (насколько дней будем прогнозировать вперёд, например, на 7 дней)
# forecast_len = 14

# # Сделаем для тренировочных и валидационных данных временные последовательности длинной 30 каждые 15 шагов (с перекрытием).
# train_val_step = 15

# # Создаем окна для всех признаков
# X_train, y_train = create_sequences(train_df, window_size, forecast_len=forecast_len, step_size=train_val_step)
# X_val, y_val = create_sequences(val_df, window_size, forecast_len=forecast_len, step_size=train_val_step)
# X_test, y_test = create_sequences(test_df, window_size, forecast_len=forecast_len, step_size=window_size)

# # Убедимся, что формы данных правильные
# print("Размерность обучающей выборки: " + str(X_train.shape) + str(y_train.shape))
# print("Размерность валидационной выборки: " + str(X_val.shape) + str(y_val.shape))
# print("Размерность валидационной выборки: " + str(X_test.shape) + str(y_test.shape))

# # Выведем пример с окном из тестового набора (возьмём первую временную последовательность)
# sample_idx = 0
# plt.figure(figsize=(12, 6))
# plt.plot(np.arange(0, window_size), X_test[sample_idx, :, 0], label='Исходные данные (X)', color='blue', marker='o')
# plt.plot(np.arange(0, forecast_len) + window_size, y_test[sample_idx], label='Целевые значения (y)', color='red', linestyle='--', marker='x')
# plt.title('Пример временной последовательности: исходные данные и целевые предсказания на тестовом наборе')
# plt.xlabel('Дата')
# plt.ylabel('Спрос')
# plt.legend()
# plt.show()

input_shape = X_train.shape[1:3] # задаём размерность входа в нашу рекуррентную нейронную сеть. размер входа нейронной сети, определяется по длине последовательности и кол-ву признаков
num_targets = 1                  # указываем кол-во целевых переменных

# Создаём обычную рекуррентную LSTM сеть. Включаем слои в режиме Seq2Seq.
# Данный вариант будет работать если длина входного временного ряда и горизонта прогноза имеет одинаковую длину.
def createRecurrentNetwork(input_shape):
  input_x = Input(shape=input_shape)                      # для создания модели необходимо предопределить вход и размерность входных данных input_shape
  lstm1 = LSTM(64, return_sequences=True)(input_x)        # добавляем рекуррентный скрытый слой с 64 нейронами (включен режим seq2seq, поскольку следующий LSTM должен принять последовательность)
  lstm2 = LSTM(32, return_sequences=True)(lstm1)          # добавляем рекуррентный скрытый слой с 32 нейронами (включаем режим seq2seq, поскольку мы хотим в итоге предсказать на выходе спрос для следующих 30 дней)
  out = Dense(num_targets)(lstm2)                         # добавляем выходной слой с num_targetss нейронами (по кол-ву целевых переменных) без функции активации (так как задача прогнозирования)
  return Model(inputs=input_x, outputs=out)               # создаём модель, задавая вход и выход и возвращаем эту модель для последующих манипуляций

# Создаём сеть с архитектурой "кодер-декодер". Эксперементируйте с кол-вом слоёв и нейронов.
# Подходит для задач, где длина входного временного ряда не совпадает с длиной горизонта прогноза.
def createEncoderDecoderNetwork(input_shape):
  input_x = Input(shape=input_shape)
  # Кодирующая сеть
  enc_1 = LSTM(128, return_sequences=True)(input_x)
  enc_1 = LSTM(64, return_sequences=False)(enc_1)
  # Слой повторяет последнее скрытое состояние n раз, где n = длина горизонта прогноза (forecast_len).
  repeat_vec = RepeatVector(n=forecast_len)(enc_1)
  # Декодирующая сеть
  dec_1 = LSTM(64, return_sequences=True)(repeat_vec)
  dec_2 = LSTM(128, return_sequences=True)(dec_1)
  out = Dense(num_targets)(dec_2)
  return Model(inputs=input_x, outputs=out)

def trainModel(X_train, y_train, X_val, y_val):
  # Сохранение только наилучшей модели с наименьшей ошибкой на валидации
  save_callback = ModelCheckpoint(filepath='best_model.weights.h5', # путь и название файла лучшей модели.
                                  monitor = 'val_loss',             # отслеживать каждую эпоху показать val_loss
                                  save_best_only = True,            # сохранять только самую лучшую модель
                                  save_weights_only=True,           # сохранять только веса модели (не всю модель);
                                  mode = 'min',                     # режим оптимизации min (найти минимальную ошибку)
                                  verbose=1)                        # 1 означает выводить информацию о нахождении и сохранении лучшей модели в процессе обучения, 0 означает не выводить.

  model = createRecurrentNetwork(input_shape) # создаётся рекуррентная сеть, принимающая на вход входные данные с размерностью input_shape

  #model = createEncoderDecoderNetwork(input_shape) # создаётся сеть кодер-декодер, принимающая на вход входные данные с размерностью input_shape

  # Компиляция модели
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error') # для задачи прогнозирования используется функция потерь MSE (среднеквадратичная ошибка), так как модель прогнозирует непрерывные значения.

  model.summary() # вывод информации о созданной модели

  model.fit(x=X_train,                       # признаки X
            y=y_train,                       # желаемый выход y
            epochs=200,                      # количество эпох обучения. эпоха представляет собой один полный проход через все данные обучающего набора.
            batch_size=8,                    # параметр, с помощью которого можно регулировать порцию подаваемых примеров для сети за одну итерацию обучения (по-умолчанию он равен 32).
            validation_data=(X_val, y_val),  # указываем валидационные данные
            callbacks=[save_callback]        # используем в процессе обучения созданный ModelCheckpoint
           )
  return model # обученная модель возвращается для тестирования и других манипуляций

# Запуск обучения нейронной сети
model = trainModel(X_train, y_train, X_val, y_val)

plt.figure()
plt.plot(model.history.history["loss"], label="training loss")
plt.plot(model.history.history["val_loss"], label="validation loss")
plt.legend()
plt.title("График изменения ошибки модели")
plt.xlabel("Эпохи")
plt.ylabel("Ошибка")
plt.show()

# Загрузка весов самой лучшей модели с самой маленькой ошибкой на валидации
model.load_weights('/content/best_model.weights.h5')

from sklearn.metrics import mean_absolute_percentage_error

# Прогноз на тестовых данных
y_test_pred = model.predict(X_test).squeeze()

# Обратная нормализация только для целевой переменной
y_test_pred = target_scaler.inverse_transform(y_test_pred.reshape(-1, 1)).flatten()
y_test_actual = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Расчет MAPE
mape = mean_absolute_percentage_error(y_test_actual, y_test_pred) * 100
print(f"Модель ошибается в среднем на {mape:.2f}% от истинных значений")

# Построение графика всех предсказаний (len_cut для обрезки)
len_cut=60
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual[0:len_cut], label='Real Demand', color='blue', marker='o')
plt.plot(y_test_pred[0:len_cut], label='Predicted Demand', color='red', linestyle='--', marker='x')
plt.title('Истинный vs Предсказанный спрос')
plt.xlabel('Дата')
plt.ylabel('Спрос')
plt.xticks(rotation=90, size = 7)
plt.legend()
plt.show()

import numpy as np
import tensorflow as tf
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, SimpleRNN, LSTM, GRU, Dense, Dropout
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

df = pd.read_csv('/content/human_activity_classification.csv') # загрузка набора данных, используя библиотеку pandas
df # вывод набора данных

# Сортируем время наблюдений по возрастанию
df = df.sort_values(by='time (s)', ascending=True)

# Выведем статистику по числовым признакам
print("\nСтатистика по числовым данным:")
df.describe().transpose()

# Проверим наличие пустых значений в данных
print("Проверка на пустые значения:")
print(df.isnull().sum())

# Разделим данные по классам и определим кол-во
classes = df['class'].unique()
num_classes = len(classes)

# Создаем графики для каждого класса
fig, axes = plt.subplots(len(classes), 4, figsize=(25, 6 * len(classes)))

# Для каждого класса строим графики
for i, class_id in enumerate(classes):
    class_data = df[df['class'] == class_id]

    # График для x-оси
    axes[i, 0].plot(class_data['time (s)'], class_data['x-axis'], label=f'Class {class_id}', color='tab:blue')
    axes[i, 0].set_title(f'Class {class_id} - X-axis')
    axes[i, 0].set_xlabel('Time')
    axes[i, 0].set_ylabel('Acceleration (m/s²)')

    # График для y-оси
    axes[i, 1].plot(class_data['time (s)'], class_data['y-axis'], label=f'Class {class_id}', color='tab:orange')
    axes[i, 1].set_title(f'Class {class_id} - Y-axis')
    axes[i, 1].set_xlabel('Time')
    axes[i, 1].set_ylabel('Acceleration (m/s²)')

    # График для z-оси
    axes[i, 2].plot(class_data['time (s)'], class_data['z-axis'], label=f'Class {class_id}', color='tab:green')
    axes[i, 2].set_title(f'Class {class_id} - Z-axis')
    axes[i, 2].set_xlabel('Time')
    axes[i, 2].set_ylabel('Acceleration (m/s²)')

    # График для w
    axes[i, 3].plot(class_data['time (s)'], class_data['w'], label=f'Class {class_id}', color='tab:red')
    axes[i, 3].set_title(f'Class {class_id} - W')
    axes[i, 3].set_xlabel('Time')
    axes[i, 3].set_ylabel('Total Acceleration')

# Подстройка графиков
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# Построим гистограмму распределения классов
plt.figure(figsize=(8, 6))
sns.countplot(x='class', data=df, palette='Set2')

# Добавим подписи
plt.title('Распределение классов')
plt.xlabel('Класс')
plt.ylabel('Частота')
plt.show()

# Построим корреляционную матрицу между непрерывными числовыми признаками:

corr = df[['x-axis', 'y-axis', 'z-axis', 'w']].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Корреляционная матрица')
plt.show()

# Удаление высоко коррелирующих признаков для уменьшения мультиколлинеарности.
# Признак z-axis сильно коррелирует с x-axis и с y-axis, что делает её избыточной с точки зрения информации, которая может быть получена от других переменных.
df = df.drop(columns=['z-axis'])

# Создаем скейлер для нормализации
scaler = MinMaxScaler(feature_range=(0, 1))

# Нормализуем x, y, z оси
df[['x-axis', 'y-axis']] = scaler.fit_transform(df[['x-axis', 'y-axis']])

# Функция для создания окон
def create_sequences(df, window_size, step_size):
    X, y = [], []
    for i in range(window_size, len(df) - window_size + 1, step_size):
        window = df.iloc[i-window_size:i]
        X.append(window.drop(columns=['time (s)', 'class']).values)
        y.append(window['class'].mode()[0])
    return np.array(X), np.array(y)

# Укажем параметр окна.
window_size = 30

# Перекрытие не используем (оно равно window_size)
train_val_step = window_size

# Создаем окна для всех признаков
X, y = create_sequences(df, window_size, step_size=train_val_step)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, stratify=y_train)

# Убедимся, что формы данных правильные
print("Размерность обучающей выборки: " + str(X_train.shape) + str(y_train.shape))
print("Размерность валидационной выборки: " + str(X_val.shape) + str(y_val.shape))
print("Размерность валидационной выборки: " + str(X_test.shape) + str(y_test.shape))

input_shape = X_train.shape[1:3] # задаём размерность входа в нашу рекуррентную нейронную сеть. размер входа нейронной сети, определяется по длине последовательности и кол-ву признаков

def createRecurrentNetwork(input_shape):
  input_x = Input(shape=input_shape)                      # для создания модели необходимо предопределить вход и размерность входных данных input_shape
  lstm1 = LSTM(64, return_sequences=True)(input_x)        # добавляем рекуррентный скрытый слой с 64 нейронами (включен режим seq2seq, поскольку следующий LSTM должен принять последовательность)
  drop1 = Dropout(0.5)(lstm1)                             # добавляем слой дропаута от переобучения с вероятностью 0.5
  lstm2 = LSTM(32, return_sequences=False)(drop1)         # добавляем рекуррентный скрытый слой с 32 нейронами (включаем режим seq2vec return_sequences=False поскольку работаем с задачей классификации)
  drop2 = Dropout(0.5)(lstm2)                             # добавляем слой дропаута от переобучения с вероятностью 0.5
  out = Dense(num_classes, activation='softmax')(drop2)   # добавляем выходной слой с num_classes нейронами (по кол-ву классов) и функцией активации софтмакс для многоклассовой классификации
  return Model(inputs=input_x, outputs=out)               # создаём модель, задавая вход и выход и возвращаем эту модель для последующих манипуляций

def trainModel(X_train, y_train, X_val, y_val):
  # Сохранение только наилучшей модели с наименьшей ошибкой на валидации
  save_callback = ModelCheckpoint(filepath='best_model.weights.h5', # путь и название файла лучшей модели.
                                  monitor = 'val_loss',             # отслеживать каждую эпоху показать val_loss
                                  save_best_only = True,            # сохранять только самую лучшую модель
                                  save_weights_only=True,           # сохранять только веса модели (не всю модель);
                                  mode = 'min',                     # режим оптимизации min (найти минимальную ошибку)
                                  verbose=1)                        # 1 означает выводить информацию о нахождении и сохранении лучшей модели в процессе обучения, 0 означает не выводить.

  model = createRecurrentNetwork(input_shape) # создаётся модель принимающая на вход входные данные с размерностью input_shape

  # Компиляция модели
  model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # алгоритм градиентного спуска, минимизирующий функцию ошибки. В нём можно задать скорость обучения (по умолчанию 0.001)
                loss='sparse_categorical_crossentropy',                  # функция ошибки categorical_crossentropy категориальная кроссэнтропия для задачи многоклассовой классификации.
                metrics=['sparse_categorical_accuracy'])                 # метрика accuracy - точность.

  model.summary() # вывод информации о созданной модели

  model.fit(x=X_train,                      # признаки X
            y=y_train,                      # желаемый выход y
            epochs=50,                      # количество эпох обучения. эпоха представляет собой один полный проход через все данные обучающего набора.
            batch_size=16,                  # параметр, с помощью которого можно регулировать порцию подаваемых примеров для сети за одну итерацию обучения (по-умолчанию он равен 32).
            validation_data=(X_val, y_val), # указываем валидационные данные
            callbacks=[save_callback]       # используем в процессе обучения созданный ModelCheckpoint
            )
  return model # обученная модель возвращается для тестирования и других манипуляций

# Запуск обучения нейронной сети
model = trainModel(X_train, y_train, X_val, y_val)

plt.figure()
plt.plot(model.history.history["sparse_categorical_accuracy"], label="training accuracy")
plt.plot(model.history.history["val_sparse_categorical_accuracy"], label="validation accuracy")
plt.legend()
plt.title("График изменения точности модели")
plt.xlabel("Эпохи")
plt.ylabel("Точность")
plt.show()

plt.figure()
plt.plot(model.history.history["loss"], label="training loss")
plt.plot(model.history.history["val_loss"], label="validation loss")
plt.legend()
plt.title("График изменения ошибки модели")
plt.xlabel("Эпохи")
plt.ylabel("Ошибка")
plt.show()

# Загрузка весов самой лучшей модели с самой маленькой ошибкой на валидации
model.load_weights('/content/best_model.weights.h5')

# Тестирование модели на тестовых данных
mlp_loss, mlp_accuracy = model.evaluate(X_test, y_test)
print("Точность модели MLP: " + str(mlp_accuracy))

from sklearn.metrics import classification_report

# Тестирование
lstm_pred = model.predict(X_test)
lstm_pred = np.argmax(lstm_pred, axis=1) # чтобы получить предсказанные классы, используйте функцию argmax, которая вернет индекс класса с наибольшей вероятностью.
print(classification_report(y_test, lstm_pred))

from sklearn.metrics import confusion_matrix

# Построим матрицу ошибок для оценки качества классификационной модели
# Матрица ошибок имеет вид таблицы, где:
# Строки представляют фактические классы (истинные метки),
# Столбцы представляют предсказанные классы (то, что модель предсказала).
cm = confusion_matrix(y_test, lstm_pred)

cm_df = pd.DataFrame(cm,
                     index=np.unique(y_test),
                     columns=np.unique(y_test))

plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', cbar=False, linewidths=0.5)
plt.title('Матрица ошибок')
plt.xlabel('Предсказанные метки')
plt.ylabel('Истинные метки')
plt.show()