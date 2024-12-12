# Работа с табличными данными
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.stats as stats

# Преобразование признаков
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Пайплайнs
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

# Модели, валидация, метрики
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedGroupKFold
from sklearn.metrics import f1_score

# Подбор гиперпараметров
import optuna
optuna.logging.set_verbosity(optuna.logging.ERROR)

import os

# Для аннотаций
from typing import List, Any


# ----------------------------------------------------------------------------------------------
# ПАРАМЕТРЫ ЧТЕНИЯ ИСХОДНЫХ ДАННЫХ И РАЗМЕТКИ

DATA_DIR = 'data/new'

N_OMG_CH = 16         # количество каналов OMG-датчиков
OMG_COL_PRFX = 'omg'  # префикс в названиях столбцов датафрейма, соответствующих OMG-датчикам
STATE_COL = 'state'   # столбец с названием жеста, соответствующего команде
CMD_COL = 'id'        # столбец с меткой команды на выполнение жеста
TS_COL = 'ts'         # столбец метки времени
N_GESTS_IN_CYCLE = 14 # количество жестов в цикле протокола (включая нейтральные)

NOGO_STATE = 'Neutral'      # статус, обозначающий нейтральный жест
BASELINE_STATE = 'Baseline' # доп. служебный статус в начале монтажа
FINISH_STATE   = 'Finish'   # доп. служебный статус в конце монтажа

# Список с названиями всех столбцов OMG
OMG_CH = [OMG_COL_PRFX + str(i) for i in range(N_OMG_CH)]

# Новые столбцы:
SYNC_COL = 'sample'   # порядковый номер размеченного жеста
GROUP_COL = 'group'   # новый группы (цикла протокола)
TARGET = 'act_label'  # таргет (метка фактически выполняемого жеста)



# ----------------------------------------------------------------------------------------------
# РАЗМЕТКА НА ФАКТИЧЕСКИЕ ЖЕСТЫ

class BasePeakMarker(BaseEstimator, TransformerMixin):

    '''
    Класс-преобразователь для добавления в данные признака `"act_label"` – метки фактически выполняемого жеста.

    ### Параметры объекта класса
    
    **sync_col**: *str, default=SYNC_COL*<br>
    Найзвание столбца с порядковым номером жеста (в соответствии с поступившей командой)

    **label_col**: *str, default=LABEL_COL*<br>
    Название столбца с меткой жеста (в соответствии с поступившей командой)

    **ts_col**: *str, default=TS_COL*<br>
    Название столбца с меткой времени

    **omg_cols**

    **target_col_name**

    **hi_val_threshold**: *float, default=0.1*<br>
    Нижний порог медианы показаний датчика (в нормализованных измерениях) 
    для отнесения его к числу *активных* датчиков
    
    **sudden_drop_threshold**: *float, default=0.1*<br>
    Верхний порог отношения первого процентиля к медиане показаний датчика 
    для отнесения его к числу датчиков, которым свойственны внезапные резкие падения сигнала 
    до околонулевых значений (такие датчики игнорируются при выполнении разметки)

    **sync_shift**: *int, default=0*<br>
    Общий сдвиг меток синхронизации (признак `sync_col`)

    **bounds_shift_extra**: *int, default=0*<br>
    Корректировка (сдвиг) найденных границ

    **clean_w**: *int, default=5*<br>
    Параметр используемый при очистке найденных максимумов (пиков): 
    если два или более пиков находятся на расстоянии не более `clean_w` измерений друг от друга, 
    то из них оставляем один самый высокий пик

    **use_grad2**: *bool, default=True*<br>
    Если True - алгоитм разметки использует локальные максимумы (пики) суммарного второго градиента 
    показаний датчиков. Иначе - используется градиент суммарного стандартного отклонения

    ## Методы
    Данный класс реализует стандартые методы классов-преобразователей *scikit-learn*:

    `fit()`, `fit_transform()` и `transform()` и может быть использован как элемент пайплайна.
    '''

    def __init__(
            self,
            sync_col: str = SYNC_COL,
            cmd_col: str = CMD_COL,
            state_col: str = STATE_COL,
            nogo_state: str = NOGO_STATE,
            ts_col: str = TS_COL,
            omg_cols: str =  OMG_CH,
            target_col_name: str = TARGET,
            hi_val_threshold: float = 0.1,
            sudden_drop_threshold: float = 0.1,
            sync_shift: int = 0,
            bounds_shift_extra: int = 0,
            clean_w: int = 5,
            use_grad2: bool = True
        ):
        self.sync_col = sync_col
        self.cmd_col = cmd_col
        self.state_col = state_col
        self.nogo_state = nogo_state
        self.ts_col = ts_col
        self.omg_cols = omg_cols
        self.target_col_name = target_col_name
        self.hi_val_threshold = hi_val_threshold
        self.sudden_drop_threshold = sudden_drop_threshold
        self.sync_shift = sync_shift
        self.clean_w = clean_w
        self.use_grad2 = use_grad2
        self.bounds_shift = (-2 if self.use_grad2 else 3) + bounds_shift_extra


    # Внутренний метод для нахождения "пиков" (второго градиента, градиента станд. отклонения)
    def _find_peaks(
        self,
        X: np.ndarray,
        window: int,
        spacing: int,
    ):
        def _peaks(arr):
            mask = np.hstack([
                [False],
                (arr[: -2] < arr[1: -1]) & (arr[2:] < arr[1: -1]),
                [False]
            ])
            peaks = arr.copy()
            peaks[~mask] = 0
            peaks[peaks < 0] = 0
            return peaks
        
        def _clean_peaks(arr, w=self.clean_w):
            peaks = arr.copy()
            # Разберемся с пиками, расположенными близко друг к другу: 
            # из нескольких пиков, помещающихся в окне w, 
            # оставим только один - максимальный
            for i in range(peaks.shape[0] - w + 1):
                slice = peaks[i: i + w]
                max_peak = np.max(slice)
                mask = slice != max_peak
                peaks[i: i + w][mask] = 0
            return peaks

        # 1) Градиенты векторов показаний датчиков
        grad1 = np.sum(np.abs(np.gradient(X, spacing, axis=0)), axis=1)
        # не забудем заполнить образовавшиеся "дырки" из NaN
        grad1 = np.nan_to_num(grad1)

        grad2 = np.gradient(grad1, spacing, axis=0)
        grad2 = np.nan_to_num(grad2)
        peaks2 = _peaks(grad2)

        # 2) Среднее стандартное отклонение и его градиент
        std = np.mean(pd.DataFrame(X).rolling(window, center=True).std(), axis=1)
        std = np.nan_to_num(std)

        std1 = np.gradient(std, 1, axis=0)
        std1 = np.nan_to_num(std1)
        peaks_std1 = _peaks(std1)

        # Возвращяем пики градиента стандартного отклонения и второго градиента
        return _clean_peaks(peaks_std1), _clean_peaks(peaks2)

    # Функция для непосредственной разметки  
    def _mark(
        self,
        X: pd.DataFrame
    ) -> np.ndarray[int]:
        
        # Сохраним исходные индексы и сбросим на время разметки
        origin_index = X.index
        X = X.reset_index(drop=True)
        
        # Сглаживание
        X_omg = pd.DataFrame(X[self.mark_sensors]).rolling(self.window, center=True).median()
        # Приведение к единому масштабу
        X_omg = MinMaxScaler((1, 1000)).fit_transform(X_omg)

        peaks_std1, peaks_grad2 = self._find_peaks(
            X_omg,
            window=self.window,
            spacing=self.spacing
        )

        peaks = peaks_grad2 if self.use_grad2 else peaks_std1

        sync = X[self.sync_col].copy()
       
        # Сдвигаем синхронизацию
        if self.sync_shift > 0:
            sync_0 = sync.iloc[0]
            sync.iloc[self.sync_shift: ] = sync.iloc[: -self.sync_shift]
            sync.iloc[: self.sync_shift] = sync_0
        
        # Искать максимальные пики будем внутри отрезков, 
        # определяемых по признаку синхронизации
        sync_mask = sync != sync.shift(-1)
        sync_index = np.append([X.index[0]], X[sync_mask].index)

        labels = [int(X.loc[idx + 1, self.cmd_col]) for idx in sync_index[:-1]]

        bounds = np.array([])

        for l, r in zip(sync_index, sync_index[1:]):
            bounds = np.append(bounds, np.argmax(peaks[l: r]) + l)

        bounds += self.bounds_shift

        X_mrk = X.copy()

        # Теперь разметим каждое измерение в наборе фактическими метками
        X_mrk[self.target_col_name] = labels[0]
        for i, lr in enumerate(zip(bounds, np.append(bounds[1:], X_mrk.index[-1]))):
            l, r = lr
            # l, r - индексы начала текущего и следующего жестов соответственно
            X_mrk.loc[l: r, self.target_col_name] = labels[i]      

        X_mrk.index = origin_index

        return X_mrk


    def fit(self, X: pd.DataFrame, y=None):
        
        # 1. Определим параметры монтажа:

        grouped = X[X[self.state_col] != self.nogo_state].groupby(self.sync_col)
        # - периодичность измерений – разность между соседними метками времени
        ts_delta = np.median((X[self.ts_col].shift(-1) - X[self.ts_col]).value_counts().index)

        # - среднее кол-во измерений на один (не нейтральный) жест
        self.ticks_per_gest = int(
            grouped[self.ts_col].count().median()
        )

        # - среднее кол-во измерений на один разделительный нейтральный жест
        self.ticks_per_nogo = int(
            ((grouped[self.ts_col].first() - grouped[self.ts_col].first().shift(1)) / ts_delta).median() - self.ticks_per_gest
        )

        # 2. Определим датчики с высоким уровнем сигнала

        omg_medians = pd.DataFrame(
            MinMaxScaler().fit_transform(pd.DataFrame(X[self.omg_cols].median(axis=0))),
            index=self.omg_cols, columns=['omg']
        )
        self.hi_val_sensors = omg_medians[omg_medians['omg'] > self.hi_val_threshold].index.to_list()


        # 3. Исключим датчики с внезапными падениями сигнала 
        # (используем для этого заданный порог отношения первого процентиля к медиане)

        # По каждому из активных датчиков посчитаем отношение первого процентиля к медиане 
        q_to_med = pd.Series(
            X[self.hi_val_sensors].quantile(0.01) / X[self.hi_val_sensors].median(),
            index=self.hi_val_sensors
        )
        # Отфильтруем датчики по заданному порогу self.sudden_drop_threshold
        sudden_drop_sensors = q_to_med[q_to_med <= self.sudden_drop_threshold].index
        self.mark_sensors = [sensor for sensor in self.hi_val_sensors if sensor not in sudden_drop_sensors]


        # 4. Исключим датчики с перегрузкой

        # Сколько идущих подряд максимальных значений считать перегрузкой
        in_a_row_threshold = 5
        # Доля перегруженного сигнала, чтобы исключить датчик из определения границ
        clip_threshold = 0.05

        clip_sensors = []

        # Для каждого из рассматриваемых датчиков найдем его максимум
        for sensor in self.mark_sensors:
            mask = X[sensor] == X[sensor].max()
            in_a_row = []
            cur = 0
            for x in mask:
                if not x:
                    if cur >= in_a_row_threshold:
                        in_a_row.append(cur)
                    cur = 0
                else:
                    cur += 1
            if cur  >= in_a_row_threshold:
                in_a_row.append(cur)
            if sum(in_a_row) / X.shape[0] > clip_threshold:
                clip_sensors.append(sensor)

        if len(clip_sensors):
            self.mark_sensors = [sensor for sensor in self.mark_sensors if sensor not in clip_sensors]


        # Теперь у нас готов список датчиков self.mark_sensors, по которым мы и будем определять границы жестов

        # Установим ширину окна (для сглаживания медианой)
        self.window = self.ticks_per_gest // 3
        # и параметр spacing для вычисления градиентов
        self.spacing = self.ticks_per_gest // 3

        return self
    
    def transform(self, X):
        if self.cmd_col in X.columns:
            return self._mark(X)
        else:
            return X.copy()
        

def read_emg8(
        montage: str, 
        dir: str = DATA_DIR, 
        sep: str = ' ',
        feature_cols: List[str] = OMG_CH,
        cmd_col: str = CMD_COL,
        state_col: str = STATE_COL,
        ts_col: str = TS_COL,
        sync_col_name: str = SYNC_COL,
        group_col_name: str = GROUP_COL,
        n_gests_in_group: int = N_GESTS_IN_CYCLE,
        states_to_drop: list = [BASELINE_STATE, FINISH_STATE],
        target_col_name: str = TARGET,
        n_holdout_groups: int = 0
        ) -> List[pd.DataFrame | pd.Series | None]:
    '''
    Осуществляет чтение файла с данными измерений монтажа .emg8.

    Перенумеровывает классы жестов в порядке их следования по протоколу монтажа.

    Добавляет в возвращаемый датафрейм признак `sample`, представляющий собой порядковый номер жеста в монтаже.

    ### Параметры

    **montage**: *str*<br>
    Имя файла для чтения

    **dir**: *str, default="data"*<br>
    Название папки, в которой находится файл

    **sep**: *str, default=' '*<br>
    Символ-разделитель в csv-файле

    **feature_cols**: *List[str], default=OMG_COLS*

    **cmd_col**: *str, default=CMD_COL*

    **ts_col**: *str, default=TS_COL*

    **sync_col_name**: *str, default=SYNC_COL*

    **drop_baseline_and_finish**: *bool, default=True*<br>
    Удалять ли в начале и в конце монтажа измерения с метками `Baseline` и `Finish` соответственно

    **mark_up**: *bool, default=True*

    **target_col_name**: *str, default=TARGET*

    **n_holdput_groups**: *int, default=0*

    ### Возвращаемый результат

    Список: **X_train**, **X_test** | None, **y_train**, **y_train** | None, **data**
    '''
    path = os.path.join(dir, montage)
    cols = feature_cols + [cmd_col, state_col, ts_col]

    data_origin = pd.read_csv(path, sep=sep, index_col=None)

    # Удалим примеры с указанными статусами 
    # (имеются ввиду служебные дополнительные статусы 'Baseline' и 'Finish')
    if not states_to_drop is None:
        mask = ~data_origin[state_col].isin(states_to_drop)
        data_origin = data_origin[mask]

    # Перенумеруем классы жестов в порядке их появления в монтаже
    states_ordered = data_origin[state_col].drop_duplicates().reset_index(drop=True)
    states_ordered = pd.Series(states_ordered.index, index=states_ordered)
    data_origin[cmd_col] = data_origin[state_col].apply(lambda state: states_ordered[state])

    # Далее работаем только с необходимыми столбцами
    data = data_origin.copy()[cols]
    
    # Добавим признак порядкового номера эпох (требуется для алгоритма разметки)
    bounds = data[data[cmd_col] != data[cmd_col].shift(1)].index
    for i, lr in enumerate(zip(bounds, np.append(bounds[1:], [data.index[-1]]))):
        l, r = lr # l, r - индексы начала текущей и следующей эпох соответственно
        data.loc[l: r - 1, sync_col_name] = i

    # Выполним разметку жестов
    marker = BasePeakMarker(
        sync_col=sync_col_name, 
        cmd_col=cmd_col, 
        ts_col=ts_col, 
        omg_cols=feature_cols,
        target_col_name=target_col_name
    )
    data = marker.fit_transform(data)

    # Перепишем признак sync_col (порядковый номер жеста в монтаже), 
    # чтобы он соответствовал разметке по фактическим границам
    bounds = data[data[target_col_name] != data[target_col_name].shift(1)].index
    for i, lr in enumerate(zip(bounds, np.append(bounds[1:], [data.index[-1]]))):
        l, r = lr # l, r - индексы начала текущей и следующей эпох соответственно
        data.loc[l: r - 1, sync_col_name] = i

    # Теперь проставим номера групп (циклов протокола) – понадобится при кросс-валидации
    group_idx = list(data[[target_col_name, sync_col_name]].drop_duplicates().index[::n_gests_in_group])
    # [последний индекс "дотянем" до конца]
    group_idx[-1] = data.index[-1] + 1
    for i, lr in enumerate(zip(group_idx, group_idx[1:])):
        l, r = lr
        data.loc[l: r - 1, group_col_name] = i
    n_groups = i


    X = data[feature_cols].copy()
    y = data[target_col_name].copy()

    if n_holdout_groups > 0:
        if n_holdout_groups >= n_groups:
            raise ValueError(f"Количество отложенных групп n_holdput_groups={n_holdout_groups} должно быть меньше общего количества групп {n_groups + 1}")
        last_train_idx = group_idx[-n_holdout_groups - 1] - 1
        X_train = X.loc[: last_train_idx, :].to_numpy()
        X_test = X.loc[last_train_idx + 1: , :].to_numpy()
        y_train = y.loc[: last_train_idx].to_numpy()
        y_test = y.loc[last_train_idx + 1: ].to_numpy()
        train_groups = data.loc[: last_train_idx, group_col_name].to_numpy()
    else:
        X_train = X.to_numpy()
        X_test = None
        y_train = y.to_numpy()
        y_test = None
        train_groups = data[group_col_name].to_numpy()

    data_origin[target_col_name] = data[target_col_name]
    data_origin[sync_col_name] = data[sync_col_name]
    data_origin[group_col_name] = data[group_col_name]

    return X_train, X_test, y_train, y_test, data_origin, train_groups


# ----------------------------------------------------------------------------------------------
# ПРЕОБРАЗОВАНИЕ И СОЗДАНИЕ ПРИЗНАКОВ

# Самый первый обработчик: переводит одномерный вектор признаков (когда нужно 
# предсказать один пример) в двумерный массив

class FixOneDimSample(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape((1, -1))
        return X

# Базовый класс для преобразования данных с помощью скользящего окна
class BaseSlidingProc(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            n_lags: int = 3,
            oper: str = 'replace' # 'add'
        ):
        self.n_lags = n_lags
        self.oper = oper

    # По умолчанию для всех дочерних классов считаем, 
    # что их работа не вносит задержку при работе в реальном времени.
    def get_realtime_shift(self):
        return 0

    def fit(self, X, y=None):
        self.n_ftr = X.shape[1]
        self.X_que = np.empty((0, self.n_ftr))
        return self
    

    # В дочерних классах данных внутренний метод 
    # должен реализовавыть фактическую обработку
    def _proc_func(self, X_sld, X):
        # В качестве заглушки просто возвращаем последний (текущий) пример
        return X_sld[:, -1]
    

    def transform(self, X):

        X = np.array(X)

        if self.n_lags < 2:
            return X

        # Если хотят предсказать один пример, 
        # все равно переводим его в двумерный массив
        #if X.ndim == 1:
            #X = X.reshape((1, self.n_ftr))
        
        # Если наш объект получает данные впервые,
        if self.X_que.shape[0] == 0:
            # накопируем первый пример нужное число раз
            self.X_que = np.tile(X[0], (self.n_lags - 1, 1))

        self.X_que = np.vstack((self.X_que, X))

        # Скользящее окно по двумерному массиву признаков 
        X_sld = sliding_window_view(self.X_que, (self.n_lags, self.n_ftr))[:, 0]

        X_proc = self._proc_func(X_sld, X)

        # В зависимости от значение oper 
        if self.oper == 'add':
            # добавляем новый признак к основному набору
            X_res = np.hstack((X, X_proc))
        else: # (self.oper == 'replace')
            # либо заменяем исходные признаки новыми значениями
            X_res = X_proc

        # Запомним в нашем объекте лаги только для будущего нового примера
        self.X_que = self.X_que[-self.n_lags + 1: ]

        return X_res
    
    # Для преобразования обучающей выборки (fit_transform)
    # нам нельзя вносить задержку во временные ряды
    def fit_transform(self, X, y=None):
        self.fit(X)
        X_proc = self.transform(X)
        self.X_que = np.empty((0, self.n_ftr))
        shift = self.get_realtime_shift()
        if shift == 0:
            return X_proc
        return np.vstack((X_proc[shift:, :], X[-shift:, :]))


# Преобразователь для сглаживания ("срезания"  выбросов)
class NoiseReduction(BaseSlidingProc):

    def __init__(
            self,
            n_lags: int = 3,
            avg: str = 'median' # 'mean'
        ):
        super().__init__(n_lags=n_lags, oper='replace')
        self.avg = avg

    def get_realtime_shift(self):
        return self.n_lags // 2

    def _proc_func(self, X_sld, X):
        
        np_avg_func = {
            'mean': np.mean,
            'median': np.median
            }[self.avg]
        X_avg = np_avg_func(X_sld, axis=(1,))
        
        return X_avg
    

# Обозачение тренда (нахождение разности с предыдущими значениями)
class AddDifference(BaseSlidingProc):

    def __init__(
            self,
            n_lags: int = 3,
            avg: str = 'mean', # 'median'
            n_features: int = N_OMG_CH
        ):
        super().__init__(n_lags=n_lags, oper='add')
        self.avg = avg
        self.n_features = n_features


    def _proc_func(self, X_sld, X):
        
        np_avg_func = {
            'mean': np.mean,
            'median': np.median
            }[self.avg]
        X_prev_avg = np_avg_func(X_sld[:, :-1, :self.n_features], axis=(1,))
        # Результат – разность признаков и их средних предыдущих значений
        return X[:, :self.n_features] - X_prev_avg
    


class AvgDiff(BaseEstimator, TransformerMixin):

    def __init__(self, scaler=None, avg='mean'):
        self.scaler = scaler
        self.avg = avg


    def fit(self, X, y=None):
        if not self.scaler is None:
            self.scaler.fit(X)
        return self
    
    
    def transform(self, X):
        if not self.scaler is None:
            X = self.scaler.transform(X)
        else:
            X = np.array(X)
        np_func = {
            'mean': np.mean,
            'median': np.median
        }[self.avg]
        X_avg = np_func(X, axis=1).reshape((-1, 1))
        return np.hstack([X, X - X_avg])
    

# ----------------------------------------------------------------------------------------------
# ОБЕРТКА МОДЕЛИ С ПОСТОБРАБОТКОЙ ПРЕДСКАЗАНИЙ

class PostprocWrapper(BaseEstimator, TransformerMixin):

    def __init__(
            self,
            estimator = LogisticRegression(C=10, max_iter=5000),
            n_lags: int = 5
        ):
        self.estimator = estimator
        self.n_lags = n_lags
        self.y_que = np.empty((0,))

        self.classes_ = []

    def get_realtime_shift(self):
        return self.n_lags - 1

    def fit(self, X, y):
        # Обучаем модель на лаговых признаках. 
        # При этом первые w - 1 примеров исходных данных пропускаются 
        # и в обучении не участвуют 
        self.estimator.fit(X, y)
        # Скопируем атрибут модели .classes_ в соответсвутующий атрибут обёртки
        try:
            self.classes_ = self.estimator.classes_
        except AttributeError:
            self.classes_ = pd.Series(y).unique().tolist()
        return self
    
    # В дочерних классах данных внутренний метод 
    # должен реализовавыть фактическую постобработку
    def _proc_func(self, yy):
        # В базовой версии возвращаем моду
        mode_res, _ = stats.mode(yy)
        return mode_res

    def predict(self, X):

        y_pred = self.estimator.predict(X)

        if self.n_lags < 2:
            return y_pred
        
        # Если наш объект-обёртка получает данные впервые,
        if self.y_que.shape[0] == 0:
            # накопируем первый ответ нужное кол-во раз
            self.y_que = np.tile(y_pred[0], self.n_lags - 1)

        self.y_que = np.hstack((self.y_que, y_pred))

        y_pred_proc = np.array(
            [self._proc_func(yy) for yy in sliding_window_view(self.y_que, self.n_lags)]
        )

        # Запомним в нашем объекте лаги только для будущего нового предсказания
        self.y_que = self.y_que[-self.n_lags + 1: ]

        return y_pred_proc
    
    def predict_proba(self, X):
        return self.estimator.predict_proba(X)
    
    def set_params(self, **params):

        self.y_que = np.empty(0)
        # Все именованные аргументы (параметры) кроме перечисленных
        # предназначены для оборачиваемой модели
        wrapper_params = ['n_lags']

        params_for_estimator = dict()
        params_for_self = dict()
        for param in params:
            if param in wrapper_params:
                params_for_self[param] = params[param]
            else:
                params_for_estimator[param] = params[param]

        self.estimator.set_params(**params_for_estimator)
        return super().set_params(**params_for_self)


# ----------------------------------------------------------------------------------------------
# ПАЙПЛАЙНЫ

MAX_TOTAL_SHIFT = 8

from sklearn.pipeline import Pipeline


# Получить суммарный лаг пайплайна (в количестве измерений)
def get_total_shift(pipeline: Pipeline):
    total_shift = 0
    for i in range(len(pipeline)):
        try:
            total_shift += pipeline[i].get_realtime_shift()
        except AttributeError:
            continue
    return total_shift

# Пайплайн на базе логистической регрессии
def create_logreg_pipeline( 
        X, y, groups=None,
        optimize_and_fit: bool = False,
        max_total_shift: int = MAX_TOTAL_SHIFT,
        n_trials: int = 100
    ):

    pl = Pipeline([
        ('fix_1dim_sample', FixOneDimSample()),
        ('noise_reduct', NoiseReduction(3)),
        #('average_diff', AvgDiff(scaler=RobustScaler())),
        ('add_diff', AddDifference(5)),
        ('scaler', MinMaxScaler()),
        ('model', PostprocWrapper(estimator=LogisticRegression(C=10, max_iter=5000)))
    ])

    def opt_func(trial: optuna.Trial, X=X, y=y, groups=groups, pl=pl, max_total_shift=max_total_shift):

        params = {
            'noise_reduct__n_lags': trial.suggest_int('noise_reduct__n_lags', 1, 7),

            #'average_diff__avg':        trial.suggest_categorical('average_diff__avg', ['mean', 'median']),

            'add_diff__n_lags':     trial.suggest_int('add_diff__n_lags', 2, 7),
            'add_diff__avg':        trial.suggest_categorical('add_diff__avg', ['mean', 'median']),

            'model__n_lags':        trial.suggest_int('model__n_lags', 3, 7, step=2),

            'model__C':             trial.suggest_int('model__C', 1, 50, log=True)
        }
        pl.set_params(**params)

        total_shift = get_total_shift(pl)
    
        if total_shift == 0:
            y_shifted = y
        else:
            total_shift = min(total_shift, max_total_shift)
            y_shifted = np.hstack((np.tile(y[0], total_shift), y[: -total_shift]))

        # Если переданы группы, то будем проводить кросс-валидацию 
        # с помощью StratifiedGroupKFold
        if not groups is None:
            n_groups = int(np.max(groups) + 1)
            kf = StratifiedGroupKFold(n_groups)
            scores = []
            for index_train, index_valid in kf.split(X, y, groups=groups):
                X_train = X[index_train]
                y_train = y_shifted[index_train]
                X_valid = X[index_valid]
                y_valid = y_shifted[index_valid]
                pl.fit(X_train, y_train)
                y_pred = pl.predict(X_valid)
                scores.append(f1_score(y_valid, y_pred, average='macro'))
            return np.mean(scores)
        # Если же группы не переданы, то используем 
        # обычную кроссвалидацию без перемешивания
        else: # (groups is None)
            return cross_val_score(pl, X, y_shifted, scoring='f1_macro').mean()

    
    if optimize_and_fit:
        study = optuna.create_study(direction='maximize')
        study.optimize(opt_func, n_trials=n_trials, show_progress_bar=True)
        pl.set_params(**study.best_params)
    pl.fit(X, y)
    
    return pl