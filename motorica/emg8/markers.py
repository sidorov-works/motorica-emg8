# Работа с табличными данными
import pandas as pd
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import scipy.stats as stats

# Преобразование признаков
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

# Пайплайн
from sklearn.base import BaseEstimator, TransformerMixin

import os

# Для аннотаций
from typing import List, Any

from motorica.emg8.constants import *


class BasePeakMarker(BaseEstimator, TransformerMixin):

    """
    Класс-преобразователь для добавления в данные метки фактически выполняемого жеста.

    ### Параметры объекта класса
    
    **sync_col**: *str, default=SYNC_COL*<br>
    Найзвание столбца с порядковым номером жеста (в соответствии с поступившей командой)

    **label_col**: *str, default=LABEL_COL*<br>
    Название столбца с меткой жеста (в соответствии с поступившей командой)

    ts_col : str, default=TS_COL
        Название столбца с меткой времени

    omg_cols : List[str]
        Список названий столбцов с показанями датчиков

    target_col_name : str, default=TARGET
        Название столбца для таргета

    hi_val_threshold :  float, default=0.1
        Нижний порог медианы показаний датчика (в нормализованных измерениях) 
    для отнесения его к числу *активных* датчиков
    
    sudden_drop_threshold : float, default=0.1
        Верхний порог отношения первого процентиля к медиане показаний датчика 
        для отнесения его к числу датчиков, которым свойственны внезапные резкие падения сигнала 
        до околонулевых значений (такие датчики игнорируются при выполнении разметки)

    sync_shift : int, default=0
        Общий сдвиг меток синхронизации (признак `sync_col`) перед поиском границ

    bounds_shift_extra : int, default=0
        Корректировка (сдвиг) найденных границ

    clean_w : int, default=5
        Параметр используемый при очистке найденных максимумов (пиков): 
        если два или более пиков находятся на расстоянии не более `clean_w` измерений друг от друга, 
        то из них оставляем один самый высокий пик

    use_grad2 : bool, default=True
        Если True - алгоитм разметки использует локальные максимумы (пики) 
        суммарного второго градиента показаний датчиков. 
        Иначе - используется градиент суммарного стандартного отклонения

    ## Методы
    Данный класс реализует стандартые методы классов-преобразователей scikit-learn:
    `fit()`, `fit_transform()` и `transform()` и может быть использован как элемент пайплайна.
    """

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

        self.peaks_grad2 = _clean_peaks(_peaks(grad2))
        self.peaks_grad2_neg = -_clean_peaks(_peaks(-grad2))


        # 2) Среднее стандартное отклонение и его градиент
        std = np.mean(pd.DataFrame(X).rolling(window, center=True).std(), axis=1)
        std = np.nan_to_num(std)

        std1 = np.gradient(std, 1, axis=0)
        std1 = np.nan_to_num(std1)
        self.peaks_std1 = _peaks(std1)
        self.peaks_std1_neg = -_clean_peaks(_peaks(-std1))

        # Cохраним в атрибутах объекта:
        # self.grad1 = grad1
        # self.grad2 = grad2
        # self.std = std
        # self.std1 = std1

        # Возвращаем пики градиента стандартного отклонения и второго градиента
        # return _clean_peaks(peaks_std1), _clean_peaks(peaks2)

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

        self._find_peaks(
            X_omg,
            window=self.window,
            spacing=self.spacing
        )

        peaks = self.peaks_grad2 if self.use_grad2 else self.peaks_std1

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
        self.window = self.ticks_per_gest // 5
        # и параметр spacing для вычисления градиентов
        self.spacing = self.ticks_per_gest // 5

        return self
    
    def transform(self, X):
        if self.cmd_col in X.columns:
            return self._mark(X)
        else:
            return X.copy()
