# shared/emg8/config.py

from typing import List
from pathlib import Path

class Config:

    DATA_ROOT = Path("data")

    # Исходные названия столбцов в файлах с измерениями (*.emg8)
    OMG_CH_CNT = 16       # количество каналов OMG-датчиков
    OMG_COL_PRFX = 'omg'  # префикс в названиях столбцов датафрейма, соответствующих OMG-датчикам
    STATE_COL = 'state'   # столбец с названием жеста, соответствующего команде
    CMD_COL = 'id'        # столбец с меткой команды на выполнение жеста
    TS_COL = 'ts'         # столбец метки времени


    # Статусы для обозначения команд жестов:

    BASELINE_STATE = 'Baseline'      # доп. служебный статус в начале монтажа
    FINISH_STATE   = 'Finish'        # доп. служебный статус в конце монтажа

    NOGO_STATE = 'Neutral'           # нейтральный жест
    THUMB_STATE = 'ThumbFingers'     #
    CLOSE_STATE = 'Close'            # сжать кулак
    OPEN_STATE = 'Open'              # раскрыть ладонь (разжать все пальцы)
    PINCH_STATE = 'Pinch'            #
    INDICATION_STATE = 'Indication'  # указательный палец
    FLEX_STATE = 'Wrist_Flex'        # согнуть запястье
    EXTEND_STATE = 'Wrist_Extend'    # выгнуть запястье

    # Статус для альтернативного подхода к разметке, при котором нас интересуют только переходы, 
    # а сами жесты считаются за один класс
    STABLE_STATE = 'Stable'          # любой жест (стабильное состояние, переход не выполняется)

    # Список с названиями всех столбцов OMG
    @property
    def OMG_CH(self) -> List[str]:
        return [self.OMG_COL_PRFX + str(i) for i in range(self.OMG_CH_CNT)]

    # Новые столбцы:
    SYNC_COL = 'sample'   # порядковый номер размеченного жеста
    GROUP_COL = 'group'   # порядковый номер цикла протокола
    TARGET = 'act_label'  # таргет (метка фактически выполняемого жеста)

    # Пороговое значение метрики f1-macro, полученной в результате кросс-валидации, 
    # выше которого монтаж принимается годным для дальнейшего использования
    CV_SCORE_OK = 0.85

config = Config()
