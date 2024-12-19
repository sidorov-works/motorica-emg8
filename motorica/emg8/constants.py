# ----------------------------------------------------------------------------------------------
# ПАРАМЕТРЫ ЧТЕНИЯ ИСХОДНЫХ ДАННЫХ И РАЗМЕТКИ

DATA_DIR = 'data/new'

N_OMG_CH = 16         # количество каналов OMG-датчиков
OMG_COL_PRFX = 'omg'  # префикс в названиях столбцов датафрейма, соответствующих OMG-датчикам
STATE_COL = 'state'   # столбец с названием жеста, соответствующего команде
CMD_COL = 'id'        # столбец с меткой команды на выполнение жеста
TS_COL = 'ts'         # столбец метки времени


# Используемые компанией статусы для обозначения команд жестов:

BASELINE_STATE = 'Baseline'      # доп. служебный статус в начале монтажа
FINISH_STATE   = 'Finish'        # доп. служебный статус в конце монтажа

NOGO_STATE = 'Neutral'           # нейтральный жест
THUMB_STATE = 'ThumbFingers'     #
CLOSE_STATE = 'Close'            #
OPEN_STATE = 'Open'              #
PINCH_STATE = 'Pinch'            #
INDICATION_STATE = 'Indication'  #
FLEX_STATE = 'Wrist_Flex'        #
EXTEND_STATE = 'Wrist_Extend'    #

STABLE_STATE = 'Stable'


# Список с названиями всех столбцов OMG
OMG_CH = [OMG_COL_PRFX + str(i) for i in range(N_OMG_CH)]

# Новые столбцы:
SYNC_COL = 'sample'   # порядковый номер размеченного жеста
GROUP_COL = 'group'   # новый группы (цикла протокола)
TARGET = 'act_label'  # таргет (метка фактически выполняемого жеста)
