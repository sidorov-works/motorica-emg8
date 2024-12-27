##  **Система распознавания жестов на основе OMG-данных**


### Классы, выполняющие разметку монтажей по фактическому выполнению жестов – в модуле [motorica.emg8.markers](https://github.com/sidorov-works/motorica-emg8/blob/main/motorica/emg8/markers.py):
- `BaseMarker` – по границам начала выполнения жеста, без выделения переходов
- `FullMarker` – переходы и жесты (метки от 0 до 7 – жесты, от 11 до 17 – переходы от нейтрали к жесту, от -1 до -7 – переходы от жеста к нейстрали)
- `TransMarker` – размечает только переходы

### Классы и функции для пайплайна – в модуле [motorica.emg8.pipeline](https://github.com/sidorov-works/motorica-emg8/blob/main/motorica/emg8/pipeline.py):
- `read_emg8()` – функция чтения, разметки и разделения данных на train/test (использует классы для разметки из модуля motorica.emg8.markers.py)
- `NoiseReduction` – сглаживание, срезание пиков (рекомендуется использовать значение n_lags от 3 до 5)
- `DiffWithPrev` – устранение тренда (n_lags от 100 до 150 работает хорошо на имеющихся данных)
- `CutOutliers` – срезание (клиппирование) выбросов (адекватно работает при lo=0.03 и up=0.97)

### Демонстрация в инференсе
Краткий демо-инференс (по шаблону заказчика) – в [ноутбуке](https://github.com/sidorov-works/motorica-emg8/blob/main/baseline_logreg_short.ipynb). Более подробный – [тут](https://nbviewer.org/github/sidorov-works/motorica-emg8/blob/main/baseline_logreg_full.ipynb)

### Презентация по докладу
[Презентация](https://docs.google.com/presentation/d/e/2PACX-1vSaOOL_UG4ZYO5E79aW-opP67UGDhKsby2HMCZQJNe04w5kp8_2bd-uKn7WBMemV4nOUaPQcHZ8XWWt/pub?start=false&loop=false&delayms=15000) для доклада по результатам стажировки.
