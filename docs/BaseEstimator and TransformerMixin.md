## Наследование от BaseEstimator и TransformerMixin

### Зачем это нужно?

При разработке собственных классов для обработки данных в экосистеме scikit-learn, наследование от `BaseEstimator` и `TransformerMixin` дает важные преимущества.

### BaseEstimator - обязательная основа

Наследование от `BaseEstimator` делает класс "совместимым" с scikit-learn:

**Автоматические методы `get_params()` и `set_params()`**
```python
class MyTransformer(BaseEstimator):
    def __init__(self, param1=10, param2='mean'):
        self.param1 = param1
        self.param2 = param2

# Автоматически работают:
transformer.get_params()  # → {'param1': 10, 'param2': 'mean'}
transformer.set_params(param1=20)  # → меняет параметр
```

**Возможность использования в Pipeline**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('my_transformer', MyTransformer()),  # Работает!
    ('classifier', LogisticRegression())
])
```

**Поддержка GridSearchCV и Optuna**
```python
# Подбор гиперпараметров
params = {
    'my_transformer__param1': [5, 10, 15],
    'my_transformer__param2': ['mean', 'median']
}
grid_search = GridSearchCV(pipeline, params)  # Работает!
```

**Клонирование через `clone()`**
```python
from sklearn.base import clone

new_transformer = clone(my_transformer)  # Работает!
```

### TransformerMixin - для преобразователей данных

Если класс **преобразует данные** (трансформер), добавляется `TransformerMixin`:

**Бесплатный метод `fit_transform()`**
```python
class MyTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X * 2

# Теперь доступно:
X_transformed = transformer.fit_transform(X)  # Один вызов вместо двух
```

**Совместимость с Pipeline**
```python
# Pipeline ожидает у трансформеров метод fit_transform()
Pipeline([
    ('scaler', MinMaxScaler()),        # есть fit_transform
    ('my_trans', MyTransformer()),     # теперь тоже есть!
    ('model', LogisticRegression())
])
```

### Практический пример

```python
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class NoiseReduction(BaseEstimator, TransformerMixin):
    """Сглаживание шума в данных"""
    
    def __init__(self, window_size=3, method='median'):
        self.window_size = window_size
        self.method = method
    
    def fit(self, X, y=None):
        # Вычисляем параметры для transform
        return self
    
    def transform(self, X):
        # Основная логика преобразования
        if self.method == 'median':
            return np.median(X, axis=0)
        else:
            return np.mean(X, axis=0)

# Использование:
transformer = NoiseReduction(window_size=5, method='mean')

# Работает как в sklearn:
X_train_transformed = transformer.fit_transform(X_train)
X_test_transformed = transformer.transform(X_test)

# Работает в Pipeline:
pipeline = Pipeline([
    ('noise', transformer),
    ('classifier', LogisticRegression())
])
```

### Важные правила

**Метод `fit()` должен возвращать `self`**
```python
def fit(self, X, y=None):
    # ... вычисления
    return self  # ОБЯЗАТЕЛЬНО!
```

**Имена параметров в `__init__()` должны совпадать с атрибутами**
```python
def __init__(self, param1=10):  # имя параметра
    self.param1 = param1        # имя атрибута (должно совпадать!)
```

**TransformerMixin - только для трансформеров**
- Используется если есть метод `transform()`
- Не используется для estimators (классификаторов, регрессоров)

### Итог

| Компонент | Когда использовать | Что дает |
|-----------|-------------------|----------|
| `BaseEstimator` | Всегда для пользовательских классов | `get_params()`, `set_params()`, совместимость с sklearn |
| `TransformerMixin` | Для классов с методом `transform()` | `fit_transform()`, совместимость с Pipeline |

Все классы в этом проекте корректно наследуются от обоих базовых классов, что обеспечивает их полную интеграцию с экосистемой scikit-learn.