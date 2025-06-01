import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
import os


class DataLoader:
    """Класс для загрузки и подготовки данных"""

    def __init__(self, file_path=None):
        self.file_path = file_path
        self.data = None

    def load_from_csv(self):
        """Загрузка данных из CSV файла"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Файл {self.file_path} не найден")
        self.data = pd.read_csv(self.file_path)
        return self

    def get_features_target(self, target_column='price'):
        """Разделение на признаки и целевую переменную"""
        if self.data is None:
            raise ValueError("Данные не загружены")
        X = self.data.drop(target_column, axis=1)
        y = self.data[target_column]
        return X, y


class DataPreprocessor:
    """Класс для предварительной обработки данных"""

    def __init__(self, numeric_features, categorical_features):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.preprocessor = None

    def create_preprocessor(self):
        """Создание пайплайна для предобработки данных"""
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), self.numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ])
        return self



class PricePredictor:
    """Основной класс для прогнозирования цен"""

    def __init__(self, model_path='price_predictor.joblib'):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None

    def train(self, X, y):
        """Обучение модели"""
        # Определяем типы признаков
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()

        # Создаем и настраиваем предобработчик
        preprocessor = DataPreprocessor(numeric_features, categorical_features).create_preprocessor()
        self.preprocessor = preprocessor.preprocessor

        # Создаем и обучаем модель
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=42))
        ])

        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Прогнозирование цен"""
        if self.model is None:
            raise ValueError("Модель не обучена")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test):
        """Оценка качества модели"""
        if self.model is None:
            raise ValueError("Модель не обучена")
        y_pred = self.predict(X_test)
        return {
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
            'R2': r2_score(y_test, y_pred)
        }

    def save_model(self):
        """Сохранение модели на диск"""
        if self.model is None:
            raise ValueError("Модель не обучена")
        dump(self.model, self.model_path)

    def load_model(self):
        """Загрузка модели с диска"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл модели {self.model_path} не найден")
        self.model = load(self.model_path)
        return self




class PredictionManager:
    """Класс для управления прогнозами"""

    def __init__(self, predictor):
        self.predictor = predictor
        self.predictions = []

    def make_prediction(self, input_data):
        """Создание прогноза"""
        if not isinstance(input_data, pd.DataFrame):
            input_data = pd.DataFrame([input_data])

        prediction = self.predictor.predict(input_data)[0]
        result = {
            'input': input_data.iloc[0].to_dict(),
            'prediction': prediction,
            'timestamp': pd.Timestamp.now()
        }
        self.predictions.append(result)
        return result

    def get_predictions_history(self):
        """Получение истории прогнозов"""
        return pd.DataFrame(self.predictions)

    def save_predictions_to_csv(self, file_path):
        """Сохранение прогнозов в CSV"""
        if not self.predictions:
            raise ValueError("Нет данных для сохранения")
        df = self.get_predictions_history()
        df.to_csv(file_path, index=False)



if __name__ == "__main__":
    try:
        # 1. Загрузка данных
        loader = DataLoader('csv_data.csv')
        X, y = loader.load_from_csv().get_features_target()

        # 2. Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 3. Обучение модели
        predictor = PricePredictor()
        predictor.train(X_train, y_train)

        # 4. Оценка модели
        evaluation = predictor.evaluate(X_test, y_test)
        print(f"Оценка модели: RMSE={evaluation['RMSE']:.2f}, R2={evaluation['R2']:.2f}")

        # 5. Сохранение модели
        predictor.save_model()

        # 6. Пример прогнозирования
        manager = PredictionManager(predictor)

        # Прогноз 1
        new_data_1 = {
            'count': 5000,
            'add_cost': 3000,
            'company': 'Tesla',
            'product': 'Galaxy'
        }
        result_1 = manager.make_prediction(new_data_1)
        print(f"Прогноз 1: {result_1['prediction']:.2f}")

        # Прогноз 2
        new_data_2 = {
            'count': 300,
            'add_cost': 2000,
            'company': 'Apple',
            'product': 'iPad'
        }
        result_2 = manager.make_prediction(new_data_2)
        print(f"Прогноз 2: {result_2['prediction']:.2f}")

        # 7. Сохранение истории прогнозов
        manager.save_predictions_to_csv('predictions_history.csv')

        # 8. Загрузка модели и повторное использование
        loaded_predictor = PricePredictor().load_model()
        new_prediction = loaded_predictor.predict(pd.DataFrame([new_data_1]))[0]
        print(f"Прогноз с загруженной моделью: {new_prediction:.2f}")

    except Exception as e:
        print(f"Ошибка: {str(e)}")

import matplotlib.pyplot as plt


class PredictionVisualizer:
    """Класс для визуализации результатов"""

    @staticmethod
    def plot_predictions_vs_actual(y_true, y_pred, title='Прогноз vs Факт'):
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--')
        plt.xlabel('Фактические значения')
        plt.ylabel('Прогнозируемые значения')
        plt.title(title)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_feature_importance(model, feature_names, top_n=10):
        """Визуализация важности признаков"""
        if hasattr(model.named_steps['regressor'], 'feature_importances_'):
            importances = model.named_steps['regressor'].feature_importances_
            indices = np.argsort(importances)[-top_n:]

            plt.figure(figsize=(10, 6))
            plt.title('Важность признаков')
            plt.barh(range(len(indices)), importances[indices], align='center')
            plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
            plt.xlabel('Относительная важность')
            plt.show()


# После обучения модели
visualizer = PredictionVisualizer()

# Визуализация прогнозов vs фактические значения
y_pred = predictor.predict(X_test)
visualizer.plot_predictions_vs_actual(y_test, y_pred)

# Визуализация важности признаков
# Получаем имена признаков после преобразования
preprocessor = predictor.model.named_steps['preprocessor']
numeric_features = preprocessor.transformers_[0][2]
categorical_features = preprocessor.transformers_[1][2]
cat_encoder = preprocessor.named_transformers_['cat']
cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
all_feature_names = numeric_features + list(cat_feature_names)

visualizer.plot_feature_importance(predictor.model, all_feature_names)



class InputValidator:
    """Класс для валидации входных данных"""

    @staticmethod
    def validate_prediction_input(input_data):
        required_fields = ['count', 'add_cost', 'company', 'product']
        for field in required_fields:
            if field not in input_data:
                raise ValueError(f"Отсутствует обязательное поле: {field}")

        if not isinstance(input_data['count'], int) or input_data['count'] <= 0:
            raise ValueError("Количество должно быть положительным целым числом")

        if not (isinstance(input_data['add_cost'], (int, float))) or input_data['add_cost'] < 0:
            raise ValueError("Затраты на продвижение должны быть положительным числом")

        if not isinstance(input_data['company'], str) or not input_data['company'].strip():
            raise ValueError("Название компании должно быть непустой строкой")

        if not isinstance(input_data['product'], str) or not input_data['product'].strip():
            raise ValueError("Название продукта должно быть непустой строкой")

        return True





def make_prediction(self, input_data):
    """Создание прогноза с валидацией входных данных"""
    InputValidator.validate_prediction_input(input_data)

    if not isinstance(input_data, pd.DataFrame):
        input_data = pd.DataFrame([input_data])

    prediction = self.predictor.predict(input_data)[0]
    result = {
        'input': input_data.iloc[0].to_dict(),
        'prediction': prediction,
        'timestamp': pd.Timestamp.now()
    }
    self.predictions.append(result)
    return result

"""
Эта
реализация
демонстрирует
чистое
ООП
решение
для
прогнозирования
цен, включая:

Загрузку
и
подготовку
данных

Обучение
и
оценку
модели

Управление
прогнозами

Визуализацию
результатов

Валидацию
входных
данных

Каждый
класс
отвечает
за
свою
конкретную
задачу, что
соответствует
принципам
SOLID.
"""
