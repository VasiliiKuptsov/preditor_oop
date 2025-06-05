import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from joblib import dump, load
import os
import matplotlib.pyplot as plt

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

class InputValidator:
    @staticmethod
    def validate_prediction_input(input_data):
        """Валидация входных данных для прогнозирования"""
        required_fields = ['count', 'add_cost', 'company', 'product']

        if isinstance(input_data, pd.DataFrame):
            # Проверяем DataFrame
            missing_cols = [col for col in required_fields if col not in input_data.columns]
            if missing_cols:
                raise ValueError(f"Отсутствуют обязательные колонки: {', '.join(missing_cols)}")

            if input_data.empty:
                raise ValueError("DataFrame не должен быть пустым")
        elif isinstance(input_data, dict):
            # Проверяем словарь
            missing_fields = [field for field in required_fields if field not in input_data]
            if missing_fields:
                raise ValueError(f"Отсутствуют обязательные поля: {', '.join(missing_fields)}")
        else:
            raise ValueError("Входные данные должны быть либо словарем, либо DataFrame")


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

    def get_predictions_history(self):
        """Получение истории прогнозов"""
        return pd.DataFrame(self.predictions)

    def save_predictions_to_csv(self, file_path):
        """Сохранение прогнозов в CSV"""
        if not self.predictions:
            raise ValueError("Нет данных для сохранения")
        df = self.get_predictions_history()
        df.to_csv(file_path, index=False)


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
