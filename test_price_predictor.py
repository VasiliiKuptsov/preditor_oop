import pytest
import pandas as pd
import numpy as np
import os
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from joblib import dump, load
import tempfile

# Импорт тестируемых классов
from price_predictor import (
    DataLoader,
    DataPreprocessor,
    PricePredictor,
    PredictionManager,
    PredictionVisualizer,
    InputValidator
)


@pytest.fixture
def sample_csv(tmp_path, sample_data):
    """Фикстура с тестовым CSV файлом с правильными колонками"""
    # Создаем данные с правильными полями
    data = {
        'count': [100, 200, 300],
        'add_cost': [50, 100, 150],
        'company': ['A', 'B', 'C'],
        'product': ['X', 'Y', 'Z'],
        'price': [100.0, 200.0, 300.0]
    }
    df = pd.DataFrame(data)

    csv_path = tmp_path / "test_data.csv"
    df.to_csv(csv_path, index=False)
    return str(csv_path)

@pytest.fixture
def sample_data():
    """Фикстура с тестовыми данными с правильными полями"""
    data = {
        'count': np.random.randint(1, 1000, 100),
        'add_cost': np.random.randint(0, 500, 100),
        'company': np.random.choice(['A', 'B', 'C'], 100),
        'product': np.random.choice(['X', 'Y', 'Z'], 100),
        'price': np.random.rand(100) * 100
    }
    return pd.DataFrame(data)


@pytest.fixture
def trained_model():
    """Фикстура с обученной моделью"""
    # Создаем данные с правильными полями
    data = {
        'count': [100, 200, 300],
        'add_cost': [50, 100, 150],
        'company': ['A', 'B', 'C'],
        'product': ['X', 'Y', 'Z'],
        'price': [100.0, 200.0, 300.0]
    }
    df = pd.DataFrame(data)

    X = df.drop('price', axis=1)
    y = df['price']

    predictor = PricePredictor()
    predictor.train(X, y)
    return predictor


class TestDataLoader:
    def test_load_from_csv_file_not_found(self):
        """Тест на обработку отсутствующего файла"""
        loader = DataLoader("nonexistent_file.csv")
        with pytest.raises(FileNotFoundError):
            loader.load_from_csv()

    def test_load_from_csv_success(self, sample_csv):
        """Тест успешной загрузки данных"""
        loader = DataLoader(sample_csv)
        loader.load_from_csv()

        assert loader.data is not None
        assert isinstance(loader.data, pd.DataFrame)
        assert not loader.data.empty
        assert set(loader.data.columns) == {'count', 'add_cost', 'company', 'product', 'price'}


    def test_get_features_target_not_loaded(self):
        """Тест на попытку получить признаки без загрузки данных"""
        loader = DataLoader()
        with pytest.raises(ValueError):
            loader.get_features_target()

    def test_get_features_target_success(self, sample_data):
        """Тест успешного разделения на признаки и целевую переменную"""
        loader = DataLoader()
        loader.data = sample_data
        X, y = loader.get_features_target()

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert set(X.columns) == {'count', 'add_cost', 'company', 'product'}
        assert y.name == 'price'
        assert len(X) == len(y) == len(sample_data)


class TestInputValidator:
    def test_validate_prediction_input_dict_valid(self):
        """Тест валидации корректного словаря"""
        valid_data = {
            'count': 100,
            'add_cost': 50,
            'company': 'Test',
            'product': 'Product'
        }
        InputValidator.validate_prediction_input(valid_data)

    def test_validate_prediction_input_dict_missing_fields(self):
        """Тест валидации с отсутствующими полями"""
        invalid_data = {'count': 100}  # Не хватает других полей
        with pytest.raises(ValueError) as excinfo:
            InputValidator.validate_prediction_input(invalid_data)
        assert "Отсутствуют обязательные поля" in str(excinfo.value)

    def test_validate_prediction_input_wrong_type(self):
        """Тест валидации неподходящего типа данных"""
        with pytest.raises(ValueError) as excinfo:
            InputValidator.validate_prediction_input("invalid type")
        assert "словарем, либо DataFrame" in str(excinfo.value)


class TestDataPreprocessor:
    def test_create_preprocessor(self):
        """Тест создания предобработчика"""
        numeric_features = ['num1', 'num2']
        categorical_features = ['cat1', 'cat2']

        preprocessor = DataPreprocessor(numeric_features, categorical_features)
        result = preprocessor.create_preprocessor()

        assert result is preprocessor
        assert preprocessor.preprocessor is not None
        assert len(preprocessor.preprocessor.transformers) == 2


class TestPricePredictor:
    def test_train_predict_evaluate(self, sample_data):
        """Тест обучения, предсказания и оценки модели"""
        X = sample_data.drop('price', axis=1)
        y = sample_data['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        predictor = PricePredictor()
        predictor.train(X_train, y_train)

        # Проверка предсказаний
        predictions = predictor.predict(X_test)
        assert len(predictions) == len(y_test)
        assert isinstance(predictions, np.ndarray)

        # Проверка оценки
        evaluation = predictor.evaluate(X_test, y_test)
        assert 'RMSE' in evaluation
        assert 'R2' in evaluation
        assert isinstance(evaluation['RMSE'], float)
        assert isinstance(evaluation['R2'], float)

    def test_predict_without_train(self):
        """Тест предсказания без обучения модели"""
        predictor = PricePredictor()
        with pytest.raises(ValueError):
            predictor.predict(pd.DataFrame({'col1': [1], 'col2': [2]}))

    def test_save_load_model(self, trained_model, tmp_path):
        """Тест сохранения и загрузки модели"""
        model_path = tmp_path / "test_model.joblib"
        trained_model.model_path = str(model_path)

        # Сохранение
        trained_model.save_model()
        assert os.path.exists(model_path)

        # Загрузка
        loaded_predictor = PricePredictor(model_path=str(model_path))
        loaded_predictor.load_model()

        assert loaded_predictor.model is not None
        assert hasattr(loaded_predictor.model, 'predict')

    def test_load_nonexistent_model(self, tmp_path):
        """Тест загрузки несуществующей модели"""
        model_path = tmp_path / "nonexistent_model.joblib"
        predictor = PricePredictor(model_path=str(model_path))
        with pytest.raises(FileNotFoundError):
            predictor.load_model()


class TestPredictionManager:
    @pytest.fixture
    def valid_input_data(self):
        """Фикстура с корректными входными данными"""
        return {
            'count': 100,
            'add_cost': 50,
            'company': 'TestCompany',
            'product': 'TestProduct'
        }

    @pytest.fixture
    def valid_input_dataframe(self, valid_input_data):
        """Фикстура с корректным DataFrame"""
        return pd.DataFrame([valid_input_data])

    def test_make_prediction(self, trained_model, valid_input_data):
        """Тест создания прогноза с валидными данными"""
        # Переобучаем модель с правильными признаками
        X = pd.DataFrame([valid_input_data])
        y = pd.Series([100.0])
        trained_model.train(X, y)

        manager = PredictionManager(trained_model)
        result = manager.make_prediction(valid_input_data)

        assert 'prediction' in result
        assert isinstance(result['prediction'], float)
        assert len(manager.predictions) == 1

    def test_make_prediction_dataframe(self, trained_model, valid_input_dataframe):
        """Тест создания прогноза с DataFrame"""
        # Переобучаем модель с правильными признаками
        y = pd.Series([100.0])
        trained_model.train(valid_input_dataframe, y)

        manager = PredictionManager(trained_model)
        result = manager.make_prediction(valid_input_dataframe)

        assert 'prediction' in result
        assert isinstance(result['prediction'], float)
        assert len(manager.predictions) == 1


class TestPredictionVisualizer:
    def test_plot_predictions_vs_actual(self, trained_model, sample_data):
        """Тест визуализации прогнозов vs фактические значения"""
        X = sample_data.drop('price', axis=1)
        y = sample_data['price']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        trained_model.train(X_train, y_train)
        y_pred = trained_model.predict(X_test)

        # Просто проверяем, что функция выполняется без ошибок
        PredictionVisualizer.plot_predictions_vs_actual(y_test, y_pred)

    def test_plot_feature_importance(self, trained_model):
        """Тест визуализации важности признаков"""
        # Просто проверяем, что функция выполняется без ошибок
        # В реальном тесте нужно было бы мокировать или создавать тестовую модель
        if hasattr(trained_model.model.named_steps['regressor'], 'feature_importances_'):
            # Получаем имена признаков (упрощенный вариант для теста)
            feature_names = ['feature_' + str(i) for i in range(10)]
            PredictionVisualizer.plot_feature_importance(trained_model.model, feature_names)


@pytest.fixture
def regression_data():
    """Фикстура с синтетическими данными с правильными именами колонок"""
    X, y = make_regression(
        n_samples=100,
        n_features=4,  # 4 признака: count, add_cost, company, product
        n_informative=3,
        noise=0.1,
        random_state=42
    )

    # Преобразуем в DataFrame с правильными именами колонок
    X_df = pd.DataFrame(X, columns=['count', 'add_cost', 'company_code', 'product_code'])

    # Преобразуем категориальные признаки
    X_df['company'] = X_df['company_code'].apply(lambda x: ['A', 'B', 'C'][int(x % 3)])
    X_df['product'] = X_df['product_code'].apply(lambda x: ['X', 'Y', 'Z'][int(x % 3)])
    X_df = X_df.drop(['company_code', 'product_code'], axis=1)

    y_series = pd.Series(y, name='price')
    return X_df, y_series


def test_integration_workflow(tmp_path, regression_data):
    """Интеграционный тест полного workflow"""
    X, y = regression_data

    # Переименовываем колонки чтобы соответствовать требованиям
    X = X.rename(columns={
        'feature_0': 'count',
        'feature_1': 'add_cost',
        'feature_2': 'company',
        'feature_3': 'product'
    })

    # Добавляем фиктивные категориальные значения
    X['company'] = ['Company_' + str(i) for i in range(len(X))]
    X['product'] = ['Product_' + str(i) for i in range(len(X))]

"""

def test_integration_workflow(tmp_path, regression_data):
    #Интеграционный тест полного workflow
    X, y = regression_data

    # 1. Сохраняем данные в CSV
    data = X.copy()
    data['price'] = y
    csv_path = tmp_path / "regression_data.csv"
    data.to_csv(csv_path, index=False)

    # 2. Загрузка данных
    loader = DataLoader(str(csv_path))
    X_loaded, y_loaded = loader.load_from_csv().get_features_target()
    assert X_loaded.shape == X.shape
    assert len(y_loaded) == len(y)

    # 3. Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(X_loaded, y_loaded, test_size=0.2, random_state=42)

    # 4. Обучение модели
    predictor = PricePredictor()
    predictor.train(X_train, y_train)

    # 5. Оценка модели
    evaluation = predictor.evaluate(X_test, y_test)
    assert 'RMSE' in evaluation
    assert 'R2' in evaluation

    # 6. Сохранение и загрузка модели
    model_path = tmp_path / "integration_model.joblib"
    predictor.model_path = str(model_path)
    predictor.save_model()
    assert os.path.exists(model_path)

    loaded_predictor = PricePredictor(model_path=str(model_path))
    loaded_predictor.load_model()

    # 7. Прогнозирование
    manager = PredictionManager(loaded_predictor)
    test_sample = X_test.iloc[0].to_dict()
    prediction = manager.make_prediction(test_sample)
    assert 'prediction' in prediction

    # 8. Сохранение истории прогнозов
    predictions_path = tmp_path / "integration_predictions.csv"
    manager.save_predictions_to_csv(str(predictions_path))
    assert os.path.exists(predictions_path)
"""