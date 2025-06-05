from sklearn.model_selection import train_test_split

from price_predictor import DataLoader, PricePredictor, PredictionManager, PredictionVisualizer
import pandas as pd
import os

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
            'count': 700,
            'add_cost': 3000,
            'company': 'Apple',
            'product': 'iPad'
        }
        result_1 = manager.make_prediction(new_data_1)
        print(f"Прогноз 1: {result_1['prediction']:.2f}")

        # Прогноз 2
        new_data_2 = {
            'count': 700,
            'add_cost': 300,
            'company': 'Google',
            'product': 'Xbox'
        }
        result_2 = manager.make_prediction(new_data_2)
        print(f"Прогноз 2: {result_2['prediction']:.2f}")

        # 7. Сохранение истории прогнозов
        manager.save_predictions_to_csv('predictions_history.csv')

        # 8. Загрузка модели и повторное использование
        loaded_predictor = PricePredictor().load_model()
        new_prediction = loaded_predictor.predict(pd.DataFrame([new_data_1]))[0]
        print(f"Прогноз с загруженной моделью: {new_prediction:.2f}")



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
    except Exception as e:
        print(f"Ошибка: {str(e)}")