import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging
from datetime import datetime
import shap
import lime
import lime.lime_tabular

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('models')

class AdverseReactionModel:
    """Класс для анализа нежелательных реакций на лекарственные препараты"""
    
    def __init__(self, model_type='random_forest', model_dir='../models'):
        """
        Инициализация модели
        
        Args:
            model_type (str): Тип модели ('random_forest', 'gradient_boosting', 'logistic_regression')
            model_dir (str): Директория для сохранения и загрузки моделей
        """
        self.model_type = model_type
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        self.metadata = None
        
        # Создаем директорию для моделей, если она не существует
        os.makedirs(model_dir, exist_ok=True)
        
        # Инициализация модели
        self._init_model()
    
    def _init_model(self):
        """Инициализация модели в зависимости от выбранного типа"""
        
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        elif self.model_type == 'logistic_regression':
            self.model = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42
            )
        else:
            logger.warning(f"Неизвестный тип модели: {self.model_type}. Используется RandomForest по умолчанию.")
            self.model_type = 'random_forest'
            self.model = RandomForestClassifier(
                n_estimators=100,
                random_state=42
            )
    
    def train(self, X, y, metadata=None, eval_size=0.2):
        """
        Обучение модели
        
        Args:
            X (numpy.ndarray): Обучающие данные
            y (numpy.ndarray): Целевые значения
            metadata (dict, optional): Метаданные для модели
            eval_size (float): Доля данных для валидации
            
        Returns:
            dict: Метрики качества модели
        """
        if X is None or y is None:
            logger.error("Переданы пустые данные для обучения")
            return None
            
        # Сохраняем метаданные
        self.metadata = metadata
        
        # Масштабирование признаков
        X_scaled = self.scaler.fit_transform(X)
        
        # Разделение на обучающую и валидационную выборки
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, test_size=eval_size, random_state=42
        )
        
        logger.info(f"Обучение модели типа {self.model_type}...")
        
        # Обучение модели
        self.model.fit(X_train, y_train)
        
        # Оценка качества на валидационной выборке
        y_pred = self.model.predict(X_val)
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, average='weighted'),
            'recall': recall_score(y_val, y_pred, average='weighted'),
            'f1': f1_score(y_val, y_pred, average='weighted')
        }
        
        logger.info(f"Обучение завершено. Метрики: {metrics}")
        
        return metrics
    
    def predict(self, X):
        """
        Прогнозирование для новых данных
        
        Args:
            X (numpy.ndarray): Данные для прогнозирования
            
        Returns:
            numpy.ndarray: Прогнозы
        """
        if self.model is None:
            logger.error("Модель не обучена")
            return None
            
        # Масштабирование признаков
        X_scaled = self.scaler.transform(X)
        
        # Прогнозирование
        predictions = self.model.predict(X_scaled)
        
        # Вероятности классов
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X_scaled)
        else:
            probabilities = None
            
        return {
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def explain(self, X, prediction_idx=None):
        """
        Объяснение прогноза модели
        
        Args:
            X (numpy.ndarray): Данные для объяснения
            prediction_idx (int, optional): Индекс прогноза для объяснения
            
        Returns:
            dict: Объяснение прогноза
        """
        if self.model is None:
            logger.error("Модель не обучена")
            return None
            
        # Масштабирование признаков
        X_scaled = self.scaler.transform(X)
        
        # Если индекс не указан, берем первую запись
        if prediction_idx is None:
            prediction_idx = 0
            
        if prediction_idx >= X.shape[0]:
            logger.error(f"Индекс {prediction_idx} выходит за пределы данных")
            return None
            
        # Получаем одну запись для объяснения
        X_single = X_scaled[prediction_idx].reshape(1, -1)
        
        # Используем SHAP для объяснения
        try:
            explainer = shap.Explainer(self.model, X_scaled)
            shap_values = explainer(X_single)
            
            # Превращаем значения SHAP в словарь
            feature_names = self.metadata.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
            
            shap_dict = {}
            for i, name in enumerate(feature_names):
                shap_dict[name] = float(shap_values.values[0, i])
                
            # Сортируем признаки по абсолютному значению влияния
            sorted_shap = sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)
            
            # Объяснение на основе SHAP
            explanation = {
                'shap_values': sorted_shap,
                'base_value': float(shap_values.base_values[0]),
                'prediction': float(self.model.predict(X_single)[0])
            }
            
            return explanation
            
        except Exception as e:
            logger.error(f"Ошибка при генерации объяснения с SHAP: {str(e)}")
            
            # Пробуем использовать LIME, если SHAP не сработал
            try:
                feature_names = self.metadata.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
                
                lime_explainer = lime.lime_tabular.LimeTabularExplainer(
                    X_scaled,
                    feature_names=feature_names,
                    class_names=self.metadata.get('class_names', ['Class 0', 'Class 1']),
                    mode='regression' if not hasattr(self.model, 'predict_proba') else 'classification'
                )
                
                lime_exp = lime_explainer.explain_instance(
                    X_scaled[prediction_idx],
                    self.model.predict if not hasattr(self.model, 'predict_proba') else self.model.predict_proba
                )
                
                # Превращаем объяснение LIME в словарь
                lime_explanation = {}
                for feature, importance in lime_exp.as_list():
                    lime_explanation[feature] = importance
                    
                sorted_lime = sorted(lime_explanation.items(), key=lambda x: abs(x[1]), reverse=True)
                
                return {
                    'lime_values': sorted_lime,
                    'prediction': float(self.model.predict(X_single)[0])
                }
                
            except Exception as e2:
                logger.error(f"Ошибка при генерации объяснения с LIME: {str(e2)}")
                
                # Если ни SHAP, ни LIME не сработали, возвращаем важность признаков
                if hasattr(self.model, 'feature_importances_'):
                    feature_names = self.metadata.get('feature_names', [f'feature_{i}' for i in range(X.shape[1])])
                    importances = self.model.feature_importances_
                    
                    feature_importance = {}
                    for i, name in enumerate(feature_names):
                        feature_importance[name] = float(importances[i])
                        
                    sorted_importance = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
                    
                    return {
                        'feature_importance': sorted_importance,
                        'prediction': float(self.model.predict(X_single)[0])
                    }
                    
                return None
    
    def save(self, filename=None):
        """
        Сохранение модели
        
        Args:
            filename (str, optional): Имя файла для сохранения
            
        Returns:
            str: Путь к сохраненной модели
        """
        if self.model is None:
            logger.error("Нечего сохранять: модель не обучена")
            return None
            
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{self.model_type}_model_{timestamp}.joblib"
            
        model_path = os.path.join(self.model_dir, filename)
        
        # Сохраняем модель вместе с метаданными и масштабатором
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'metadata': self.metadata,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, model_path)
        logger.info(f"Модель сохранена в {model_path}")
        
        return model_path
    
    def load(self, filepath):
        """
        Загрузка модели
        
        Args:
            filepath (str): Путь к файлу модели
            
        Returns:
            bool: Успешна ли загрузка
        """
        try:
            model_data = joblib.load(filepath)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.metadata = model_data['metadata']
            self.model_type = model_data['model_type']
            
            logger.info(f"Модель успешно загружена из {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Ошибка при загрузке модели из {filepath}: {str(e)}")
            return False
    
    def analyze_csv(self, csv_path):
        """
        Анализ данных из CSV-файла
        
        Args:
            csv_path (str): Путь к CSV-файлу
            
        Returns:
            dict: Результаты анализа
        """
        try:
            # Загрузка данных
            df = pd.read_csv(csv_path)
            
            # Базовая валидация
            if df.empty:
                return {"error": "Файл не содержит данных"}
                
            logger.info(f"Начало анализа данных из {csv_path}")
            
            # Демонстрационный анализ (в реальной системе здесь была бы настоящая обработка)
            # Здесь должна быть реальная предобработка данных и применение модели
            
            # Пример демонстрационного результата
            results = {
                "total_records": len(df),
                "analyzed_drugs": len(df['drug_name'].unique()) if 'drug_name' in df.columns else 0,
                "detected_reactions": [
                    {
                        "drug_name": "Аспирин",
                        "reactions": [
                            {
                                "name": "Желудочно-кишечное кровотечение",
                                "probability": 0.85,
                                "factors": ["Возраст > 65 лет", "Одновременный прием антикоагулянтов"],
                                "recommendations": "Назначить ингибиторы протонной помпы"
                            },
                            {
                                "name": "Бронхоспазм",
                                "probability": 0.62,
                                "factors": ["Астма в анамнезе"],
                                "recommendations": "Избегать применения у пациентов с астмой"
                            }
                        ]
                    },
                    {
                        "drug_name": "Метформин",
                        "reactions": [
                            {
                                "name": "Лактатацидоз",
                                "probability": 0.43,
                                "factors": ["Почечная недостаточность", "Высокая доза"],
                                "recommendations": "Контроль функции почек, снижение дозы"
                            }
                        ]
                    }
                ],
                "overall_recommendations": [
                    "Мониторинг уровня креатинина у пациентов старше 75 лет",
                    "Рассмотреть альтернативные препараты для пациентов с астмой"
                ]
            }
            
            logger.info(f"Анализ завершен, выявлено {len(results['detected_reactions'])} лекарств с потенциальными нежелательными реакциями")
            
            return results
            
        except Exception as e:
            logger.error(f"Ошибка при анализе CSV: {str(e)}")
            return {"error": f"Не удалось проанализировать файл: {str(e)}"} 