import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import logging
from datetime import datetime
import os

# Настройка логгирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_processor')

class DataProcessor:
    """Класс для обработки и подготовки данных о нежелательных реакциях"""
    
    def __init__(self, input_dir='../data/raw', output_dir='../data/processed'):
        """
        Инициализация процессора данных
        
        Args:
            input_dir (str): Директория с сырыми данными
            output_dir (str): Директория для сохранения обработанных данных
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        
        # Создаем директории, если они не существуют
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        # Инициализация инструментов NLTK
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
        except LookupError:
            logger.info("Загрузка необходимых ресурсов NLTK...")
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('wordnet')
        
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('russian'))
    
    def load_data(self, filepath=None):
        """
        Загрузка данных из CSV файла
        
        Args:
            filepath (str, optional): Путь к файлу для загрузки. Если не указан,
                                    загружаются все файлы из input_dir
                                    
        Returns:
            pandas.DataFrame: Загруженные данные
        """
        if filepath:
            logger.info(f"Загрузка данных из файла {filepath}")
            return pd.read_csv(filepath)
        else:
            # Загрузка всех CSV файлов из директории
            all_files = [f for f in os.listdir(self.input_dir) if f.endswith('.csv')]
            
            if not all_files:
                logger.warning(f"В директории {self.input_dir} не найдено CSV файлов")
                return pd.DataFrame()
            
            logger.info(f"Загрузка данных из {len(all_files)} файлов")
            
            dfs = []
            for file in all_files:
                file_path = os.path.join(self.input_dir, file)
                try:
                    df = pd.read_csv(file_path)
                    dfs.append(df)
                except Exception as e:
                    logger.error(f"Ошибка при загрузке файла {file}: {str(e)}")
            
            if not dfs:
                return pd.DataFrame()
                
            return pd.concat(dfs, ignore_index=True)
    
    def preprocess_text(self, text):
        """
        Предобработка текстовых данных
        
        Args:
            text (str): Текст для обработки
            
        Returns:
            str: Обработанный текст
        """
        if not isinstance(text, str):
            return ""
            
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление специальных символов
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Токенизация
        tokens = word_tokenize(text)
        
        # Удаление стоп-слов и лемматизация
        processed_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(processed_tokens)
    
    def process_data(self, data=None, save=True):
        """
        Обработка данных о нежелательных реакциях
        
        Args:
            data (pandas.DataFrame, optional): Данные для обработки. Если не указаны,
                                            данные загружаются с помощью load_data()
            save (bool): Сохранять ли обработанные данные
            
        Returns:
            pandas.DataFrame: Обработанные данные
        """
        if data is None:
            data = self.load_data()
            
        if data.empty:
            logger.warning("Нет данных для обработки")
            return data
            
        logger.info(f"Начало обработки {len(data)} записей")
        
        # Обработка текстовых полей
        if 'title' in data.columns:
            data['processed_title'] = data['title'].apply(self.preprocess_text)
            
        if 'adverse_reaction' in data.columns:
            data['processed_reaction'] = data['adverse_reaction'].apply(self.preprocess_text)
        
        # Нормализация названий лекарств (приведение к стандартному виду)
        if 'drug_name' in data.columns:
            data['normalized_drug'] = data['drug_name'].str.lower()
            
            # Словарь для стандартизации названий (в реальной системе был бы более полным)
            drug_mapping = {
                'аспирин': 'ацетилсалициловая кислота',
                'аспирин кардио': 'ацетилсалициловая кислота',
                'ибупрофен': 'ибупрофен',
                'нурофен': 'ибупрофен',
                'парацетамол': 'парацетамол',
                'панадол': 'парацетамол',
                'тайленол': 'парацетамол'
            }
            
            # Применяем маппинг
            data['normalized_drug'] = data['normalized_drug'].map(lambda x: drug_mapping.get(x, x))
            
        # Добавление дополнительных признаков
        data['processing_date'] = datetime.now().strftime('%Y-%m-%d')
        
        if save:
            self.save_processed_data(data)
            
        logger.info(f"Обработка данных завершена")
        return data
    
    def save_processed_data(self, data, filename=None):
        """
        Сохранение обработанных данных
        
        Args:
            data (pandas.DataFrame): Данные для сохранения
            filename (str, optional): Имя файла. Если не указано, генерируется автоматически
            
        Returns:
            str: Путь к сохраненному файлу
        """
        if data.empty:
            logger.warning("Нет данных для сохранения")
            return None
            
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"processed_adverse_reactions_{timestamp}.csv"
            
        output_path = os.path.join(self.output_dir, filename)
        
        data.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"Обработанные данные сохранены в {output_path}")
        
        return output_path
        
    def prepare_for_model(self, data=None):
        """
        Подготовка данных для модели машинного обучения
        
        Args:
            data (pandas.DataFrame, optional): Данные для подготовки
            
        Returns:
            dict: Словарь с подготовленными данными и метаданными
        """
        if data is None:
            data = self.process_data(save=False)
            
        if data.empty:
            logger.warning("Нет данных для подготовки к моделированию")
            return {'X': None, 'metadata': None}
            
        # В реальной системе здесь была бы более сложная подготовка 
        # данных для моделирования, включая преобразование категориальных 
        # переменных, выделение признаков из текста и т.д.
        
        # Простой пример для демонстрации
        if 'processed_reaction' in data.columns and 'normalized_drug' in data.columns:
            # Создание словарей для кодирования категориальных переменных
            drug_mapping = {drug: idx for idx, drug in enumerate(data['normalized_drug'].unique())}
            reaction_mapping = {reaction: idx for idx, reaction in enumerate(data['adverse_reaction'].unique())}
            
            # Кодирование категориальных переменных
            data['drug_code'] = data['normalized_drug'].map(drug_mapping)
            data['reaction_code'] = data['adverse_reaction'].map(reaction_mapping)
            
            # Формирование входных данных для модели
            X = data[['drug_code', 'reaction_code']].values
            
            return {
                'X': X,
                'metadata': {
                    'drug_mapping': drug_mapping,
                    'reaction_mapping': reaction_mapping,
                    'inverse_drug_mapping': {v: k for k, v in drug_mapping.items()},
                    'inverse_reaction_mapping': {v: k for k, v in reaction_mapping.items()}
                }
            }
        
        logger.warning("Не найдены необходимые колонки для подготовки к моделированию")
        return {'X': None, 'metadata': None} 