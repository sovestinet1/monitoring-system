import requests
from bs4 import BeautifulSoup
import re
import pandas as pd
import time
import random
import logging
import os
import json
import numpy as np
from datetime import datetime
from urllib.parse import urlparse

# Добавляем условный импорт для PHARMBerta
TORCH_AVAILABLE = False
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
    from transformers import AutoModelForSequenceClassification, AutoConfig
    TORCH_AVAILABLE = True
except ImportError:
    logging.warning("PyTorch не установлен. Расширенная функциональность PHARMBerta недоступна.")

# Пытаемся импортировать spacy
SPACY_AVAILABLE = False
try:
    import spacy
    from collections import defaultdict
    SPACY_AVAILABLE = True
except ImportError:
    logging.warning("Spacy не установлен. Будут использоваться базовые методы извлечения информации.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('data_collector')

class ArticleCollector:
    """Класс для сбора данных о нежелательных реакциях из интернет-источников"""
    
    def __init__(self, sources=None, output_dir='../data/raw', use_pharm_berta=True):
        """
        Инициализация сборщика данных
        
        Args:
            sources (list): Список источников для сбора данных
            output_dir (str): Директория для сохранения собранных данных
            use_pharm_berta (bool): Использовать ли PHARMBerta для анализа текста
        """
        self.sources = sources or []
        self.output_dir = output_dir
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        self.collected_data = []
        
        # Проверяем доступность необходимых библиотек
        self.use_pharm_berta = use_pharm_berta and TORCH_AVAILABLE
        
        if self.use_pharm_berta:
            logger.info("Инициализация моделей PHARMBerta...")
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Создаем директорию для моделей
            os.makedirs('../models/pharm_berta', exist_ok=True)
            
            try:
                self._initialize_pharm_berta()
                logger.info("Модели PHARMBerta успешно инициализированы")
            except Exception as e:
                logger.error(f"Ошибка при инициализации PHARMBerta: {str(e)}")
                logger.warning("Переключение на базовые методы извлечения данных")
                self.use_pharm_berta = False
        else:
            logger.info("Используются базовые методы извлечения данных (без PHARMBerta)")
        
        # Инициализируем SpaCy для лингвистического анализа
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("SpaCy модель успешно загружена")
            except:
                logger.warning("Не удалось загрузить модель SpaCy. Выполнение команды: python -m spacy download en_core_web_sm")
    
    def _initialize_pharm_berta(self):
        """Инициализация моделей PHARMBerta"""
        if not TORCH_AVAILABLE:
            logger.warning("PyTorch недоступен. Невозможно инициализировать PHARMBerta.")
            return
            
        # Загружаем модель PHARMBerta для распознавания сущностей (NER)
        self.ner_tokenizer = AutoTokenizer.from_pretrained("alvaroalon2/biobert_diseases_ner", use_auth_token=False)
        self.ner_model = AutoModelForTokenClassification.from_pretrained("alvaroalon2/biobert_diseases_ner").to(self.device)
        self.ner_pipeline = pipeline('ner', model=self.ner_model, tokenizer=self.ner_tokenizer, device=0 if self.device.type == 'cuda' else -1)
        
        # Загружаем модель PHARMBerta для анализа отношений между сущностями
        self.relation_tokenizer = AutoTokenizer.from_pretrained("gsarti/biobert-nli", use_auth_token=False)
        self.relation_model = AutoModelForSequenceClassification.from_pretrained("gsarti/biobert-nli").to(self.device)
        
        # Модель для определения частоты реакций
        self.frequency_model = AutoModelForSequenceClassification.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").to(self.device)
        self.frequency_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", use_auth_token=False)
    
    def add_source(self, source_url, source_type='medical_website'):
        """
        Добавить источник данных
        
        Args:
            source_url (str): URL источника
            source_type (str): Тип источника (medical_website, scientific_journal, etc.)
        """
        self.sources.append({
            'url': source_url,
            'type': source_type
        })
        logger.info(f"Добавлен источник: {source_url} (тип: {source_type})")
    
    def collect_from_all_sources(self):
        """Сбор данных из всех добавленных источников"""
        
        logger.info(f"Начало сбора данных из {len(self.sources)} источников")
        
        for i, source in enumerate(self.sources):
            try:
                logger.info(f"Обработка источника {i+1}/{len(self.sources)}: {source['url']}")
                source_data = self._collect_from_source(source)
                self.collected_data.extend(source_data)
                
                # Пауза между запросами для снижения нагрузки на сервер
                time.sleep(random.uniform(1.0, 3.0))
                
            except Exception as e:
                logger.error(f"Ошибка при обработке источника {source['url']}: {str(e)}")
        
        logger.info(f"Сбор данных завершен. Собрано {len(self.collected_data)} записей")
        return self.collected_data
    
    def _collect_from_source(self, source):
        """
        Сбор данных из конкретного источника
        
        Args:
            source (dict): Информация об источнике
            
        Returns:
            list: Список собранных данных
        """
        url = source['url']
        source_type = source['type']
        source_data = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Определяем тип источника и выбираем соответствующую стратегию парсинга
            if source_type == 'scientific_journal':
                # Для научных журналов используем более сложную логику
                source_data = self._parse_scientific_journal(soup, url)
            elif source_type == 'medical_website':
                # Для медицинских сайтов
                source_data = self._parse_medical_website(soup, url)
            else:
                # Общий случай
                source_data = self._parse_general_source(soup, url)
                
            logger.info(f"Обработан источник {url}. Найдено {len(source_data)} записей")
            
        except Exception as e:
            logger.error(f"Ошибка при сборе данных с {url}: {str(e)}")
        
        return source_data
    
    def _parse_scientific_journal(self, soup, url):
        """Анализ научной статьи"""
        data = []
        
        # Извлекаем заголовок, авторов и абстракт
        title = soup.find('h1') or soup.find('h2', class_='article-title')
        title_text = title.get_text().strip() if title else "Заголовок не найден"
        
        # Извлекаем полный текст статьи или абстракт
        abstract = soup.find('div', class_='abstract') or soup.find('section', class_='abstract')
        abstract_text = abstract.get_text().strip() if abstract else ""
        
        # Находим основной текст статьи
        article_sections = soup.find_all(['div', 'section'], class_=['article-section', 'body', 'fulltext'])
        full_text = ""
        
        # Собираем текст из всех секций
        for section in article_sections:
            section_text = section.get_text().strip()
            full_text += section_text + " "
        
        # Если не нашли основной текст, используем весь текст страницы
        if not full_text:
            main_content = soup.find('main') or soup.find('article') or soup.find('body')
            full_text = main_content.get_text().strip() if main_content else ""
        
        # Обрабатываем собранный текст
        if abstract_text or full_text:
            # Разделяем текст на параграфы для более эффективного анализа
            paragraphs = []
            
            if abstract_text:
                paragraphs.append(abstract_text)
                
            if full_text:
                # Разбиваем на разделы по подзаголовкам
                sections = re.split(r'\n\n|\r\n\r\n|<h[2-4]>', full_text)
                paragraphs.extend([p.strip() for p in sections if p.strip()])
            
            # Анализируем каждый параграф с помощью PharmBERTa
            for paragraph in paragraphs:
                if self.use_pharm_berta:
                    paragraph_data = self._extract_relations_with_pharm_berta(paragraph, title_text, url)
                    data.extend(paragraph_data)
                else:
                    # Используем традиционный подход
                    drugs = self._extract_drugs(paragraph)
                    reactions = self._extract_adverse_reactions(paragraph)
                    
                    if drugs and reactions:
                        for drug in drugs:
                            for reaction in reactions:
                                data.append({
                                    'source_url': url,
                                    'source_type': 'scientific_journal',
                                    'title': title_text,
                                    'drug_name': drug,
                                    'adverse_reaction': reaction,
                                    'confidence': 0.5,  # Доверие по умолчанию
                                    'frequency': 'Неизвестно',
                                    'severity': 'Неизвестно',
                                    'collection_date': datetime.now().strftime('%Y-%m-%d')
                                })
        
        return data
    
    def _parse_medical_website(self, soup, url):
        """Анализ страницы медицинского сайта"""
        data = []
        
        # Извлекаем заголовок
        title = soup.find('h1') or soup.find('h2')
        title_text = title.get_text().strip() if title else "Заголовок не найден"
        
        # Ищем основной контент
        main_content = soup.find('main') or soup.find('article') or soup.find('div', class_=['content', 'main-content'])
        
        if main_content:
            # Извлекаем текст контента
            content_text = main_content.get_text().strip()
            
            # Ищем таблицы с побочными эффектами
            tables = main_content.find_all('table')
            for table in tables:
                table_text = table.get_text()
                # Если таблица потенциально содержит информацию о побочных эффектах
                if re.search(r'(побочн|нежелательн|adverse|side effect)', table_text, re.IGNORECASE):
                    # Анализируем таблицу
                    table_data = self._extract_from_table(table, url, title_text)
                    data.extend(table_data)
            
            # Анализируем основной текст
            if self.use_pharm_berta:
                content_data = self._extract_relations_with_pharm_berta(content_text, title_text, url)
                data.extend(content_data)
            else:
                # Традиционный подход
                drugs = self._extract_drugs(content_text)
                reactions = self._extract_adverse_reactions(content_text)
                
                if drugs and reactions:
                    for drug in drugs:
                        for reaction in reactions:
                            data.append({
                                'source_url': url,
                                'source_type': 'medical_website',
                                'title': title_text,
                                'drug_name': drug,
                                'adverse_reaction': reaction,
                                'confidence': 0.5,
                                'frequency': 'Неизвестно',
                                'severity': 'Неизвестно',
                                'collection_date': datetime.now().strftime('%Y-%m-%d')
                            })
        
        return data
    
    def _parse_general_source(self, soup, url):
        """Анализ общего источника"""
        data = []
        
        title = soup.find('h1') or soup.find('h2') or soup.find('title')
        title_text = title.get_text().strip() if title else "Заголовок не найден"
        
        # Извлекаем весь текст страницы
        body_text = soup.body.get_text() if soup.body else soup.get_text()
        
        # Разбиваем на абзацы
        paragraphs = re.split(r'\n\n|\r\n\r\n', body_text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        # Анализируем каждый абзац
        for paragraph in paragraphs:
            if self.use_pharm_berta:
                paragraph_data = self._extract_relations_with_pharm_berta(paragraph, title_text, url)
                data.extend(paragraph_data)
            else:
                # Традиционный подход
                drugs = self._extract_drugs(paragraph)
                reactions = self._extract_adverse_reactions(paragraph)
                
                if drugs and reactions:
                    for drug in drugs:
                        for reaction in reactions:
                            data.append({
                                'source_url': url,
                                'source_type': 'general',
                                'title': title_text,
                                'drug_name': drug,
                                'adverse_reaction': reaction,
                                'confidence': 0.5,
                                'frequency': 'Неизвестно',
                                'severity': 'Неизвестно',
                                'collection_date': datetime.now().strftime('%Y-%m-%d')
                            })
        
        return data
    
    def _extract_from_table(self, table, url, title_text):
        """Извлечение данных о нежелательных реакциях из таблицы"""
        data = []
        
        # Извлечение заголовков таблицы
        headers = []
        header_row = table.find('tr')
        if header_row:
            headers = [th.get_text().strip().lower() for th in header_row.find_all(['th', 'td'])]
        
        # Индексы столбцов с нужной информацией
        drug_idx = next((i for i, h in enumerate(headers) if re.search(r'(препарат|лекарств|drug|medication)', h)), None)
        reaction_idx = next((i for i, h in enumerate(headers) if re.search(r'(реакц|побочн|эффект|reaction|effect|side)', h)), None)
        frequency_idx = next((i for i, h in enumerate(headers) if re.search(r'(частот|frequency|rate|incidence)', h)), None)
        severity_idx = next((i for i, h in enumerate(headers) if re.search(r'(тяжесть|severity|seriousness)', h)), None)
        
        # Если нашли хотя бы препарат и реакцию
        if drug_idx is not None and reaction_idx is not None:
            # Перебираем строки таблицы
            for row in table.find_all('tr')[1:]:  # Пропускаем заголовок
                cells = row.find_all(['td', 'th'])
                if len(cells) > max(drug_idx, reaction_idx):
                    drug = cells[drug_idx].get_text().strip()
                    reaction = cells[reaction_idx].get_text().strip()
                    
                    # Если есть информация о частоте и тяжести
                    frequency = cells[frequency_idx].get_text().strip() if frequency_idx is not None and frequency_idx < len(cells) else 'Неизвестно'
                    severity = cells[severity_idx].get_text().strip() if severity_idx is not None and severity_idx < len(cells) else 'Неизвестно'
                    
                    if drug and reaction:
                        data.append({
                            'source_url': url,
                            'source_type': 'table',
                            'title': title_text,
                            'drug_name': drug,
                            'adverse_reaction': reaction,
                            'confidence': 0.8,  # Информация из таблицы обычно достоверна
                            'frequency': frequency,
                            'severity': severity,
                            'collection_date': datetime.now().strftime('%Y-%m-%d')
                        })
        
        return data
    
    def _extract_relations_with_pharm_berta(self, text, title_text, url):
        """
        Извлечение отношений между препаратами и нежелательными реакциями с помощью PHARMBerta
        
        Args:
            text (str): Текст для анализа
            title_text (str): Заголовок статьи
            url (str): URL источника
            
        Returns:
            list: Список найденных отношений
        """
        data = []
        
        # Если PharmBERTa недоступна, используем традиционный подход
        if not self.use_pharm_berta or not TORCH_AVAILABLE:
            logger.info("PharmBERTa недоступна, используется базовый анализ текста")
            drugs = self._extract_drugs(text)
            reactions = self._extract_adverse_reactions(text)
            
            if drugs and reactions:
                for drug in drugs:
                    for reaction in reactions:
                        data.append({
                            'source_url': url,
                            'source_type': 'fallback',
                            'title': title_text,
                            'drug_name': drug,
                            'adverse_reaction': reaction,
                            'confidence': 0.5,
                            'frequency': 'Неизвестно',
                            'severity': 'Неизвестно',
                            'collection_date': datetime.now().strftime('%Y-%m-%d')
                        })
            return data
        
        try:
            # Если текст слишком длинный, разбиваем его на части
            max_length = 512  # Максимальная длина для BERT
            chunks = self._split_text_into_chunks(text, max_length)
            
            # Обнаруженные сущности из всего текста
            drugs_found = []
            reactions_found = []
            
            # Анализируем каждый фрагмент текста
            for chunk in chunks:
                # Шаг 1: Находим все упоминания лекарств и реакций с помощью NER
                ner_results = self.ner_pipeline(chunk)
                
                # Собираем сущности и их позиции
                drug_entities = []
                reaction_entities = []
                
                # Группируем токены в единые сущности
                current_entity = {"text": "", "entity": "", "start": -1, "end": -1}
                
                for result in ner_results:
                    if result['entity'].startswith('B-'):  # Начало новой сущности
                        # Сохраняем предыдущую сущность, если она была
                        if current_entity["text"]:
                            if current_entity["entity"] in ["B-DRUG", "I-DRUG"]:
                                drug_entities.append(current_entity.copy())
                            elif current_entity["entity"] in ["B-ADR", "I-ADR", "B-REACTION", "I-REACTION"]:
                                reaction_entities.append(current_entity.copy())
                        
                        # Начинаем новую сущность
                        current_entity = {
                            "text": result['word'],
                            "entity": result['entity'][2:],  # Убираем B-
                            "start": result['start'],
                            "end": result['end'],
                            "score": result['score']
                        }
                    elif result['entity'].startswith('I-') and current_entity["text"]:  # Продолжение сущности
                        # Проверяем, что это продолжение той же сущности
                        if result['entity'][2:] == current_entity["entity"]:
                            current_entity["text"] += " " + result['word']
                            current_entity["end"] = result['end']
                            current_entity["score"] = (current_entity["score"] + result['score']) / 2  # Средний скор
                
                # Не забываем про последнюю сущность
                if current_entity["text"]:
                    if current_entity["entity"] in ["DRUG"]:
                        drug_entities.append(current_entity.copy())
                    elif current_entity["entity"] in ["ADR", "REACTION"]:
                        reaction_entities.append(current_entity.copy())
                
                # Отфильтровываем дубликаты и объединяем с общими найденными сущностями
                for drug in drug_entities:
                    if drug["text"].lower() not in [d.lower() for d in drugs_found]:
                        drugs_found.append(drug["text"])
                
                for reaction in reaction_entities:
                    if reaction["text"].lower() not in [r.lower() for r in reactions_found]:
                        reactions_found.append(reaction["text"])
                
                # Шаг 2: Анализируем отношения для конкретного фрагмента
                # Для каждой пары drug-reaction в пределах одного фрагмента, проверяем связь
                for drug in drug_entities:
                    for reaction in reaction_entities:
                        # Создаем контекст из текста между (и включая) препаратом и реакцией
                        start = min(drug["start"], reaction["start"])
                        end = max(drug["end"], reaction["end"])
                        
                        # Расширяем контекст для захвата окружающих слов
                        context_start = max(0, start - 50)
                        context_end = min(len(chunk), end + 50)
                        
                        relation_context = chunk[context_start:context_end]
                        
                        # Проверяем наличие причинно-следственной связи с помощью модели
                        hypothesis = f"{drug['text']} вызывает {reaction['text']}"
                        
                        # Определяем, есть ли причинно-следственная связь
                        relation_score = self._check_relation(relation_context, hypothesis)
                        
                        # Если модель подтверждает связь с достаточной уверенностью
                        if relation_score > 0.7:  # Порог можно настроить
                            # Определяем частоту и тяжесть
                            frequency = self._extract_frequency(relation_context, reaction['text'])
                            severity = self._extract_severity(relation_context, reaction['text'])
                            
                            data.append({
                                'source_url': url,
                                'source_type': 'scientific_article',
                                'title': title_text,
                                'drug_name': drug['text'],
                                'adverse_reaction': reaction['text'],
                                'confidence': relation_score,
                                'frequency': frequency,
                                'severity': severity,
                                'context': relation_context,
                                'collection_date': datetime.now().strftime('%Y-%m-%d')
                            })
            
            # Шаг 3: После анализа всех фрагментов, ищем потенциальные связи между сущностями из разных фрагментов
            if len(chunks) > 1 and drugs_found and reactions_found:
                # Проверяем связи для препаратов и реакций, которые могли быть в разных фрагментах
                for drug in drugs_found:
                    for reaction in reactions_found:
                        # Проверяем, не добавлена ли уже эта пара
                        if not any(d['drug_name'] == drug and d['adverse_reaction'] == reaction for d in data):
                            # Формируем гипотезу
                            hypothesis = f"{drug} вызывает {reaction}"
                            
                            # Используем заголовок или первый фрагмент как контекст
                            relation_score = self._check_relation(title_text or chunks[0], hypothesis)
                            
                            # Если есть уверенность в связи
                            if relation_score > 0.6:  # Меньший порог для связей между фрагментами
                                data.append({
                                    'source_url': url,
                                    'source_type': 'cross_fragment',
                                    'title': title_text,
                                    'drug_name': drug,
                                    'adverse_reaction': reaction,
                                    'confidence': relation_score,
                                    'frequency': 'Неизвестно',
                                    'severity': 'Неизвестно',
                                    'collection_date': datetime.now().strftime('%Y-%m-%d')
                                })
            
        except Exception as e:
            logger.error(f"Ошибка при анализе текста с PHARMBerta: {str(e)}")
            
            # Если PHARMBerta не сработала, используем традиционный подход
            drugs = self._extract_drugs(text)
            reactions = self._extract_adverse_reactions(text)
            
            if drugs and reactions:
                for drug in drugs:
                    for reaction in reactions:
                        data.append({
                            'source_url': url,
                            'source_type': 'fallback',
                            'title': title_text,
                            'drug_name': drug,
                            'adverse_reaction': reaction,
                            'confidence': 0.5,
                            'frequency': 'Неизвестно',
                            'severity': 'Неизвестно',
                            'collection_date': datetime.now().strftime('%Y-%m-%d')
                        })
        
        return data
    
    def _split_text_into_chunks(self, text, max_length=512):
        """Разбиваем текст на фрагменты, не превышающие максимальную длину"""
        # Для начала разбиваем текст на предложения
        if self.nlp and SPACY_AVAILABLE:
            doc = self.nlp(text)
            sentences = [sent.text for sent in doc.sents]
        else:
            # Если SpaCy не доступен, используем простое разбиение
            sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Если предложение само по себе длиннее max_length, разбиваем его
            if len(sentence) > max_length:
                # Добавляем текущий чанк, если он не пустой
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                
                # Разбиваем длинное предложение на части
                for i in range(0, len(sentence), max_length):
                    chunks.append(sentence[i:i + max_length])
                
            # Если текущий чанк и предложение вместе не превышают max_length
            elif len(current_chunk) + len(sentence) <= max_length:
                current_chunk += " " + sentence if current_chunk else sentence
            
            # Если текущий чанк и предложение вместе превышают max_length
            else:
                chunks.append(current_chunk)
                current_chunk = sentence
        
        # Не забываем про последний чанк
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _check_relation(self, context, hypothesis):
        """
        Проверяет наличие причинно-следственной связи между препаратом и реакцией
        
        Args:
            context (str): Контекст, в котором упоминаются препарат и реакция
            hypothesis (str): Гипотеза о причинно-следственной связи
            
        Returns:
            float: Уверенность в наличии связи (0-1)
        """
        # Если PyTorch недоступен, используем эвристический подход
        if not self.use_pharm_berta or not TORCH_AVAILABLE:
            # Извлекаем препарат и реакцию из гипотезы
            match = re.match(r"(.+?) вызывает (.+)", hypothesis)
            if not match:
                return 0.0
                
            drug = match.group(1).strip().lower()
            reaction = match.group(2).strip().lower()
            
            # Проверяем, упоминаются ли оба слова в одном предложении
            sentences = re.split(r'(?<=[.!?])\s+', context.lower())
            for sentence in sentences:
                if drug in sentence and reaction in sentence:
                    # Проверяем наличие слов, указывающих на причинно-следственную связь
                    causal_words = ['вызывает', 'приводит', 'провоцирует', 'вызвал', 'вызывать', 
                                   'является причиной', 'связан с', 'ведет к', 'может вызвать']
                    
                    for word in causal_words:
                        if word in sentence:
                            # Если найдены оба термина и причинно-следственное слово в одном предложении
                            return 0.8
                            
                    # Если нашли оба термина в одном предложении, но без явной причинно-следственной связи
                    return 0.6
                    
            # Если препарат и реакция упоминаются в разных предложениях
            if drug in context.lower() and reaction in context.lower():
                return 0.4
                
            return 0.1
                
        try:
            # Токенизируем текст
            inputs = self.relation_tokenizer(context, hypothesis, return_tensors="pt", truncation=True, max_length=512).to(self.device)
            
            # Отключаем расчет градиента для экономии памяти
            with torch.no_grad():
                outputs = self.relation_model(**inputs)
                
            # Получаем логиты и преобразуем их в вероятности
            logits = outputs.logits
            probabilities = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
            
            # Для моделей NLI, обычно вероятности соответствуют: [отвержение, нейтрально, подтверждение]
            # Возвращаем вероятность подтверждения (entailment)
            return float(probabilities[2])
        
        except Exception as e:
            logger.error(f"Ошибка при проверке отношения: {str(e)}")
            return 0.0
    
    def _extract_frequency(self, text, reaction):
        """
        Извлекает информацию о частоте нежелательной реакции
        
        Args:
            text (str): Контекст, в котором упоминается реакция
            reaction (str): Название реакции
            
        Returns:
            str: Описание частоты реакции
        """
        # Ключевые слова для определения частоты
        frequency_patterns = {
            r'\b(очень часто|very common|very frequent)\b': 'Очень часто',
            r'\b(часто|common|frequent)\b': 'Часто',
            r'\b(нечасто|uncommon|infrequent)\b': 'Нечасто',
            r'\b(редко|rare)\b': 'Редко',
            r'\b(очень редко|very rare)\b': 'Очень редко',
            r'\b(\d+(?:\.\d+)?(?:\s*[-–—]\s*\d+(?:\.\d+)?)?%)\b': 'Частота: {}'
        }
        
        # Ищем упоминания частоты рядом с названием реакции
        for pattern, label in frequency_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                # Проверяем, что упоминание частоты находится достаточно близко к реакции
                if reaction.lower() in text[max(0, match.start() - 100):min(len(text), match.end() + 100)].lower():
                    if '{}' in label:
                        return label.format(match.group(0))
                    return label
        
        return 'Неизвестно'
    
    def _extract_severity(self, text, reaction):
        """
        Извлекает информацию о тяжести нежелательной реакции
        
        Args:
            text (str): Контекст, в котором упоминается реакция
            reaction (str): Название реакции
            
        Returns:
            str: Описание тяжести реакции
        """
        # Ключевые слова для определения тяжести
        severity_patterns = {
            r'\b(легк|mild|minor)\b': 'Легкая',
            r'\b(средн|умеренн|moderate)\b': 'Средняя',
            r'\b(тяжел|серьезн|severe|serious)\b': 'Тяжелая',
            r'\b(опасн|угрожа|life.threatening|смерт)\b': 'Жизнеугрожающая'
        }
        
        # Ищем упоминания тяжести рядом с названием реакции
        for pattern, label in severity_patterns.items():
            if re.search(pattern + r'.*?' + re.escape(reaction.lower()), text.lower(), re.IGNORECASE) or \
               re.search(re.escape(reaction.lower()) + r'.*?' + pattern, text.lower(), re.IGNORECASE):
                return label
        
        return 'Неизвестно'
    
    def _extract_drugs(self, text):
        """
        Извлечение названий лекарств из текста (классический подход)
        
        В реальной системе здесь был бы более сложный алгоритм NER (Named Entity Recognition)
        с использованием моделей NLP
        """
        # Простая демонстрационная логика
        common_drugs = [
            "аспирин", "парацетамол", "ибупрофен", "амоксициллин", "метформин",
            "омепразол", "диазепам", "варфарин", "симвастатин", "метотрексат"
        ]
        
        found_drugs = []
        for drug in common_drugs:
            if re.search(r'\b' + re.escape(drug) + r'\b', text.lower()):
                found_drugs.append(drug)
                
        return found_drugs
    
    def _extract_adverse_reactions(self, text):
        """
        Извлечение описаний нежелательных реакций из текста (классический подход)
        
        В реальной системе здесь был бы более сложный алгоритм
        """
        # Простая демонстрационная логика
        common_reactions = [
            "головная боль", "тошнота", "рвота", "диарея", "аллергическая реакция",
            "кожная сыпь", "головокружение", "усталость", "бессонница", "боль в желудке",
            "анафилактический шок", "отек квинке", "аритмия", "печеночная недостаточность"
        ]
        
        found_reactions = []
        for reaction in common_reactions:
            if re.search(r'\b' + re.escape(reaction) + r'\b', text.lower()):
                found_reactions.append(reaction)
                
        return found_reactions
        
    def save_collected_data(self, filename=None):
        """
        Сохранение собранных данных в CSV
        
        Args:
            filename (str, optional): Имя файла для сохранения
        """
        if not self.collected_data:
            logger.warning("Нет данных для сохранения")
            return None
            
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"adverse_reactions_data_{timestamp}.csv"
        
        output_path = f"{self.output_dir}/{filename}"
        
        df = pd.DataFrame(self.collected_data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Данные сохранены в {output_path}")
        return output_path
        
    def export_to_json(self, filename=None):
        """
        Экспорт собранных данных в JSON
        
        Args:
            filename (str, optional): Имя файла для сохранения
        """
        if not self.collected_data:
            logger.warning("Нет данных для сохранения")
            return None
            
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"adverse_reactions_data_{timestamp}.json"
        
        output_path = f"{self.output_dir}/{filename}"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.collected_data, f, ensure_ascii=False, indent=4)
        
        logger.info(f"Данные сохранены в JSON: {output_path}")
        return output_path 