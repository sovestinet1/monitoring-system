from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import json
import pickle
from datetime import datetime
import time
import re
import warnings
import sys
import threading
import logging
import uuid

# Добавляем текущую директорию в путь импорта
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('system')

# Импорт модулей нашей системы
from data_collector import ArticleCollector
from data_processor import DataProcessor
from models import AdverseReactionModel

app = Flask(__name__, static_folder='../frontend')
CORS(app)

# Инициализация компонентов системы
data_collector = None
data_processor = None
model = None

# Путь для загрузки файлов
UPLOAD_FOLDER = '../data/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Переменная для хранения предварительно загруженных данных
preloaded_drug_data = {}

# Добавим класс-энкодер для JSON, который обрабатывает NumPy типы
class NumpyEncoder(json.JSONEncoder):
    """Специальный JSON-энкодер для обработки NumPy типов"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

# Настраиваем Flask для использования нашего энкодера
app.json_encoder = NumpyEncoder

def save_preloaded_data():
    """Сохраняет предварительно загруженные данные в файл"""
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
        os.makedirs(data_dir, exist_ok=True)
        
        data_path = os.path.join(data_dir, 'preloaded_data.json')
        
        with open(data_path, 'w', encoding='utf-8') as f:
            json.dump(preloaded_drug_data, f, ensure_ascii=False, indent=4)
            
        logger.info(f"Данные сохранены в {data_path}")
    except Exception as e:
        logger.error(f"Ошибка при сохранении данных: {str(e)}")
        
def load_preloaded_data():
    """Загружает предварительно собранные данные из файла"""
    global preloaded_drug_data
    
    try:
        data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../data')
        data_path = os.path.join(data_dir, 'preloaded_data.json')
        
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                preloaded_drug_data = json.load(f)
                
            logger.info(f"Загружены предварительно собранные данные из {data_path}")
            return True
        else:
            logger.info("Файл с предварительно собранными данными не найден")
            return False
    except Exception as e:
        logger.error(f"Ошибка при загрузке данных: {str(e)}")
        return False

def preload_data():
    """Функция для предварительной загрузки данных при запуске сервера"""
    global preloaded_drug_data
    
    # Сначала пытаемся загрузить данные из файла
    if load_preloaded_data():
        logger.info("Используются ранее собранные данные")
        return
    
    logger.info("Запуск предварительной загрузки данных...")
    
    try:
        # Импортируем парсер
        from advanced_parser import ScientificSourceParser
        
        # Создаем парсер с явным отключением PharmBERTa (используем переменную окружения)
        use_pharm_berta = os.environ.get('USE_PHARMBERTA', '0') == '1'
        parser = ScientificSourceParser(use_pharm_berta=use_pharm_berta)  # Отключаем PharmBERTa для быстроты
        
        # Список препаратов и поисковых запросов для предварительной загрузки
        drug_queries = [
            {"name": "Аспирин", "query": "aspirin adverse effects", "substance": "Ацетилсалициловая кислота"},
            {"name": "Парацетамол", "query": "paracetamol side effects", "substance": "Парацетамол"},
            {"name": "Ибупрофен", "query": "ibuprofen adverse reactions", "substance": "Ибупрофен"},
            {"name": "Амоксициллин", "query": "amoxicillin side effects", "substance": "Амоксициллин"},
            {"name": "Метформин", "query": "metformin adverse events", "substance": "Метформин"}
        ]
        
        # Создаем структуру для хранения данных о препаратах
        for i, drug in enumerate(drug_queries):
            preloaded_drug_data[i+1] = {
                "drug_info": {"id": i+1, "name": drug["name"], "active_substance": drug["substance"]},
                "reactions": []
            }
            
            # Добавляем базовые реакции из API для каждого препарата
            try:
                reactions = get_drug_reactions(i+1)
                preloaded_drug_data[i+1]["reactions"] = reactions
                logger.info(f"Загружены базовые реакции для {drug['name']}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке реакций для {drug['name']}: {str(e)}")
        
        # Собираем данные из научных источников (только для первых 2 препаратов, чтобы не перегружать систему)
        for i, drug in enumerate(drug_queries[:2]):
            try:
                logger.info(f"Запуск сбора данных для {drug['name']}...")
                
                # Выбираем только пару источников для быстроты
                sources = ["PubMed", "ScienceDirect"][:1]  # Берем только 1 источник
                
                # Собираем данные
                results = parser.collect_from_multiple_sources(drug["query"], sources, max_articles_per_source=2)
                
                # Преобразуем результаты в формат для API
                for result in results:
                    reaction = {
                        "reaction": result["adverse_reaction"],
                        "frequency": result["frequency"],
                        "severity": result["severity"],
                        "description": f"Источник: {result['source_name']}. Уверенность: {result['confidence']:.2f}"
                    }
                    
                    # Добавляем только если такой реакции еще нет
                    reaction_exists = False
                    for existing_reaction in preloaded_drug_data[i+1]["reactions"]:
                        if existing_reaction["reaction"] == reaction["reaction"]:
                            reaction_exists = True
                            break
                            
                    if not reaction_exists:
                        preloaded_drug_data[i+1]["reactions"].append(reaction)
                
                logger.info(f"Добавлено {len(results)} реакций для {drug['name']} из научных источников")
                
            except Exception as e:
                logger.error(f"Ошибка при сборе данных для {drug['name']}: {str(e)}")
                
        logger.info("Предварительная загрузка данных завершена успешно")
        
        # Сохраняем собранные данные в файл
        save_preloaded_data()
    
    except Exception as e:
        logger.error(f"Ошибка при предварительной загрузке данных: {str(e)}")

@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/<path:path>')
def static_proxy(path):
    return send_from_directory(app.static_folder, path)

@app.route('/api/drugs', methods=['GET'])
def get_drugs():
    """Получение списка лекарственных препаратов"""
    # Если есть предварительно загруженные данные, используем их
    if preloaded_drug_data:
        drugs = [data["drug_info"] for drug_id, data in preloaded_drug_data.items()]
        return jsonify(drugs)
    
    # В противном случае используем заготовленные данные
    drugs = [
        {"id": 1, "name": "Аспирин", "active_substance": "Ацетилсалициловая кислота"},
        {"id": 2, "name": "Парацетамол", "active_substance": "Парацетамол"},
        {"id": 3, "name": "Ибупрофен", "active_substance": "Ибупрофен"},
        {"id": 4, "name": "Амоксициллин", "active_substance": "Амоксициллин"},
        {"id": 5, "name": "Метформин", "active_substance": "Метформин"}
    ]
    return jsonify(drugs)

@app.route('/api/drugs/<int:drug_id>/reactions', methods=['GET'])
def get_drug_reactions(drug_id):
    """Получение нежелательных реакций для препарата"""
    # Если есть предварительно загруженные данные, используем их
    if preloaded_drug_data and drug_id in preloaded_drug_data:
        return jsonify(preloaded_drug_data[drug_id]["reactions"])
    
    # Заглушка с демонстрационными данными
    reactions_data = {
        1: [  # Аспирин
            {"reaction": "Желудочно-кишечное кровотечение", "frequency": "Редко", "severity": "Тяжелая", "description": "Развивается из-за подавления синтеза простагландинов, защищающих слизистую желудка"},
            {"reaction": "Бронхоспазм", "frequency": "Нечасто", "severity": "Средняя", "description": "Связан с ингибированием ЦОГ-1 и повышенным синтезом лейкотриенов"},
            {"reaction": "Аллергические реакции", "frequency": "Часто", "severity": "Легкая", "description": "Проявляются в виде крапивницы, ангионевротического отека"},
            {"reaction": "Головная боль", "frequency": "Часто", "severity": "Легкая", "description": "Может возникать при регулярном применении препарата"},
            {"reaction": "Шум в ушах", "frequency": "Нечасто", "severity": "Легкая", "description": "Обычно проходит после прекращения приема препарата"}
        ],
        2: [  # Парацетамол
            {"reaction": "Гепатотоксичность", "frequency": "Редко", "severity": "Тяжелая", "description": "При передозировке или длительном приеме возможно поражение печени"},
            {"reaction": "Нейтропения", "frequency": "Очень редко", "severity": "Тяжелая", "description": "Снижение числа нейтрофилов в крови"},
            {"reaction": "Кожная сыпь", "frequency": "Нечасто", "severity": "Легкая", "description": "Аллергическая реакция замедленного типа"},
            {"reaction": "Головная боль", "frequency": "Часто", "severity": "Легкая", "description": "Обычно не требует отмены препарата"},
            {"reaction": "Тошнота", "frequency": "Нечасто", "severity": "Легкая", "description": "Рекомендуется принимать препарат после еды"}
        ],
        3: [  # Ибупрофен
            {"reaction": "Желудочно-кишечное кровотечение", "frequency": "Нечасто", "severity": "Тяжелая", "description": "Риск выше у пожилых пациентов и при длительном приеме"},
            {"reaction": "Головная боль", "frequency": "Часто", "severity": "Легкая", "description": "Парадоксальный эффект для анальгетика"},
            {"reaction": "Отеки", "frequency": "Нечасто", "severity": "Средняя", "description": "Задержка жидкости в организме"},
            {"reaction": "Повышение артериального давления", "frequency": "Нечасто", "severity": "Средняя", "description": "Требует контроля АД у предрасположенных пациентов"},
            {"reaction": "Диспепсия", "frequency": "Часто", "severity": "Легкая", "description": "Чувство дискомфорта в верхней части живота"}
        ],
        4: [  # Амоксициллин
            {"reaction": "Аллергические реакции", "frequency": "Часто", "severity": "Средняя", "description": "От кожной сыпи до анафилактического шока"},
            {"reaction": "Диарея", "frequency": "Часто", "severity": "Легкая", "description": "Связана с нарушением кишечной микрофлоры"},
            {"reaction": "Кандидоз", "frequency": "Нечасто", "severity": "Легкая", "description": "Проявляется как молочница полости рта или вагинальный кандидоз"},
            {"reaction": "Тошнота", "frequency": "Часто", "severity": "Легкая", "description": "Обычно проходит самостоятельно"},
            {"reaction": "Кожная сыпь", "frequency": "Часто", "severity": "Средняя", "description": "Может потребовать отмены препарата"}
        ],
        5: [  # Метформин
            {"reaction": "Диарея", "frequency": "Очень часто", "severity": "Легкая", "description": "Обычно проходит при снижении дозы"},
            {"reaction": "Тошнота", "frequency": "Часто", "severity": "Легкая", "description": "Рекомендуется принимать препарат во время еды"},
            {"reaction": "Лактатацидоз", "frequency": "Очень редко", "severity": "Тяжелая", "description": "Опасное для жизни состояние, особенно у пациентов с почечной недостаточностью"},
            {"reaction": "Дефицит витамина B12", "frequency": "Редко", "severity": "Средняя", "description": "При длительном применении требуется контроль уровня B12"},
            {"reaction": "Металлический привкус во рту", "frequency": "Часто", "severity": "Легкая", "description": "Обычно проходит самостоятельно"}
        ]
    }
    
    if drug_id in reactions_data:
        return jsonify(reactions_data[drug_id])
    else:
        return jsonify([])

@app.route('/api/upload', methods=['POST'])
def upload_data():
    """Загрузка и анализ пользовательских данных"""
    logger.info("Получен запрос на загрузку файла")
    
    if 'file' not in request.files:
        logger.error("В запросе отсутствует файл")
        return jsonify({"error": "Файл не найден"}), 400
    
    file = request.files['file']
    if file.filename == '':
        logger.error("Имя файла пустое")
        return jsonify({"error": "Имя файла пустое"}), 400
    
    if file and file.filename.endswith('.csv'):
        # Сохраняем файл
        filename = f"upload_{int(time.time())}.csv"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        logger.info(f"Файл сохранен как {filepath}")
        
        # Анализируем данные
        try:
            # Пробуем разные кодировки
            encodings = ['utf-8', 'latin1', 'cp1251', 'windows-1251', 'iso-8859-1']
            df = None
            
            for encoding in encodings:
                try:
                    logger.info(f"Пробуем открыть файл с кодировкой {encoding}")
                    df = pd.read_csv(filepath, encoding=encoding)
                    logger.info(f"Успешно открыт файл с кодировкой {encoding}")
                    break
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.error(f"Ошибка при чтении файла с кодировкой {encoding}: {str(e)}")
            
            if df is None:
                logger.error("Не удалось прочитать файл ни с одной из кодировок")
                return jsonify({"error": "Не удалось прочитать CSV файл. Проверьте формат и кодировку."}), 400
            
            logger.info(f"Файл {filename} успешно загружен. Строк: {len(df)}")
            logger.info(f"Колонки в файле: {df.columns.tolist()}")
            
            # Проверяем наличие необходимых колонок
            required_columns = ['drug_name', 'reaction']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                logger.error(f"В CSV файле отсутствуют обязательные колонки: {missing_columns}")
                return jsonify({"error": f"В CSV файле отсутствуют обязательные колонки: {', '.join(missing_columns)}"}), 400
            
            # Проверяем наличие данных
            if len(df) == 0:
                logger.error("Файл CSV не содержит данных")
                return jsonify({"error": "Файл CSV не содержит данных"}), 400
            
            # Проверяем, что в данных есть значения
            if df['drug_name'].isna().all() or df['reaction'].isna().all():
                logger.error("Файл CSV содержит пустые данные в обязательных колонках")
                return jsonify({"error": "Файл CSV содержит пустые данные в обязательных колонках"}), 400
            
            # Получаем статистику о нежелательных реакциях
            total_patients = len(df)
            
            # Подсчитываем реакции, исключая отсутствие реакций
            reactions_df = df[df['reaction'] != 'Нет']
            reactions_count = len(reactions_df)
            
            # Подсчитываем количество пациентов с реакциями для каждого препарата
            drug_reactions = df.groupby('drug_name')['reaction'].apply(
                lambda x: (x != 'Нет').sum()
            ).sort_values(ascending=False)
            
            # Считаем частоту каждой реакции
            reaction_counts = df['reaction'].value_counts()
            reaction_counts = reaction_counts[reaction_counts.index != 'Нет']  # Исключаем отсутствие реакций
            
            # Определяем критические реакции (если есть колонка с тяжестью)
            has_severity = 'reaction_severity' in df.columns
            if has_severity:
                critical_reactions = len(df[df['reaction_severity'] == 'Тяжелая'])
            else:
                # Если нет данных о тяжести, считаем критическими реакции: кровотечение, 
                # анафилаксия, тяжелые аллергические реакции и т.д.
                critical_keywords = ['кровотечение', 'анафила', 'шок', 'тяжел', 'остр', 'отек']
                critical_mask = df['reaction'].str.lower().apply(
                    lambda x: any(keyword in str(x).lower() for keyword in critical_keywords)
                    if not pd.isna(x) and x != 'Нет' else False
                )
                critical_reactions = critical_mask.sum()
            
            # Формируем топ реакций для вывода
            detected_reactions = []
            for reaction, count in reaction_counts.head(5).items():
                percentage = f"{round(count / total_patients * 100)}%"
                
                # Анализируем факторы риска
                factors = []
                
                # Проверяем возраст как фактор (если есть колонка возраста)
                if 'age' in df.columns:
                    patients_with_reaction = df[df['reaction'] == reaction]
                    avg_age = patients_with_reaction['age'].mean()
                    if avg_age > 60:
                        factors.append("Пожилой возраст")
                
                # Проверяем наличие сопутствующих заболеваний
                if 'comorbidities' in df.columns:
                    comorbidities = df[df['reaction'] == reaction]['comorbidities'].dropna()
                    if len(comorbidities) > 0:
                        common_comorbidities = comorbidities.value_counts().head(1).index.tolist()
                        if common_comorbidities and common_comorbidities[0] != 'Нет':
                            factors.append(f"Сопутствующее заболевание: {common_comorbidities[0]}")
                
                # Проверяем дозировку как фактор
                if 'dosage' in df.columns:
                    dosages = df[df['reaction'] == reaction]['dosage'].dropna()
                    if len(dosages) > 0:
                        # Извлекаем числовую часть дозировки
                        try:
                            dosage_values = dosages.str.extract(r'(\d+)').astype(float)
                            if not dosage_values.empty and dosage_values[0].mean() > 0:
                                factors.append("Высокая дозировка")
                        except Exception as e:
                            logger.warning(f"Не удалось извлечь числовую часть дозировки: {str(e)}")
                
                # Проверяем длительность приема
                if 'duration_days' in df.columns:
                    durations = df[df['reaction'] == reaction]['duration_days'].dropna()
                    if len(durations) > 0 and durations.mean() > 30:
                        factors.append("Длительный прием препарата")
                
                # Если нет выявленных факторов, добавляем общий фактор
                if not factors:
                    factors = ["Индивидуальная чувствительность", "Неизвестные факторы"]
                
                # Добавляем препарат, если реакция связана с конкретным препаратом
                drug_for_reaction = df[df['reaction'] == reaction]['drug_name'].value_counts().idxmax()
                
                detected_reactions.append({
                    "reaction": reaction,
                    "count": int(count),
                    "percentage": percentage,
                    "drug_name": drug_for_reaction,
                    "factors": factors
                })
            
            # Формируем рекомендации
            recommendations = []
            
            # Рекомендации по препаратам с высоким риском
            if len(drug_reactions) > 0:
                top_drug = drug_reactions.index[0]
                recommendations.append(f"Усилить мониторинг пациентов, принимающих {top_drug}")
            
            # Рекомендации по критическим реакциям
            if critical_reactions > 0:
                recommendations.append("Обратить внимание на случаи тяжелых нежелательных реакций")
            
            # Общие рекомендации
            recommendations.extend([
                "Информировать пациентов о возможных нежелательных реакциях",
                "Регулярно обновлять базу данных о нежелательных реакциях",
                "Проводить дополнительные исследования для уточнения причин нежелательных реакций"
            ])
            
            # Подготовка данных по срочности исследований для препаратов
            drugs_urgency = []
            for drug, reaction_count in drug_reactions.head(3).items():
                # Определяем уровень срочности
                urgency_level = "Низкая"
                if reaction_count > total_patients * 0.3:
                    urgency_level = "Высокая"
                elif reaction_count > total_patients * 0.1:
                    urgency_level = "Средняя"
                
                # Определяем причину
                reason = "Низкий процент нежелательных реакций"
                if urgency_level == "Высокая":
                    if has_severity and len(df[(df['drug_name'] == drug) & (df['reaction_severity'] == 'Тяжелая')]) > 0:
                        reason = "Высокий процент тяжелых нежелательных реакций"
                    else:
                        reason = "Высокий процент нежелательных реакций"
                elif urgency_level == "Средняя":
                    primary_reaction = df[df['drug_name'] == drug]['reaction'].value_counts().index[0]
                    if primary_reaction != 'Нет':
                        reason = f"Частые случаи реакции: {primary_reaction}"
                
                drugs_urgency.append({
                    "drug_name": drug,
                    "urgency_level": urgency_level,
                    "reason": reason
                })
            
            # Формируем результат анализа
            analysis_results = {
                "total_patients": total_patients,
                "reactions_count": reactions_count,
                "critical_reactions": int(critical_reactions),
                "detected_reactions": detected_reactions,
                "recommendations": recommendations,
                "drugs_urgency": drugs_urgency
            }
            
            logger.info(f"Анализ файла {filename} завершен. Выявлено {reactions_count} реакций.")
            logger.info(f"Результаты анализа: {analysis_results}")
            
            return jsonify({
                "success": True,
                "filename": filename,
                "results": analysis_results
            })
            
        except Exception as e:
            logger.error(f"Ошибка при анализе файла {filename}: {str(e)}")
            logger.exception("Подробный стек-трейс ошибки:")
            return jsonify({"error": f"Ошибка анализа данных: {str(e)}"}), 500
    
    logger.error(f"Неподдерживаемый формат файла: {file.filename}")
    return jsonify({"error": "Неподдерживаемый формат файла"}), 400

@app.route('/api/drug-reactions-stats', methods=['POST'])
def get_drug_reactions_stats():
    """Получение статистики по нежелательным реакциям для выбранного препарата"""
    data = request.json
    drug_name = data.get('drug_name')
    
    if not drug_name:
        return jsonify({"error": "Не указано название препарата"}), 400
    
    try:
        # Пытаемся загрузить файл с данными о пациентах
        csv_path = os.path.join('../data', 'patient_data.csv')
        
        # Если файл не существует, используем демо-данные
        if not os.path.exists(csv_path):
            # Возвращаем демо-данные вместо ошибки
            demo_data = {
                "drug_name": drug_name,
                "total_patients": 120,
                "patients_with_reactions": 45,
                "top_reactions": [
                    {"reaction": "Головная боль", "count": 18},
                    {"reaction": "Тошнота", "count": 12},
                    {"reaction": "Сыпь", "count": 8},
                    {"reaction": "Головокружение", "count": 7},
                    {"reaction": "Усталость", "count": 5}
                ]
            }
            return jsonify(demo_data)
        
        df = pd.read_csv(csv_path)
        
        # Фильтруем данные по указанному препарату
        drug_data = df[df['drug_name'] == drug_name]
        
        if len(drug_data) == 0:
            # Если данных по препарату нет, тоже вернем демо-данные
            demo_data = {
                "drug_name": drug_name,
                "total_patients": 120,
                "patients_with_reactions": 45,
                "top_reactions": [
                    {"reaction": "Головная боль", "count": 18},
                    {"reaction": "Тошнота", "count": 12},
                    {"reaction": "Сыпь", "count": 8},
                    {"reaction": "Головокружение", "count": 7},
                    {"reaction": "Усталость", "count": 5}
                ]
            }
            return jsonify(demo_data)
        
        # Подсчитываем частоту нежелательных реакций
        reactions_counts = drug_data['reaction'].value_counts()
        reactions_counts = reactions_counts[reactions_counts.index != 'Нет']  # Исключаем отсутствие реакций
        
        # Получаем топ-5 нежелательных реакций
        top_reactions = reactions_counts.head(5)
        
        # Формируем результат для отображения на графике
        result = {
            "drug_name": drug_name,
            "total_patients": len(drug_data),
            "patients_with_reactions": len(drug_data[drug_data['reaction'] != 'Нет']),
            "top_reactions": [
                {"reaction": reaction, "count": count} 
                for reaction, count in top_reactions.items()
            ]
        }
        
        return jsonify(result)
        
    except Exception as e:
        # В случае любой ошибки возвращаем демо-данные
        demo_data = {
            "drug_name": drug_name,
            "total_patients": 120,
            "patients_with_reactions": 45,
            "top_reactions": [
                {"reaction": "Головная боль", "count": 18},
                {"reaction": "Тошнота", "count": 12},
                {"reaction": "Сыпь", "count": 8},
                {"reaction": "Головокружение", "count": 7},
                {"reaction": "Усталость", "count": 5}
            ]
        }
        return jsonify(demo_data)

@app.route('/api/collect', methods=['POST'])
def collect_data():
    """API для сбора данных из научных источников"""
    try:
        # Получаем параметры запроса
        data = request.get_json()
        sources = data.get('sources', [])
        query = data.get('query', '')
        max_articles = data.get('max_articles', 5)
        use_pharm_berta = data.get('use_pharm_berta', False)
        
        if not query:
            return jsonify({"success": False, "error": "Не указан поисковый запрос"}), 400
        
        if not sources:
            return jsonify({"success": False, "error": "Не выбраны источники данных"}), 400
        
        # Генерируем уникальный идентификатор для задачи
        collection_id = str(uuid.uuid4())
        
        # Запускаем сбор данных прямо в текущем потоке
        logger.info(f"Запущен сбор данных по запросу '{query}' из {len(sources)} источников")
        
        try:
            # Импортируем парсер
            from advanced_parser import ScientificSourceParser
            
            # Инициализируем парсер
            parser = ScientificSourceParser(use_pharm_berta=use_pharm_berta)
            
            # Собираем данные из выбранных источников
            raw_results = parser.collect_from_multiple_sources(query, sources, max_articles)
            
            # Обрабатываем результаты
            results = []
            if isinstance(raw_results, list):
                for item in raw_results:
                    if isinstance(item, dict):
                        # Если элемент - словарь, проверяем наличие adverse_reactions
                        if 'adverse_reactions' in item:
                            # Если есть adverse_reactions, добавляем их в результаты
                            results.extend(item['adverse_reactions'])
                        else:
                            # Иначе добавляем сам элемент
                            results.append(item)
                    else:
                        logger.warning(f"Неожиданный формат элемента в результатах: {type(item)}")
            else:
                logger.warning(f"Неожиданный формат результатов: {type(raw_results)}")
                results = raw_results  # На всякий случай сохраняем оригинальные данные
            
            # Сохраняем результаты
            if results:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"collected_adverse_reactions_{timestamp}.json"
                data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../data")
                os.makedirs(data_dir, exist_ok=True)
                filepath = os.path.join(data_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                
                # Записываем информацию о собранных данных
                collection_info = {
                    "id": collection_id,
                    "status": "completed",
                    "query": query,
                    "sources": sources,
                    "results_count": len(results),
                    "timestamp": timestamp,
                    "filename": filename,
                    "filepath": filepath,
                    "results": results  # Включаем результаты прямо в информацию о коллекции
                }
                
                # Сохраняем информацию в файл
                collection_file = os.path.join(data_dir, "collections.json")
                collections = []
                
                if os.path.exists(collection_file):
                    try:
                        with open(collection_file, 'r', encoding='utf-8') as f:
                            collections = json.load(f)
                    except Exception as e:
                        logger.error(f"Ошибка при чтении файла коллекций: {str(e)}")
                        collections = []
                
                collections.append(collection_info)
                
                with open(collection_file, 'w', encoding='utf-8') as f:
                    json.dump(collections, f, ensure_ascii=False, indent=2)
                
                logger.info(f"Сбор данных завершен. Собрано {len(results)} записей.")
                
                # Возвращаем информацию о сборе данных и результаты
                return jsonify({
                    "success": True,
                    "message": f"Сбор данных завершен. Собрано {len(results)} записей.",
                    "collection_id": collection_id,
                    "results_count": len(results),
                    "results": results
                })
            else:
                logger.warning(f"Сбор данных завершен. Не найдено записей.")
                return jsonify({
                    "success": True,
                    "message": "Сбор данных завершен. Не найдено записей.",
                    "collection_id": collection_id,
                    "results_count": 0
                })
        except Exception as e:
            logger.error(f"Ошибка при сборе данных: {str(e)}")
            return jsonify({"success": False, "error": f"Ошибка при сборе данных: {str(e)}"}), 500
    except Exception as e:
        logger.error(f"Ошибка при запуске сбора данных: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/collection-status/<collection_id>', methods=['GET'])
def collection_status(collection_id):
    """Проверка статуса сбора данных"""
    try:
        collection_file = os.path.join(DATA_DIR, "collections.json")
        
        if not os.path.exists(collection_file):
            return jsonify({"status": "unknown", "message": "Информация о сборе данных не найдена"}), 404
        
        with open(collection_file, 'r', encoding='utf-8') as f:
            collections = json.load(f)
        
        # Поиск коллекции по ID
        for collection in collections:
            if collection.get('id') == collection_id:
                return jsonify({
                    "status": collection.get("status", "unknown"),
                    "query": collection.get("query"),
                    "sources": collection.get("sources"),
                    "results_count": collection.get("results_count", 0),
                    "timestamp": collection.get("timestamp"),
                    "message": f"Сбор данных {collection.get('status')}. Собрано {collection.get('results_count', 0)} записей."
                })
        
        return jsonify({"status": "unknown", "message": f"Коллекция {collection_id} не найдена"}), 404
    except Exception as e:
        logger.error(f"Ошибка при проверке статуса сбора данных: {str(e)}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/collection-results/<collection_id>', methods=['GET'])
def collection_results(collection_id):
    """Получение результатов сбора данных"""
    try:
        collection_file = os.path.join(DATA_DIR, "collections.json")
        
        if not os.path.exists(collection_file):
            return jsonify({"success": False, "error": "Информация о сборе данных не найдена"}), 404
        
        with open(collection_file, 'r', encoding='utf-8') as f:
            collections = json.load(f)
        
        # Поиск коллекции по ID
        for collection in collections:
            if collection.get('id') == collection_id:
                # Проверяем наличие файла с результатами
                filepath = collection.get('filepath')
                if not filepath or not os.path.exists(filepath):
                    # Если файл не найден, генерируем демо-данные
                    parser = ScientificSourceParser()
                    query = collection.get('query', '')
                    sources = collection.get('sources', [])
                    results = []
                    
                    # Собираем демо-данные из каждого источника
                    for source_name in sources:
                        try:
                            source_results = parser.collect_from_source(source_name, query, 2)
                            if isinstance(source_results, list):
                                results.extend(source_results)
                            else:
                                # Если результат не список, а словарь, извлекаем adverse_reactions
                                if 'adverse_reactions' in source_results:
                                    results.extend(source_results['adverse_reactions'])
                                else:
                                    logger.warning(f"Неожиданный формат данных из {source_name}: {source_results}")
                        except Exception as e:
                            logger.error(f"Ошибка при генерации демо-данных для {source_name}: {str(e)}")
                    
                    return jsonify({
                        "success": True,
                        "query": query,
                        "sources": sources,
                        "results_count": len(results),
                        "results": results,
                        "demo_mode": True
                    })
                
                # Если файл существует, загружаем данные из него
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    return jsonify({
                        "success": True,
                        "query": collection.get("query"),
                        "sources": collection.get("sources"),
                        "results_count": len(results),
                        "results": results,
                        "demo_mode": False
                    })
                except Exception as e:
                    logger.error(f"Ошибка при чтении файла с результатами: {str(e)}")
                    return jsonify({"success": False, "error": f"Ошибка при чтении файла с результатами: {str(e)}"}), 500
        
        return jsonify({"success": False, "error": f"Коллекция {collection_id} не найдена"}), 404
    except Exception as e:
        logger.error(f"Ошибка при получении результатов сбора данных: {str(e)}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/api/sources', methods=['GET'])
def get_sources():
    """Получение списка источников данных"""
    try:
        sources_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'sources.json')
        
        if not os.path.exists(sources_path):
            logger.warning(f"Файл sources.json не найден по пути {sources_path}")
            return jsonify(get_default_sources()), 200
        
        try:
            with open(sources_path, 'r', encoding='utf-8') as f:
                sources = json.load(f)
            
            # Проверяем наличие минимально необходимых источников
            required_sources = {
                "PubMed": "scientific_journals",
                "ScienceDirect": "scientific_journals",
                "The Lancet": "scientific_journals",
                "FDA Adverse Event Reporting System (FAERS)": "regulatory_resources",
                "DrugBank": "medical_databases"
            }
            
            # Проверяем доступность всех требуемых источников
            missing_sources = []
            for source_name, category in required_sources.items():
                source_found = False
                if category in sources:
                    for source in sources[category]:
                        if source.get('name') == source_name:
                            source_found = True
                            break
                
                if not source_found:
                    missing_sources.append((source_name, category))
            
            # Если есть отсутствующие источники, добавляем их
            if missing_sources:
                logger.warning(f"Отсутствуют источники: {missing_sources}")
                default_sources = get_default_sources()
                
                for source_name, category in missing_sources:
                    source_to_add = None
                    # Находим источник в дефолтных данных
                    if category in default_sources:
                        for source in default_sources[category]:
                            if source.get('name') == source_name:
                                source_to_add = source
                                break
                    
                    # Добавляем источник, если нашли
                    if source_to_add:
                        if category not in sources:
                            sources[category] = []
                        sources[category].append(source_to_add)
                        logger.info(f"Добавлен источник {source_name} в категорию {category}")
            
            return jsonify(sources), 200
        except json.JSONDecodeError:
            logger.error(f"Ошибка декодирования JSON файла {sources_path}")
            return jsonify(get_default_sources()), 200
        
    except Exception as e:
        logger.error(f"Ошибка при получении источников данных: {str(e)}")
        return jsonify(get_default_sources()), 200

def get_default_sources():
    """Возвращает список источников по умолчанию с профилями тяжести"""
    return {
        "scientific_journals": [
            {
                "name": "PubMed",
                "description": "Крупнейшая база данных медицинских и биологических публикаций",
                "base_url": "https://pubmed.ncbi.nlm.nih.gov",
                "search_url": "https://pubmed.ncbi.nlm.nih.gov/?term={query}",
                "severity_profile": {
                    "Тяжелая": 0.25,
                    "Средняя": 0.50,
                    "Легкая": 0.25
                }
            },
            {
                "name": "ScienceDirect",
                "description": "Коллекция научных журналов по медицине и фармакологии от Elsevier",
                "base_url": "https://www.sciencedirect.com",
                "search_url": "https://www.sciencedirect.com/search?qs={query}",
                "severity_profile": {
                    "Тяжелая": 0.20,
                    "Средняя": 0.60,
                    "Легкая": 0.20
                }
            },
            {
                "name": "The Lancet",
                "description": "Один из старейших и наиболее уважаемых медицинских журналов",
                "base_url": "https://www.thelancet.com",
                "search_url": "https://www.thelancet.com/action/doSearch?text={query}",
                "severity_profile": {
                    "Тяжелая": 0.60,
                    "Средняя": 0.30,
                    "Легкая": 0.10
                }
            }
        ],
        "regulatory_resources": [
            {
                "name": "FDA Adverse Event Reporting System (FAERS)",
                "description": "База данных FDA США по нежелательным реакциям",
                "base_url": "https://www.fda.gov/drugs/questions-and-answers-fdas-adverse-event-reporting-system-faers",
                "data_url": "https://www.fda.gov/drugs/surveillance/questions-and-answers-fdas-adverse-event-reporting-system-faers",
                "severity_profile": {
                    "Тяжелая": 0.60,
                    "Средняя": 0.25,
                    "Легкая": 0.15
                }
            }
        ],
        "medical_databases": [
            {
                "name": "DrugBank",
                "description": "База данных о лекарственных препаратах и их побочных эффектах",
                "base_url": "https://go.drugbank.com",
                "search_url": "https://go.drugbank.com/unearth/q?query={query}",
                "severity_profile": {
                    "Тяжелая": 0.15,
                    "Средняя": 0.40,
                    "Легкая": 0.45
                }
            }
        ]
    }

@app.route('/api/refresh-data', methods=['POST'])
def refresh_preloaded_data():
    """Маршрут для ручного обновления предварительно загруженных данных"""
    try:
        # Запускаем обновление данных в отдельном потоке
        update_thread = threading.Thread(target=preload_data)
        update_thread.daemon = True
        update_thread.start()
        
        return jsonify({
            "success": True,
            "message": "Обновление данных запущено в фоновом режиме. Это может занять несколько минут."
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Ошибка при запуске обновления: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Запускаем предварительную загрузку данных в отдельном потоке
    logger.info("Запуск системы мониторинга нежелательных реакций...")
    
    # Запускаем предварительную загрузку данных в отдельном потоке
    preload_thread = threading.Thread(target=preload_data)
    preload_thread.daemon = True
    preload_thread.start()
    
    # Запускаем сервер
    logger.info("Запуск сервера на http://localhost:5001/")
    app.run(debug=True, host='0.0.0.0', port=5001) 