# Система мониторинга нежелательных реакций на лекарственные препараты

Автоматизированная система мониторинга и анализа нежелательных реакций на лекарственные препараты на основе данных из интернета и пользовательских данных.

## Возможности системы

- Сбор данных о нежелательных реакциях с медицинских сайтов и научных статей
- Анализ собранных данных с помощью NLP и методов машинного обучения
- Веб-интерфейс для просмотра информации о нежелательных реакциях по препаратам
- Функция загрузки пользовательских данных (CSV) для анализа
- Интерпретация результатов модели с объяснением возможных причин нежелательных реакций

## Структура проекта

```
adverse_reactions_system/
├── backend/            # Бэкенд сервер и API на Flask
├── frontend/           # Веб-интерфейс
├── data/               # Собранные и предобработанные данные
├── models/             # Обученные модели для анализа
├── requirements.txt    # Зависимости проекта
└── README.md           # Описание проекта
```

## Установка и запуск

1. Установите зависимости проекта:
   ```
   pip install -r requirements.txt
   ```

2. Запустите бэкенд-сервер:
   ```
   python backend/app.py
   ```

3. Откройте веб-интерфейс:
   http://localhost:5000/

## Использование

1. **Просмотр данных о нежелательных реакциях**
   - Выберите лекарственный препарат из списка
   - Просмотрите список возможных нежелательных реакций и их частоту
   - Изучите детальную информацию и рекомендации

2. **Анализ пользовательских данных**
   - Загрузите CSV-файл с данными о пациентах и их реакциях
   - Система проанализирует данные и представит результаты в виде графиков и таблиц
   - Для каждой выявленной нежелательной реакции будут указаны возможные причины

## Модель анализа данных

В системе используются следующие модели машинного обучения:
- Модель классификации нежелательных реакций на основе клинических данных
- NLP-модель для извлечения информации из текстовых источников
- Модель интерпретации для выявления причинно-следственных связей

## Разработка и расширение

Для добавления новых источников данных или расширения функциональности, обратитесь к документации по разработке в директории `/docs`. 