// Ожидаем загрузку DOM
document.addEventListener('DOMContentLoaded', function() {
    // Элементы навигации
    const navLinks = document.querySelectorAll('nav ul li a');
    const sections = document.querySelectorAll('section');
    
    // Кнопки главной страницы
    const btnExploreDrugs = document.getElementById('btn-explore-drugs');
    const btnUploadData = document.getElementById('btn-upload-data');
    
    // Кнопки возврата на главную страницу
    const btnBackToDrugs = document.getElementById('back-to-home-drugs');
    const btnBackToUpload = document.getElementById('back-to-home-upload');
    const btnBackToAbout = document.getElementById('back-to-home-about');
    
    // Логотип
    const logoHome = document.getElementById('logo-home');
    
    // Страница с препаратами
    const drugSearch = document.getElementById('drug-search');
    const searchBtn = document.getElementById('search-btn');
    const drugsList = document.querySelector('.drugs-list');
    const drugDetails = document.getElementById('drug-details');
    const drugName = document.getElementById('drug-name');
    const drugSubstance = document.getElementById('drug-substance');
    const reactionsList = document.getElementById('reactions-list');
    
    // Страница загрузки файлов
    const uploadArea = document.getElementById('upload-area');
    const fileUpload = document.getElementById('file-upload');
    const fileInfo = document.getElementById('file-info');
    const fileName = document.getElementById('file-name');
    const fileSize = document.getElementById('file-size');
    const btnRemoveFile = document.getElementById('btn-remove-file');
    const btnAnalyze = document.getElementById('btn-analyze');
    const analysisResults = document.getElementById('analysis-results');
    
    // Глобальные переменные
    let selectedFile = null;
    let loadedDrugs = [];
    let selectedDrug = null;
    let reactionChart = null; // Переменная для хранения экземпляра графика
    
    // Переключение между секциями
    function switchSection(sectionId) {
        // Удаляем активные классы
        navLinks.forEach(link => link.classList.remove('active'));
        sections.forEach(section => section.classList.remove('active-section'));
        sections.forEach(section => section.classList.add('hidden-section'));
        
        // Добавляем активные классы
        document.getElementById(`nav-${sectionId}`).classList.add('active');
        document.getElementById(`section-${sectionId}`).classList.remove('hidden-section');
        document.getElementById(`section-${sectionId}`).classList.add('active-section');
        
        // Если переходим на страницу препаратов, загружаем их
        if (sectionId === 'drugs' && loadedDrugs.length === 0) {
            loadDrugs();
        }
        
        // Если переходим на страницу загрузки, загружаем источники
        if (sectionId === 'upload') {
            loadDataSources();
        }
    }
    
    // Навигация
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const sectionId = this.id.replace('nav-', '');
            switchSection(sectionId);
        });
    });
    
    // Кнопки главной страницы
    if (btnExploreDrugs) {
        btnExploreDrugs.addEventListener('click', function() {
            switchSection('drugs');
        });
    }
    
    if (btnUploadData) {
        btnUploadData.addEventListener('click', function() {
            switchSection('upload');
        });
    }
    
    // Кнопки возврата на главную страницу
    if (btnBackToDrugs) {
        btnBackToDrugs.addEventListener('click', function() {
            switchSection('home');
        });
    }
    
    if (btnBackToUpload) {
        btnBackToUpload.addEventListener('click', function() {
            switchSection('home');
        });
    }
    
    if (btnBackToAbout) {
        btnBackToAbout.addEventListener('click', function() {
            switchSection('home');
        });
    }
    
    // Клик по логотипу - переход на главную страницу
    if (logoHome) {
        logoHome.addEventListener('click', function(e) {
            e.preventDefault();
            switchSection('home');
        });
    }
    
    // Загрузка списка препаратов
    function loadDrugs() {
        // В реальном приложении здесь был бы запрос к API
        drugsList.innerHTML = '<div class="loading">Загрузка препаратов...</div>';
        
        fetch('/api/drugs')
            .then(response => response.json())
            .then(data => {
                loadedDrugs = data;
                renderDrugsList(data);
            })
            .catch(error => {
                console.error('Ошибка при загрузке препаратов:', error);
                drugsList.innerHTML = '<div class="error">Не удалось загрузить список препаратов. Пожалуйста, попробуйте позже.</div>';
                
                // Для демонстрации добавим имитацию загрузки
                setTimeout(() => {
                    const mockDrugs = [
                        {id: 1, name: "Аспирин", active_substance: "Ацетилсалициловая кислота"},
                        {id: 2, name: "Парацетамол", active_substance: "Парацетамол"},
                        {id: 3, name: "Ибупрофен", active_substance: "Ибупрофен"},
                        {id: 4, name: "Амоксициллин", active_substance: "Амоксициллин"},
                        {id: 5, name: "Метформин", active_substance: "Метформин"},
                        {id: 6, name: "Омепразол", active_substance: "Омепразол"},
                        {id: 7, name: "Диазепам", active_substance: "Диазепам"},
                        {id: 8, name: "Варфарин", active_substance: "Варфарин"}
                    ];
                    loadedDrugs = mockDrugs;
                    renderDrugsList(mockDrugs);
                }, 1000);
            });
    }
    
    // Отображение списка препаратов
    function renderDrugsList(drugs) {
        if (!drugsList) return;
        
        if (drugs.length === 0) {
            drugsList.innerHTML = '<div class="no-results">Препараты не найдены</div>';
            return;
        }
        
        let html = '';
        drugs.forEach(drug => {
            html += `
                <div class="drug-card" data-id="${drug.id}">
                    <h3>${drug.name}</h3>
                    <div class="substance">${drug.active_substance}</div>
                </div>
            `;
        });
        
        drugsList.innerHTML = html;
        
        // Добавляем обработчики событий на карточки препаратов
        document.querySelectorAll('.drug-card').forEach(card => {
            card.addEventListener('click', function() {
                const drugId = parseInt(this.getAttribute('data-id'));
                loadDrugDetails(drugId);
            });
        });
    }
    
    // Загрузка подробной информации о препарате
    function loadDrugDetails(drugId) {
        // В реальном приложении здесь был бы запрос к API
        fetch(`/api/drugs/${drugId}/reactions`)
            .then(response => response.json())
            .then(data => {
                renderDrugDetails(drugId, data);
            })
            .catch(error => {
                console.error(`Ошибка при загрузке информации о препарате ${drugId}:`, error);
                
                // Для демонстрации добавим имитацию загрузки
                setTimeout(() => {
                    const mockReactions = [
                        {
                            reaction: "Желудочно-кишечное кровотечение",
                            frequency: "Редко",
                            severity: "Тяжелая",
                            description: "Развивается из-за подавления синтеза простагландинов, защищающих слизистую желудка"
                        },
                        {
                            reaction: "Бронхоспазм",
                            frequency: "Нечасто",
                            severity: "Средняя",
                            description: "Связан с ингибированием ЦОГ-1 и повышенным синтезом лейкотриенов"
                        },
                        {
                            reaction: "Аллергические реакции",
                            frequency: "Часто",
                            severity: "Легкая",
                            description: "Проявляются в виде крапивницы, ангионевротического отека"
                        }
                    ];
                    renderDrugDetails(drugId, mockReactions);
                }, 500);
            });
    }
    
    // Отображение подробной информации о препарате
    function renderDrugDetails(drugId, reactions) {
        if (!drugDetails || !drugName || !drugSubstance || !reactionsList) return;
        
        // Находим информацию о выбранном препарате
        selectedDrug = loadedDrugs.find(drug => drug.id === drugId);
        
        if (!selectedDrug) return;
        
        // Обновляем информацию о препарате
        drugName.textContent = selectedDrug.name;
        drugSubstance.textContent = selectedDrug.active_substance;
        
        // Обновляем таблицу с нежелательными реакциями
        if (reactions.length === 0) {
            reactionsList.innerHTML = '<tr><td colspan="4">Нежелательные реакции не найдены</td></tr>';
        } else {
            let html = '';
            reactions.forEach(reaction => {
                html += `
                    <tr>
                        <td>${reaction.reaction}</td>
                        <td>${reaction.frequency}</td>
                        <td>${reaction.severity}</td>
                        <td>${reaction.description}</td>
                    </tr>
                `;
            });
            reactionsList.innerHTML = html;
        }
        
        // Отображаем блок с деталями
        drugDetails.classList.remove('hidden');
        
        // Загружаем статистику о реакциях для графика
        loadDrugReactionsStats(selectedDrug.name);
    }
    
    // Загрузка статистики о нежелательных реакциях для выбранного препарата
    function loadDrugReactionsStats(drugName) {
        // Проверяем, существует ли контейнер для графика
        if (!document.getElementById('reactions-chart-container')) {
            // Создаем контейнер для графика, если его нет
            const chartContainer = document.createElement('div');
            chartContainer.id = 'reactions-chart-container';
            chartContainer.className = 'chart-container';
            chartContainer.innerHTML = `
                <h4>Топ-5 нежелательных реакций</h4>
                <canvas id="reactions-chart"></canvas>
            `;
            
            // Добавляем контейнер после таблицы реакций
            drugDetails.appendChild(chartContainer);
        }
        
        // Отправляем запрос к API для получения статистики
        fetch('/api/drug-reactions-stats', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ drug_name: drugName })
        })
        .then(response => response.json())
        .then(data => {
            // Проверяем, есть ли ошибка в ответе
            if (data.error) {
                console.warn('Ошибка API:', data.error);
                // Создаем демо-данные
                data = {
                    drug_name: drugName,
                    total_patients: 120,
                    patients_with_reactions: 45,
                    top_reactions: [
                        {reaction: "Головная боль", count: 18},
                        {reaction: "Тошнота", count: 12},
                        {reaction: "Сыпь", count: 8},
                        {reaction: "Головокружение", count: 7},
                        {reaction: "Усталость", count: 5}
                    ]
                };
            }
            renderReactionsChart(data);
        })
        .catch(error => {
            console.error('Ошибка при загрузке статистики о реакциях:', error);
            // Создаем демо-данные
            const demoData = {
                drug_name: drugName,
                total_patients: 120,
                patients_with_reactions: 45,
                top_reactions: [
                    {reaction: "Головная боль", count: 18},
                    {reaction: "Тошнота", count: 12},
                    {reaction: "Сыпь", count: 8},
                    {reaction: "Головокружение", count: 7},
                    {reaction: "Усталость", count: 5}
                ]
            };
            renderReactionsChart(demoData);
        });
    }
    
    // Отображение графика с нежелательными реакциями
    function renderReactionsChart(data) {
        if (!data.top_reactions || data.top_reactions.length === 0) {
            document.getElementById('reactions-chart-container').innerHTML = `
                <h4>Топ-5 нежелательных реакций</h4>
                <div class="no-data">Нет данных о нежелательных реакциях</div>
            `;
            return;
        }
        
        // Подготавливаем данные для графика
        const labels = data.top_reactions.map(item => item.reaction);
        const counts = data.top_reactions.map(item => item.count);
        
        // Если график уже существует, уничтожаем его
        if (reactionChart) {
            reactionChart.destroy();
        }
        
        // Создаем новый график
        const ctx = document.getElementById('reactions-chart').getContext('2d');
        reactionChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Количество случаев',
                    data: counts,
                    backgroundColor: 'rgba(54, 162, 235, 0.6)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: `Нежелательные реакции на ${data.drug_name}`
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Количество случаев: ${context.raw}`;
                            }
                        }
                    }
                }
            }
        });
    }
    
    // Поиск препаратов
    if (searchBtn && drugSearch) {
        searchBtn.addEventListener('click', function() {
            searchDrugs();
        });
        
        drugSearch.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                searchDrugs();
            }
        });
    }
    
    // Функция поиска препаратов
    function searchDrugs() {
        const query = drugSearch.value.trim().toLowerCase();
        
        if (query === '') {
            renderDrugsList(loadedDrugs);
            return;
        }
        
        const filteredDrugs = loadedDrugs.filter(drug => 
            drug.name.toLowerCase().includes(query) || 
            drug.active_substance.toLowerCase().includes(query)
        );
        
        renderDrugsList(filteredDrugs);
    }
    
    // Обработка загрузки файла
    if (uploadArea) {
        console.log('Настройка обработчиков для загрузки файла');
        
        // Клик на область загрузки
        uploadArea.addEventListener('click', function() {
            console.log('Клик на область загрузки');
            fileUpload.click();
        });
        
        // Drag & Drop
        uploadArea.addEventListener('dragover', function(e) {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });
        
        uploadArea.addEventListener('dragleave', function() {
            uploadArea.classList.remove('dragover');
        });
        
        uploadArea.addEventListener('drop', function(e) {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            console.log('Файл перетащен в область загрузки');
            
            if (e.dataTransfer.files.length > 0) {
                console.log('Найден перетащенный файл:', e.dataTransfer.files[0].name);
                handleFileSelect(e.dataTransfer.files[0]);
            }
        });
        
        // Выбор файла через диалог
        fileUpload.addEventListener('change', function() {
            console.log('Событие change на fileUpload, files:', this.files);
            if (this.files.length > 0) {
                console.log('Выбран файл через диалог:', this.files[0].name);
                handleFileSelect(this.files[0]);
            }
        });
        
        // Клик на ссылку "выберите файл"
        const uploadLink = document.querySelector('.upload-link');
        if (uploadLink) {
            console.log('Найдена upload-link');
            uploadLink.addEventListener('click', function(e) {
                console.log('Клик на upload-link');
                e.stopPropagation();
                fileUpload.click();
            });
        } else {
            console.error('Не найден элемент .upload-link');
        }
        
        // Удаление выбранного файла
        if (btnRemoveFile) {
            btnRemoveFile.addEventListener('click', function() {
                console.log('Клик на кнопку удаления файла');
                removeFile();
            });
        }
        
        // Анализ данных
        if (btnAnalyze) {
            btnAnalyze.addEventListener('click', function() {
                console.log('Клик на кнопку анализа');
                analyzeFile();
            });
        }
    }
    
    // Обработка выбора файла
    function handleFileSelect(file) {
        console.log('handleFileSelect вызван с файлом:', file);
        if (!file) {
            console.error('Файл не передан в handleFileSelect');
            return;
        }
        
        // Проверяем тип файла
        if (!file.name.endsWith('.csv')) {
            alert('Пожалуйста, выберите файл CSV');
            return;
        }
        
        console.log('Файл прошел проверку, устанавливаем selectedFile');
        selectedFile = file;
        
        // Обновляем информацию о файле
        fileName.textContent = file.name;
        fileSize.textContent = formatFileSize(file.size);
        
        console.log('Отображаем информацию о файле');
        // Отображаем информацию о файле и активируем кнопку анализа
        fileInfo.classList.remove('hidden');
        btnAnalyze.disabled = false;
        
        // Скрываем результаты предыдущего анализа
        analysisResults.classList.add('hidden');
    }
    
    // Удаление выбранного файла
    function removeFile() {
        console.log('removeFile вызван');
        selectedFile = null;
        fileUpload.value = '';
        fileInfo.classList.add('hidden');
        btnAnalyze.disabled = true;
        analysisResults.classList.add('hidden');
    }
    
    // Форматирование размера файла
    function formatFileSize(bytes) {
        if (bytes < 1024) {
            return bytes + ' байт';
        } else if (bytes < 1048576) {
            return (bytes / 1024).toFixed(1) + ' КБ';
        } else {
            return (bytes / 1048576).toFixed(1) + ' МБ';
        }
    }
    
    // Анализ файла
    function analyzeFile() {
        console.log('analyzeFile вызван, selectedFile:', selectedFile);
        if (!selectedFile) {
            console.error('selectedFile отсутствует в analyzeFile');
            return;
        }
        
        // Создаем объект FormData для отправки файла
        const formData = new FormData();
        formData.append('file', selectedFile);
        console.log('FormData создан, добавлен файл:', selectedFile.name);
        
        // Блокируем кнопку на время анализа
        btnAnalyze.disabled = true;
        btnAnalyze.textContent = 'Идет анализ...';
        
        // Отправляем запрос к API
        console.log('Отправляем запрос к /api/upload');
        fetch('/api/upload', {
            method: 'POST',
            body: formData
        })
        .then(response => {
            console.log('Получен ответ от сервера, статус:', response.status);
            if (!response.ok) {
                throw new Error(`Ошибка HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log('Данные получены:', data);
            renderAnalysisResults(data);
        })
        .catch(error => {
            console.error('Ошибка при анализе файла:', error);
            alert('Произошла ошибка при анализе файла: ' + error.message);
        })
        .finally(() => {
            btnAnalyze.textContent = 'Анализировать';
            btnAnalyze.disabled = false;
        });
    }
    
    // Отображение результатов анализа
    function renderAnalysisResults(data) {
        console.log('renderAnalysisResults вызван с данными:', data);
        
        if (!data) {
            console.error('В renderAnalysisResults не переданы данные');
            alert('Ошибка при анализе данных: нет данных от сервера');
            return;
        }
        
        if (!data.success || !data.results) {
            console.error('Ошибка в данных:', data);
            alert('Ошибка при анализе данных: ' + (data.error || 'неизвестная ошибка'));
            return;
        }
        
        const results = data.results;
        console.log('Обрабатываем результаты:', results);
        
        try {
            // Обновляем сводную информацию
            const totalPatientsEl = document.getElementById('total-patients');
            if (!totalPatientsEl) {
                console.error('Не найден элемент #total-patients');
            } else {
                totalPatientsEl.textContent = results.total_patients;
            }
            
            const reactionsCountEl = document.getElementById('reactions-count');
            if (!reactionsCountEl) {
                console.error('Не найден элемент #reactions-count');
            } else {
                reactionsCountEl.textContent = results.detected_reactions.length;
            }
            
            // Считаем количество критических реакций (условно, если count > 10)
            const criticalReactions = results.detected_reactions.filter(r => r.count > 10).length;
            
            const criticalReactionsEl = document.getElementById('critical-reactions');
            if (!criticalReactionsEl) {
                console.error('Не найден элемент #critical-reactions');
            } else {
                criticalReactionsEl.textContent = criticalReactions;
            }
            
            // Обновляем список выявленных реакций
            const detectedReactions = document.getElementById('detected-reactions');
            if (!detectedReactions) {
                console.error('Не найден элемент #detected-reactions');
                return;
            }
            
            detectedReactions.innerHTML = '';
            
            results.detected_reactions.forEach(reaction => {
                const card = document.createElement('div');
                card.className = 'reaction-card';
                
                // Добавляем препарат, связанный с реакцией, если есть эта информация
                const drugInfo = reaction.drug_name ? `<div class="reaction-drug">Препарат: ${reaction.drug_name}</div>` : '';
                
                card.innerHTML = `
                    <div class="reaction-name">${reaction.reaction}</div>
                    <div class="reaction-count">${reaction.count} пациентов (${reaction.percentage})</div>
                    ${drugInfo}
                    <div class="reaction-factors">
                        <h5>Факторы риска:</h5>
                        <ul>
                            ${reaction.factors.map(f => `<li>${f}</li>`).join('')}
                        </ul>
                    </div>
                `;
                
                detectedReactions.appendChild(card);
            });
            
            // Обновляем рекомендации
            const recommendationsList = document.getElementById('recommendations-list');
            if (!recommendationsList) {
                console.error('Не найден элемент #recommendations-list');
                return;
            }
            
            recommendationsList.innerHTML = '';
            
            // Добавляем стандартные рекомендации
            results.recommendations.forEach(recommendation => {
                const li = document.createElement('li');
                li.textContent = recommendation;
                recommendationsList.appendChild(li);
            });
            
            // Добавляем информацию о срочности дополнительных исследований, если есть
            if (results.drugs_urgency && results.drugs_urgency.length > 0) {
                // Добавляем заголовок для срочности исследований
                const urgencyHeader = document.createElement('li');
                urgencyHeader.className = 'urgency-header';
                urgencyHeader.innerHTML = '<strong>Срочность проведения дополнительных исследований:</strong>';
                recommendationsList.appendChild(urgencyHeader);
                
                // Добавляем информацию о каждом препарате
                results.drugs_urgency.forEach(drugUrgency => {
                    const li = document.createElement('li');
                    li.className = `urgency-item urgency-${drugUrgency.urgency_level.toLowerCase()}`;
                    li.innerHTML = `
                        <strong>${drugUrgency.drug_name}:</strong> 
                        Срочность: <span class="urgency-level">${drugUrgency.urgency_level}</span> - 
                        ${drugUrgency.reason}
                    `;
                    recommendationsList.appendChild(li);
                });
            } else {
                // Если данных по срочности нет, добавляем общую рекомендацию
                const liUrgency = document.createElement('li');
                liUrgency.innerHTML = '<strong>Рекомендуется:</strong> Провести дополнительные исследования для препаратов с высоким процентом нежелательных реакций';
                recommendationsList.appendChild(liUrgency);
            }
            
            // Добавляем график распределения реакций
            addReactionsChart(results.detected_reactions);
            
            // Отображаем результаты
            console.log('Отображаем блок с результатами анализа');
            const resultsBlock = document.getElementById('analysis-results');
            if (!resultsBlock) {
                console.error('Не найден элемент #analysis-results');
                return;
            }
            resultsBlock.classList.remove('hidden');
            
        } catch (error) {
            console.error('Ошибка при отображении результатов:', error);
            alert('Произошла ошибка при отображении результатов анализа: ' + error.message);
        }
    }
    
    // Функция для добавления графика распределения реакций
    function addReactionsChart(reactions) {
        // Проверяем, существует ли контейнер для графика
        if (!document.getElementById('analysis-chart-container')) {
            // Создаем контейнер для графика, если его нет
            const chartContainer = document.createElement('div');
            chartContainer.id = 'analysis-chart-container';
            chartContainer.className = 'chart-container';
            chartContainer.innerHTML = `
                <h4>Распределение нежелательных реакций</h4>
                <canvas id="analysis-chart"></canvas>
            `;
            
            // Добавляем контейнер перед рекомендациями
            const recommendations = document.querySelector('.recommendations');
            recommendations.parentNode.insertBefore(chartContainer, recommendations);
        }
        
        // Подготавливаем данные для графика
        const labels = reactions.map(item => item.reaction);
        const counts = reactions.map(item => item.count);
        const percentages = reactions.map(item => parseFloat(item.percentage.replace('%', '')));
        const backgroundColors = [
            'rgba(54, 162, 235, 0.6)',
            'rgba(255, 99, 132, 0.6)',
            'rgba(255, 206, 86, 0.6)',
            'rgba(75, 192, 192, 0.6)',
            'rgba(153, 102, 255, 0.6)'
        ];
        
        // Если график уже существует, уничтожаем его
        if (window.analysisChart) {
            window.analysisChart.destroy();
        }
        
        // Создаем новый график
        const ctx = document.getElementById('analysis-chart').getContext('2d');
        window.analysisChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Количество случаев',
                    data: counts,
                    backgroundColor: backgroundColors,
                    borderColor: backgroundColors.map(color => color.replace('0.6', '1')),
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: true,
                aspectRatio: 2,
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            precision: 0
                        }
                    }
                },
                plugins: {
                    title: {
                        display: true,
                        text: 'Частота нежелательных реакций'
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return `Количество случаев: ${context.raw} (${percentages[context.dataIndex]}%)`;
                            }
                        }
                    }
                }
            }
        });
    }

    // Функция запуска сбора данных из научных источников
    function startDataCollection() {
        const selectedSources = Array.from(document.querySelectorAll('.source-item input:checked'))
            .map(checkbox => checkbox.value);
        
        const query = document.getElementById('collection-query').value;
        const maxArticles = parseInt(document.getElementById('max-articles').value);
        const usePharmBerta = document.getElementById('use-pharm-berta').checked;
        
        if (!query) {
            showNotification('Ошибка', 'Введите поисковый запрос', 'error');
            return;
        }
        
        if (selectedSources.length === 0) {
            showNotification('Ошибка', 'Выберите хотя бы один источник данных', 'error');
            return;
        }
        
        // Показываем индикатор загрузки
        document.getElementById('collection-status').innerHTML = '<div class="spinner"></div><p>Выполняется сбор данных, пожалуйста, подождите...</p>';
        
        // Отправляем запрос на сервер
        fetch('/api/collect', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                sources: selectedSources,
                query: query,
                max_articles: maxArticles,
                use_pharm_berta: usePharmBerta
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`Ошибка HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                showNotification('Успех', data.message, 'success');
                document.getElementById('collection-status').innerHTML = `<p>${data.message}</p>`;
                
                // Если результаты уже есть в ответе, отображаем их сразу
                if (data.results) {
                    renderCollectionResults({
                        success: true,
                        query: query,
                        sources: selectedSources,
                        results_count: data.results_count,
                        results: data.results,
                        demo_mode: false
                    });
                }
                // Если результатов нет, не делаем дополнительных запросов
                else {
                    document.getElementById('collection-results').innerHTML = '<p>Не найдено данных о нежелательных реакциях по вашему запросу.</p>';
                }
            } else {
                showNotification('Ошибка', data.error || 'Неизвестная ошибка', 'error');
                document.getElementById('collection-status').innerHTML = `<p class="error">Ошибка: ${data.error || 'Неизвестная ошибка'}</p>`;
            }
        })
        .catch(error => {
            console.error('Ошибка:', error);
            showNotification('Ошибка', error.message, 'error');
            document.getElementById('collection-status').innerHTML = `<p class="error">Ошибка: ${error.message}</p>`;
        });
    }

    // Отслеживание статуса сбора данных
    let activeCollectionId = null;
    let statusCheckTimer = null;

    function checkCollectionStatus(collectionId) {
        // Если уже запущен таймер, останавливаем его
        if (statusCheckTimer) {
            clearTimeout(statusCheckTimer);
        }
        
        // Проверяем статус
        fetch(`/api/collection-status/${collectionId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Ошибка HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            // Обновляем статус на странице
            const statusElem = document.getElementById('collection-status');
            
            if (data.status === 'completed') {
                // Если сбор завершен, загружаем результаты
                statusElem.innerHTML = `<p>Сбор данных завершен. Собрано ${data.results_count} записей. Загружаем результаты...</p>`;
                loadCollectionResults(collectionId);
            } else if (data.status === 'error') {
                // Если произошла ошибка
                statusElem.innerHTML = `<p class="error">Ошибка при сборе данных: ${data.message}</p>`;
            } else {
                // Если сбор в процессе, показываем статус и индикатор загрузки
                statusElem.innerHTML = `
                    <div class="spinner"></div>
                    <p>${data.message || 'Сбор данных в процессе...'}</p>
                `;
                
                // Устанавливаем таймер для повторной проверки через 3 секунды
                statusCheckTimer = setTimeout(() => checkCollectionStatus(collectionId), 3000);
            }
        })
        .catch(error => {
            console.error('Ошибка при проверке статуса:', error);
            
            // При ошибке, пробуем загрузить результаты напрямую
            // Это нужно для демо-режима, где статус не всегда корректно отслеживается
            loadCollectionResults(collectionId);
        });
    }

    // Загрузка результатов сбора данных
    function loadCollectionResults(collectionId) {
        fetch(`/api/collection-results/${collectionId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`Ошибка HTTP: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            if (data.success) {
                // Отображаем результаты
                renderCollectionResults(data);
            } else {
                showNotification('Ошибка', data.error || 'Не удалось загрузить результаты', 'error');
                document.getElementById('collection-status').innerHTML = `<p class="error">Ошибка: ${data.error}</p>`;
            }
        })
        .catch(error => {
            console.error('Ошибка при загрузке результатов:', error);
            showNotification('Ошибка', error.message, 'error');
            
            // В демо-режиме, пробуем снова через 3 секунды
            // Это помогает если API только запустился и еще не готов
            statusCheckTimer = setTimeout(() => checkCollectionStatus(collectionId), 3000);
        });
    }

    // Отображение результатов сбора данных
    function renderCollectionResults(data) {
        const resultsContainer = document.getElementById('collection-results');
        const statusContainer = document.getElementById('collection-status');
        
        // Обновляем статус
        statusContainer.innerHTML = `<p>Сбор данных завершен. Собрано ${data.results_count} записей${data.demo_mode ? ' (демо-режим)' : ''}.</p>`;
        
        // Создаем HTML для отображения результатов
        let html = `
            <div class="collection-results">
                <h4>Результаты по запросу "${data.query}" из ${data.sources.length} источников</h4>
        `;
        
        if (data.results && data.results.length > 0) {
            // Группируем результаты по препаратам
            const drugGroups = {};
            
            data.results.forEach(item => {
                if (!item) return;
                
                const drugName = item.drug_name || 'Неизвестный препарат';
                
                if (!drugGroups[drugName]) {
                    drugGroups[drugName] = [];
                }
                
                drugGroups[drugName].push(item);
            });
            
            // Выводим результаты сгруппированные по препаратам
            Object.keys(drugGroups).sort().forEach(drugName => {
                const drugItems = drugGroups[drugName];
                
                html += `<div class="drug-group">
                    <h5>${drugName} (${drugItems.length})</h5>
                    <ul>`;
                
                drugItems.forEach(item => {
                    const sourceInfo = item.source_name ? `<strong>${item.source_name}</strong>` : '';
                    const reaction = item.adverse_reaction || item.reaction || 'Нет данных';
                    const severity = item.severity ? `<span class="severity-${item.severity.toLowerCase()}">${item.severity}</span>` : '';
                    const url = item.source_url ? `<a href="${item.source_url}" target="_blank">Источник</a>` : '';
                    
                    html += `<li>
                        ${reaction} ${severity} ${sourceInfo} ${url}
                    </li>`;
                });
                
                html += `</ul></div>`;
            });
        } else {
            html += `<p>Не найдено данных о нежелательных реакциях по вашему запросу.</p>`;
        }
        
        html += `</div>`;
        
        // Отображаем результаты
        resultsContainer.innerHTML = html;
        
        // Прокручиваем до результатов
        resultsContainer.scrollIntoView({ behavior: 'smooth' });
    }

    // Загрузка списка источников данных
    function loadDataSources() {
        const sourcesContainer = document.getElementById('sources-list');
        if (!sourcesContainer) return;
        
        sourcesContainer.innerHTML = '<div class="loading sources-loading">Загрузка источников...</div>';
        
        fetch('/api/sources')
            .then(response => response.json())
            .then(data => {
                sourcesContainer.innerHTML = '';
                
                // Проверка наличия данных
                if (!data || Object.keys(data).length === 0) {
                    sourcesContainer.innerHTML = '<div class="error">Не удалось загрузить источники данных</div>';
                    return;
                }
                
                // Группируем источники по категориям
                Object.keys(data).forEach(category => {
                    const categoryTitle = category.replace('_', ' ').split(' ')
                        .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                        .join(' ');
                    
                    const categorySection = document.createElement('div');
                    categorySection.className = 'source-category';
                    
                    const categoryHeader = document.createElement('h4');
                    categoryHeader.textContent = categoryTitle;
                    categorySection.appendChild(categoryHeader);
                    
                    const sourcesList = document.createElement('div');
                    sourcesList.className = 'sources-grid';
                    
                    data[category].forEach(source => {
                        const sourceItem = document.createElement('div');
                        sourceItem.className = 'source-item';
                        
                        const checkbox = document.createElement('input');
                        checkbox.type = 'checkbox';
                        checkbox.id = `source-${source.name.replace(/\s+/g, '-').toLowerCase()}`;
                        checkbox.value = source.name;
                        
                        const label = document.createElement('label');
                        label.htmlFor = checkbox.id;
                        label.textContent = source.name;
                        
                        const description = document.createElement('p');
                        description.className = 'source-description';
                        description.textContent = source.description;
                        
                        sourceItem.appendChild(checkbox);
                        sourceItem.appendChild(label);
                        sourceItem.appendChild(description);
                        sourcesList.appendChild(sourceItem);
                    });
                    
                    categorySection.appendChild(sourcesList);
                    sourcesContainer.appendChild(categorySection);
                });
            })
            .catch(error => {
                console.error('Ошибка при загрузке источников:', error);
                sourcesContainer.innerHTML = '<div class="error">Ошибка при загрузке источников данных</div>';
            });
    }

    // Функция для отображения уведомлений
    function showNotification(title, message, type = 'info') {
        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.innerHTML = `
            <h4>${title}</h4>
            <p>${message}</p>
            <button class="close-notification"><i class="fas fa-times"></i></button>
        `;
        
        document.body.appendChild(notification);
        
        // Анимация появления
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Автоматическое скрытие через 5 секунд
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 5000);
        
        // Обработка кнопки закрытия
        notification.querySelector('.close-notification').addEventListener('click', () => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        });
    }

    // Обработчик для кнопки обновления данных о препаратах
    function setupRefreshButton() {
        const refreshButton = document.getElementById('refresh-drugs-data');
        if (refreshButton) {
            refreshButton.addEventListener('click', function() {
                // Добавляем класс загрузки
                refreshButton.classList.add('loading');
                refreshButton.disabled = true;
                refreshButton.querySelector('i').classList.remove('fa-sync-alt');
                refreshButton.querySelector('i').classList.add('fa-spinner');
                
                // Отправляем запрос на обновление данных
                fetch('/api/refresh-data', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    }
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        showNotification('Обновление данных', data.message, 'info');
                        
                        // Через 10 секунд обновляем страницу с препаратами
                        setTimeout(() => {
                            loadDrugs();
                            
                            // Восстанавливаем кнопку
                            refreshButton.classList.remove('loading');
                            refreshButton.disabled = false;
                            refreshButton.querySelector('i').classList.remove('fa-spinner');
                            refreshButton.querySelector('i').classList.add('fa-sync-alt');
                        }, 10000);
                    } else {
                        showNotification('Ошибка', data.error || 'Не удалось обновить данные', 'error');
                        
                        // Восстанавливаем кнопку
                        refreshButton.classList.remove('loading');
                        refreshButton.disabled = false;
                        refreshButton.querySelector('i').classList.remove('fa-spinner');
                        refreshButton.querySelector('i').classList.add('fa-sync-alt');
                    }
                })
                .catch(error => {
                    console.error('Ошибка при обновлении данных:', error);
                    showNotification('Ошибка', 'Не удалось обновить данные', 'error');
                    
                    // Восстанавливаем кнопку
                    refreshButton.classList.remove('loading');
                    refreshButton.disabled = false;
                    refreshButton.querySelector('i').classList.remove('fa-spinner');
                    refreshButton.querySelector('i').classList.add('fa-sync-alt');
                });
            });
        }
    }

    // Добавим вызов setupRefreshButton() в инициализацию
    function setupDrugsSection() {
        // Поиск препаратов
        if (searchBtn && drugSearch) {
            searchBtn.addEventListener('click', function() {
                searchDrugs();
            });
            
            drugSearch.addEventListener('keypress', function(e) {
                if (e.key === 'Enter') {
                    searchDrugs();
                }
            });
        }
        
        // Настраиваем кнопку обновления данных
        setupRefreshButton();
    }
    
    // Обработчики для страницы загрузки
    function setupUploadSection() {
        // Обработка загрузки файла
        if (uploadArea) {
            console.log('Настройка обработчиков для загрузки файла');
            
            // Клик на область загрузки
            uploadArea.addEventListener('click', function() {
                console.log('Клик на область загрузки');
                fileUpload.click();
            });
            
            // Drag & Drop
            uploadArea.addEventListener('dragover', function(e) {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });
            
            uploadArea.addEventListener('dragleave', function() {
                uploadArea.classList.remove('dragover');
            });
            
            uploadArea.addEventListener('drop', function(e) {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                console.log('Файл перетащен в область загрузки');
                
                if (e.dataTransfer.files.length > 0) {
                    console.log('Найден перетащенный файл:', e.dataTransfer.files[0].name);
                    handleFileSelect(e.dataTransfer.files[0]);
                }
            });
            
            // Выбор файла через диалог
            fileUpload.addEventListener('change', function() {
                console.log('Событие change на fileUpload, files:', this.files);
                if (this.files.length > 0) {
                    console.log('Выбран файл через диалог:', this.files[0].name);
                    handleFileSelect(this.files[0]);
                }
            });
            
            // Клик на ссылку "выберите файл"
            const uploadLink = document.querySelector('.upload-link');
            if (uploadLink) {
                console.log('Найдена upload-link');
                uploadLink.addEventListener('click', function(e) {
                    console.log('Клик на upload-link');
                    e.stopPropagation();
                    fileUpload.click();
                });
            } else {
                console.error('Не найден элемент .upload-link');
            }
            
            // Удаление выбранного файла
            if (btnRemoveFile) {
                btnRemoveFile.addEventListener('click', function() {
                    console.log('Клик на кнопку удаления файла');
                    removeFile();
                });
            }
            
            // Анализ данных
            if (btnAnalyze) {
                btnAnalyze.addEventListener('click', function() {
                    console.log('Клик на кнопку анализа');
                    analyzeFile();
                });
            }
        } else {
            console.error('Не найден элемент uploadArea');
        }
        
        // Обработчики для раздела сбора данных
        const btnStartCollection = document.getElementById('start-collection');
        if (btnStartCollection) {
            btnStartCollection.addEventListener('click', startDataCollection);
        }
    }
    
    // Инициализация
    loadDrugs();
    setupDrugsSection();
    setupUploadSection();
    
    // Остальные функции и обработчики...
    // ... код из файла ...
}); 