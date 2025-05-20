#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run.py - Скрипт запуска системы мониторинга нежелательных реакций на лекарственные препараты
Версия 2025.1
"""

import os
import sys
import logging
import subprocess
import time
import signal

# Настройка логирования
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('system')

# Отключаем использование PharmBertA по умолчанию из-за проблем с совместимостью
os.environ['USE_PHARMBERTA'] = '0'
logger.info("PharmBertA отключена для обеспечения совместимости")

def is_port_in_use(port):
    """Проверяет, занят ли порт"""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def kill_process_on_port(port):
    """Завершает процесс, который использует указанный порт"""
    try:
        # Пытаемся найти PID процесса, который слушает порт
        if sys.platform == 'win32':
            # Для Windows
            cmd = f'netstat -ano | findstr :{port}'
            output = subprocess.check_output(cmd, shell=True).decode('utf-8')
            if output:
                # Извлекаем PID из последней колонки
                pid = output.strip().split()[-1]
                # Завершаем процесс
                subprocess.run(f'taskkill /PID {pid} /F', shell=True)
                logger.info(f"Завершен процесс с PID {pid}, который использовал порт {port}")
        else:
            # Для Unix/Linux/Mac
            cmd = f"lsof -i :{port} | grep LISTEN | awk '{{print $2}}'"
            try:
                output = subprocess.check_output(cmd, shell=True).decode('utf-8')
                if output:
                    pids = output.strip().split('\n')
                    for pid in pids:
                        if pid:
                            os.kill(int(pid), signal.SIGTERM)
                            logger.info(f"Завершен процесс с PID {pid}, который использовал порт {port}")
            except subprocess.CalledProcessError:
                # Команда не нашла процессов
                pass
    except Exception as e:
        logger.warning(f"Не удалось завершить процесс на порту {port}: {str(e)}")

def main():
    """Основная функция запуска системы"""
    port = 5001
    
    # Проверяем, занят ли порт
    if is_port_in_use(port):
        logger.warning(f"Порт {port} уже используется. Пытаемся остановить процесс...")
        kill_process_on_port(port)
        # Даем время на освобождение порта
        time.sleep(1)
        if is_port_in_use(port):
            logger.error(f"Не удалось освободить порт {port}. Пожалуйста, проверьте вручную.")
            print(f"Ошибка: Порт {port} занят другой программой. Остановите программу или измените порт.")
            sys.exit(1)
    
    # Проверяем наличие папки для загруженных файлов
    upload_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'uploads')
    os.makedirs(upload_dir, exist_ok=True)
    
    # Запускаем приложение Flask
    try:
        print("Запуск системы мониторинга нежелательных реакций...")
        
        # Определяем путь к backend/app.py
        backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend')
        app_path = os.path.join(backend_dir, 'app.py')
        
        # Проверяем наличие виртуального окружения
        venv_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'venv')
        venv_python = os.path.join(venv_dir, 'bin', 'python') if not sys.platform.startswith('win') else os.path.join(venv_dir, 'Scripts', 'python.exe')
        
        if os.path.exists(venv_python):
            # Если есть виртуальное окружение, используем его
            logger.info("Использование виртуального окружения для запуска")
            # Устанавливаем FLASK_APP и запускаем
            os.environ['FLASK_APP'] = app_path
            os.environ['FLASK_ENV'] = 'development'
            # Добавляем backend в путь PYTHONPATH
            python_path = os.environ.get('PYTHONPATH', '')
            os.environ['PYTHONPATH'] = f"{backend_dir}:{python_path}" if python_path else backend_dir
            
            cmd = [venv_python, '-m', 'flask', 'run', '--host=0.0.0.0', f'--port={port}']
            subprocess.run(cmd)
        else:
            # Иначе используем системный Python
            logger.info("Использование системного Python для запуска")
            os.environ['FLASK_APP'] = app_path
            os.environ['FLASK_ENV'] = 'development'
            # Добавляем backend в путь PYTHONPATH
            python_path = os.environ.get('PYTHONPATH', '')
            os.environ['PYTHONPATH'] = f"{backend_dir}:{python_path}" if python_path else backend_dir
            
            # Запускаем Flask напрямую
            subprocess.run([sys.executable, '-m', 'flask', 'run', '--host=0.0.0.0', f'--port={port}'])
    except Exception as e:
        logger.error(f"Ошибка при запуске сервера: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 