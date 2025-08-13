import logging
import logging.handlers
import sys
import os
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue

# Попытка импортировать psutil для мониторинга памяти
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
logger = logging.getLogger(__name__)

def get_memory_usage() -> str:
    """Возвращает текущее использование RSS-памяти процессом."""
    if not PSUTIL_AVAILABLE:
        return "N/A (psutil not installed)"
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss
    return f"{mem_bytes / 1024 / 1024:.2f} MB"

def _handle_exception(exc_type, exc_value, exc_traceback):
    """Логирует неперехваченные исключения."""
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logger.critical("Uncaught exception:", exc_info=(exc_type, exc_value, exc_traceback))

def setup_logging(log_dir: Path = Path("logs")) -> Path:
    """
    Настраивает потокобезопасное логирование:
    - В файл — подробный лог
    - В консоль — компактный цветной лог (если терминал поддерживает ANSI)
    """
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    pid = os.getpid()
    log_file_path = log_dir / f"session-{timestamp}-{pid}.log"

    log_queue = Queue(-1)

    # Форматтер для файла (подробный)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s:%(funcName)s] - %(message)s'
    )

    # Цветной форматтер для консоли (компактный)
    class ColorFormatter(logging.Formatter):
        COLORS = {
            logging.DEBUG: "\033[37m",     # серый
            logging.INFO: "\033[36m",      # голубой
            logging.WARNING: "\033[33m",   # жёлтый
            logging.ERROR: "\033[31m",     # красный
            logging.CRITICAL: "\033[41m",  # белый на красном фоне
        }
        RESET = "\033[0m"

        def __init__(self, fmt, enable_colors=True):
            super().__init__(fmt)
            self.enable_colors = enable_colors

        def format(self, record):
            message = super().format(record)
            if self.enable_colors:
                color = self.COLORS.get(record.levelno, self.RESET)
                return f"{color}{message}{self.RESET}"
            return message

    enable_colors = sys.stdout.isatty()

    # Обработчик для файла
    file_handler = logging.FileHandler(log_file_path, encoding='utf-8')
    file_handler.setFormatter(file_formatter)

    # Обработчик для консоли (компактно)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        ColorFormatter('%(levelname)s: %(message)s',
                       enable_colors=enable_colors)
    )

    # Слушатель
    listener = logging.handlers.QueueListener(log_queue, file_handler, console_handler)
    listener.start()

    # Корневой логгер
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Чистим старые обработчики
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.addHandler(logging.handlers.QueueHandler(log_queue))

    # Глобальный хук исключений
    sys.excepthook = _handle_exception
    if hasattr(threading, 'excepthook'):
        threading.excepthook = lambda args: _handle_exception(args.exc_type, args.exc_value, args.exc_traceback)

    logger.info(f"Логирование настроено. Файл сессии: {log_file_path}")
    logger.info(f"Начальное использование памяти: {get_memory_usage()}")

    return log_file_path

def resource_path(relative_path: str) -> str:
    """
    Возвращает абсолютный путь к ресурсу (иконкам, изображениям, моделям и т.д.).
    Работает как при обычном запуске, так и в собранном .exe.
    """
    try:
        base_path = sys._MEIPASS  # PyInstaller
    except AttributeError:
        base_path = os.path.abspath(os.path.dirname(__file__))  # Папка animesub
    return os.path.join(base_path, relative_path)