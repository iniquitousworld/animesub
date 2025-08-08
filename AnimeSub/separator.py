import subprocess
import logging
import shutil
from pathlib import Path
import torchaudio

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def separate_vocals(input_path: str, device: str = "cpu", output_dir: str = None, demucs_model: str = "htdemucs") -> str:
    """
    Отделяет вокал из аудио/видео файла, используя Demucs.

    Args:
        input_path: Путь к входному аудио/видео файлу.
        device: Устройство для вычислений ('cpu' или 'cuda').
        output_dir: Директория для сохранения результата.
        demucs_model: Модель Demucs (например, 'htdemucs', 'mdx_extra_q').

    Returns:
        Путь к файлу с вокалом или None в случае ошибки.
    """
    logging.info(f"Отделение вокала из {input_path} с моделью {demucs_model}...")
    
    # Проверка наличия FFmpeg
    if not shutil.which("ffmpeg"):
        logging.critical("FFmpeg не найден. Установите FFmpeg через 'conda install ffmpeg -c conda-forge' или добавьте в PATH.")
        return None

    # Установка бэкенда torchaudio
    try:
        torchaudio.set_audio_backend("ffmpeg")
        logging.info("Бэкенд torchaudio установлен: ffmpeg")
    except Exception as e:
        logging.error(f"Ошибка установки бэкенда ffmpeg в torchaudio: {e}")
        return None

    input_path = Path(input_path)
    if not input_path.exists():
        logging.error(f"Входной файл не найден: {input_path}")
        return None

    if output_dir is None:
        output_dir = input_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    output_subdir = output_dir / demucs_model / input_path.stem
    vocals_path = output_subdir / "vocals.wav"

    if vocals_path.exists():
        logging.info(f"Вокал уже существует: {vocals_path}")
        return str(vocals_path)

    command = [
        "demucs",
        "--two-stems=vocals",
        f"-n={demucs_model}",
        f"--device={device}",
        "-o",
        str(output_dir),
        str(input_path)
    ]

    logging.info(f"Запуск команды: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.debug(f"Вывод demucs: {result.stdout}")
        logging.debug(f"Ошибки demucs (если есть): {result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Ошибка при выполнении demucs: {e}")
        logging.error(f"Вывод: {e.stdout}")
        logging.error(f"Ошибки: {e.stderr}")
        return None
    except FileNotFoundError:
        logging.critical("Команда 'demucs' не найдена. Убедитесь, что пакет demucs установлен ('pip install demucs').")
        return None

    if not vocals_path.exists():
        logging.error(f"Файл вокала не создан: {vocals_path}")
        return None

    logging.info(f"Вокал успешно отделен и сохранен в {vocals_path}")
    return str(vocals_path)