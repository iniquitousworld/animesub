import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple

import torchaudio

torchaudio.set_audio_backend("ffmpeg")


def convert_to_wav(input_path: str, output_path: str, sr: int = 44100) -> None:
    """
    Конвертирует любой видео или аудиофайл в WAV с помощью ffmpeg.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-ar", str(sr),
        "-ac", "1",
        "-vn",
        output_path,
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)


def separate_vocals(input_path: str, model_name: str = "htdemucs", device: str = "cuda") -> Tuple[str, str]:
    """
    Выполняет вокальную сепарацию с помощью Demucs. 
    Fallback: если не получилось, возвращается исходный wav.
    
    Returns:
        Tuple[str, str]: путь к выходному аудио, путь к временной директории.
    """
    logging.info(f"Отделение вокала из {input_path} с моделью {model_name}...")

    temp_dir = tempfile.mkdtemp()
    wav_path = os.path.join(temp_dir, "converted.wav")

    try:
        logging.debug("Конвертация входного файла в WAV...")
        convert_to_wav(input_path, wav_path)
    except subprocess.CalledProcessError:
        logging.error("Не удалось сконвертировать файл в WAV через ffmpeg.")
        return input_path, temp_dir

    try:
        logging.debug("Запуск Demucs...")
        subprocess.run(
            [
                "demucs",
                "--two-stems=vocals",
                f"-n={model_name}",
                f"--device={device}",
                "-o", temp_dir,
                wav_path,
            ],
            check=True
        )
        # Получение пути к результату
        track_name = Path(wav_path).stem
        vocals_path = os.path.join(temp_dir, model_name, track_name, "vocals.wav")
        if os.path.exists(vocals_path):
            logging.info("Вокал успешно извлечён.")
            return vocals_path, temp_dir
        else:
            raise FileNotFoundError("Файл vocals.wav не найден.")
    except Exception as e:
        logging.warning(f"Demucs не смог обработать файл: {e}")
        logging.warning("Будет использован исходный .wav без отделения вокала.")
        return wav_path, temp_dir
