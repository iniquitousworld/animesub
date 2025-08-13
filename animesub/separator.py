import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Tuple
import soundfile as sf  # для определения длительности

logger = logging.getLogger(__name__)

CHUNK_DURATION_SEC = 8 * 60  # 8 минут

if sys.platform == "win32":
    creationflags = subprocess.CREATE_NO_WINDOW
else:
    creationflags = 0  # Для не-Windows систем

def check_cancel(cancel_event):
    if cancel_event and hasattr(cancel_event, "is_set") and cancel_event.is_set():
        raise InterruptedError("Отмена в separator.py")

def get_audio_duration(path: str) -> float:
    """Возвращает длительность аудио в секундах."""
    try:
        f = sf.SoundFile(path)
        return len(f) / f.samplerate
    except Exception:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries",
            "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True,
        creationflags=creationflags)
        return float(result.stdout.strip())

def run_demucs_subprocess(wav_path: str, model_name: str, device: str, temp_dir: str):
    logger.info(f"Запуск Demucs в подпроцессе для {wav_path}")
    python_exe = sys._base_executable if hasattr(sys, "_base_executable") else sys.executable
    cmd = [
        python_exe, "-m", "demucs.separate",
        "--two-stems", "vocals",
        "-n", model_name,
        "--device", device,
        "-o", temp_dir,
        wav_path
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", creationflags=creationflags)
        logger.debug(f"Demucs stdout: {result.stdout}")
        logger.info("Demucs успешно завершил работу.")
    except subprocess.CalledProcessError as e:
        logger.error(f"Demucs завершился с ошибкой (код {e.returncode}).")
        logger.error(f"Demucs stderr: {e.stderr}")
        raise RuntimeError("Ошибка выполнения Demucs. Подробности в логе.")

def process_single_file(wav_path, model_name, device, temp_dir, cancel_event):
    check_cancel(cancel_event)
    run_demucs_subprocess(wav_path, model_name, device, temp_dir)
    check_cancel(cancel_event)
    vocals_path = os.path.join(temp_dir, model_name, "input", "vocals.wav")
    if os.path.exists(vocals_path):
        return vocals_path
    return wav_path

def separate_vocals(
    input_path: str,
    model_name: str = "htdemucs",
    device: str = "cuda",
    existing_temp_dir: str = None,
    cancel_event=None
) -> Tuple[str, str]:
    check_cancel(cancel_event)
    logger.info(f"Начало отделения вокала из {input_path} с моделью {model_name}.")

    temp_dir = existing_temp_dir or tempfile.mkdtemp(prefix="demucs_")

    wav_path = os.path.join(temp_dir, "input.wav")
    if not input_path.lower().endswith(".wav"):
        logger.info(f"Конвертация {input_path} в {wav_path} для Demucs.")
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, "-ar", "44100", "-ac", "2", "-vn", wav_path
        ], check=True, capture_output=True, creationflags=creationflags)
    else:
        if Path(input_path) != Path(wav_path):
            subprocess.run(["ffmpeg", "-y", "-i", input_path, wav_path],
                           check=True, capture_output=True, creationflags=creationflags)

    duration = get_audio_duration(wav_path)
    logger.info(f"Длительность аудио: {duration:.2f} сек.")

    if duration > CHUNK_DURATION_SEC:
        logger.info("Аудио дольше 8 минут — выполняем разбиение на чанки.")
        chunks_dir = os.path.join(temp_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)

        subprocess.run([
            "ffmpeg", "-i", wav_path, "-f", "segment", "-segment_time", str(CHUNK_DURATION_SEC),
            "-c", "copy", os.path.join(chunks_dir, "chunk_%03d.wav")
        ], check=True, capture_output=True, creationflags=creationflags)

        processed_chunks = []
        for chunk_file in sorted(Path(chunks_dir).glob("chunk_*.wav")):
            logger.info(f"Обработка чанка {chunk_file}")
            chunk_temp = tempfile.mkdtemp(prefix="demucs_chunk_", dir=temp_dir)
            processed_vocals = process_single_file(str(chunk_file), model_name, device, chunk_temp, cancel_event)
            processed_chunks.append(processed_vocals)

        concat_list = os.path.join(temp_dir, "concat_list.txt")
        with open(concat_list, "w", encoding="utf-8") as f:
            for p in processed_chunks:
                f.write(f"file '{p}'\n")

        merged_vocals = os.path.join(temp_dir, "vocals_merged.wav")
        subprocess.run([
            "ffmpeg", "-f", "concat", "-safe", "0", "-i", concat_list, "-c", "copy", merged_vocals
        ], check=True, capture_output=True, creationflags=creationflags)

        logger.info("Вокал успешно извлечён и склеен из чанков.")
        return merged_vocals, temp_dir
    else:
        logger.info("Аудио короче или равно 8 минутам — обрабатываем целиком.")
        vocals_path = process_single_file(wav_path, model_name, device, temp_dir, cancel_event)
        return vocals_path, temp_dir
