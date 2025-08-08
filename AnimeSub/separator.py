import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple, List

import torchaudio
import soundfile as sf
import numpy as np

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


def chunk_audio(wav_path: str, chunk_duration: float = 90.0) -> List[str]:
    """
    Делит WAV-файл на чанки фиксированной длины.
    Возвращает список путей к чанкам.
    """
    waveform, sample_rate = torchaudio.load(wav_path)
    samples_per_chunk = int(chunk_duration * sample_rate)

    chunk_paths = []
    for i in range(0, waveform.size(1), samples_per_chunk):
        chunk = waveform[:, i:i + samples_per_chunk]
        chunk_path = str(Path(wav_path).with_stem(f"{Path(wav_path).stem}_chunk_{len(chunk_paths)}"))
        chunk_path += ".wav"
        torchaudio.save(chunk_path, chunk, sample_rate)
        chunk_paths.append(chunk_path)
    return chunk_paths


def concat_chunks(chunk_paths: List[str], output_path: str) -> None:
    """
    Склеивает чанки в один WAV-файл.
    """
    audio_data = []
    sample_rate = None

    for path in chunk_paths:
        data, sr = sf.read(path)
        audio_data.append(data)
        sample_rate = sr

    combined = np.concatenate(audio_data, axis=0)
    sf.write(output_path, combined, sample_rate)


def separate_vocals(input_path: str, model_name: str = "htdemucs", device: str = "cuda") -> Tuple[str, str]:
    """
    Выполняет вокальную сепарацию с помощью Demucs.
    Если длинный файл на CPU — делит на чанки.
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
        # Получим длительность аудио
        info = torchaudio.info(wav_path)
        duration = info.num_frames / info.sample_rate
        logging.debug(f"Длительность WAV-файла: {duration:.2f} секунд")

        use_chunking = device == "cpu" and duration > 300

        if use_chunking:
            logging.info("Файл длинный и используется CPU — включено разбиение на чанки.")
            chunk_paths = chunk_audio(wav_path, chunk_duration=90.0)

            vocals_chunks = []
            for chunk_path in chunk_paths:
                logging.debug(f"Обработка чанка: {chunk_path}")
                subprocess.run(
                    [
                        "demucs",
                        "--two-stems=vocals",
                        f"-n={model_name}",
                        f"--device={device}",
                        "-o", temp_dir,
                        chunk_path,
                    ],
                    check=True
                )
                track_name = Path(chunk_path).stem
                vocals_path = os.path.join(temp_dir, model_name, track_name, "vocals.wav")
                if os.path.exists(vocals_path):
                    vocals_chunks.append(vocals_path)
                else:
                    raise FileNotFoundError(f"Вокал не найден для чанка: {chunk_path}")

            final_vocals_path = os.path.join(temp_dir, "vocals_final.wav")
            logging.debug("Склейка всех вокальных чанков...")
            concat_chunks(vocals_chunks, final_vocals_path)
            logging.info("Вокал успешно извлечён по чанкам.")
            return final_vocals_path, temp_dir

        else:
            logging.info("Обычный режим обработки (без чанков)...")
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
