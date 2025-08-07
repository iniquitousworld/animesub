import os
import argparse
import tempfile
import shutil
import torch
import logging
import sys
from typing import List, Dict, Tuple

from .asr_whisper import transcribe_segments  # Для других моделей
from .asr_kotoba import transcribe_segments as transcribe_kotoba  # Специфично для kotoba
from .separator import separate_vocals
from .vad_detector import detect_speech_segments
from .punctuator import add_punctuation_with_xlm
from .srt_formatter import segments_to_srt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_close_segments(timestamps: List[Dict], max_silence_s: float = 0.6) -> List[Dict]:
    if not timestamps:
        return []
    merged_timestamps = []
    current_segment = timestamps[0].copy()
    for next_segment in timestamps[1:]:
        if (next_segment['start'] - current_segment['end']) < max_silence_s:
            current_segment['end'] = next_segment['end']
        else:
            merged_timestamps.append(current_segment)
            current_segment = next_segment.copy()
    merged_timestamps.append(current_segment)
    return merged_timestamps

def process_audio(input_path: str, output_path: str, model_name: str, device: str):
    """
    Полный конвейер обработки аудиофайла для генерации субтитров с выбором ASR движка.
    """
    logging.info("--- Запуск процесса создания субтитров ---")
    logging.info(f"Текущее время: 07:52 AM JST, 08 августа 2025 года")

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.info(f"Создана временная директория: {temp_dir}")

        # --- Шаг 1: Отделение вокала ---
        logging.info(f"[1/5] Отделение вокала из файла: {input_path}")
        vocals_path = separate_vocals(input_path, device=device, output_dir=temp_dir)
        if not vocals_path:
            logging.critical("Не удалось отделить вокал. Процесс прерван.")
            return
        logging.info(f"Вокал успешно сохранен в: {vocals_path}")

        # --- Шаг 2: Детекция речи (VAD) ---
        logging.info("[2/5] Обнаружение сегментов речи...")
        speech_timestamps, waveform, sample_rate = detect_speech_segments(vocals_path)
        if not speech_timestamps:
            logging.critical("Речь не обнаружена. Процесс прерван.")
            return
        logging.info(f"Обнаружено {len(speech_timestamps)} больших сегментов речи.")

        # --- Шаг 3: Транскрибация речи (ASR) ---
        logging.info("[3/5] Транскрибация сегментов...")
        if model_name.lower() == ("kotoba-whisper-v2.2"):
            transcribed_segments = transcribe_kotoba(
                audio_path=vocals_path,
                speech_timestamps=speech_timestamps,
                waveform=waveform,
                sample_rate=sample_rate,
                model_name=model_name,
                device=device
            )
        else:
            transcribed_segments = transcribe_segments(
                audio_path=vocals_path,
                speech_timestamps=speech_timestamps,
                waveform=waveform,
                sample_rate=sample_rate,
                model_name=model_name,
                device=device
            )

        if not transcribed_segments:
            logging.critical("Транскрипция не удалась. Процесс прерван.")
            return
        logging.info("Транскрибация завершена.")

        # --- Шаг 4: Добавление пунктуации ---
        logging.info("[4/5] Добавление пунктуации...")
        texts_to_punctuate = [seg['text'] for seg in transcribed_segments if seg['text']]
        if texts_to_punctuate:
            punctuated_lists = add_punctuation_with_xlm(texts_to_punctuate)
            punctuated_texts = ["".join(sentences) for sentences in punctuated_lists]  # Без пробелов для японского
            text_iterator = iter(punctuated_texts)
            for segment in transcribed_segments:
                if segment['text']:
                    try:
                        segment['text'] = next(text_iterator)
                    except StopIteration:
                        logging.warning("Предупреждение: количество сегментов с текстом и результатов пунктуации не совпадает.")
                        break
        logging.info("Пунктуация добавлена.")

        # --- Шаг 5: Форматирование в SRT и сохранение ---
        logging.info("[5/5] Форматирование в SRT и сохранение файла...")
        srt_content = segments_to_srt(transcribed_segments)
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            logging.info("\n--- Процесс успешно завершен! ---")
            logging.info(f"Субтитры сохранены в файл: {output_path}")
        except IOError as e:
            logging.critical(f"Ошибка при записи в файл '{output_path}': {e}")
    logging.info("Временные файлы очищены.")

def main():
    """
    Главная функция для запуска из командной строки.
    """
    parser = argparse.ArgumentParser(
        description="Создает субтитры (.srt) из любого видео или аудиофайла с выбором ASR движка.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Путь к исходному видео/аудио файлу."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        help="Путь к итоговому .srt файлу (по умолчанию: имя_файла.srt)."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default="kotoba-whisper-v2.2",
        help="Имя модели для выбранного движка (по умолчанию: 'kotoba-whisper-v2.2')."
    )
    parser.add_argument(
        "-d", "--device",
        type=str,
        default=None,
        choices=['cpu', 'cuda'],
        help="Устройство для вычислений (cpu или cuda). По умолчанию определяется автоматически."
    )
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        logging.critical("Ошибка: FFmpeg не найден. Пожалуйста, установите FFmpeg и убедитесь, что он доступен в системном PATH.")
        return
    
    if not os.path.exists(args.input_file):
        logging.critical(f"Ошибка: Входной файл не найден по пути: {args.input_file}")
        return

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.warning(f"Устройство не указано, используется автоопределение: {device.upper()}")
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            logging.error("Ошибка: Выбрано устройство 'cuda', но оно недоступно. Используется 'cpu'.")
            device = "cpu"

    if args.output:
        output_file_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_file_path = f"{base_name}.srt"
        
    process_audio(args.input_file, output_file_path, args.model, device)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    main()