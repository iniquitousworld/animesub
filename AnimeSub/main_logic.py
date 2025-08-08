import os
import argparse
import tempfile
import shutil
import torch
import logging
import sys
from typing import List, Dict

from .asr_whisper import transcribe_segments
from .asr_kotoba import transcribe_segments as transcribe_kotoba
from .separator import separate_vocals
from .vad_detector import detect_speech_segments
from .punctuator import add_punctuation_with_xlm
from .srt_formatter import segments_to_srt

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def merge_close_segments(timestamps: List[Dict], max_silence_s: float = 0.6) -> List[Dict]:
    logging.debug(f"Объединение сегментов, max_silence_s={max_silence_s}")
    if not timestamps:
        logging.warning("Список таймстампов пуст")
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
    logging.debug(f"Объединено {len(merged_timestamps)} сегментов")
    return merged_timestamps

def process_audio(input_path: str, output_path: str, model_name: str, device: str, demucs_model: str = "htdemucs"):
    logging.info("--- Запуск процесса создания субтитров ---")
    logging.info(f"Входной файл: {input_path}, Выходной файл: {output_path}, Модель: {model_name}, Устройство: {device}")

    with tempfile.TemporaryDirectory() as temp_dir:
        logging.debug(f"Создана временная директория: {temp_dir}")

        # Шаг 1: Отделение вокала
        logging.info("[1/5] Отделение вокала...")
        vocals_path = separate_vocals(input_path, device=device, output_dir=temp_dir, demucs_model=demucs_model)
        if not vocals_path:
            logging.error("Не удалось отделить вокал. Процесс прерван.")
            return
        logging.info(f"Вокал сохранен в: {vocals_path}")

        # Шаг 2: Детекция речи (VAD)
        logging.info("[2/5] Обнаружение сегментов речи...")
        speech_timestamps, waveform, sample_rate = detect_speech_segments(vocals_path)
        if not speech_timestamps:
            logging.error("Речь не обнаружена. Процесс прерван.")
            return
        logging.info(f"Обнаружено {len(speech_timestamps)} сегментов речи")

        # Шаг 3: Транскрипция
        logging.info("[3/5] Транскрипция сегментов...")
        try:
            if model_name.lower() in ("kotoba-whisper", "kotoba-whisper-v2.2"):
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
        except Exception as e:
            logging.error(f"Ошибка транскрипции: {e}")
            return

        if not transcribed_segments:
            logging.error("Транскрипция не удалась: пустой результат. Процесс прерван.")
            return
        logging.info(f"Транскрибировано {len(transcribed_segments)} сегментов")

        # Шаг 4: Добавление пунктуации
        logging.info("[4/5] Добавление пунктуации...")
        texts_to_punctuate = [seg['text'] for seg in transcribed_segments if seg['text']]
        if texts_to_punctuate:
            try:
                punctuated_lists = add_punctuation_with_xlm(texts_to_punctuate)
                punctuated_texts = ["".join(sentences) for sentences in punctuated_lists]
                text_iterator = iter(punctuated_texts)
                for segment in transcribed_segments:
                    if segment['text']:
                        try:
                            segment['text'] = next(text_iterator)
                        except StopIteration:
                            logging.warning("Несоответствие количества текстов и сегментов")
                            break
            except Exception as e:
                logging.error(f"Ошибка пунктуации: {e}")
                return
        else:
            logging.warning("Нет текстов для пунктуации")
        logging.info("Пунктуация завершена")

        # Шаг 5: Форматирование в SRT
        logging.info("[5/5] Форматирование в SRT...")
        try:
            srt_content = segments_to_srt(transcribed_segments)
            if not srt_content:
                logging.error("SRT-контент пуст. Файл не будет создан.")
                return
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(srt_content)
            logging.info(f"Субтитры успешно сохранены в: {output_path}")
        except Exception as e:
            logging.error(f"Ошибка записи SRT файла: {e}")
            return
    logging.info("Транскрибация прошла успешно. Временные файлы удалены")

def main():
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
    parser.add_argument(
        "--demucs-model",
        type=str,
        default="htdemucs",
        choices=['htdemucs', 'mdx_extra_q'],
        help="Модель Demucs для отделения вокала (по умолчанию: 'htdemucs')."
    )
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        logging.critical("Ошибка: FFmpeg не найден. Пожалуйста, установите FFmpeg.")
        return
    
    if not os.path.exists(args.input_file):
        logging.critical(f"Ошибка: Входной файл не найден: {args.input_file}")
        return

    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.warning(f"Устройство не указано, используется: {device.upper()}")
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            logging.error("Ошибка: CUDA недоступна. Используется CPU.")
            device = "cpu"

    if args.output:
        output_file_path = args.output
    else:
        base_name = os.path.splitext(os.path.basename(args.input_file))[0]
        output_file_path = f"{base_name}.srt"
        
    process_audio(args.input_file, output_file_path, args.model, device, demucs_model=args.demucs_model)

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(current_dir)
    main()