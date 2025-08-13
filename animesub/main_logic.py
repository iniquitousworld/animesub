import os
import argparse
import logging
import shutil
import tempfile
import torch
from typing import List, Dict, Callable, Optional

from animesub.model_manager import ModelManager
from animesub.utils import get_memory_usage


logger = logging.getLogger(__name__)

def check_cancel(cancel_event):
    """Проверяет флаг отмены и выбрасывает исключение для остановки."""
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Процесс был отменён пользователем.")

def merge_close_segments(timestamps: List[Dict], max_silence_s: float = 0.6) -> List[Dict]:
    if not timestamps:
        logger.warning("Список таймстампов для объединения пуст.")
        return []
    logger.debug(f"Объединение сегментов с максимальной паузой {max_silence_s}s")
    merged_timestamps = []
    current_segment = timestamps[0].copy()
    for next_segment in timestamps[1:]:
        if (next_segment['start'] - current_segment['end']) < max_silence_s:
            current_segment['end'] = next_segment['end']
        else:
            merged_timestamps.append(current_segment)
            current_segment = next_segment.copy()
    merged_timestamps.append(current_segment)
    logger.info(f"{timestamps} Старые сегменты")
    logger.info(f"Объединено {len(timestamps)} -> {len(merged_timestamps)} сегментов.")
    return merged_timestamps

def build_subtitles_by_words_merged(asr_results, max_chars=20, max_gap=0.5):
    """
    Собирает субтитры на основе word timestamps из нескольких сегментов подряд.
    Объединяет слова между сегментами, если пауза между ними меньше max_gap.
    """
    subtitles = []
    current_line = ""
    current_start = None
    last_word_end = None

    # Проходим все слова всех сегментов как один список
    for seg in asr_results:
        for w in seg.get('words', []):
            # Определяем gap между текущим словом и предыдущим
            gap = 0
            if last_word_end is not None and w['start'] is not None:
                gap = w['start'] - last_word_end

            # Если gap большой — закрываем предыдущую реплику
            if gap > max_gap and current_line:
                subtitles.append({
                    "start": current_start,
                    "end": last_word_end,
                    "text": current_line.strip()
                })
                current_line = ""
                current_start = None

            # Начало новой реплики
            if current_start is None:
                current_start = w['start']

            # Добавляем слово
            if current_line:
                current_line += " "
            current_line += w['word']

            # Если превышен лимит символов — закрываем реплику
            if len(current_line) >= max_chars:
                subtitles.append({
                    "start": current_start,
                    "end": w['end'],
                    "text": current_line.strip()
                })
                current_line = ""
                current_start = None

            last_word_end = w['end']

    # Закрываем последнюю строку
    if current_line and current_start is not None:
        subtitles.append({
            "start": current_start,
            "end": last_word_end,
            "text": current_line.strip()
        })

    return subtitles


def process_audio(
    input_path: Optional[str],
    output_path: Optional[str],
    model_name: str,
    device: str,
    url: Optional[str],
    demucs_model: str = "htdemucs",
    merge_silence: float = 0.4,
    progress_callback: Optional[Callable[[float, str], None]] = None,
    cancel_event=None
):
    """Основная логика обработки аудио с централизованным управлением моделями."""
    from animesub.download_video import download_audio_wav
    from animesub.asr_whisper import transcribe_segments
    from animesub.asr_kotoba import transcribe_segments as transcribe_kotoba
    from animesub.separator import separate_vocals
    from animesub.vad_detector import detect_speech_segments
    from animesub.punctuator import add_punctuation_with_xlm
    from animesub.srt_formatter import _format_srt_time, _clean_text

    def _report_progress(value: float, message: str):
        logger.info(f"Progress {int(value*100)}%: {message}")
        if progress_callback:
            progress_callback(value, message)

    _report_progress(0.0, "Инициализация...")
    logger.info("--- Запуск нового процесса создания субтитров ---")

    temp_dir = tempfile.mkdtemp(prefix="animesub_")
    model_manager = ModelManager()

    try:
        check_cancel(cancel_event)
        
        # --- Шаг 0: Загрузка аудио ---
        if url:
            _report_progress(0.05, "Загрузка аудио...")
            wav_path = download_audio_wav(url=url, download_path=temp_dir, cancel_event=cancel_event)
            if not wav_path:
                raise RuntimeError("Не удалось скачать аудио или получить его название.")
            input_path = os.path.join(temp_dir, f"{wav_path}")
            # Если output_path это папка, создаем полный путь к файлу
            if os.path.isdir(output_path):
                 output_path = os.path.join(output_path, "output.srt")
        
        logger.info(f"Вход: {input_path}, Выход: {output_path}, Модель: {model_name}, Устройство: {device}")
        
        check_cancel(cancel_event)

        # --- Шаг 1: Отделение вокала ---
        _report_progress(0.1, "Отделение вокала...")
        vocals_path, _ = separate_vocals(
            input_path, model_name=demucs_model, device=device, 
            existing_temp_dir=temp_dir, cancel_event=cancel_event
        )
        if not vocals_path or not os.path.exists(vocals_path):
            raise RuntimeError("Не удалось отделить вокал.")
        logger.info(f"Вокал сохранен в: {vocals_path}")
        logger.debug(f"Память после Demucs: {get_memory_usage()}")
        
        check_cancel(cancel_event)

        # --- Шаг 2: Детекция речи (VAD) ---
        _report_progress(0.3, "Обнаружение речи (VAD)...")
        vad_model_pack = model_manager.load_model("vad", device=device)
        speech_timestamps, waveform, sample_rate = detect_speech_segments(
            vocals_path, vad_model_pack['model'], vad_model_pack['utils'], cancel_event
        )
        model_manager.unload_model("vad") 
        if not speech_timestamps:
            raise RuntimeError("Речь в аудио не обнаружена.")
            
        check_cancel(cancel_event)

        # --- Шаг 3: Транскрипция (ASR) ---
        _report_progress(0.4, f"Транскрипция с {model_name}...")
        is_kotoba_model = "kotoba" in model_name.lower()
        asr_model_type = "asr_kotoba" if is_kotoba_model else "asr"
        
        # Уточняем ID модели для Kotoba
        asr_model_id = f"kotoba-tech/{model_name}" if "kotoba-whisper" in model_name else model_name

        asr_model = model_manager.load_model(asr_model_type, model_name=asr_model_id, device=device)

        if is_kotoba_model:
            speech_timestamps = merge_close_segments(speech_timestamps, max_silence_s=merge_silence)
            transcription_iterator = transcribe_kotoba(
                speech_timestamps, waveform, sample_rate, asr_model, cancel_event
            )
        else:
            from animesub.asr_whisper import BatchedInferencePipeline
            batched_model = BatchedInferencePipeline(model=asr_model)
            transcription_iterator = transcribe_segments(
                speech_timestamps, waveform, sample_rate, batched_model, cancel_event
            )

        all_segments = list(transcription_iterator)
        model_manager.unload_model(asr_model_type)

        if not all_segments:
            raise RuntimeError("Транскрипция не дала результатов.")

        check_cancel(cancel_event)
        subtitles_data = build_subtitles_by_words_merged(all_segments, max_chars=100, max_gap=merge_silence)

        # --- Шаг 4: Пунктуация и сохранение ---
        _report_progress(0.7, "Расстановка пунктуации...")
        punctuator_model = model_manager.load_model("punctuator")

        # Берём только непустые реплики
        texts_to_punctuate = [s['text'] for s in subtitles_data if s and s.get('text')]

        # Подаём весь список сразу — группировка и чистка теперь внутри add_punctuation_with_xlm
        punctuated_results = add_punctuation_with_xlm(
            punctuator_model,
            texts_to_punctuate,
            cancel_event=cancel_event
        )

        with open(output_path, 'w', encoding='utf-8') as f:
            for idx, sub in enumerate(subtitles_data, start=1):
                if not sub or not sub.get('text'):
                    continue

                punctuated_text = "".join(punctuated_results[idx - 1])
                start_time = _format_srt_time(sub['start'])
                end_time = _format_srt_time(sub['end'])
                cleaned_text = _clean_text(punctuated_text)

                if cleaned_text:
                    f.write(f"{idx}\n{start_time} --> {end_time}\n{cleaned_text}\n\n")

        
        logger.info(f"Субтитры успешно сохранены в: {output_path}")

    except InterruptedError as e:
        logger.warning(str(e))
        _report_progress(0.0, "Отменено")
    except Exception as e:
        logger.critical(f"Критическая ошибка в процессе обработки аудио: {e}", exc_info=True)
        _report_progress(0.0, f"Ошибка: {e}")
        raise
    finally:
        model_manager.unload_all()
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Временная директория удалена: {temp_dir}")
            except Exception as e:
                logger.error(f"Не удалось удалить временную директорию {temp_dir}: {e}")

    _report_progress(1.0, "Готово!")
    logger.info("--- Процесс создания субтитров завершен ---")

def main():
    from animesub.utils import setup_logging
    setup_logging()

    parser = argparse.ArgumentParser(description="Создает субтитры (.srt) из медиафайла.")
    parser.add_argument("-i", "--input_file", type=str)
    parser.add_argument("-o", "--output", type=str)
    parser.add_argument("-m", "--model", type=str, default="small")
    parser.add_argument("-d", "--device", type=str, default=None, choices=['cpu', 'cuda'])
    parser.add_argument("-u", "--url", type=str, default=None)
    parser.add_argument("--demucs-model", type=str, default="htdemucs", choices=['htdemucs', 'mdx_extra_q'])
    parser.add_argument("--merge-silence", type=float, default=0.6)
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        logging.critical("FFmpeg не найден.")
        return

    if args.input_file is None and args.url is None:
        logging.critical("Нужно указать файл или URL.")
        return

    output_file_path = args.output

    if args.input_file:
        if not os.path.exists(args.input_file):
            logging.critical(f"Файл не найден: {args.input_file}")
            return
        if not output_file_path:
            output_file_path = f"{os.path.splitext(os.path.basename(args.input_file))[0]}.srt"

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.device == "cuda" and not torch.cuda.is_available():
        logging.error("CUDA недоступна, используется CPU.")
        device = "cpu"

    process_audio(
        input_path=args.input_file, output_path=output_file_path,
        model_name=args.model, device=device, url=args.url,
        demucs_model=args.demucs_model, merge_silence=args.merge_silence
    )

if __name__ == '__main__':
    main()