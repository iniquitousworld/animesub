import logging
import torch
from pathlib import Path
from typing import List, Dict, Union
from faster_whisper import WhisperModel

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_IDS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "kotoba-faster": "kotoba-tech/kotoba-whisper-v2.0-faster"
}

def transcribe_segments(
    audio_path: str,
    speech_timestamps: List[Dict[str, float]],
    waveform: torch.Tensor,
    sample_rate: int,
    model_name: str,
    device: str
) -> List[Dict[str, Union[float, str]]]:
    logging.info(f"Запуск транскрипции для {audio_path} с моделью {model_name}")

    model_id = MODEL_IDS.get(model_name.lower())
    if not model_id:
        logging.error(f"Неизвестное имя модели: {model_name}. Доступные модели: {', '.join(MODEL_IDS.keys())}")
        return []

    # Проверка входного аудио
    if waveform.numel() == 0 or sample_rate <= 0:
        logging.error("Некорректные аудиоданные: пустая волна или неверная частота дискретизации")
        return []

    compute_type = "float32"
    if device == "cuda" and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        compute_type = "float16" if capability[0] >= 7 else "float32"
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}, compute_type={compute_type}")
    else:
        device = "cpu"
        logging.warning("CUDA недоступна. Используется CPU.")

    logging.info(f"Загрузка модели {model_id} на {device}...")
    try:
        model = WhisperModel(model_id, device=device, compute_type=compute_type)
    except Exception as e:
        logging.error(f"Ошибка загрузки модели {model_id}: {e}")
        return []

    audio_path = Path(audio_path)
    if not audio_path.exists():
        logging.error(f"Аудиофайл не найден: {audio_path}")
        return []

    transcriptions = []
    for segment in speech_timestamps:
        start_time = segment['start']
        end_time = segment['end']
        segment_duration = end_time - start_time
        logging.debug(f"Обработка сегмента [{start_time:.2f}s - {end_time:.2f}s]")

        if segment_duration < 0.3:
            logging.warning(f"Пропущен сегмент: слишком короткий ({segment_duration:.2f}s)")
            continue

        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio_segment = waveform[:, start_sample:end_sample]

        if audio_segment.shape[0] > 1:
            audio_segment = torch.mean(audio_segment, dim=0, keepdim=True)

        if audio_segment.numel() == 0:
            logging.warning(f"Пустой аудиосегмент [{start_time:.2f}s - {end_time:.2f}s]")
            continue

        try:
            segments, info = model.transcribe(
                audio=audio_segment.numpy().squeeze(),
                language="ja",
                word_timestamps=True,
                beam_size=5,
                vad_filter=True,  # Включен VAD для большей точности
                vad_parameters=dict(min_silence_duration_ms=500)
            )
            logging.debug(f"Язык: {info.language}, Вероятность: {info.language_probability}")
            for s in segments:
                if s.text.strip():
                    transcriptions.append({
                        'start': s.start + start_time,
                        'end': s.end + start_time,
                        'text': s.text.strip()
                    })
                else:
                    logging.warning(f"Пустой текст в сегменте: {s.start + start_time:.2f}s - {s.end + start_time:.2f}s")
        except Exception as e:
            logging.error(f"Ошибка транскрипции сегмента [{start_time:.2f}s - {end_time:.2f}s]: {e}")
            transcriptions.append({
                'start': start_time,
                'end': end_time,
                'text': ''
            })

    transcriptions = [seg for seg in transcriptions if seg['text']]
    logging.info(f"Транскрибировано {len(transcriptions)} сегментов")
    return transcriptions