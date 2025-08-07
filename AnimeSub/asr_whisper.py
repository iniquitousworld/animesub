import logging
import torch
import torchaudio 
from pathlib import Path
from typing import List, Dict, Union
from faster_whisper import WhisperModel

# Словарь для преобразования коротких имен моделей в их полные ID
MODEL_IDS = {
    "tiny": "tiny",
    "base": "base",
    "small": "small",
    "medium": "medium",
    "large": "large",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "kotoba-whisper": "kotoba-tech/kotoba-whisper-v2.0-faster"
}

def transcribe_segments(
    audio_path: str,
    speech_timestamps: List[Dict[str, float]],
    waveform: torch.Tensor,
    sample_rate: int,
    model_name: str,
    device: str
) -> List[Dict[str, Union[float, str]]]:
    """
    Транскрибирует аудиосегменты, используя faster-whisper с таймстампами слов.
    """
    model_id = MODEL_IDS.get(model_name.lower())
    if not model_id:
        logging.error(f"Ошибка: Неизвестное имя модели: {model_name}")
        return []

    compute_type = "float32"
    if device == "cuda" and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        if capability[0] >= 7:
            compute_type = "float16"
            logging.info("GPU поддерживает float16. Используется быстрый режим float16.")
        else:
            logging.warning("GPU не поддерживает float16. Используется универсальный, но более медленный режим float32 на GPU.")
            compute_type = "float32"
    else:
        logging.warning("CUDA недоступна или не выбрана. Вычисления будут производиться на CPU.")
        device = "cpu"

    logging.info(f"Загрузка модели ASR: {model_id} на устройство {device} с типом {compute_type}...")
    
    try:
        model = WhisperModel(model_id, device=device, compute_type=compute_type)
    except Exception as e:
        if "CUDA_OUT_OF_MEMORY" in str(e):
            logging.error(f"Недостаточно видеопамяти для модели {model_id} с типом {compute_type}. Попробуйте модель поменьше.")
        else:
            logging.error(f"Ошибка загрузки модели: {e}. Попробуйте запустить с --device cpu.")
        return []

    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
    
    transcriptions = []
    
    # Мы проходим по большим сегментам, которые дал Silero VAD
    for segment in speech_timestamps:
        start_time = segment['start']
        end_time = segment['end']
        
        # Вырезаем аудио-фрагмент из полного файла
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio_segment = waveform[:, start_sample:end_sample]

        if audio_segment.shape[0] > 1:
            audio_segment = torch.mean(audio_segment, dim=0, keepdim=True)

        try:
            # Транскрипция с таймстампами слов,
            # что позволит Whisper самому разбивать на предложения
            segments, _ = model.transcribe(
                audio=audio_segment.numpy().squeeze(),
                language="ja",
                word_timestamps=True, # Получаем таймстапы для каждого слова
                beam_size=5
            )
            
            # Собираем сегменты, сгенерированные Whisper, и корректируем их время
            for s in segments:
                transcriptions.append({
                    'start': s.start + start_time,
                    'end': s.end + start_time,
                    'text': s.text.strip()
                })

        except Exception as e:
            logging.exception(f"Ошибка при транскрипции сегмента с помощью Whisper [{start_time:.2f}s - {end_time:.2f}s]: {e}")
            transcriptions.append({
                'start': start_time,
                'end': end_time,
                'text': ''
            })
            
    return transcriptions