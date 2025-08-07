import logging
import torch
import torchaudio
from pathlib import Path
from typing import List, Dict, Union
from transformers import pipeline

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Словарь для преобразования коротких имен моделей в их полные ID
MODEL_IDS = {
    "kotoba-whisper-v2.2": "kotoba-tech/kotoba-whisper-v2.2"
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
    Транскрибирует аудиосегменты, используя kotoba-whisper-v2.2 через transformers.pipeline.

    Args:
        audio_path (str): Путь к аудиофайлу.
        speech_timestamps (List[Dict[str, float]]): Список таймстампов речи от VAD.
        waveform (torch.Tensor): Аудиоданные.
        sample_rate (int): Частота дискретизации.
        model_name (str): Имя модели (например, "kotoba-whisper-v2.2").
        device (str): Устройство для вычислений ("cpu" или "cuda").

    Returns:
        List[Dict[str, Union[float, str]]]: Список словарей с полями 'start', 'end', 'text'.
    """
    model_id = MODEL_IDS.get(model_name.lower())
    if not model_id:
        logging.error(f"Неизвестное имя модели: {model_name}. Доступные модели: {', '.join(MODEL_IDS.keys())}")
        return []

    # Динамический выбор compute_type
    if device == "cuda" and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        compute_type = torch.float16 if capability[0] >= 7 else torch.float32
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}, compute_type={compute_type}")
    else:
        compute_type = torch.float32
        device = "cpu"
        logging.warning("CUDA недоступна или не выбрана. Используется CPU.")

    logging.info(f"Загрузка модели ASR: {model_id} на устройство {device} с типом {compute_type}...")

    model_kwargs = {"attn_implementation": "sdpa"} if device == "cuda" and torch.cuda.is_available() else {}

    try:
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            torch_dtype=compute_type,
            device=device,
            model_kwargs=model_kwargs,
            batch_size=8,
            trust_remote_code=True,
        )
    except Exception as e:
        if "CUDA_OUT_OF_MEMORY" in str(e):
            logging.error(f"Недостаточно видеопамяти для модели {model_id} с типом {compute_type}. Попробуйте модель поменьше или --device cpu.")
        else:
            logging.error(f"Ошибка загрузки модели: {e}. Попробуйте запустить с --device cpu.")
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

        # Пропускаем слишком короткие сегменты
        if segment_duration < 0.1:
            logging.warning(f"Пропущен сегмент [{start_time:.2f}s - {end_time:.2f}s]: слишком короткий ({segment_duration:.2f}s).")
            continue

        # Вырезаем аудиофрагмент
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        audio_segment = waveform[:, start_sample:end_sample]

        if audio_segment.shape[0] > 1:
            audio_segment = torch.mean(audio_segment, dim=0, keepdim=True)

        try:
            # Транскрипция с таймстампами слов и перекрытием чанков
            result = pipe(
                audio_segment.numpy().squeeze(),
                chunk_length_s=5,  # Уменьшенная длина чанка для точности
                stride_length_s=1,  # Перекрытие для плавных переходов
                return_timestamps="word",  # Таймстампы на уровне слов
                generate_kwargs={"language": "ja", "task": "transcribe"}
            )

            if 'chunks' in result:
                for chunk in result['chunks']:
                    if chunk['timestamp'][0] is not None and chunk['timestamp'][1] is not None:
                        transcriptions.append({
                            'start': chunk['timestamp'][0] + start_time,
                            'end': chunk['timestamp'][1] + start_time,
                            'text': chunk['text'].strip()
                        })
                    else:
                        logging.warning(f"Недействительные таймстампы для чанка в сегменте [{start_time:.2f}s - {end_time:.2f}s]: {chunk}")
            else:
                logging.warning(f"Чанки не найдены для сегмента [{start_time:.2f}s - {end_time:.2f}s]. Используется полный текст.")
                transcriptions.append({
                    'start': start_time,
                    'end': end_time,
                    'text': result.get('text', '').strip()
                })

        except Exception as e:
            logging.exception(f"Ошибка транскрипции сегмента [{start_time:.2f}s - {end_time:.2f}s]: {e}")
            transcriptions.append({
                'start': start_time,
                'end': end_time,
                'text': ''
            })

    # Фильтрация пустых транскрипций
    transcriptions = [seg for seg in transcriptions if seg['text']]

    # Логирование результатов
    logging.info(f"Транскрибировано {len(transcriptions)} сегментов для аудиофайла {audio_path}")
    return transcriptions