import logging
import torch
import torchaudio
from pathlib import Path
from typing import List, Dict, Union
from transformers import pipeline

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_IDS = {
    "kotoba-whisper": "kotoba-tech/kotoba-whisper-v1.1",
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
    logging.info(f"Запуск транскрипции для {audio_path} с моделью {model_name}")

    model_id = MODEL_IDS.get(model_name.lower())
    if not model_id:
        logging.error(f"Неизвестное имя модели: {model_name}. Доступные модели: {', '.join(MODEL_IDS.keys())}")
        return []

    # Проверка входного аудио
    if waveform.numel() == 0 or sample_rate <= 0:
        logging.error("Некорректные аудиоданные: пустая волна или неверная частота дискретизации")
        return []

    # Проверка частоты дискретизации
    if sample_rate != 16000:
        logging.info(f"Ресемплирование аудио с {sample_rate} Гц до 16000 Гц")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    # Динамический выбор compute_type
    if device == "cuda" and torch.cuda.is_available():
        capability = torch.cuda.get_device_capability(0)
        compute_type = torch.float16 if capability[0] >= 7 else torch.float32
        logging.info(f"GPU: {torch.cuda.get_device_name(0)}, compute_type={compute_type}")
    else:
        compute_type = torch.float32
        device = "cpu"
        logging.warning("CUDA недоступна. Используется CPU.")

    logging.info(f"Загрузка модели {model_id} на {device}...")
    try:
        pipe = pipeline(
            task="automatic-speech-recognition",
            model=model_id,
            torch_dtype=compute_type,
            device=device,
            model_kwargs={"attn_implementation": "sdpa"} if device == "cuda" else {},
            batch_size=2,  
            trust_remote_code=True,
        )
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
            audio_input = audio_segment.squeeze(0).detach().cpu().numpy()
            result = pipe(
                audio_input,
                chunk_length_s=15,  
                stride_length_s=3,  
                return_timestamps="word",
                generate_kwargs={"language": "ja", "task": "transcribe"}
            )
            logging.debug(f"Результат транскрипции: {result}")

            if 'chunks' in result and result['chunks']:
                for chunk in result['chunks']:
                    if chunk['timestamp'][0] is not None and chunk['timestamp'][1] is not None:
                        transcriptions.append({
                            'start': chunk['timestamp'][0] + start_time,
                            'end': chunk['timestamp'][1] + start_time,
                            'text': chunk['text'].strip()
                        })
                    else:
                        logging.warning(f"Недействительные таймстампы в чанке: {chunk}")
            else:
                logging.warning(f"Чанки не найдены для сегмента [{start_time:.2f}s - {end_time:.2f}s]. Используется полный текст.")
                text = result.get('text', '').strip()
                if text:
                    transcriptions.append({
                        'start': start_time,
                        'end': end_time,
                        'text': text
                    })

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