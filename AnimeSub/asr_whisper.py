import logging
import torch
import torchaudio
from pathlib import Path
from typing import List, Dict, Union
from transformers import pipeline

# Словарь для преобразования коротких имен моделей в их полные ID
MODEL_IDS = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large",
    "large-v2": "openai/whisper-large-v2",
    "large-v3": "openai/whisper-large-v3",
    "kotoba-whisper": "kotoba-tech/kotoba-whisper-v2.0"
}

def transcribe_segments(
    audio_path: str,
    speech_timestamps: List[Dict[str, float]],
    model_name: str,
    device: str
) -> List[Dict[str, Union[float, str]]]:
    """
    Транскрибирует аудиосегменты, используя Whisper из библиотеки transformers.
    """
    model_id = MODEL_IDS.get(model_name.lower())
    if not model_id:
        logging.error(f"Ошибка: Неизвестное имя модели: {model_name}")
        return []

    logging.info(f"Загрузка модели ASR: {model_id} на устройство {device}...")
    
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    
    transcriptions = []
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError(f"Аудиофайл не найден: {audio_path}")
    
    full_waveform, sample_rate = torchaudio.load(audio_path)

    for segment in speech_timestamps:
        start_time = segment['start']
        end_time = segment['end']
        
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        try:
            audio_segment = full_waveform[:, start_sample:end_sample]

            if audio_segment.shape[0] > 1:
                audio_segment = torch.mean(audio_segment, dim=0, keepdim=True)
            
            result = pipe(
                {"sampling_rate": sample_rate, "array": audio_segment.squeeze().numpy()},
                chunk_length_s=30,
                stride_length_s=5,
                return_timestamps=False,
                generate_kwargs={"language": "japanese"}
            )
            
            transcriptions.append({
                'start': start_time,
                'end': end_time,
                'text': result['text'].strip()
            })
        except Exception as e:
            logging.exception(f"Ошибка при транскрипции сегмента с помощью Whisper [{start_time:.2f}s - {end_time:.2f}s]: {e}")
            transcriptions.append({
                'start': start_time,
                'end': end_time,
                'text': ''
            })
            
    return transcriptions