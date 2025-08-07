import torch
import torchaudio
from omegaconf import OmegaConf

# Загрузка модели silero-vad
model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    force_reload=False,
    onnx=False
)
(get_speech_timestamps, save_audio, read_audio, VADIterator, collect_chunks) = utils

def detect_speech_segments(audio_path: str):
    """
    Детектирует сегменты речи в аудиофайле с помощью silero-vad,
    игнорируя короткие паузы.
    
    Args:
        audio_path (str): Путь к аудиофайлу.
        
    Returns:
        list: Список словарей с начальным и конечным временем
              объединенных сегментов речи.
    """
    # Загрузка и обработка аудиофайла
    waveform, sample_rate = torchaudio.load(audio_path)
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000
    
    # Определение сегментов речи с увеличенной продолжительностью паузы
    # min_silence_duration_ms=1000 — это 1 секунда.
    speech_timestamps = get_speech_timestamps(
        waveform,
        model,
        sampling_rate=sample_rate,
        return_seconds=True,
        min_silence_duration_ms=1000 
    )
    
    return speech_timestamps, waveform, sample_rate