import torch
import torchaudio
import logging

logger = logging.getLogger(__name__)

def check_cancel(cancel_event):
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Отмена в vad_detector.py")

def detect_speech_segments(audio_path: str, model, utils, cancel_event=None):
    """
    Детектирует сегменты речи, принимая загруженную модель VAD (silero).
    Сделан более "щедрый" режим: ниже порог, больше паддинг, чтобы не 
    резать короткие/высокие фразы.
    Возвращает (speech_timestamps, waveform, sample_rate).
    """
    check_cancel(cancel_event)
    (get_speech_timestamps, _, _, _, _) = utils

    logger.info("VAD: загрузка и предобработка аудио...")
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # VAD требует 16kHz
    if sample_rate != 16000:
        check_cancel(cancel_event)
        logger.info(f"VAD: ресемплирование {sample_rate} -> 16000 Гц.")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    logger.info("VAD: запуск детекции речи (silero-vad)...")
    check_cancel(cancel_event)

    device = next(model.parameters()).device
    waveform = waveform.to(device)

    speech_timestamps = get_speech_timestamps(
        waveform, model,
        sampling_rate=sample_rate,
        threshold=0.30,
        return_seconds=True,
        min_silence_duration_ms=500,
        min_speech_duration_ms=80,
        speech_pad_ms=500
    )

    logger.info(f"VAD обнаружил {len(speech_timestamps)} сегментов речи.")
    return speech_timestamps, waveform, sample_rate