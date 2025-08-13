import logging
import torch
import torchaudio
from typing import List, Dict, Union, Generator
from faster_whisper import BatchedInferencePipeline

logger = logging.getLogger(__name__)

MAX_CHUNK_DURATION = 30.0  # максимум длительности чанка в секундах
CHUNK_OVERLAP = 0.8        # перекрытие между чанками в секундах

def check_cancel(cancel_event):
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Отмена в asr_whisper.py")

def rms_normalize(audio, target_rms=0.1, tolerance=0.03):
    """Нормализация RMS с допустимым отклонением."""
    rms = torch.sqrt(torch.mean(audio**2))
    if abs(rms.item() - target_rms) < tolerance:
        return audio
    gain = target_rms / (rms + 1e-9)
    return audio * gain

def transcribe_segments(
    speech_timestamps: List[Dict[str, float]],
    waveform: torch.Tensor,
    sample_rate: int,
    batched_model: BatchedInferencePipeline,
    cancel_event=None,
    pitch_steps: int = -2
) -> Generator[Dict[str, Union[float, str, List[Dict[str, Union[str, float]]]]], None, None]:

    logger.info(f"Запуск транскрипции Whisper для {len(speech_timestamps)} сегментов.")

    if waveform.numel() == 0 or sample_rate <= 0:
        logger.error("Некорректные аудиоданные для транскрипции.")
        return

    with torch.no_grad():
        for segment in speech_timestamps:
            check_cancel(cancel_event)
            start_time = segment['start']
            end_time = segment['end']
            seg_duration = end_time - start_time

            if seg_duration < 0.2:
                logger.warning(f"Пропущен слишком короткий сегмент: {seg_duration:.2f}s")
                continue

            # Разбиваем длинный сегмент на чанки с перекрытием
            chunk_starts = []
            pos = start_time
            while pos < end_time:
                chunk_starts.append(pos)
                pos += MAX_CHUNK_DURATION - CHUNK_OVERLAP

            chunk_ends = [
                min(cs + MAX_CHUNK_DURATION, end_time) for cs in chunk_starts
            ]

            for chunk_start, chunk_end in zip(chunk_starts, chunk_ends):
                check_cancel(cancel_event)

                start_sample = int(chunk_start * sample_rate)
                end_sample = int(chunk_end * sample_rate)
                audio_segment = waveform[0, start_sample:end_sample]

                if audio_segment.numel() == 0:
                    logger.warning(f"Пустой аудиочанк для [{chunk_start:.2f}s - {chunk_end:.2f}s]")
                    continue

                try:
                    # Снижение тона
                    if pitch_steps != 0: 
                        audio_segment = torchaudio.functional.pitch_shift(
                            audio_segment.unsqueeze(0),
                            sample_rate,
                            n_steps=pitch_steps 
                        ).squeeze(0)

                    # RMS нормализация
                    audio_input = rms_normalize(audio_segment.cpu(), target_rms=0.1)
                    audio_input = audio_input.numpy()

                    # Транскрипция чанка
                    segments, info = batched_model.transcribe(
                        audio=audio_input,
                        language="ja",
                        word_timestamps=True,
                        vad_filter=False,
                        no_repeat_ngram_size=5,
                        batch_size=8,
                        beam_size=10
                    )

                    logger.debug(
                        f"Язык: {info.language}, Вероятность: {info.language_probability:.2f}"
                    )

                    for s in segments:
                        check_cancel(cancel_event)
                        text = s.text.strip()

                        # Глобальные таймкоды слов
                        words = []
                        for w in getattr(s, "words", []):
                            words.append({
                                "word": w.word,
                                "start": (w.start + chunk_start) if w.start is not None else None,
                                "end": (w.end + chunk_start) if w.end is not None else None
                            })

                        if text:
                            yield {
                                'start': s.start + chunk_start,
                                'end': s.end + chunk_start,
                                'text': text,
                                'words': words
                            }

                except Exception as e:
                    if isinstance(e, InterruptedError):
                        raise
                    logger.error(
                        f"Ошибка транскрипции чанка [{chunk_start:.2f}s - {chunk_end:.2f}s]: {e}",
                        exc_info=True
                    )
                    yield {
                        'start': chunk_start,
                        'end': chunk_end,
                        'text': '',
                        'words': []
                    }
