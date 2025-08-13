import logging
import torch
import torchaudio
from typing import List, Dict, Union, Generator

logger = logging.getLogger(__name__)

def check_cancel(cancel_event):
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Отмена в asr_kotoba.py")

def transcribe_segments(
    speech_timestamps: List[Dict[str, float]],
    waveform: torch.Tensor,
    sample_rate: int,
    pipe,  
    cancel_event=None
) -> Generator[Dict[str, Union[float, str]], None, None]:

    logger.info(f"Запуск транскрипции Kotoba для {len(speech_timestamps)} сегментов.")
    
    if waveform.numel() == 0 or sample_rate <= 0:
        logger.error("Некорректные аудиоданные для транскрипции.")
        return
        
    if sample_rate != 16000:
        logger.info(f"Ресемплирование аудио с {sample_rate} Гц до 16000 Гц для Kotoba.")
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
        sample_rate = 16000

    with torch.no_grad():
        for segment in speech_timestamps:
            check_cancel(cancel_event)
            start_time = segment['start']
            end_time = segment['end']
            
            start_sample = int(start_time * sample_rate)
            end_sample = int(end_time * sample_rate)
            audio_segment = waveform[0, start_sample:end_sample]

            if audio_segment.numel() < 160: # меньше 0.01с
                logger.warning("Пропущен слишком короткий аудиосегмент для Kotoba.")
                continue

            try:
                audio_input = audio_segment.cpu().numpy()
                result = pipe(
                    audio_input,
                    chunk_length_s=30,
                    stride_length_s=5,
                    return_timestamps="word",
                    generate_kwargs={"language": "ja", "task": "transcribe"}
                )
                
                if 'chunks' in result and result['chunks']:
                    for chunk in result['chunks']:
                        if chunk['timestamp'][0] is not None and chunk['timestamp'][1] is not None:
                            yield {
                                'start': chunk['timestamp'][0] + start_time,
                                'end': chunk['timestamp'][1] + start_time,
                                'text': chunk['text'].strip()
                            }
                elif result.get('text', '').strip():
                     logger.warning(f"Чанки не найдены, используется полный текст для сегмента [{start_time:.2f}s - {end_time:.2f}s].")
                     yield {'start': start_time, 'end': end_time, 'text': result['text'].strip()}

            except Exception as e:
                if isinstance(e, InterruptedError): raise
                logger.error(f"Ошибка транскрипции Kotoba сегмента [{start_time:.2f}s - {end_time:.2f}s]: {e}", exc_info=True)
                yield {'start': start_time, 'end': end_time, 'text': ''}