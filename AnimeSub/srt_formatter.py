import re
from typing import List, Dict

def _format_srt_time(seconds: float) -> str:
    """
    Форматирует время в секундах в формат SRT (HH:MM:SS,MMM).
    """
    total_milliseconds = int(seconds * 1000)
    hours, remainder = divmod(total_milliseconds, 3600000)
    minutes, remainder = divmod(remainder, 60000)
    seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{milliseconds:03}"

def _clean_text(text: str) -> str:
    """
    Очищает текст субтитров, удаляя нежелательные западные знаки препинания
    и дублирующиеся японские знаки, сохраняя только '、' и '。' для японского текста.
    """
    # Удаляем дублирующиеся знаки препинания (например, 、、 или 。。)
    cleaned_text = re.sub(r'([、。])\1+', r'\1', text)

    # Удаляем западные знаки препинания (., !, ?, ,, ; и т.д.), сохраняя японские
    cleaned_text = re.sub(r'[.,!?;]', '', cleaned_text)

    # Удаляем лишние пробелы (для японского текста пробелы обычно не нужны)
    cleaned_text = re.sub(r'\s+', '', cleaned_text).strip()

    return cleaned_text

def segments_to_srt(segments: List[Dict]) -> str:
    """
    Преобразует список сегментов с текстом и временными метками в строку формата SRT.

    Args:
        segments (List[Dict]): Список словарей с ключами 'start', 'end', 'text'.

    Returns:
        str: Строка в формате SRT.
    """
    srt_blocks = []
    for i, segment in enumerate(segments):
        text = segment.get('text', '')
        if not text:
            continue

        # Очищаем текст
        cleaned_text = _clean_text(text)
        if not cleaned_text:
            continue

        start_time = _format_srt_time(segment['start'])
        end_time = _format_srt_time(segment['end'])

        block = f"{i + 1}\n{start_time} --> {end_time}\n{cleaned_text}\n"
        srt_blocks.append(block)

    return "\n".join(srt_blocks)