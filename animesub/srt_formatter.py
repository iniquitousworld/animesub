import re
import logging

logger = logging.getLogger(__name__)

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
    # Удаляем подряд идущие знаки '。' и '、', оставляя только один
    cleaned_text = re.sub(r'[。、]{2,}', '。', text)
    cleaned_text = re.sub(r'[、。]{2,}', '。', text)

    # Удаляем дублирующиеся знаки препинания (например, 、、 или 。。)
    cleaned_text = re.sub(r'([、。])\1+', r'\1', text)

    # Удаляем западные знаки препинания (., !, ?, ,, ; и т.д.), сохраняя японские
    cleaned_text = re.sub(r'[.,!?;]', '', cleaned_text)

    # Удаляем лишние пробелы (для японского текста пробелы обычно не нужны)
    cleaned_text = re.sub(r'\s+', '', cleaned_text).strip()

    return cleaned_text

def create_srt_file(output_path, valid_subs, punctuated_results):
    with open(output_path, "w", encoding="utf-8") as f:
        for idx, (sub, punct_parts) in enumerate(zip(valid_subs, punctuated_results), start=1):

            punctuated_text = "".join(punct_parts)
            cleaned_text = _clean_text(punctuated_text)

            if cleaned_text:
                f.write(
                    f"{idx}\n"
                    f"{_format_srt_time(sub['start'])} --> {_format_srt_time(sub['end'])}\n"
                    f"{cleaned_text}\n\n"
                )