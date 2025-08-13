import logging
import re
from typing import List

logger = logging.getLogger(__name__)

def check_cancel(cancel_event):
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Отмена в punctuator.py")

def _clean_punctuation(text: str) -> str:
    # Убираем повторы пунктуации: 。。 или .. и т.д.
    text = re.sub(r"([。、，,.!?！？])\1+", r"\1", text)
    # Убираем пробелы перед знаками пунктуации
    text = re.sub(r"\s+([。、，,.!?！？])", r"\1", text)
    return text.strip()

def add_punctuation_with_xlm(model, texts: List[str], cancel_event=None) -> List[List[str]]:
    """
    Добавляет пунктуацию с объединением текста в блоки для большего контекста
    и последующим восстановлением границ реплик.
    """
    check_cancel(cancel_event)
    if not texts:
        return []

    try:
        grouped_results = []
        group_size = 10  # кол-во реплик в одной группе для контекста

        for i in range(0, len(texts), group_size):
            # Текущая группа
            chunk = texts[i:i + group_size]
            check_cancel(cancel_event)

            # Склеиваем реплики в один блок с маркерами <SPLIT>
            block = " <SPLIT> ".join(chunk)

            # Прогоняем через пунктуатор
            punctuated_block_list = model.infer([block])
            check_cancel(cancel_event)

            # model.infer возвращает список списков/строк — приводим к строке
            if punctuated_block_list and isinstance(punctuated_block_list[0], list):
                punctuated_block = "".join(punctuated_block_list[0])
            else:
                punctuated_block = str(punctuated_block_list[0])

            # Разбиваем обратно по маркеру
            split_parts = [part.strip() for part in punctuated_block.split("<SPLIT>")]

            # Чистим каждую реплику
            for part in split_parts:
                cleaned = _clean_punctuation(part)
                # Убираем ложную точку, если она оказалась в конце, а следующее предложение явно продолжается
                grouped_results.append([cleaned])

        return grouped_results

    except Exception as e:
        if isinstance(e, InterruptedError):
            raise
        logger.error(f"Ошибка при расстановке пунктуации: {e}", exc_info=True)
        return [[_clean_punctuation(text)] for text in texts]
