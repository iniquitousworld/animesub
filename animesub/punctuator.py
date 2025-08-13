import logging
import re
from typing import List

logger = logging.getLogger(__name__)

# Частицы/союзы, с которых редко начинается новое предложение — скорее продолжение.
CONTINUATION_MARKERS = (
    'が', 'を', 'に', 'は', 'も', 'と', 'で', 'へ', 'や',
    'けど', 'けれど', 'から', 'ので', 'し', 'たり',
    'また', 'そうな'
)

# Стабильный маркер разбиения для батчевой пунктуации
MARKER = "<<<SPLIT>>>"
# Регулярка для «восстановления» маркера
MARKER_VARIANTS_RE = re.compile(r"[<＜]{1,3}\s*SPLIT\s*[>＞]{1,3}")

# Классы символов японского письма
_JA_CHAR = r"\u3040-\u30FF\u4E00-\u9FFF"  # хирагана/катакана/кандзи
_OPEN_QUOTES = "「『（(［[｢『【〈《〔（"
_CLOSE_QUOTES = "」』）)］]｣』】〉》〕）"

# Список слов, перед которыми желательно ставить точку
BREAK_BEFORE = ("だから", "でも", "しかし", "それでも", "けれども", "なのに", "ところが")


def check_cancel(cancel_event):
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Отмена в punctuator.py")


def _to_single_string(model_output) -> str:
    if isinstance(model_output, list):
        if model_output and isinstance(model_output[0], list):
            return "".join(model_output[0])
        return "".join(str(x) for x in model_output)
    return str(model_output)


def _normalize_markers(text: str) -> str:
    if MARKER in text:
        return text
    return MARKER_VARIANTS_RE.sub(MARKER, text)


def _insert_break_before_connectives(text: str) -> str:
    """Вставляет точку перед некоторыми соединительными словами."""
    for conn in BREAK_BEFORE:
        text = re.sub(
            rf"(?<![。！？!?])({conn})",
            r"。\1",
            text
        )
    return text


def _clean_and_format(text: str) -> str:
    if not text:
        return ""
    # Базовая чистка пробелов
    text = text.replace(" ", "").replace("　", "")
    # Убираем ведущую пунктуацию в начале строки
    text = re.sub(r"^[。！？!?、，,]+", "", text)
    # Спец-фикс ложных вопросительных
    connective = ("なら", "ならば", "たら", "から", "ので", "けど", "けれど", "し", "たり")
    conn_pat = "|".join(connective)
    text = re.sub(rf"({conn_pat})？(?=.)", r"\1", text)
    # Убираем точку после этих конструкций перед японским символом
    text = re.sub(rf"({conn_pat})。(?=[{_JA_CHAR}])", r"\1", text)
    # Убираем лишние точки/запятые перед закрывающими кавычками
    text = re.sub(rf"([。！？!?、，,])(?=[{_CLOSE_QUOTES}])", "", text)
    # Вставляем точку перед некоторыми словами
    text = _insert_break_before_connectives(text)
    # Убираем пробелы вокруг пунктуации и дубликаты знаков
    text = re.sub(r"\s*([。、!?！？，,])\s*", r"\1", text)
    text = re.sub(r"([。、!?！？，,])\1+", r"\1", text)
    return text.strip()


def _merge_split_sentences_preserve_len(phrases: List[str]) -> List[str]:
    if len(phrases) < 2:
        return phrases
    out = phrases[:]
    for i in range(len(phrases) - 1):
        left = phrases[i].strip()
        right = phrases[i + 1].lstrip(_OPEN_QUOTES)
        if not left:
            continue
        if left.endswith('。'):
            base = left[:-1]
            if len(base) <= 4 and right.startswith(CONTINUATION_MARKERS):
                out[i] = base
    return out


def add_punctuation_with_xlm(model, texts: List[str], cancel_event=None) -> List[List[str]]:
    check_cancel(cancel_event)
    if not texts:
        return []
    try:
        final_results: List[List[str]] = []
        group_size = 10
        for i in range(0, len(texts), group_size):
            chunk = texts[i:i + group_size]
            check_cancel(cancel_event)
            block_with_markers = f" {MARKER} ".join(chunk)
            raw = model.infer([block_with_markers])
            check_cancel(cancel_event)
            punctuated_block = _to_single_string(raw)
            punctuated_block = _normalize_markers(punctuated_block)
            parts = [p.strip() for p in punctuated_block.split(MARKER)]
            if len(parts) != len(chunk):
                logger.warning(
                    f"Маркеров {MARKER} недостаточно: ожидалось {len(chunk)}, получили {len(parts)}. "
                    "Фоллбэк на поэлементную пунктуацию."
                )
                raw_parts = model.infer(chunk)
                check_cancel(cancel_event)
                parts = [_to_single_string(rp) for rp in raw_parts]
            parts = _merge_split_sentences_preserve_len(parts)
            for part in parts:
                final_results.append([_clean_and_format(part)])
        return final_results
    except Exception as e:
        if isinstance(e, InterruptedError):
            raise
        logger.error(f"Ошибка при расстановке пунктуации: {e}", exc_info=True)
        return [[_clean_and_format(t)] for t in texts]
