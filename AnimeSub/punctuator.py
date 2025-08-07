from typing import List
from punctuators.models import PunctCapSegModelONNX


model: PunctCapSegModelONNX = PunctCapSegModelONNX.from_pretrained(
    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
)

def add_punctuation_with_xlm(texts: List[str]) -> List[List[str]]:
    """
    Добавляет японскую пунктуацию в список текстов с помощью
    модели xlm-roberta.

    Args:
        texts (List[str]): Список строк без пунктуации.

    Returns:
        List[List[str]]: Список, где каждый элемент — это список
                         предложений с пунктуацией.
    """
    if not texts:
        return []
    
    # Модель обрабатывает список текстов и возвращает список списков предложений
    punctuated_texts = model.infer(texts)
    
    return punctuated_texts
