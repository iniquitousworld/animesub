import logging
import gc
import torch
from animesub.utils import get_memory_usage

logger = logging.getLogger(__name__)

class ModelManager:
    """
    Класс для управления жизненным циклом моделей (загрузка, кэширование, выгрузка).
    """
    def __init__(self):
        self.models = {}
        self.device = "cpu"
        self.compute_type = "float32"

    def _setup_device(self, device: str):
        self.device = device
        if self.device == "cuda" and torch.cuda.is_available():
            capability = torch.cuda.get_device_capability()
            # Для Kotoba и других современных моделей float16 является стандартом
            self.compute_type = "float16" if capability[0] >= 7 else "float32"
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}, compute_type={self.compute_type}")
        else:
            self.device = "cpu"
            self.compute_type = "int8" # Для CPU используем INT8 для лучшей производительности
            logger.warning("CUDA недоступна или не выбрана. Используется CPU с compute_type=int8.")
    
    def load_model(self, model_key: str, loader_func: callable, **kwargs):
        if model_key in self.models:
            return self.models[model_key]

        try: 
            model = loader_func(**kwargs) 
        except Exception as e:
            logger.error(f"Не удалось загрузить модель '{model_key}': {e}", exc_info=True)
            self.unload_model(model_key) # Попытка очистки в случае ошибки
            raise

        if model:
            self.models[model_key] = model
            logger.info(f"Модель '{model_key}' успешно загружена.")
            logger.debug(f"Использование памяти после загрузки: {get_memory_usage()}")
            return model
        else:
            raise ValueError(f"Неизвестный тип или имя модели: {model_key}")
        
    def unload_model(self, model_key: str):
        """Выгружает модель по ее ключу и очищает память."""
        if model_key in self.models:
            logger.info(f"Выгрузка модели '{model_key}'...")
            logger.debug(f"Использование памяти перед выгрузкой: {get_memory_usage()}")
            
            model = self.models.pop(model_key)
            del model
            gc.collect()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Модель '{model_key}' выгружена.")
            logger.debug(f"Использование памяти после выгрузки: {get_memory_usage()}")

    def unload_all(self):
        """Выгружает все загруженные модели."""
        logger.info("Выгрузка всех кэшированных моделей...")
        for model_key in list(self.models.keys()):
            self.unload_model(model_key)
        logger.info("Все модели выгружены.")