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
            self.compute_type = "float16" if capability[0] >= 7 else "float32"
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}, compute_type={self.compute_type}")
        else:
            self.device = "cpu"
            self.compute_type = "float32"
            logger.warning("CUDA недоступна или не выбрана. Используется CPU.")
    
    def load_model(self, model_type: str, model_name: str = None, device: str = "cpu"):
        """Ленивая загрузка и кэширование модели."""
        if model_type in self.models:
            return self.models[model_type]

        self._setup_device(device)
        logger.info(f"Загрузка модели типа '{model_type}' (имя: {model_name or 'default'})...")
        logger.debug(f"Использование памяти перед загрузкой: {get_memory_usage()}")
        
        model = None
        try:
            if model_type == "vad":
                model, utils = torch.hub.load(
                    repo_or_dir="snakers4/silero-vad", model="silero_vad",
                    force_reload=False, trust_repo=True, onnx=False
                )
                model = {"model": model, "utils": utils}

            elif model_type == "asr":
                from faster_whisper import WhisperModel
                model = WhisperModel(model_name, device=self.device, compute_type=self.compute_type)
            
            elif model_type == "asr_kotoba":
                from transformers import pipeline
                model = pipeline(
                    task="automatic-speech-recognition", model=model_name,
                    torch_dtype=torch.float16 if self.compute_type == 'float16' else torch.float32,
                    device=self.device, model_kwargs={"attn_implementation": "sdpa"} if self.device == "cuda" else {},
                    trust_remote_code=True,
                )
            
            elif model_type == "punctuator":
                from punctuators.models import PunctCapSegModelONNX
                model = PunctCapSegModelONNX.from_pretrained(
                    "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
                )

            if model:
                self.models[model_type] = model
                logger.info(f"Модель '{model_type}' успешно загружена.")
                logger.debug(f"Использование памяти после загрузки: {get_memory_usage()}")
                return model
            else:
                raise ValueError(f"Неизвестный тип или имя модели: {model_type}, {model_name}")

        except Exception as e:
            logger.error(f"Не удалось загрузить модель '{model_type}': {e}", exc_info=True)
            self.unload_model(model_type) # Попытка очистки в случае ошибки
            raise

    def unload_model(self, model_type: str):
        """Выгружает модель и очищает память."""
        if model_type in self.models:
            logger.info(f"Выгрузка модели '{model_type}'...")
            logger.debug(f"Использование памяти перед выгрузкой: {get_memory_usage()}")
            
            model = self.models.pop(model_type)
            del model
            gc.collect()
            if self.device == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()

            logger.info(f"Модель '{model_type}' выгружена.")
            logger.debug(f"Использование памяти после выгрузки: {get_memory_usage()}")

    def unload_all(self):
        """Выгружает все загруженные модели."""
        logger.info("Выгрузка всех кэшированных моделей...")
        # Создаем копию ключей, так как словарь будет изменяться во время итерации
        for model_type in list(self.models.keys()):
            self.unload_model(model_type)
        logger.info("Все модели выгружены.")