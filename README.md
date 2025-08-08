# AnimeSub

Инструмент для **автоматического создания субтитров** из любого видео- или аудиофайла.  
Оптимизирован для японского языка (подходит для аниме, интервью, и т.п.).

---

## 📦 Установка

Установите из PyPI:

```bash
pip install animesub
````

💡 Для использования CUDA рекомендуется установить `torch` и `torchaudio` с поддержкой вашей версии CUDA вручную:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Также требуется `ffmpeg` и `demucs`:

```bash
conda install ffmpeg -c conda-forge
pip install demucs
```

---

## 🚀 Использование

Создайте субтитры по умолчанию (с моделью `kotoba-whisper-v2.2`):

```bash
animesub input_file.mp4
```

Это создаст файл `input_file.srt` в текущей директории.

### 🔧 Параметры CLI

| Аргумент          | Описание |
|-------------------|----------|
| `input_file`      | Путь к входному видео или аудио |
| `-o`, `--output`  | Путь к выходному `.srt` (по умолчанию: `<имя_файла>.srt`) |
| `-m`, `--model`   | Название модели Whisper: ↩<br>`tiny`, `base`, `small`, `medium`, `large`, ↩<br>`kotoba-whisper`, `kotoba-faster`, `kotoba-whisper-v2.2` ↩<br>(по умолчанию: `kotoba-whisper-v2.2`) |
| `-d`, `--device`  | `cpu` или `cuda` (по умолчанию: определяется автоматически) |
| `--demucs-model`  | Модель вокальной сепарации: ↩<br>`htdemucs` или `mdx_extra_q` ↩<br>(по умолчанию: `htdemucs`) |
| `--merge-silence` | Максимальная пауза между VAD-сегментами для их объединения ↩<br>**(только для kotoba-моделей)**, например: `0.6` ↩<br>(по умолчанию: `0.6`) |

### 📌 Примеры

#### С базовой моделью на CPU:

```bash
animesub input.mp3 -m base -d cpu
```

#### С моделью `kotoba` и кастомной паузой объединения:

```bash
animesub input.mkv -m kotoba-whisper-v2.2 --merge-silence 0.8 -d cuda
```

#### С сохранением результата в указанный файл:

```bash
animesub episode.mp4 -o subtitles.srt
```

---

## 🎯 Как это работает

Процесс состоит из 5 этапов:

1. 🎵 Отделение вокала (Demucs)
2. 🎙️ Обнаружение сегментов речи (Silero VAD)
3. ✍️ Транскрипция аудио (Whisper или Kotoba)
4. 🔡 Постобработка: японская пунктуация (XLM-RoBERTa)
5. 📝 Экспорт в `.srt` (с форматированием и очисткой текста)

---

## 🧠 Поддерживаемые модели

* **Faster-Whisper (от OpenAI):**

  * `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`
* **Faster-Whisper (kotoba):**

  * `kotoba-faster` (на основе kotoba-tech/kotoba-whisper-v2.0-faster)
* **Kotoba-Whisper (через transformers):**

  * `kotoba-whisper`, `kotoba-whisper-v2.2`

---

## 🧪 Тестирование

Убедитесь, что установлены зависимости из `pyproject.toml` или вручную:

```bash
pip install torch torchaudio faster-whisper transformers demucs punctuators
```

---

## 🔗 Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iniquitousworld/animesub/blob/main/animesub_colab.ipynb)

---

## 🛠️ Использование как библиотеки

Можно импортировать и использовать в своём Python-скрипте:

```python
from AnimeSub.main_logic import process_audio

process_audio(
    input_path="video.mp4",
    output_path="subs.srt",
    model_name="kotoba-whisper-v2.2",
    device="cuda",
    merge_silence=0.6
)
```

---

## 📜 Лицензия

MIT

---

## 👤 Автор

**Ivan Tyumentsev**
📧 [ivanfufa184@gmail.com](mailto:ivanfufa184@gmail.com)
🔗 [GitHub](https://github.com/iniquitousworld)

```
