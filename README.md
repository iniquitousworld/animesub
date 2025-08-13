# AnimeSub

Инструмент для **автоматического создания субтитров** из видео- или аудиофайлов.  
Оптимизирован для японского языка (подходит для аниме, интервью и т.п.).

---

## ⚡ Быстрый старт

**Локальный файл → субтитры:**
```bash
animesub -i input_file.mp4
```

**YouTube → субтитры:**
```bash
animesub -u "https://youtube.com/watch?v=XXXX"
```

**Запуск в Colab**
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/iniquitousworld/animesub/blob/main/animesub_colab.ipynb)

---

## 📦 Установка

Из PyPI:
```bash
pip install animesub
```

💡 Для работы на CUDA желательно поставить `torch` и `torchaudio` под свою версию CUDA:
```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
```

Не забудьте установить `ffmpeg` и `demucs`:
```bash
conda install ffmpeg -c conda-forge
pip install demucs
```

---

## 🚀 Использование

### Локальный файл
```bash
animesub -i input_file.mp4
```
Создаст `input_file.srt` в текущей папке.

### Скачивание по URL (YouTube и др.)
```bash
animesub -u "https://youtube.com/watch?v=XXXX"
```
Сохранит результат в `output.srt`.

---

## 🔧 Аргументы CLI

| Аргумент               | Описание |
|------------------------|----------|
| `-i`, `--input_file`   | Путь к локальному видео или аудио |
| `-u`, `--url`          | URL видео/аудио (YouTube) |
| `-o`, `--output`       | Путь к выходному `.srt` (по умолчанию: `<имя_файла>.srt` или `output.srt` для URL) |
| `-m`, `--model`        | Модель ASR (см. список ниже) (по умолчанию: `small`) |
| `-d`, `--device`       | `cpu` или `cuda` (по умолчанию определяется автоматически) |
| `--demucs-model`       | Модель сепарации вокала: `htdemucs` или `mdx_extra_q` (по умолчанию: `htdemucs`) |
| `--merge-silence`      | Максимальная пауза между VAD-сегментами для объединения (по умолчанию: `0.6`) |

---

🎤 Поддерживаемые модели
1. Официальные OpenAI Whisper

tiny, base
small, medium
large-v1, large-v2, large-v3, large-v3-turbo

2. Kotoba-Tech (оптимизированы под японский)

kotoba-whisper-v1.0
kotoba-whisper-v2.0
kotoba-whisper-v3.0

kotoba-faster

(для HuggingFace-версий указывайте kotoba-tech/имя_модели)

**3. HuggingFace faster-whisper**

Примеры:

Systran/faster-whisper-small
Systran/faster-whisper-medium
Systran/faster-whisper-large-v2
Systran/faster-whisper-large-v3

**4. Локальные модели**

Укажите путь к .bin или .gguf файлу:

```bash
animesub -i file.mp4 -m "models/ggml-large-v3.bin"
```

---

## 📌 Примеры

**CPU + base модель**
```bash
animesub -i input.mp3 -m base -d cpu
```

**Kotoba-модель с кастомной паузой**
```bash
animesub -i anime.mkv -m kotoba-whisper-v2.2 --merge-silence 0.8 -d cuda
```

**С YouTube**
```bash
animesub -u "https://youtube.com/watch?v=XXXX" -m kotoba-faster -d cuda
```

**С указанием файла вывода**
```bash
animesub -i episode.mp4 -o subs/episode01.srt
```

---

## 🎯 Как работает

1. **Отделение вокала** (Demucs)  
2. **Детекция речи** (Silero VAD)  
3. **Транскрипция** (Whisper или Kotoba-Whisper)  
4. **Пунктуация** (XLM-RoBERTa через `punctuators`)  
5. **Экспорт в `.srt`** с форматированием

---

## 🛠️ Использование как библиотеки

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
