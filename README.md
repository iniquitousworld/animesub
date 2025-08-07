# AnimeSub

Инструмент для автоматического создания субтитров для аниме из видео- или аудиофайлов.

## Установка

```bash
pip install animesub
```

## Использование
Чтобы создать субтитры, используйте следующую команду:

```bash
animesub input_file.mp4
```

Это создаст файл input_file.srt в той же директории. ВАЖНО: Для использования cuda, вам нужно установить нужные зависимости (torch, torchaudio) с нужной версией cuda.

Либо запустите ячейку следуя инструкциям в Google Colab:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/liokar/animesub/blob/main/animesub_colab.ipynb)

Дополнительные параметры:

-o, --output: Путь к итоговому .srt файлу.

-m, --model: Имя модели Whisper (например, tiny, base, small, medium, large, kotoba-whisper). По умолчанию: small.

-d, --device: Устройство для вычислений (cpu или cuda). По умолчанию: определяется автоматически.

Пример:

```bash
animesub my_anime_episode.mkv -o subtitles.srt -m medium -d cuda
```