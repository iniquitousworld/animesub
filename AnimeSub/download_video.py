import yt_dlp
import os
import logging

logger = logging.getLogger(__name__)

def check_cancel(cancel_event):
    if cancel_event and cancel_event.is_set():
        raise InterruptedError("Отмена в download_video.py")

def download_audio_wav(url: str, download_path: str, cancel_event=None) -> str:
    """
    Скачивает аудио из URL и конвертирует в WAV с фиксированным именем 'input.wav'.
    Возвращает путь к этому файлу.
    """
    check_cancel(cancel_event)
    logger.info(f"Начало загрузки аудио с URL: {url}")

    # Гарантируем, что папка назначения существует
    os.makedirs(download_path, exist_ok=True)

    # Путь к готовому .wav файлу
    output_wav_path = os.path.join(download_path, "input.wav")

    # Хук для отмены загрузки
    progress_hooks = []
    if cancel_event:
        def cancel_hook(d):
            if d["status"] == "downloading":
                check_cancel(cancel_event)
        progress_hooks.append(cancel_hook)

    # Настройки yt-dlp — всегда сохраняем как input.wav
    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(download_path, 'input'),
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "wav",
        }],
        "progress_hooks": progress_hooks,
        "quiet": True,
        "logger": logging.getLogger("yt_dlp"),
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Получаем инфо о видео (чтобы можно было залогировать)
            info_dict = ydl.extract_info(url, download=False)
            video_title = info_dict.get("title", "Без названия")

            check_cancel(cancel_event)

            logger.info(f"Загрузка и конвертация видео '{video_title}' в WAV...")
            ydl.download([url])

            check_cancel(cancel_event)
            logger.info(f"Загрузка завершена: {output_wav_path}")

            if not os.path.exists(output_wav_path):
                raise FileNotFoundError(f"Файл {output_wav_path} не найден после загрузки.")

            return output_wav_path

    except InterruptedError:
        logger.warning("Загрузка видео была отменена пользователем.")
        raise
    except Exception as e:
        logger.error(f"Ошибка при загрузке видео с {url}: {e}", exc_info=True)
        raise RuntimeError(f"Ошибка yt-dlp: {e}")
