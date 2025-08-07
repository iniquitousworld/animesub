import os
import subprocess
import logging

def separate_vocals(audio_path: str, device: str, output_dir: str = "separated"):
    """
    Separates vocal track from an audio file using Demucs.
    
    Args:
        audio_path (str): The path to the input audio file.
        output_dir (str): The directory where the separated tracks will be saved.
    
    Returns:
        str: The path to the separated vocal track.
    """
    logging.info(f"Отделение вокала из {audio_path}...")
    
    # Создаем выходную директорию, если она не существует
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Команда для Demucs, которая разделяет аудио на 4 дорожки (вокал, барабаны, бас, другое)
    command = [
        "demucs",
        "--device", device, # Добавляем флаг устройства
        audio_path,
        "-o", output_dir
    ]
    
    # Запускаем команду
    subprocess.run(command, check=True)
    
    # Формируем путь к файлу с вокалом
    vocal_track_name = os.path.splitext(os.path.basename(audio_path))[0]
    vocal_track_path = os.path.join(output_dir, "htdemucs", vocal_track_name, "vocals.wav")
    
    if os.path.exists(vocal_track_path):
        logging.info(f"Вокал успешно отделен и сохранен в {vocal_track_path}")
        return vocal_track_path
    else:
        logging.critical("Ошибка: файл с вокалом не найден.")
        return None
