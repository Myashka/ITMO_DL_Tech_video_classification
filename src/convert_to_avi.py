import os
import subprocess
import argparse
from tqdm import tqdm

def convert_to_avi(source_path, target_path):
    """ Конвертирует видеофайл в формат AVI с помощью FFmpeg. """
    command = ['ffmpeg', '-i', source_path, '-c:v', 'libx264', '-c:a', 'copy', target_path]
    try:
        subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        os.remove(source_path)  # Удаление исходного файла после конвертации
        print(f"Конвертировано и удалено исходное: {target_path}")
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при конвертации {source_path}: {e}")

def scan_and_convert_directory(directory):
    """ Сканирует директорию и конвертирует все MP4 видео в AVI. """
    files_to_convert = [os.path.join(root, file) for root, dirs, files in os.walk(directory) for file in files if file.endswith('.mp4')]
    for file_path in tqdm(files_to_convert, desc="Конвертация видео"):
        target_path = file_path.rsplit('.', 1)[0] + '.avi'
        if not os.path.exists(target_path):  # Проверяем, не было ли уже выполнено преобразование
            convert_to_avi(file_path, target_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=str, help="Путь к директории, которую нужно сканировать.")
    
    args = parser.parse_args()
    scan_and_convert_directory(args.directory)

if __name__ == "__main__":
    main()
