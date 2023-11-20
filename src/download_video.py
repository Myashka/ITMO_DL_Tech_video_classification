import json
import os
from tqdm import tqdm
import subprocess


output_base_path = "../data/videos"
json_path = "../data/filtered_COIN.json"
failed_downloads = {}

if not os.path.exists(output_base_path):
    os.makedirs(output_base_path)

data = json.load(open(json_path, "r"))["database"]

for youtube_id, info in tqdm(data.items()):
    domain = info["domain"].replace(" ", "")
    subset = info["subset"]
    subset_dir = (
        "train" if subset == "training" else "val" if subset == "testing" else None
    )

    if subset_dir:
        vid_class_dir = os.path.join(output_base_path, subset_dir, domain)
        os.makedirs(vid_class_dir, exist_ok=True)

        url = info["video_url"].split("/")[-1]
        vid_output_path = os.path.join(vid_class_dir, f"{youtube_id}.mp4")

        command = (
            f'yt-dlp -o "{vid_output_path}" "https://www.youtube.com/watch?v={url}"'
        )
        result = subprocess.run(
            command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        if result.returncode != 0:
            failed_downloads.setdefault(domain, []).append(youtube_id)

# Сохраняем информацию о неудачных попытках скачивания
with open("../data/failed_downloads.json", "w") as f:
    json.dump(failed_downloads, f, indent=4)

# Выводим количество неудачных попыток по категориям
for domain, ids in failed_downloads.items():
    print(f"Не удалось скачать {len(ids)} видео в категории '{domain}'")
