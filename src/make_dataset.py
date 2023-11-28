import os
import pytorchvideo.data

from pytorchvideo.transforms import (
    ApplyTransformToKey,
    Normalize,
    RandomShortSideScale,
    UniformTemporalSubsample,
)

from torchvision.transforms import v2


def make_datasets(image_processor, model):
    image_processor.size["shortest_edge"] = 112
    mean = image_processor.image_mean
    std = image_processor.image_std
    print(f"Mean: {mean}\nStd: {std}")
    if "shortest_edge" in image_processor.size:
        height = width = image_processor.size["shortest_edge"]
    else:
        height = image_processor.size["height"]
        width = image_processor.size["width"]
    resize_to = (height, width)
    print(f"Resize to {resize_to}")

    num_frames_to_sample = model.config.num_frames
    sample_rate = 4
    fps = 30
    clip_duration = num_frames_to_sample * sample_rate / fps
    print(f"Num frames to sample: {num_frames_to_sample}")
    print(f"Sample rate: {sample_rate}")
    print(f"Duration: {clip_duration} sec")
    train_transform = v2.Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=v2.Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        v2.Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        RandomShortSideScale(min_size=112, max_size=160),
                        v2.RandomCrop(resize_to),
                        v2.RandomHorizontalFlip(p=0.5),
                    ]
                ),
            ),
        ]
    )

    val_transform = v2.Compose(
        [
            ApplyTransformToKey(
                key="video",
                transform=v2.Compose(
                    [
                        UniformTemporalSubsample(num_frames_to_sample),
                        v2.Lambda(lambda x: x / 255.0),
                        Normalize(mean, std),
                        v2.Resize(resize_to),
                    ]
                ),
            )
        ]
    )
    dataset_root_path = "/content/home/myashka/dl_programming_tech/coin_dataset_classification/data/videos/"
    train_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "train"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("random", clip_duration),
        decode_audio=False,
        transform=train_transform,
    )

    val_dataset = pytorchvideo.data.Ucf101(
        data_path=os.path.join(dataset_root_path, "val"),
        clip_sampler=pytorchvideo.data.make_clip_sampler("uniform", clip_duration),
        decode_audio=False,
        transform=val_transform,
    )
    print(train_dataset.num_videos, val_dataset.num_videos)
    print(f"Test data: {round(val_dataset.num_videos/(train_dataset.num_videos+val_dataset.num_videos)*100, 2)}%")
    return train_dataset, val_dataset
