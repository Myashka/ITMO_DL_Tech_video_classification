import torch


def collate_fn(examples):
    pixel_values_list = [example["video"].permute(1, 0, 2, 3) for example in examples]
    labels_list = [example["label"] for example in examples]

    pixel_values = torch.stack(pixel_values_list)
    labels = torch.tensor(labels_list)

    return {"pixel_values": pixel_values, "labels": labels}
    # pixel_values = torch.stack(
    #     [example["video"] for example in examples]
    # )
    # labels = torch.tensor([example["label"] for example in examples])
    # pixel_values, labels = mix_up(pixel_values, labels)
    # # permute to (num_frames, num_channels, height, width)
    # pixel_values = torch.stack(
    #     [pixel_value.permute(1, 0, 2, 3) for pixel_value in pixel_values]
    # )
    # labels = torch.argmax(labels, axis=1)
    # return {"pixel_values": pixel_values, "labels": labels}
