import os
import sys
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from transformers import TrainingArguments, Trainer
from transformers import VideoMAEConfig

from .metrics import compute_metrics
from .make_dataset import make_datasets
from .utils import collate_fn



def main():
    class_labels = sorted(
        os.listdir(
            "/content/home/myashka/dl_programming_tech/coin_dataset_classification/data/videos/train"
        )
    )
    label2id = {label: i for i, label in enumerate(class_labels)}
    id2label = {i: label for label, i in label2id.items()}

    print(f"Unique classes {len(list(label2id.keys()))}: {list(label2id.keys())}.")

    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}

    model_ckpt = "MCG-NJU/videomae-base"
    config = VideoMAEConfig(
        image_size=112,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,
    )
    image_processor = VideoMAEImageProcessor.from_pretrained(model_ckpt, config=config)

    model = VideoMAEForVideoClassification.from_pretrained(
        model_ckpt,
        config=config,
        torch_dtype=torch.float16,
        device_map=device_map,
    )

    train_dataset, val_dataset = make_datasets(image_processor, model)
    
    os.environ["WANDB_PROJECT"] = "ITMO_Video_lab"
    os.environ["WANDB_LOG_MODEL"] = "checkpoint"

    for param in model.parameters():
        param.requires_grad = False

    for param in model.classifier.parameters():
        param.requires_grad = True
    
    for param in model.videomae.encoder.layers[-1].parameters():
        param.requires_grad = True
    
    model.enable_input_require_grads()
    model_name = model_ckpt.split("/")[-1]
    new_model_name = f"{model_name}-finetuned-coin_domains-112"
    num_epochs = 10
    batch_size = 24

    args = TrainingArguments(
        new_model_name,
        remove_unused_columns=False,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        max_steps=(train_dataset.num_videos // batch_size) * num_epochs,
        seed=42,

        learning_rate=5e-5,
        weight_decay=0.05,
        max_grad_norm=1,
        # optim="adafactor",

        lr_scheduler_type="cosine",
        warmup_ratio=0.1,

        fp16=True, # torch.float16 wieghts
        gradient_checkpointing=True,
        # gradient_accumulation_steps: 1

        evaluation_strategy="steps",
        eval_steps=train_dataset.num_videos // batch_size,
        save_strategy="steps",
        save_steps=train_dataset.num_videos // batch_size,

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_total_limit=2,
        hub_model_id='Myashka/videomae-base-coin-domains-no_mixup-112',
        push_to_hub=True,

        logging_steps=1,
        report_to='wandb',
        run_name="train-videomae-domains-lr5e_5-no_mixup-not_fr_11_cls-bs_24"
    )
    trainer = Trainer(
        model,
        args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=image_processor,
        compute_metrics=compute_metrics,
        data_collator=collate_fn,
    )

    model.gradient_checkpointing_enable()
    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    train_results = trainer.train()
    print(train_results)

if __name__ == "__main__":
    main()