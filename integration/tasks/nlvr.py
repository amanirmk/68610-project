import json
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional
from transformers import AutoProcessor, AutoModelForVision2Seq, AdamW, get_linear_schedule_with_warmup
from PIL import Image
import torch
import random
import numpy as np
from tqdm import tqdm
from minicons import scorer
from integration.abstract import Object
from integration.util import load_scorer, format_prompt, load_model_and_processor, to_scorer


class NLVR(Object):
    pass


def load_nlvr_data(file_path: str, split: str) -> Tuple[list, Dict[str, str]]:
    NLVR.info(f"Loading data from {file_path}")

    if not Path(file_path).exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        labeled_examples = [json.loads(line) for line in f if line]

    NLVR.info(f"Loaded {len(labeled_examples)} examples")

    image_paths = {}
    for example in labeled_examples:
        base_dir = Path(file_path).parent
        img_dir = base_dir / "images" / example["directory"]

        if not img_dir.exists():
            NLVR.warn(f"Image directory does not exist: {img_dir}")
            continue

        for i in range(6):
            identifier = f"{example['identifier']}-{i}"
            img_filename = f"{split}-{example['identifier']}-{i}.png"
            img_path = img_dir / img_filename
            image_paths[identifier] = str(img_path)

    NLVR.info(f"Generated {len(image_paths)} image paths")
    NLVR.info("Sample image paths:")
    for i, path in enumerate(list(image_paths.values())[:3]):
        NLVR.info(f"Sample {i+1}: {path}")

    return labeled_examples, image_paths


def zeroshot_nlvr(
    model_name: str, save_intermediate: bool = False
) -> Tuple[float, float]:
    vlm_scorer = load_scorer(model_name)
    return test_nlvr(model_name, vlm_scorer, save_intermediate=save_intermediate, tag='zeroshot')


def test_nlvr(model_name: str, vlm_scorer: scorer.VLMScorer, save_intermediate: bool = False, tag: Optional[str] = None) -> Tuple[float, float]:
    NLVR.info("Loading data...")
    labeled_examples, image_paths = load_nlvr_data("nlvr/nlvr/dev/dev.json", "dev")

    predictions = {}
    failed_loads = []

    for example in tqdm(labeled_examples, desc="Collecting NLVR1 Predictions"):
        sentence = example["sentence"].rstrip(".")

        prefix = f"Answer the following question about the image.\nQuestion: Is it true or false that `{sentence}`?\nAnswer:"
        prefix = format_prompt(prefix, model_name)
        stimuli = ["True", "False"]

        for i in range(6):
            identifier = f"{example['identifier']}-{i}"

            if not Path(image_paths[identifier]).exists():
                NLVR.info(f"Image file does not exist: {image_paths[identifier]}")
                failed_loads.append(identifier)
                continue

            try:
                image = Image.open(image_paths[identifier])
                if not image.mode == "RGB":
                    image = image.convert("RGB")

                true_logprob, false_logprob = vlm_scorer.conditional_score(
                    prefix=[prefix, prefix],
                    stimuli=stimuli,
                    image=[image, image],
                    reduction=lambda x: x.sum(0).item(),
                )

                prediction = "true" if true_logprob > false_logprob else "false"
                predictions[identifier] = prediction

            except Exception as e:
                NLVR.error(f"Error processing {identifier}: {e}")
                failed_loads.append(identifier)
                continue

    NLVR.info("Saving predictions.")
    filename = model_name.replace("/", "--") + f"-nlvr1-{tag + '-' if tag else ''}answers.csv"
    with open(filename, "w") as f:
        for identifier, pred in predictions.items():
            f.write(f"split-{identifier}.png,{pred}\n")

    NLVR.info("Analyzing predictions.")
    result = subprocess.run(
        ["python", "nlvr/nlvr/metrics_images.py", filename, "nlvr/nlvr/dev/dev.json"],
        capture_output=True,
        text=True,
    )

    if not save_intermediate:
        subprocess.run(["rm", filename])

    precision = 0.0
    consistency = 0.0
    for line in result.stdout.split("\n"):
        if "precision=" in line:
            precision = float(line.split("=")[1])
        elif "consistency=" in line:
            consistency = float(line.split("=")[1])
    return precision, consistency


def finetune_nlvr(
    model_name: str,
    save_intermediate: bool = False,
    num_epochs: int = 1,
    learning_rate: float = 1e-7,
    train_limit: Optional[int] = None,
    gradient_accumulation_steps: int = 4,
    warmup_steps: int = 100,
) -> None:
    model, processor = load_model_and_processor(model_name)

    labeled_examples, image_paths = load_nlvr_data("nlvr/nlvr/train/train.json", "train")
    NLVR.info(f"THERE ARE A TOTAL OF {len(labeled_examples)} LABELED EXAMPLES (ie {6 * len(labeled_examples)} items)")
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    np.random.shuffle(labeled_examples)
    if train_limit is not None:
        labeled_examples = labeled_examples[:train_limit]

    NLVR.info(f"Training on {len(labeled_examples)} examples for {num_epochs} epoch(s).")

    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(labeled_examples) * num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )

    model.train()
    for _ in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(labeled_examples, desc="Finetuning on nlvr")
        for step, example in enumerate(progress_bar):
            sentence = example["sentence"].rstrip(".").capitalize()

            prefix = f"Answer the following question about the image.\nQuestion: Is it true or false that `{sentence}`?\nAnswer:"
            prefix = format_prompt(prefix, model_name)
            prefixes = [prefix]*6

            identifiers = [f"{example['identifier']}-{i}" for i in range(6)]
            images = [Image.open(image_paths[i]).convert('RGB') for i in identifiers]
            
            labels = [example["label"].capitalize()]*6

            processor.tokenizer.padding_side = 'left' # keep text tokens recent
            inputs = processor(
                images=images,
                text=prefixes,
                return_tensors="pt",
                padding=True
            ).to("cuda", torch.float16)

            # Process targets
            processor.tokenizer.padding_side = 'right' # immediately predict text tokens
            targets = processor.tokenizer( # use tokenizer so no image tokens added
                text=labels,
                return_tensors="pt",
                padding=True,
                add_special_tokens=False # don't start with </s>
            ).to('cuda')

            # Forward pass
            outputs = model(
                **inputs,
                labels=targets.input_ids,
                decoder_attention_mask=targets.attention_mask
            )
          
            loss = outputs.loss / gradient_accumulation_steps
            loss.backward()

            # Gradient accumulation
            if (step + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            total_loss += loss.item() * gradient_accumulation_steps
            progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})

        avg_loss = total_loss / len(labeled_examples)
        NLVR.info(f"Average loss: {avg_loss:.4f}")

    if save_intermediate:
        output_dir = f"{'finetuned/' + model_name.replace('/', '--')}-nlvr-finetuned"
        model.save_pretrained(output_dir)
        processor.save_pretrained(output_dir)

    vlm_scorer = to_scorer(model, processor)
    return test_nlvr(model_name, vlm_scorer, save_intermediate=save_intermediate, tag='finetune')

