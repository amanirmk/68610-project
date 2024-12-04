import json
import subprocess
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple
from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch
from tqdm import tqdm
from integration.abstract import Object


class NLVR(Object):
    pass


def load_nlvr_data(file_path: str) -> Tuple[list, Dict[str, str]]:
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

        split_name = "dev" if "dev" in str(file_path) else "test"
        for i in range(6):
            identifier = f"{example['identifier']}-{i}"
            img_filename = f"{split_name}-{example['identifier']}-{i}.png"
            img_path = img_dir / img_filename
            image_paths[identifier] = str(img_path)

    NLVR.info(f"Generated {len(image_paths)} image paths")
    NLVR.info("Sample image paths:")
    for i, path in enumerate(list(image_paths.values())[:3]):
        NLVR.info(f"Sample {i+1}: {path}")

    return labeled_examples, image_paths


def zeroshot_nlvr(model_name: str, device: str) -> Tuple[float, float]:
    NLVR.info(f"Loading model: {model_name}")
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForVision2Seq.from_pretrained(model_name).to(device)

    NLVR.info("Loading data...")
    labeled_examples, image_paths = load_nlvr_data("nlvr/nlvr/dev/dev.json")

    predictions = {}
    failed_loads = []

    NLVR.info(f"Processing {len(labeled_examples)} examples...")

    true_token = processor.tokenizer.encode("true", add_special_tokens=False)[0]
    false_token = processor.tokenizer.encode("false", add_special_tokens=False)[0]

    for example in tqdm(labeled_examples, desc="Collecting NLVR1 Predictions"):
        sentence = example["sentence"]
        prompt = f"True or false: {sentence.rstrip('.') + '.'}\nAnswer: "

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

                inputs = processor(images=image, text=prompt, return_tensors="pt").to(
                    device
                )
                with torch.no_grad():
                    outputs = model(**inputs)

                logits = outputs.logits[0, -1]
                true_score = logits[true_token].item()
                false_score = logits[false_token].item()

                prediction = "true" if true_score > false_score else "false"
                predictions[identifier] = prediction

            except Exception as e:
                NLVR.error(f"Error processing {identifier}: {e}")
                failed_loads.append(identifier)
                continue

    NLVR.info("Saving predictions.")
    filename = model_name.replace("/", "--") + "-nlvr1-answers.csv"
    with open(filename, "w") as f:
        for identifier, pred in predictions.items():
            f.write(f"split-{identifier}.png,{pred}\n")

    NLVR.info("Analyzing predictions.")
    result = subprocess.run(
        ["python", "nlvr/nlvr/metrics_images.py", filename, "nlvr/nlvr/dev/dev.json"],
        capture_output=True,
        text=True,
    )

    precision = 0.0
    consistency = 0.0
    for line in result.stdout.split("\n"):
        if "precision=" in line:
            precision = float(line.split("=")[1])
        elif "consistency=" in line:
            consistency = float(line.split("=")[1])
    return precision, consistency
