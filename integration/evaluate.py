from typing import Dict, Tuple
import re
import os
import json
import pandas as pd
from PIL import Image
from minicons import scorer


def evaluate(args) -> None:
    rows = []
    with open(args.stimuli_file, encoding="utf-8") as f:
        stimuli = json.load(f)
    for model_name in args.model_names:
        model = scorer.VLMScorer(model_name, device=args.device)
        for stimulus in stimuli['items']:
            prompt = stimulus_to_prompt(stimulus)
            scores = prompt_vlm(model, prompt)
            rows.append(
                {
                    "model": model_name,
                    "plus_amb_score": scores[0],
                    "minus_amb_score": scores[1],
                    "plus_amb_img_score": scores[2],
                    "minus_amb_img_score": scores[3],
                }
            )
    df = pd.DataFrame(rows)
    df.to_csv("results.csv", index=False)


def stimulus_to_prompt(stimulus: Dict[str, str]) -> Tuple[Image.Image, str, str, str]:
    assert stimulus.keys() == frozenset(["image", "text", "disambig"])

    # unclear if need jpg format or etc.
    image = Image.open(stimulus["image"])
    if not image.mode == 'RGB':
        image = image.convert('RGB')

    prefix, critical, _ = stimulus["text"].split("|")
    plus_amb = re.sub("<disambig>", "", prefix)
    minus_amb = re.sub("<disambig>", stimulus["disambig"], prefix)

    plus_amb = re.sub(r"\s+", " ", plus_amb).strip()
    minus_amb = re.sub(r"\s+", " ", minus_amb).strip()

    return image, plus_amb, minus_amb, critical


def prompt_vlm(
    model: scorer.VLMScorer, prompt: Tuple[Image.Image, str, str, str]
) -> Tuple[float, float, float, float]:
    image, plus_amb, minus_amb, critical = prompt
    
    plus_amb_img_score, minus_amb_img_score = model.conditional_score(
        prefix=[plus_amb, minus_amb],
        stimuli=[critical, critical],
        image=[image, image],
        reduction=lambda x: x.sum(0).item(),
    )

    plus_amb_score, minus_amb_score = model.conditional_score(
        prefix=[plus_amb, minus_amb],
        stimuli=[critical, critical],
        image=None,  # FIX WITH PATH TO BLANK IMAGE
        reduction=lambda x: x.sum(0).item(),
    )

    return plus_amb_score, minus_amb_score, plus_amb_img_score, minus_amb_img_score