from typing import Dict, Tuple
import re
import json
import pandas as pd
from PIL import Image
from minicons import scorer
from integration.abstract import Object

class Eval(Object):
    pass

def evaluate(args) -> None:
    rows = []
    with open(args.stimuli_file, encoding="utf-8") as f:
        stimuli = json.load(f)
    for model_name in args.model_names:
        try:
            model = scorer.VLMScorer(model_name, device=args.device)
            for stimulus in stimuli["items"]:
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
        except Exception as e:
            Eval.error(f"Failed to evaluate {model_name}: {e}")
    df = pd.DataFrame(rows)
    compute_and_save_scores(df)


def stimulus_to_prompt(stimulus: Dict[str, str]) -> Tuple[Image.Image, str, str, str]:
    assert stimulus.keys() == frozenset(["image", "text", "disambig"])

    image = Image.open(stimulus["image"])
    if not image.mode == "RGB":
        image = image.convert("RGB")

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
    blank = Image.new("RGB", image.size, color="black")

    plus_amb_img_score, minus_amb_img_score = model.conditional_score(
        prefix=[plus_amb, minus_amb],
        stimuli=[critical, critical],
        image=[image, image],
        reduction=lambda x: -x.sum(0).item(),
    )

    plus_amb_score, minus_amb_score = model.conditional_score(
        prefix=[plus_amb, minus_amb],
        stimuli=[critical, critical],
        image=[blank, blank],
        reduction=lambda x: -x.sum(0).item(),
    )

    return plus_amb_score, minus_amb_score, plus_amb_img_score, minus_amb_img_score


def compute_and_save_scores(df: pd.DataFrame) -> None:
    df["DT"] = df["plus_amb_score"] - df["minus_amb_score"]
    df["DV"] = df["plus_amb_score"] - df["plus_amb_img_score"]
    df["DV_adj"] = df["DV"] - (df["minus_amb_score"] - df["minus_amb_img_score"])
    df["ViPr"] = df["DV"] / df["DT"]
    df["ViPr_adj"] = df["DV_adj"] / df["DT"]
    df.to_csv("all_scores.csv", index=False)

    summary = df.groupby("model")[["ViPr", "ViPr_adj"]].mean().reset_index()
    summary.to_csv("summary.csv", index=False)
