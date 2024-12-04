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
    model_sizes = []
    with open(args.stimuli_file, encoding="utf-8") as f:
        stimuli = json.load(f)
    for model_name in args.model_names:
        try:
            model = scorer.VLMScorer(model_name, device=args.device)
            model_sizes.append(
                {"model": model_name, "model_size": model.model.num_parameters()}
            )
            model_rows = []
            for stimulus in stimuli["items"]:
                prompt = stimulus_to_prompt(stimulus)
                scores = prompt_vlm(model, prompt)
                row = {
                    "model": model_name,
                    "plus_amb_score": scores[0],
                    "minus_amb_score": scores[1],
                    "plus_amb_img_score": scores[2],
                    "minus_amb_img_score": scores[3],
                }
                rows.append(row)
                model_rows.append(row)
            pd.DataFrame(model_rows).to_csv(
                f"{model_name.replace('/', '--')}-vipr-save-check.csv", index=False
            )
        except Exception as e:
            Eval.error(f"Failed to evaluate {model_name}: {e}")

    model_size_df = pd.DataFrame(model_sizes)
    model_size_df.to_csv(
        f"{'_'.join(m.split('/')[1] for m in model_size_df['model'].unique())}_sizes.csv",
        index=False,
    )

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
    df["ViPr_abs"] = df["DV"].abs() / df["DT"].abs()
    df["ViPr_abs_adj"] = df["DV_adj"].abs() / df["DT"].abs()

    eps = 1e-3
    df["DT+"] = df["DT"].copy()
    df["DT+"][df["DT+"] <= 0] = eps
    df["DV+"] = df["DV"].copy()
    df["DV+"][df["DV+"] < 0] = 0
    df["DV_adj+"] = df["DV_adj"].copy()
    df["DV_adj+"][df["DV_adj"] < 0] = 0
    df["ViPr+"] = df["DV+"] / df["DT+"]
    df["ViPr_adj+"] = df["DV_adj+"] / df["DT+"]

    df.to_csv(
        f"{'_'.join(m.split('/')[1] for m in df['model'].unique())}_scores.csv",
        index=False,
    )

    summary = (
        df.groupby("model")[
            [
                "ViPr",
                "ViPr_adj",
                "ViPr_abs",
                "ViPr_abs_adj",
                "ViPr+",
                "ViPr_adj+",
                "DV",
                "DV+",
                "DV_adj",
                "DV_adj+",
            ]
        ]
        .mean()
        .reset_index()
    )
    summary.to_csv(
        f"{'_'.join(m.split('/')[1] for m in df['model'].unique())}_summary.csv",
        index=False,
    )
