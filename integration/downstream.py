import pandas as pd
from transformers import AutoModelForVision2Seq, AutoModelForVisualQuestionAnswering


def get_downstream_performance(
    model_name: str, output_file: str, device: str = "cuda"
) -> None:
    tasks = {
        "VQA": test_vqa,
        "NLVR": test_nlvr,
        # so on...
    }
    rows = []
    for task_name, task_func in tasks.items():
        rows.append({"model": model_name, task_name: task_func(model_name, device)})
    df = pd.DataFrame(rows)
    df.to_csv(output_file)


def test_vqa(model_name, device) -> float:
    model = AutoModelForVisualQuestionAnswering.from_pretrained(model_name).to(device)
    return 0.0


def test_nlvr(model_name, device) -> float:
    model = AutoModelForVisualQuestionAnswering.from_pretrained(model_name).to(device)
    return 0.0
