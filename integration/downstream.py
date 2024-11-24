def get_downstream_performance(
    model_name: str, task_name: str, device: str = "cuda"
) -> float:
    tasks = {
        "VQA": test_vqa,
        "NLVR": test_nlvr,
        # so on...
    }

    # load the VLM (with huggingface)
    model = ...
    # apply the task-specific evaluation function
    return tasks[task_name](model)


def test_vqa(model) -> float:
    return 0.0


def test_nlvr(model) -> float:
    return 0.0
