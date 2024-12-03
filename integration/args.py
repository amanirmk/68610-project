import dataclasses
from typing import List


@dataclasses.dataclass
class Arguments:
    stimuli_file: str = dataclasses.field(
        default="stimuli/stimuli.json",
    )
    model_names: List[str] = dataclasses.field(
        default_factory=lambda: [
            "Salesforce/blip2-opt-2.7b",
            "Salesforce/blip2-opt-6.7b",
            "liuhaotian/llava-v1.5-7b",
            "liuhaotian/llava-v1.5-13b"
            "liuhaotian/llava-v1.6-vicuna-7b",
            "liuhaotian/llava-v1.6-vicuna-13b",
            "dhansmair/flamingo-mini",
            "dhansmair/flamingo-tiny",
            "microsoft/git-base",
            "microsoft/git-large",
            "llava-hf/llava-onevision-qwen2-0.5b-ov-hf",
            "llava-hf/llava-onevision-qwen2-7b-ov-hf",
            "facebook/chameleon-7b",
            "HuggingFaceM4/Idefics3-8B-Llama3",
            "Qwen/Qwen2-VL-2B-Instruct",
            "Qwen/Qwen2-VL-7B-Instruct",
        ],
    )
    device: str = dataclasses.field(
        default="cuda",
    )
