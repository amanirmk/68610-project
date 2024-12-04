import dataclasses
from typing import List


@dataclasses.dataclass
class Arguments:
    stimuli_file: str = dataclasses.field(
        default="stimuli/stimuli.json",
    )
    model_names: List[str] = dataclasses.field(
        default_factory=lambda: [
            # "Salesforce/blip2-opt-2.7b",
            # "Salesforce/blip2-opt-6.7b",
            # "Salesforce/blip2-opt-2.7b-coco",
            # "Salesforce/blip2-opt-6.7b-coco",
            # "Salesforce/blip-image-captioning-base",
            # "Salesforce/blip-image-captioning-large",
            # "microsoft/git-base",
            # "microsoft/git-large",
            # "microsoft/git-base-coco",
            # "microsoft/git-large-coco",
            # "microsoft/git-base-textcaps",
            # "microsoft/git-large-textcaps",
            # "microsoft/git-base-msrvtt-qa",
            # "microsoft/git-large-msrvtt-qa",
            # "microsoft/kosmos-2-patch14-224",
            # "google/paligemma-3b-pt-224",
            # "google/paligemma-3b-pt-448",
            # "google/paligemma-3b-mix-224",
            # "google/paligemma-3b-mix-448",
            # "google/paligemma-3b-ft-ocrvqa-224",
            # "google/paligemma-3b-ft-ocrvqa-448",
        ],
        metadata={"nargs": "+"},
    )
    device: str = dataclasses.field(
        default="cuda",
    )
