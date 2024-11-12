import dataclasses
from typing import List, Optional


@dataclasses.dataclass
class Arguments:
    stimuli_file: str = dataclasses.field(
        default="stimuli/stimuli.json",
    )
    model_names: List[str] = dataclasses.field(
        default_factory=lambda: ["Salesforce/blip2-opt-2.7b"],
    )
    device: str = dataclasses.field(
        default="cuda",
    )
