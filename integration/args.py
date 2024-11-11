import dataclasses
from typing import Optional


@dataclasses.dataclass
class Arguments:
    example_arg: Optional[str] = dataclasses.field(
        default="example value",
    )
