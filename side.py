
from dataclasses import dataclass, field   # dataclass

@dataclass
class Side:
    name: str
    points: list[list] = field(default_factory=list)
    side_number: int = 0
    in_out: str = 'in'
    size: tuple = (0, 0)
