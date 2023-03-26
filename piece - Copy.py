# piece dataclass

from dataclasses import dataclass, field

from side import Side   # dataclass

@dataclass
class Piece:
    name: str
    corners: list[list] = field(default_factory=list)
    sides: list[Side] = field(default_factory=list)
    

    