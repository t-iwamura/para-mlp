from dataclasses import dataclass


@dataclass
class ModelParams:
    use_force: bool = False
    use_stress: bool = False
    composite_num: int = 2
