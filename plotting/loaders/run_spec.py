# plotting/loaders/run_spec.py
from dataclasses import dataclass
from .params import extract_params


@dataclass
class RunSpec:
    results_folder: str
    results_parent: str
    method: str
    tmax: float | None = None
    index: int = 0
    w_c: float | None = None
    w_lr: float | None = None
    w_2: float | None = None
    w_3: float | None = None

    def __post_init__(self):
        if self.tmax is None:
            self.tmax = extract_params(["tmax"], self.results_parent).get("tmax")
        
        if self.w_c is None:
            self.w_c = extract_params(["m", "lr"], self.results_parent).get("m")
        if self.w_lr is None:
            self.w_lr = extract_params(["m", "lr"], self.results_parent).get("lr")
        
        if self.w_2 is None:
            self.w_2 = extract_params(["secondW", "thirdW"], self.results_parent).get("secondW")
        if self.w_3 is None:
            self.w_3 = extract_params(["secondW", "thirdW"], self.results_parent).get("thirdW")

    @property
    def window_length(self) -> float:
        # tmin = 0.5 + 0.14，如果这个常量别处也用，抽到 config
        return self.tmax - 0.64