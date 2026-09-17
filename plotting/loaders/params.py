# plotting/loaders/params.py
import re

def extract_params(keys: list[str], text: str) -> dict[str, int | float] | None:
    def to_num(s):
        try:
            return int(s)
        except ValueError:
            return round(float(s), 10)

    regex = "_".join([k + r'(\d+(?:\.\d+)?)' for k in keys])
    match = re.match(regex, text)
    
    if not match:
        return {}
    values = match.groups()
    return {k: to_num(v) for k, v in zip(keys, values)}