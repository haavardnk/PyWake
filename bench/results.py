import json
import re

GB = 1e9
FARM_SIZE = re.compile(r"(?:grid|scale)([1-9]\d*)$")
MODELS = {
    "turbopark": "TurbOPark",
    "niayifar": "Niayifar",
    "fugamultilut": "FugaMultiLUT",
    "noj": "NOJ",
    "nojlocal": "NOJLocalDeficit",
    "bastankhah": "Bastankhah",
    "fuga": "Fuga",
}


def read(path):
    with open(path) as fid:
        return json.load(fid)


def farm_size(farm):
    match = FARM_SIZE.match(farm)
    return int(match.group(1)) if match else None


def comparisons(paths):
    rows = {}
    for path in paths:
        for case, sides in read(path)["results"].items():
            model, farm = case.split("-")
            size = farm_size(farm)
            if size and model in MODELS:
                rows[size, model] = {side: sides[side] for side in ("base", "head")}
    return dict(sorted(rows.items(), key=lambda item: (item[0][0], list(MODELS).index(item[0][1]))))
