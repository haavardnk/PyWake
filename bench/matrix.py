MODEL_NAMES = ("turbopark", "nojlocal", "fugamultilut", "noj", "bastankhah", "niayifar", "fuga")
SINGLE_TYPE_MODELS = {"fuga"}
DEFAULT_FARMS = ("hr1", "grid400", "aligned100")

MATRIX = [
    (model, farm)
    for model in MODEL_NAMES
    for farm in DEFAULT_FARMS
    if not (model in SINGLE_TYPE_MODELS and farm != "hr1")
]


def names(selection=None):
    if not selection:
        return [f"{model}-{farm}" for model, farm in MATRIX]
    return [name for item in selection for name in item.split(",")]
