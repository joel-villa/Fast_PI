from enum import Enum

class BoundsType(Enum):
    """The types of bounds allowed

    Args:
        Enum (_type_): BoundsType inherates methods from teh Enum class
    """
    POWER = 0
    STRICT = 1
