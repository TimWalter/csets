from dataclasses import dataclass


@dataclass
class Config:
    """
    Global settings.

    Attributes:
        enable_grad: Whether operations keep what they need to be differentiated.
        tolerance: Tolerance on constraint residuals.
    """
    enable_grad: bool = True
    tolerance: float = 1e-6


config = Config()
