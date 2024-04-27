from typing import Optional, List

from pydantic import BaseModel

from models.input_models import MammogramAction, UltrasoundAction


class PointDTO(BaseModel):
    x: int
    y: int


class MaskDTO(BaseModel):
    number_of_lesions: int
    mask: List[List[PointDTO]] = None


class PredictionDTO(BaseModel):
    action: MammogramAction | UltrasoundAction
    severity: Optional[str] = None
    mask: Optional[MaskDTO] = None
    severity_mask: Optional[str] = None
