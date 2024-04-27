from typing import Optional, List
from pydantic import BaseModel, Field

from models.dto import PredictionDTO, PointDTO, MaskDTO
from models.input_models import MammogramAction, UltrasoundAction


class PointResponse(BaseModel):
    x: int = Field(..., alias='X')
    y: int = Field(..., alias='Y')

    @classmethod
    def from_dto(cls, dto: PointDTO):
        return cls(X=dto.x, Y=dto.y)


class MaskResponse(BaseModel):
    number_of_lesions: int = Field(..., alias='numberOfLesions')
    mask: List[List[PointResponse]] = Field(default=None, alias='mask')

    @classmethod
    def from_dto(cls, dto: MaskDTO):
        mask = []
        if dto.mask:
            for lesion in dto.mask:
                mask.append([PointResponse.from_dto(point_dto) for point_dto in lesion])
        return cls(numberOfLesions=dto.number_of_lesions, mask=mask)


class PredictionMGResponse(BaseModel):
    modality: str = 'MG'
    action: MammogramAction = Field(..., alias='action')
    severity: Optional[str] = Field(default=None, alias='severity')
    mask: Optional[MaskResponse] = Field(default=None, alias='mask')

    @classmethod
    def from_dto(cls, dto: PredictionDTO):
        return cls(
            action=dto.action,
            severity=dto.severity,
            mask=MaskResponse.from_dto(dto.mask) if dto.mask else None
        )


class PredictionUSResponse(BaseModel):
    modality: str = 'US'
    action: UltrasoundAction = Field(..., alias='action')
    severity: Optional[str] = Field(default=None, alias='severity')
    severity_mask: Optional[str] = Field(default=None, alias='severityMask')
    mask: Optional[MaskResponse] = Field(default=None, alias='mask')

    @classmethod
    def from_dto(cls, dto: PredictionDTO):
        return cls(
            action=dto.action,
            severity=dto.severity,
            severityMask=dto.severity_mask,
            mask=MaskResponse.from_dto(dto.mask) if dto.mask else None
        )
