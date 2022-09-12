# generated by fastapi-codegen:
#   filename:  app/openapi/openapi.yaml
#   timestamp: 2022-09-12T11:57:37+00:00

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class RequestExample(BaseModel):
    pclass: str = Field(..., title='Pclass')
    sex: str = Field(..., title='Sex')
    age: float = Field(..., title='Age')


class ResponseExample(BaseModel):
    survived: str = Field(..., title='Survived')


class ValidationError(BaseModel):
    loc: List[str] = Field(..., title='Location')
    msg: str = Field(..., title='Message')
    type: str = Field(..., title='Error Type')


class HTTPValidationError(BaseModel):
    detail: Optional[List[ValidationError]] = Field(None, title='Detail')