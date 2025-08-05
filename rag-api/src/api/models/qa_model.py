from typing import Optional, List
from pydantic import BaseModel, Field
from uuid import uuid4


class QaInputModel(BaseModel):
    session_id: Optional[str] = Field(default=uuid4().hex)
    question: str = Field(max_length=1024)


class References(BaseModel):
    filename: str
    page_number: Optional[int] = -1


class QaOutputModel(BaseModel):
    output: str
    references: Optional[List[References]] = None
