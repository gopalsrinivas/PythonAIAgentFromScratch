from pydantic import BaseModel
from typing import List


class ResearchRequest(BaseModel):
    query: str


class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: List[str]
    tools_used: List[str]
