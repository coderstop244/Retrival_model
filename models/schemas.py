from pydantic import BaseModel
from typing import List

class HackRxRequest(BaseModel):
    """
    Model for the incoming API request body.
    """
    documents: str
    questions: List[str]

class HackRxResponse(BaseModel):
    """
    Model for the outgoing API response body.
    """
    answers: List[str]
