from typing import List, Dict, Any, Optional

from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field


class CandidateSummary(BaseModel):
    name: str = Field(..., description="the candidate name as specified at their cv")
    title: str = Field(..., description="job title as inferred from their cv")
    email: str = Field(..., description="email address as specified at their cv")
    linkedin: Optional[str] = Field(None, description="LinkedIn profile URL as specified at their cv")
    summary: str = Field(..., description="important facts up to 15 words")
    explanation: str = Field(..., description="short explanation for why this candidate is the best fit for the job")

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "title": self, "email": self.email, "summary": self.summary,
                "linkedin": self.linkedin, "explanation": self.explanation}

candidate_summary_parser = PydanticOutputParser(pydantic_object=CandidateSummary)