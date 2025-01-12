from typing import List, Tuple, Optional, Set

from pydantic import BaseModel


class Term(BaseModel):
    token: str
    pos_ctb: str
    pos_pku: str


class AnalysisReq(BaseModel):
    """
    for ctb pos, NR and NN turns out to be useful for keywords
    # https://hanlp.hankcs.com/docs/annotations/pos/ctb.html
    """
    text: str
    allow_pos_ctb: Optional[Set[str]] = None
    allow_pos_pku: Optional[Set[str]] = None


class NamedEntity(BaseModel):
    entity: str
    type: str
    offset: Tuple[int, int]


class AnalysisResponse(BaseModel):
    terms: List[Term]
    named_entities: Optional[List[NamedEntity]]
