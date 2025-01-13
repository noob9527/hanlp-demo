from typing import List, Tuple, Optional, Set

from pydantic import BaseModel


class Term(BaseModel):
    token: str
    pos_ctb: str
    pos_pku: str


class AnalysisReq(BaseModel):
    """
    for ctb pos, NR and NN turns out to be useful for keywords
    NN: 支付宝，财付通
    NR: 其他专有名词
    VV: 动词，大部分没用，少数可能有用 e.g. 支付, 汇付
    JJ, 其他名词修饰语 大部分没用，有用的有 e.g. 大型, 连锁，二手
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


class FineCoarseAnalysisResponse(BaseModel):
    """
    同时包含细粒度和粗粒度的分析结果
    """
    fine: AnalysisResponse
    coarse: AnalysisResponse