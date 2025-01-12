from typing import List, Tuple, Optional, Set

import hanlp

from src.analysis.models import AnalysisResponse, Term, NamedEntity

TOK_FINE = "tok_fine"
POS_CTB = "pos_ctb"
POS_PKU = "pos_pku"
NAMED_ENTITIES = "named_entities"
TERMS = "terms"

__tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)

__ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
__pos_ctb9 = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
__pos_pku = hanlp.load(hanlp.pretrained.pos.PKU_POS_ELECTRA_SMALL)


def __sum(sentences: List[str]):
    return sum(sentences, [])


def __zip_terms(*args) -> List[Term]:
    tok_fine, pos_ctb9, pos_pku = args
    res = []
    for tok, pos9, posp in zip(tok_fine, pos_ctb9, pos_pku):
        for t, p9, pp in zip(tok, pos9, posp):
            res.append(Term(
                token=t,
                pos_ctb=p9,
                pos_pku=pp
            ))
    return res


def _filter_terms(
        terms: List[Term],
        allow_pos_ctb: Optional[Set[str]] = None,
        allow_pos_pku: Optional[Set[str]] = None,
) -> List[Term]:
    res = terms
    if allow_pos_ctb:
        res = [term for term in terms if term.pos_ctb in allow_pos_ctb]
    if allow_pos_pku:
        res = [term for term in terms if term.pos_pku in allow_pos_pku]
    return res


__kw_analysis_pipeline = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
    .append(__tok_fine, input_key='sentences', output_key=TOK_FINE) \
    .append(__pos_ctb9, input_key=TOK_FINE, output_key=POS_CTB) \
    .append(__pos_pku, input_key=TOK_FINE, output_key=POS_PKU) \
    .append(__ner, input_key=TOK_FINE, output_key=NAMED_ENTITIES) \
    .append(__sum, input_key=NAMED_ENTITIES, output_key=NAMED_ENTITIES) \
    .append(__zip_terms, input_key=(TOK_FINE, POS_CTB, POS_PKU), output_key=TERMS)


def _filter_named_entities(
        items: List[Tuple[str, str, int, int]]
) -> List[Tuple[str, str, int, int]]:
    """
    ('英伟达', 'ORGANIZATION', 9, 10)
    ('西欧', 'LOCATION', 27, 28)
    Returns:

    """
    return [item for item in items if item[1] == 'ORGANIZATION']


def analysis(
        text: str,
        allow_pos_ctb: Optional[Set[str]] = None,
        allow_pos_pku: Optional[Set[str]] = None,
) -> AnalysisResponse:
    docs = __kw_analysis_pipeline(text)
    terms = _filter_terms(
        docs[TERMS],
        allow_pos_ctb=allow_pos_ctb,
        allow_pos_pku=allow_pos_pku
    )

    named_entities = _filter_named_entities(docs[NAMED_ENTITIES])
    ne_response = [
        NamedEntity(
            entity=ne[0],
            type=ne[1],
            offset=ne[2:4],
        )
        for ne in named_entities
    ]

    return AnalysisResponse(
        terms=terms,
        named_entities=ne_response
    )
