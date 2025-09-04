import logging
from typing import List, Tuple, Optional, Set

import hanlp
import torch

from src.analysis.models import AnalysisResponse, Term, NamedEntity, \
    FineCoarseAnalysisResponse
from src.split_sentence import split_sentence_with_index

# Add logging configuration near the top of the file
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def has_gpu() -> bool:
    """
    Check if a GPU is available for use
    Returns: True if GPU is available, False otherwise
    """
    return torch.cuda.is_available()


# Add this before loading models
if has_gpu():
    logger.info("GPU is available - models will run on GPU")
else:
    logger.info("No GPU detected - models will run on CPU")

TOKEN_WITH_SPAN = "token_with_span"
TOKEN = "token"
POS_CTB = "pos_ctb"
POS_PKU = "pos_pku"
NAMED_ENTITIES = "named_entities"
TERMS = "terms"

# Text length threshold for using paragraph pipeline (in characters)
TEXT_LENGTH_THRESHOLD = 120

# Load models with device info logging
__tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
logger.info(f"Fine tokenizer loaded on device: {__tok_fine.device}")

# 有时 粗分表现更好
# e.g. 麻烦找一下新氧相关的专家：
# 1）新氧公司的各种专家，包括高管、负责内容的、负责BD的等；
# 2）大众点评、天猫、百度负责医美广告的；
# 3）更美、美呗等竞对；
# 4）医美机构负责广告投放的
# 有时 粗分表现较差
# 粗分能分对 更美、美呗
# e.g.
# 巴比食品、三津汤包、庆丰包子、武汉好礼客、上海早阳、南京青露、杭州甘其食等中国早餐包子店行业从业公司
# 杭州甘其食 可能更应该算作两个 token.
__tok_coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)
logger.info(f"Coarse tokenizer loaded on device: {__tok_coarse.device}")

# HanLP支持输出每个单词在文本中的原始位置，以便用于搜索引擎等场景。
# 在词法分析中，非语素字符（空格、换行、制表符等）会被剔除，此时需要额外的位置信息才能定位每个单词
# 通过 config.output_spans = True
# 返回格式为三元组（单词，单词的起始下标，单词的终止下标），下标以字符级别计量。
# https://github.com/hankcs/HanLP/issues/1802#issuecomment-1399534301
# 但由于我们会把文本先拆成句子，此时返回位置是句子的位置，因此就作用不大了
__tok_fine.config.output_spans = True
__tok_coarse.config.output_spans = True

__ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
logger.info(f"NER model loaded on device: {__ner.device}")

__pos_ctb9 = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
logger.info(f"CTB POS tagger loaded on device: {__pos_ctb9.device}")

__pos_pku = hanlp.load(hanlp.pretrained.pos.PKU_POS_ELECTRA_SMALL)
logger.info(f"PKU POS tagger loaded on device: {__pos_pku.device}")


def __sum(sentences: List):
    return sum(sentences, [])


def __zip_sentence(*args) -> List[Term]:
    tok, pos9, posp = args
    res = []
    for t, p9, pp in zip(tok, pos9, posp):
        res.append(Term(
            token=t,
            pos_ctb=p9,
            pos_pku=pp,
            span=None
        ))
    return res


def __zip_sentence_with_span(*args) -> List[Term]:
    tok, pos9, posp = args
    res = []
    for t, p9, pp in zip(tok, pos9, posp):
        token, start, end = t
        res.append(Term(
            token=token,
            pos_ctb=p9,
            pos_pku=pp,
            span=(start, end)
        ))
    return res


def __zip_for_paragraph(*args) -> List[Term]:
    token, pos_ctb9, pos_pku = args
    res = []
    for tok, pos9, posp in zip(token, pos_ctb9, pos_pku):
        ret = __zip_sentence_with_span(tok, pos9, posp)
        res.extend(ret)
    return res


def __zip_for_sentence(*args) -> List[List[Term]]:
    token, pos_ctb9, pos_pku = args
    res = []
    for tok, pos9, posp in zip(token, pos_ctb9, pos_pku):
        ret = __zip_sentence_with_span(tok, pos9, posp)
        res.append(ret)
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


def __remove_span(token_with_span):
    res = []
    for items in token_with_span:
        res.append([token for token, start, end in items])
    return res

def __token_with_indices(token_fn):
    def __token_with_indices_fn(sent_with_index):
        sents = [item[0] for item in sent_with_index]
        res = token_fn(sents)
        for i, items in enumerate(res):
            index = sent_with_index[i][1]
            for item in items:
                item[1] = item[1] + index
                item[2] = item[2] + index
        return res
    return __token_with_indices_fn


# 注意，因为分句会丢失上下文信息，所以可以在一定程度上对分词结果有不好的影响
# e.g. 2）大众点评、天猫、百度负责医美广告的； 3）更美、美呗等竞对；
# 在粗分情况下 2)和3) 在不在一行，决定了"更美""美呗"能否被分对。。
# __fine_analysis_paragraph_pipeline = hanlp.pipeline() \
#     .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
#     .append(__tok_fine, input_key='sentences', output_key=TOKEN_WITH_SPAN) \
#     .append(__remove_span, input_key=TOKEN_WITH_SPAN, output_key=TOKEN) \
#     .append(__pos_ctb9, input_key=TOKEN, output_key=POS_CTB) \
#     .append(__pos_pku, input_key=TOKEN, output_key=POS_PKU) \
#     .append(__ner, input_key=TOKEN, output_key=NAMED_ENTITIES) \
#     .append(__sum, input_key=NAMED_ENTITIES, output_key=NAMED_ENTITIES) \
#     .append(__zip_for_paragraph, input_key=(TOKEN, POS_CTB, POS_PKU),
#             output_key=TERMS)
__fine_analysis_paragraph_pipeline = hanlp.pipeline() \
    .append(split_sentence_with_index, output_key='sentences_with_indices') \
    .append(__token_with_indices(__tok_fine), input_key='sentences_with_indices', output_key=TOKEN_WITH_SPAN) \
    .append(__remove_span, input_key=TOKEN_WITH_SPAN, output_key=TOKEN) \
    .append(__pos_ctb9, input_key=TOKEN, output_key=POS_CTB) \
    .append(__pos_pku, input_key=TOKEN, output_key=POS_PKU) \
    .append(__ner, input_key=TOKEN, output_key=NAMED_ENTITIES) \
    .append(__sum, input_key=NAMED_ENTITIES, output_key=NAMED_ENTITIES) \
    .append(__zip_for_paragraph, input_key=(TOKEN_WITH_SPAN, POS_CTB, POS_PKU),
            output_key=TERMS)

__fine_analysis_sentence_pipeline = hanlp.pipeline() \
    .append(__tok_fine, output_key=TOKEN_WITH_SPAN) \
    .append(__remove_span, input_key=TOKEN_WITH_SPAN, output_key=TOKEN) \
    .append(__pos_ctb9, input_key=TOKEN, output_key=POS_CTB) \
    .append(__pos_pku, input_key=TOKEN, output_key=POS_PKU) \
    .append(__ner, input_key=TOKEN, output_key=NAMED_ENTITIES) \
    .append(__zip_for_sentence, input_key=(TOKEN_WITH_SPAN, POS_CTB, POS_PKU),
            output_key=TERMS)

# __fine_analysis_pipeline_with_span = hanlp.pipeline() \
#     .append(hanlp.utils.rules.split_sentence, output_key='sentences') \
#     .append(__tok_fine, input_key='sentences', output_key=TOKEN_WITH_SPAN) \
#     .append(__remove_span, input_key=TOKEN_WITH_SPAN, output_key=TOKEN) \
#     .append(__pos_ctb9, input_key=TOKEN, output_key=POS_CTB) \
#     .append(__pos_pku, input_key=TOKEN, output_key=POS_PKU) \
#     .append(__ner, input_key=TOKEN, output_key=NAMED_ENTITIES) \
#     .append(__sum, input_key=NAMED_ENTITIES, output_key=NAMED_ENTITIES) \
#     .append(__zip_for_paragraph, input_key=(TOKEN_WITH_SPAN, POS_CTB, POS_PKU),
#             output_key=TERMS)

__coarse_analysis_paragraph_pipeline = hanlp.pipeline() \
    .append(split_sentence_with_index, output_key='sentences_with_indices') \
    .append(__token_with_indices(__tok_coarse), input_key='sentences_with_indices', output_key=TOKEN_WITH_SPAN) \
    .append(__remove_span, input_key=TOKEN_WITH_SPAN, output_key=TOKEN) \
    .append(__pos_ctb9, input_key=TOKEN, output_key=POS_CTB) \
    .append(__pos_pku, input_key=TOKEN, output_key=POS_PKU) \
    .append(__ner, input_key=TOKEN, output_key=NAMED_ENTITIES) \
    .append(__sum, input_key=NAMED_ENTITIES, output_key=NAMED_ENTITIES) \
    .append(__zip_for_paragraph, input_key=(TOKEN_WITH_SPAN, POS_CTB, POS_PKU),
            output_key=TERMS)

__coarse_analysis_sentence_pipeline = hanlp.pipeline() \
    .append(__tok_coarse, output_key=TOKEN_WITH_SPAN) \
    .append(__remove_span, input_key=TOKEN_WITH_SPAN, output_key=TOKEN) \
    .append(__pos_ctb9, input_key=TOKEN, output_key=POS_CTB) \
    .append(__pos_pku, input_key=TOKEN, output_key=POS_PKU) \
    .append(__ner, input_key=TOKEN, output_key=NAMED_ENTITIES) \
    .append(__zip_for_sentence, input_key=(TOKEN_WITH_SPAN, POS_CTB, POS_PKU),
            output_key=TERMS)


def _filter_named_entities(
        items: List[Tuple[str, str, int, int]]
) -> List[Tuple[str, str, int, int]]:
    """
    ('英伟达', 'ORGANIZATION', 9, 10)
    ('西欧', 'LOCATION', 27, 28)
    Returns:

    """
    return [item for item in items if item[1] == 'ORGANIZATION']


def _process_sentences(
        docs: dict,
        allow_pos_ctb: Optional[Set[str]] = None,
        allow_pos_pku: Optional[Set[str]] = None,
) -> List[AnalysisResponse]:
    res = []
    for terms, named_entities in zip(docs[TERMS], docs[NAMED_ENTITIES]):
        tmp1 = {
            TERMS: terms,
            NAMED_ENTITIES: named_entities
        }
        tmp2 = _process_paragraph(
            docs=tmp1,
            allow_pos_ctb=allow_pos_ctb,
            allow_pos_pku=allow_pos_pku
        )
        res.append(tmp2)
    return res


def _process_paragraph(
        docs: dict,
        allow_pos_ctb: Optional[Set[str]] = None,
        allow_pos_pku: Optional[Set[str]] = None,
) -> AnalysisResponse:
    """
    处理分析结果，提取terms和named entities
    """
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


def _should_use_paragraph_pipeline(text: str) -> bool:
    """
    Determine whether to use paragraph pipeline based on text length.
    Returns True if text length exceeds threshold
    """
    return len(text) > TEXT_LENGTH_THRESHOLD


def _analysis_batch(
        texts: List[str],
        paragraph_pipeline,
        sentence_pipeline,
        allow_pos_ctb: Optional[Set[str]] = None,
        allow_pos_pku: Optional[Set[str]] = None,
) -> List[AnalysisResponse]:
    """
    Generic batch processing function for text analysis.

    Args:
        texts: List of input texts to analyze
        paragraph_pipeline: Pipeline to use for long texts
        sentence_pipeline: Pipeline to use for short texts
        allow_pos_ctb: Optional set of allowed CTB POS tags to filter
        allow_pos_pku: Optional set of allowed PKU POS tags to filter

    Returns:
        List of AnalysisResponse objects in the same order as input texts
    """
    # Split texts based on length threshold
    long_texts = [text for text in texts if len(text.strip()) and _should_use_paragraph_pipeline(text)]
    short_texts = [text for text in texts if len(text.strip()) and not _should_use_paragraph_pipeline(text)]

    analysis_results = {}

    # Process longer texts individually using paragraph pipeline
    for text in long_texts:
        pipeline_output = paragraph_pipeline(text)
        analysis_result = _process_paragraph(
            pipeline_output,
            allow_pos_ctb=allow_pos_ctb,
            allow_pos_pku=allow_pos_pku
        )
        analysis_results[text] = analysis_result

    # Process shorter texts in batch using sentence pipeline
    if short_texts:
        pipeline_output = sentence_pipeline(short_texts)
        batch_results = _process_sentences(
            pipeline_output,
            allow_pos_ctb=allow_pos_ctb,
            allow_pos_pku=allow_pos_pku
        )
        # Map results back to their original texts
        for text, result in zip(short_texts, batch_results):
            analysis_results[text] = result

    empty_result = AnalysisResponse(
        terms=[],
        named_entities=[]
    )

    # Preserve original text order in output
    return [analysis_results.get(text, empty_result) for text in texts]


def fine_analysis_batch(
        texts: List[str],
        allow_pos_ctb: Optional[Set[str]] = None,
        allow_pos_pku: Optional[Set[str]] = None,
) -> List[AnalysisResponse]:
    return _analysis_batch(
        texts,
        __fine_analysis_paragraph_pipeline,
        __fine_analysis_sentence_pipeline,
        allow_pos_ctb,
        allow_pos_pku
    )


def coarse_analysis_batch(
        texts: List[str],
        allow_pos_ctb: Optional[Set[str]] = None,
        allow_pos_pku: Optional[Set[str]] = None,
) -> List[AnalysisResponse]:
    return _analysis_batch(
        texts,
        __coarse_analysis_paragraph_pipeline,
        __coarse_analysis_sentence_pipeline,
        allow_pos_ctb,
        allow_pos_pku
    )


def fine_coarse_analysis_batch(
        texts: List[str],
        allow_pos_ctb: Optional[Set[str]] = None,
        allow_pos_pku: Optional[Set[str]] = None,
) -> List[FineCoarseAnalysisResponse]:
    """
    Batch process multiple texts using both fine and coarse-grained tokenization
    """
    fine_results = fine_analysis_batch(texts, allow_pos_ctb, allow_pos_pku)
    coarse_results = coarse_analysis_batch(texts, allow_pos_ctb, allow_pos_pku)

    return [
        FineCoarseAnalysisResponse(fine=fine, coarse=coarse)
        for fine, coarse in zip(fine_results, coarse_results)
    ]


def fine_analysis(
        text: str,
        allow_pos_ctb: Optional[Set[str]] = None,
        allow_pos_pku: Optional[Set[str]] = None,
) -> AnalysisResponse:
    results = fine_analysis_batch([text], allow_pos_ctb, allow_pos_pku)
    return results[0]


def coarse_analysis(
        text: str,
        allow_pos_ctb: Optional[Set[str]] = None,
        allow_pos_pku: Optional[Set[str]] = None,
) -> AnalysisResponse:
    results = coarse_analysis_batch([text], allow_pos_ctb, allow_pos_pku)
    return results[0]


def fine_coarse_analysis(
        text: str,
        allow_pos_ctb: Optional[Set[str]] = None,
        allow_pos_pku: Optional[Set[str]] = None,
) -> FineCoarseAnalysisResponse:
    results = fine_coarse_analysis_batch([text], allow_pos_ctb, allow_pos_pku)
    return results[0]
