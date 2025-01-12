from typing import List, Union
from xml.dom.minidom import Document

import hanlp

WORD = "word"
POS_CTB9 = "pos_ctb9"
POS_PKU = "pos_pku"
NAMED_ENTITIES = "named_entities"
TERMS = "terms"

__tok_fine = hanlp.load(hanlp.pretrained.tok.FINE_ELECTRA_SMALL_ZH)
# __tok_coarse = hanlp.load(hanlp.pretrained.tok.COARSE_ELECTRA_SMALL_ZH)

__ner = hanlp.load(hanlp.pretrained.ner.MSRA_NER_ELECTRA_SMALL_ZH)
__pos_ctb9 = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
__pos_pku = hanlp.load(hanlp.pretrained.pos.PKU_POS_ELECTRA_SMALL)

# .append(__tok_coarse, output_key="tok/coarse") \
_text_pipeline = hanlp.pipeline() \
    .append(hanlp.utils.rules.split_sentence) \
    .append(__tok_fine, output_key="tok/fine") \
    .append(__pos_ctb9, input_key="tok/fine", output_key="pos/ctb") \
    .append(__pos_pku, input_key="tok/fine", output_key='pos/pku') \
    .append(__ner, input_key="tok/fine", output_key="ner/msra")

_sentences_pipeline = hanlp.pipeline() \
    .append(__tok_fine, output_key="tok/fine") \
    .append(__pos_ctb9, input_key="tok/fine", output_key="pos/ctb") \
    .append(__pos_pku, input_key="tok/fine", output_key='pos/pku') \
    .append(__ner, input_key="tok/fine", output_key="ner/msra")


class HanLPUtil:
    def __init__(self):
        self.text_pipeline = _text_pipeline
        self.sentences_pipeline = _sentences_pipeline
        # self.hanlp = hanlp.load(
        #     hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_SMALL_ZH
        # )

    def parse(
            self,
            text: Union[str, List[str]] = None,
    ) -> Document:
        if isinstance(text, str):
            # For raw text input, use text pipeline that includes sentence splitting
            res = self.text_pipeline(text)
        elif isinstance(text, list):
            # For list of sentences, use sentences pipeline without sentence splitting
            res = self.sentences_pipeline(text)
        else:
            raise ValueError("Text must be either a string or list of strings")
        # res = self.hanlp(
        #     text,
        #     tokens=tokens,
        #     tasks=tasks,
        #     skip_tasks=skip_tasks,
        #     language=language
        # )
        return res

