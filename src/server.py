import argparse
from typing import List, Union

import uvicorn
from fastapi import FastAPI, APIRouter
from pydantic import BaseModel

from .analysis.analysis import fine_analysis, coarse_analysis, fine_coarse_analysis
from .analysis.models import AnalysisReq, AnalysisResponse, \
    FineCoarseAnalysisResponse, BatchAnalysisReq, BatchAnalysisResponse, \
    BatchFineCoarseAnalysisResponse

app = FastAPI(title="HanLP Server")
router = APIRouter(prefix="/api/v1")


# nlp = HanLPUtil()


@router.get("/health")
def health_check():
    return {"status": "ok"}


class TextRequest(BaseModel):
    text: Union[str, List[str]]


# @app.post("/parse")
# def parse(request: TextRequest):
#     res = nlp.parse(request.text)
#     return res


@router.post("/analysis/fine")
def analyze_fine(request: AnalysisReq) -> AnalysisResponse:
    """
    使用细粒度分词进行分析
    """
    return fine_analysis(
        text=request.text,
        allow_pos_ctb=request.allow_pos_ctb,
        allow_pos_pku=request.allow_pos_pku,
    )


@router.post("/analysis/fine/batch")
def analyze_fine_batch(request: BatchAnalysisReq) -> BatchAnalysisResponse:
    """
    使用细粒度分词进行批量分析
    """
    results = [
        fine_analysis(
            text=text,
            allow_pos_ctb=request.allow_pos_ctb,
            allow_pos_pku=request.allow_pos_pku,
        )
        for text in request.texts
    ]
    return BatchAnalysisResponse(results=results)


@router.post("/analysis/coarse")
def analyze_coarse(request: AnalysisReq) -> AnalysisResponse:
    """
    使用粗粒度分词进行分析
    """
    return coarse_analysis(
        text=request.text,
        allow_pos_ctb=request.allow_pos_ctb,
        allow_pos_pku=request.allow_pos_pku,
    )


@router.post("/analysis/coarse/batch")
def analyze_coarse_batch(request: BatchAnalysisReq) -> BatchAnalysisResponse:
    """
    使用粗粒度分词进行批量分析
    """
    results = [
        coarse_analysis(
            text=text,
            allow_pos_ctb=request.allow_pos_ctb,
            allow_pos_pku=request.allow_pos_pku,
        )
        for text in request.texts
    ]
    return BatchAnalysisResponse(results=results)


@router.post("/analysis/fine-coarse")
def analyze_fine_coarse(request: AnalysisReq) -> FineCoarseAnalysisResponse:
    """
    同时进行细粒度和粗粒度分词分析
    """
    return fine_coarse_analysis(
        text=request.text,
        allow_pos_ctb=request.allow_pos_ctb,
        allow_pos_pku=request.allow_pos_pku,
    )


@router.post("/analysis/fine-coarse/batch")
def analyze_fine_coarse_batch(request: BatchAnalysisReq) -> BatchFineCoarseAnalysisResponse:
    """
    同时进行细粒度和粗粒度分词的批量分析
    """
    results = [
        fine_coarse_analysis(
            text=text,
            allow_pos_ctb=request.allow_pos_ctb,
            allow_pos_pku=request.allow_pos_pku,
        )
        for text in request.texts
    ]
    return BatchFineCoarseAnalysisResponse(results=results)


def parse_args():
    parser = argparse.ArgumentParser(description='HanLP Server')
    parser.add_argument('--port', type=int, default=5012,
                        help='Port to run the server on (default: 5012)')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='Host to bind the server to (default: 0.0.0.0)')
    return parser.parse_args()


# Include the router in the main app
app.include_router(router)


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
