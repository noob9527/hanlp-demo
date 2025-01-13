import argparse
from typing import List, Union

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from .hanlp_util import HanLPUtil
from .analysis.analysis import analysis
from .analysis.models import AnalysisReq, AnalysisResponse

app = FastAPI(title="HanLP Server")
nlp = HanLPUtil()


@app.get("/health")
async def health_check():
    return {"status": "ok"}


class TextRequest(BaseModel):
    text: Union[str, List[str]]


@app.post("/parse")
async def parse(request: TextRequest):
    res = nlp.parse(request.text)
    return res


@app.post("/analysis")
async def analyze_keywords(request: AnalysisReq) -> AnalysisResponse:
    return analysis(
        text=request.text,
        allow_pos_ctb=request.allow_pos_ctb,
        allow_pos_pku=request.allow_pos_pku,
    )


def parse_args():
    parser = argparse.ArgumentParser(description='HanLP Server')
    parser.add_argument('--port', type=int, default=5012,
                        help='Port to run the server on (default: 5012)')
    parser.add_argument('--host', type=str, default="0.0.0.0",
                        help='Host to bind the server to (default: 0.0.0.0)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(app, host=args.host, port=args.port)
