from fastapi import APIRouter, HTTPException
from agent_core.llm_client import explain_text

router = APIRouter()


@router.post("/explain")
def explain(payload: dict):
    input_text = payload.get("inputText")

    if not input_text:
        raise HTTPException(
            status_code=400,
            detail="inputText is required"
        )

    return explain_text(input_text)
