import time
import json
from pathlib import Path
from http import HTTPStatus

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response

from local_llm.prompting import (
    Prompt,
    ChatPrompt,
    ChatMessage,
)

from local_llm.rest_api.openai.data_models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    CompletionResponse,
    ChatCompletionResponseChoice,
    CompletionRequest,
    UsageInfo,
    CompletionResponseChoice,
    ErrorResponse,
)
from local_llm.llm import create_model_4bit, generate
from local_llm.utils import random_uuid
from local_llm.profiling import profile_llm_generation

app = FastAPI()

TIMEOUT_KEEP_ALIVE = 5 # seconds

MODEL_NAME = None
MODEL_PIPELINE = None
MODEL_PROMPT = None
CHAT_PIPELINE = None
CHAT_PROMPT = None

def init_model(model_dir):
    global MODEL_NAME
    global MODEL_PIPELINE
    global MODEL_PROMPT
    global CHAT_PIPELINE
    global CHAT_PROMPT

    MODEL_NAME = Path(model_dir).stem
    MODEL_PIPELINE = create_model_4bit(model_dir)
    MODEL_PROMPT = Prompt(MODEL_NAME)
    CHAT_PIPELINE = MODEL_PIPELINE # Use the same pipeline for chat and non-chat models.
    CHAT_PROMPT = MODEL_PROMPT # Use the same prompt for the chat model.
    # CHAT_PROMPT = ChatPrompt(MODEL_NAME) # Use a different prompt for the chat model.

def create_error_response(status_code: HTTPStatus,
                          message: str) -> JSONResponse:
    return JSONResponse(ErrorResponse(message=message,
                                      type="invalid_request_error").dict(),
                        status_code=status_code.value)


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/v1/models")
async def show_available_models():
    """Show available models. Right now we only have one model."""
    return ["mistral-7b-instruct"]


@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest, raw_request: Request):
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())

    prompt = CHAT_PROMPT(request.messages)
    response = generate(CHAT_PIPELINE, prompt)
 
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    prompt = MODEL_PROMPT(prompt)
    response = generate(MODEL_PIPELINE, prompt)
    #TODO: put in real finish_reason
    result_message = ChatMessage(role="assistant", content=response)
    choices = [ChatCompletionResponseChoice(index=0, message=result_message, finish_reason="stop")]#: Optional[Literal["stop", "length"]] = None

    num_prompt_tokens = len(CHAT_PIPELINE.tokenizer.encode(prompt))
    num_generated_tokens = len(CHAT_PIPELINE.tokenizer.encode(response)) - num_prompt_tokens
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=MODEL_NAME,
        choices=choices,
        usage=usage,
    )
    return response


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):

    if isinstance(request.prompt, list):
        if len(request.prompt) == 0:
            return create_error_response(HTTPStatus.BAD_REQUEST,
                                         "please provide at least one prompt")
        first_element = request.prompt[0]
        if isinstance(first_element, (str, list)):
            # TODO: handles multiple prompt case in list[list[int]]
            if len(request.prompt) > 1:
                return create_error_response(
                    HTTPStatus.BAD_REQUEST,
                    "multiple prompts in a batch is not currently supported")
            prompt = request.prompt[0]
    else:
        prompt = request.prompt


    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())
    prompt = MODEL_PROMPT(prompt)
    response = generate(MODEL_PIPELINE, prompt)
    #TODO: put in real finish_reason
    choices = [CompletionResponseChoice(index=0, text=response, finish_reason="stop")]

    num_prompt_tokens = len(CHAT_PIPELINE.tokenizer.encode(prompt))
    num_generated_tokens = len(CHAT_PIPELINE.tokenizer.encode(response)) - num_prompt_tokens
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = CompletionResponse(
        id=request_id,
        created=created_time,
        model=MODEL_NAME,
        choices=choices,
        usage=usage,
    )

    return response

@app.get("/profile/resources")
def profile_resources():
    input_length_tests = [100, 3000]
    output_length = 10
    traces = []
    for input_length in input_length_tests:
        profile_llm_generation(MODEL_PIPELINE, input_length, output_length)
        with open(f"trace_{input_length}_input.json") as _f:
            traces.append(json.load(_f))
    return traces


if __name__ == "__main__":
    init_model()
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
