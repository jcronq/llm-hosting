import time
import json

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
    ChatCompletionResponseChoice,
    CompletionRequest,
    UsageInfo,
    CompletionResponseChoice
)
from local_llm.llm import create_model_4bit, generate
from local_llm.utils import random_uuid
from local_llm.profiling import profile_llm_generation

app = FastAPI()

TIMEOUT_KEEP_ALIVE = 5 # seconds

model_name = "mistral-7b-instruct"

model_pipeline = create_model_4bit(f"models/{model_name}")
model_prompt = Prompt(model_name)

chat_pipeline = model_pipeline
chat_prompt = ChatPrompt(model_name)


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

    prompt = chat_prompt(request.messages)
    response = generate(chat_pipeline, prompt)
    result_message = ChatMessage("assistant", response)
    choices = [ChatCompletionResponseChoice(0, result_message, "stop")]#: Optional[Literal["stop", "length"]] = None

    num_prompt_tokens = len(chat_pipeline.tokenizer.encode(prompt))
    num_generated_tokens = len(chat_pipeline.tokenizer.encode(response)) - num_prompt_tokens
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )
    return response


@app.post("/v1/completions")
async def create_completion(request: CompletionRequest, raw_request: Request):
    request_id = f"cmpl-{random_uuid()}"
    created_time = int(time.monotonic())

    prompt = model_prompt(request.messages)
    response = generate(model_pipeline, prompt)
    choices = [CompletionResponseChoice(0, response)]

    num_prompt_tokens = len(chat_pipeline.tokenizer.encode(prompt))
    num_generated_tokens = len(chat_pipeline.tokenizer.encode(response)) - num_prompt_tokens
    usage = UsageInfo(
        prompt_tokens=num_prompt_tokens,
        completion_tokens=num_generated_tokens,
        total_tokens=num_prompt_tokens + num_generated_tokens,
    )
    response = ChatCompletionResponse(
        id=request_id,
        created=created_time,
        model=model_name,
        choices=choices,
        usage=usage,
    )

@app.get("/profile/resources")
def profile_resources():
    input_length_tests = [100, 3000]
    output_length = 10
    traces = []
    for input_length in input_length_tests:
        profile_llm_generation(model_pipeline, input_length, output_length)
        with open(f"trace_{input_length}_input.json") as _f:
            traces.append(json.load(_f))
    return traces


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
