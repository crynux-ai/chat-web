from typing import Literal, List

from typing_extensions import TypedDict

__all__ = ["Message", "GenerationConfig", "GPTTaskResponse"]


class Message(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class GenerationConfig(TypedDict, total=False):
    max_new_tokens: int

    temperature: float
    top_p: float
    top_k: float


class Usage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ResponseChoice(TypedDict):
    index: int
    message: Message
    finish_reason: Literal["stop", "length"]


class GPTTaskResponse(TypedDict):
    model: str
    choices: List[ResponseChoice]
    usage: Usage
