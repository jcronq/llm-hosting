from abc import ABC, abstractmethod
from pydantic import BaseModel

from typing import List

class ChatMessage(BaseModel):
    role: str
    content: str


class PromptBase(ABC):
    @abstractmethod
    def __call__(self, **kwargs):
        pass

class Prompt(PromptBase):
    def __init__(self, model_name):
        self.model_name = model_name
    
    def __call__(self, text, **kwargs):
        match(self.model_name):
            case "mistral-7b-instruct":
                return f"<s>[INST]{text}[/INST]"
            case _:
                raise ValueError(f"Unknown model name {self.model_name}")

class ChatPrompt(PromptBase):
    def __init__(self, model_name):
        self.model_name = model_name
    
    def __call__(self, messages: List[ChatMessage], **kwargs):
        conversation = ""
        for message in messages:
            conversation += f"{message.role}: {message.content}\n"
        conversation += "Assistant: "
        match(self.model_name):
            case "mistral-7b-instruct":
                return f"<s>[INST]You are a ChatBot. You are in a Conversation with User. User messages are labeled USER:. Your responses are labeled ASSISTANT. Respond conversationally to the user's latest message.[/INST]\n{conversation}"
            case _:
                raise ValueError(f"Unknown model name {self.model_name}")