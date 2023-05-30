from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uvicorn
import gpt4all


class Message(BaseModel):
    role: str
    content: str


class ResponseModel(BaseModel):
    message: Message


app = FastAPI()
gptj = gpt4all.GPT4All("ggml-gpt4all-l13b-snoozy.bin")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/chat")
async def chat_endpoint(messages: List[Message]):
    message_dicts = [msg.dict() for msg in messages]
    ret = gptj.chat_completion(message_dicts)
    return {"message": ret["choices"][0]["message"]}


@app.options("/chat")
async def chat_options_endpoint(request: Request):
    response = Response()
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
