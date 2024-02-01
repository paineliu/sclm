from dataclasses import dataclass
from typing import Union
import uvicorn
from fastapi import FastAPI, Depends, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.exceptions import HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
from chatbot import ChatBot, InferConfig

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    global chat_bot
    print("load chat bot")
    chat_bot = ChatBot(infer_config=InferConfig())
    print("success")
    yield
    # Clean up the ML models and release the resources
    pass

app = FastAPI(lifespan=lifespan)

class ChatInput(BaseModel):
    input_txt: str

@app.post("/api/chat")
async def chat(post_data: ChatInput) -> dict:
    """
    post 输入: {'input_txt': '输入的文本'}
    response: {'response': 'chatbot文本'}
    """
    input_txt = post_data.input_txt
    outs = ''

    if len(input_txt) > 0:
        outs = chat_bot.chat(input_txt)

    if len(outs) == 0:
            outs = "我是一个参数很少的AI模型🥺，知识库较少，无法直接回答您的问题，换个问题试试吧👋"

    return {'response': outs}

if __name__ == '__main__':
    
    # 服务方式启动：
    # 命令行输入：uvicorn api_demo:app --host 0.0.0.0 --port 8094 --workers 8
    # --reload：在代码更改后重新启动服务器。 只有在开发时才使用这个参数，此时多进程设置会无效
    uvicorn.run(
        'api_demo:app',
        host='0.0.0.0',
        port=8094,
        reload=True,
        workers=8,
        log_level='info')
