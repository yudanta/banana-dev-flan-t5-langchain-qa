import uvicorn
import subprocess
from fastapi import FastAPI
from pydantic import BaseModel

import app as user_model
from decorators import logger

# init model first
user_model.init()


class PromptInput(BaseModel):
    context: str = ""
    question: str = ""

http_api = FastAPI()

@http_api.get("/healthcheck")
async def healthcheck():
    gpu = False
    out = subprocess.run("nvidia-smi", shell=True)
    if out.returncode == 0:
        gpu = True

    return {"state": "healthy", "gpu": gpu}

@http_api.post("/")
async def inference(prompt: PromptInput):
    output = {}

    try:
        output = user_model.inference(prompt.dict())
    except Exception as e:
        logger.error(str(e))

    return output


if __name__ == "__main__":
    uvicorn.run(http_api, host="0.0.0.0", port=8000)

