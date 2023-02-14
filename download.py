import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline


def download_model():
    """download model during docker image  setup
    so it will be in cache before app.py and http_api kick-in"""

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-xl", 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer
    )


if __name__ == "__main__":
    download_model()
