import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, pipeline
from langchain import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from decorators import logger, timeit

@timeit
def init():
    global model 
    global qa_chain
    
    logger.info("initializing flan-t5 model...")

    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xl")
    model = T5ForConditionalGeneration.from_pretrained(
        "google/flan-t5-xl", 
        device_map="auto", 
        torch_dtype=torch.float16
    )
    
    model = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer
    )

    logger.info("initializing flan-t5 model is done!")

    # init langchain for QA using t5 model pipeline 
    logger.info("init langchain for QA with prompt template")
    
    prompt_template = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

    {context}

    Question: {question}
    Answer:"""

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa_chain  = LLMChain(prompt=prompt, llm=HuggingFacePipeline(pipeline=model))
    logger.info("init QA prompt done!")


@timeit
def inference(model_inputs: dict) -> dict:
    global model
    global qa_chain

    context = model_inputs.get("context", None)
    question = model_inputs.get("question", None)

    if not all([context, question]):
        return  {
            "msg": "question and context are required!"
        }
    
    result = qa_chain.run(context=context, question=question)
    return result

    
