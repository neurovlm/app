import re
import pandas as pd
import torch
from torch import nn
from transformers import AutoTokenizer
from adapters import AutoAdapterModel
from llama_cpp import Llama
import threading
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn
from neurovlm.models import Specter

data_dir = "/home/rphammonds/neurovlm/static/models"

# Llama for SmolLM3-3B quantized
llm = Llama(
    model_path=f"{data_dir}/SmolLM3-Q4_K_M.gguf",
    n_ctx=8192,
    verbose=False
)

# Load
df = pd.read_parquet(f"{data_dir}/publications_less.parquet")[['name', 'description']]
specter = Specter()
aligner = torch.load(f"{data_dir}/aligner_specter_adhoc_query.pt", weights_only=False, map_location="cpu")
latent_text = torch.load(f"{data_dir}/latent_text_specter2_adhoc_query.pt", weights_only=True).to("cpu")
latent_text /= latent_text.norm(dim=1)[:, None] # unit norm

# App
app = FastAPI()

def generate_tokens(model, tokenizer, inputs, **kwargs):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(target=model.generate, args=(inputs,), kwargs={**kwargs, 'streamer': streamer})
    thread.start()
    for token in streamer:
        yield token
    thread.join()

def stream_words(query):

    # Encode query with specter than rank publications
    #   Collect related publications to pass to LM
    encoded_text = specter(query)[0].detach()
    encoded_text_norm = encoded_text / encoded_text.norm()
    cos_sim = latent_text @ encoded_text_norm
    inds = torch.argsort(cos_sim, descending=True)
    papers = "\n".join(
        [f"[{ind + 1}] " + df.iloc[int(i)]["name"] + "\n" + re.sub(r'\s+', ' ', df.iloc[int(i)]["description"].replace("\n", "")) + "\n"
        for ind, i in enumerate(inds[:5])]
    )

    # Context to give the LM
    context = "Summarize the high level ideas of the topic in."
    # # Additional instructions
    # context += "Select a subset of related publications. "
    # context += "Use in text citations, [1], [2], etc, at the end of related statments. "

    # # Enable thinking
    # context += "<think></think>"

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": f"Topic: {query}.\n\nSupplemental publications: " + "\n" + papers}
    ]

    stream = llm.create_chat_completion(messages=messages, stream=True, temperature=0.6)
    start_stream = False # start stream after thinking
    for chunk in stream:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta and start_stream:
            yield delta['content']
        elif 'content' in delta:
            if delta['content'] == "</think>":
                start_stream = True

@app.post("/llm/stream")
async def stream(request: Request):
    data = await request.json()
    return StreamingResponse(stream_words(data["text"]), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)