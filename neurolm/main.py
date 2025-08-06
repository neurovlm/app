import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer, TextIteratorStreamer
import threading
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
import uvicorn

# global
checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = "cpu"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# app
app = FastAPI()

def generate_tokens(model, tokenizer, inputs, **kwargs):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    thread = threading.Thread(target=model.generate, args=(inputs,), kwargs={**kwargs, 'streamer': streamer})
    thread.start()
    for token in streamer:
        yield token
    thread.join()

def stream_words(text):

    context = "You are a neuroscientist expert."
    context += "Keep explanations concise and focus on cognition and neuroscience."

    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": text}
    ]

    # Apply chat template to get formatted text
    input_text = tokenizer.apply_chat_template(messages, tokenize=False)
    inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

    kwargs = dict(
        max_length=300,
        top_p=0.9,
        do_sample=True,
        temperature=0.2,
        pad_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.2
    )

    for idx, i in enumerate(generate_tokens(model, tokenizer, inputs, **kwargs)):
        if idx > 3:
            yield i

@app.post("/llm/stream")
async def stream(request: Request):
    data = await request.json()
    return StreamingResponse(stream_words(data["text"]), media_type="text/event-stream")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)