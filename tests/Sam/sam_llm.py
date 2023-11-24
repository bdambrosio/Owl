import sys, os
#add local dir to search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Any, Dict

from exllamav2 import (
    ExLlamaV2,
    ExLlamaV2Config,
    ExLlamaV2Cache,
    ExLlamaV2Tokenizer,
)

from exllamav2.generator import (
    ExLlamaV2StreamingGenerator,
    ExLlamaV2Sampler
)

import time

# Initialize model and cache
models_dir = "/home/bruce/Downloads/models/"

subdirs = [d for d in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, d))]
models = [d for d in subdirs if ('exl2' in d or 'gptq' in d.lower() or 'zephyr' in d.lower() or 'dolphin' in d.lower())]
models.append('orca-2-13b-16bit')

templates = {"CodeLlama-34B-instruct-exl2":"",
             "Mistral-7B-OpenOrca-exl2":"",
             "platypus2-70b-instruct-gptq":"alpaca",
             "platypus2-70b-instruct-exl2":"alpaca",
             "zephyr-7b-beta":"zephyr",
             "openchat-3.5-8bpw-h8-exl2":"openchat",
             "Spicyboros-70b-22-GPTQ":"llama-2(?)",
             "dolphin-34b-exl2":"llama-2(?)",
             "ShiningValiant-4bpw-h6-exl2":"llama-2(?)",
             "Airoboros-L2-70b-312-GPTQ":"llama-2",
             "airoboros-c34b-3.1.2-8.0bpq-h6-exl2":"llama-2",
             "mistral-airoboros-7b-GPTQ":"?",
             "orca-2-13b-16bit":"chatml"
             }

model_number = -1
while model_number < 0 or model_number > len(models) -1:
    print(f'Available models:')
    for i in range(len(models)):
        template = ''
        if models[i] in templates:
            template = templates[models[i]]
        print(f'{i}. {models[i]} template: {template}')
    number = input('input model # to load: ')
    try:
        model_number = int(number)
    except:
        print(f'Enter a number between 0 and {len(models)-1}')


#if 'dolphin' in models[model_number]:
#    cache = ExLlamaV2Cache(model, max_seq_len=8192)
config = ExLlamaV2Config()
config.scale_alpha_value=1.5
config.model_dir = models_dir+models[model_number]
config.prepare()

model = ExLlamaV2(config)
print("Loading model: " + models[model_number])
model.load([19, 24])

tokenizer = ExLlamaV2Tokenizer(config)

cache = ExLlamaV2Cache(model, max_seq_len=6144)
#cache = ExLlamaV2Cache(model, max_seq_len=4096)

# Initialize generator

generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

# Settings

settings = ExLlamaV2Sampler.Settings()
settings.temperature = 0.1
settings.top_k = 50
settings.top_p = 0.8
settings.token_repetition_penalty = 1.15
settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])
max_new_tokens = 250

# Make sure CUDA is initialized so we can measure performance
generator.warmup()

async def stream_data(query: Dict[Any, Any], max_new_tokens, stop_on_json=False):
    generated_tokens = 0
    # this is a very sloppy heuristic for complete json form, but since chunks are short, maybe ok?
    # or maybe they aren't so short? Think they are, at least for first json form...
    open_braces = 0
    open_brace_seen = False
    complete_json_seen = False
    text = ''
    while True:
        chunk, eos, _ = generator.stream()
        generated_tokens += 1
        if stop_on_json:
            open_braces += chunk.count('{')
            if open_braces > 0:
                open_brace_seen = True
                close_braces = chunk.count('}')
                open_braces -= close_braces
            if open_brace_seen and open_braces == 0:
                complete_json_seen = True
        print (chunk, end = "")
        text += chunk
        yield chunk
                    
        # note test for '[INST]' is a hack because we don't send template-specific stop strings yes
        if (eos or generated_tokens == max_new_tokens or
            (stop_on_json and complete_json_seen) or
            ('[INST]' in text or '</s>' in text or '<|endoftext|>' in text)):
            print('\n')
            break

    
host = socket.gethostname()
host = ''
port = 5004  # use port above 1024

app = FastAPI()
print(f"starting server")
    
@app.post("/")
async def get_stream(request: Request):
    global generator, settings, tokenizer
    query = await request.json()
    print(f'request: {query}')
    message_j = query

    temp = 0.1
    if 'temp' in message_j.keys():
        temp = message_j['temp']
    settings.temperature = temp

    top_p = 1.0
    if 'top_p' in message_j.keys():
        top_p = message_j['top_p']
    settings.top_p = top_p

    max_tokens = 100
    if 'max_tokens' in message_j.keys():
        max_tokens = message_j['max_tokens']
    stop_conditions = ['###','<|endoftext|>', "Reference(s):"]
    if 'eos' in message_j.keys():
        stop_conditions = message_j['eos']

    stop_on_json = False
    if 'json' in stop_conditions:
        stop_on_json=True
        stop_conditions = stop_conditions.remove('json') # s/b no exceptions, since we know 'json' is in the list
    if 'stop_on_json' in message_j.keys() and message_j['stop_on_json']==True:
        stop_on_json=True

    prompt = message_j['prompt']
    input_ids = tokenizer.encode(prompt)
    print(f'input_ids {input_ids.shape}')
    generator.set_stop_conditions(stop_conditions)
    generator.begin_stream(input_ids, settings)
    
    return StreamingResponse(stream_data(query, max_new_tokens = max_tokens, stop_on_json=stop_on_json))

