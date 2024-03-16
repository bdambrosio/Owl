import sys, os
import json
import requests
import subprocess
#add local dir to search path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import socket
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from typing import Any, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM

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
models = [d for d in subdirs if ('exl2' in d or 'gptq' in d.lower() or 'phi-2' in d or 'xDAN' in d or 'miqu' in d or 'gguf' in d)]
#print(models)

templates = {"bagel-dpo-34b-v0.2-6.5bpw-h8-exl2": "llama-2",
             "miqu-1-70b-sf-6.0bpw-h6-exl2":"llama-2",
             "Mistral-7b-instruct": "llama-2",
             "mixtral-8bpw-exl2": "llama-2",
             "MixtralOrochi8x7B-8.0bpw-h8-exl2": "alpaca",
             "mixtral-gguf": 'llama-2',
             "Mixtral-8x7B-instruct-8.0bpw-exl2": "llama-2",
             "Mixtral-8x7B-Instruct-v0.1-7.0bpw-h6-exl2": "llama-2",
             "Nous-Hermes-2-Mixtral-8x7B-DPO-6.0bpw-h6-loneStriker-exl2": "chatml",
             "Nous-Hermes-2-Mixtral-8x7B-SFT-8bpw-h8-qeternity-exl2":"chatml",
             "phi-2":"phi-2",
             "openchat-3.5-8bpw-h8-exl2":"openchat",
             "OpenHermes-Mixtral-8x7B-6.0bpw-h6-exl2":"llama-2",
             "orca-2-13b-16bit":"chatml",
             "Qwen1.5-72B-5.0bpw-h6-exl2-liberated":"chatml",
             "Qwen1.5-72B-6.0bpw-h6-exl2-liberated":"chatml",
             "orca-2-13b-16bit":"chatml",
             "Senku-70B-Full-6.0bpw-h6-exl2": "chatml",
             "Smaug-Mixtral-8.0bpw-exl2": "llama-2",
             "Smaug-Mixtral-6.5bpw-exl2": "llama-2",
             "tulu-2-dpo-70b-4.65bpw-h6-exl2": "zephyr"
}

model_number = -1
template = ''
while model_number < 0 or model_number > len(models) -1:
    print(f'Available models:')
    for i in range(len(models)):
        try:
            with open(models_dir+models[i]+'/config.json', 'r') as j:
                json_config = json.load(j)
                context_size = json_config["max_position_embeddings"]
        except Exception as e:
            print(f'failure to load json.config {str(e)}\n setting context to 4096')
            context_size = 4096
        if models[i] in templates:
            template = templates[models[i]]
            print(f'{i}. {models[i]}, context: {context_size}, template: {template}')
        else:
            print(f'{i}. {models[i]}, context: {context_size}')
    
    number = input('input model # to load: ')
    try:
        model_number = int(number)
    except:
        print(f'Enter a number between 0 and {len(models)-1}')

model_name=models[model_number]
model_prompt_template = ''
if model_name in templates:
    model_prompt_template = templates[model_name]
print(f"Loading model: {model_name} prompt_template {model_prompt_template}")
json_config = None
context_size = 16384
max_new_tokens = 250

# get context size from model config
try:
    with open(models_dir+models[model_number]+'/config.json', 'r') as j:
        json_config = json.load(j)
        context_size = json_config["max_position_embeddings"]
        print(f'loaded json.config, found context {context_size}')
except Exception as e:
    print(f'failure to load config.json {str(e)}\n setting context to 4096')

if model_name.startswith('miqu-1-70b'):
    context_size=10000
    
context_size = min(16384, context_size)
print(f'loaded json.config, setting context to {context_size}')

if model_name == 'phi-2':
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
    model.to('cuda')
    
elif model_name.startswith('mixtral-gguf'):
    # launch the llama.cpp server on localhost 8080
    llama_cpp = subprocess.Popen(['/home/bruce/Downloads/llama.cpp/build/bin/server',
                                  '-m', '/home/bruce/Downloads/models/mixtral-gguf/mixtral-8x7b-instruct-v0.1.Q8_0.gguf',
                                  '-c', '16384',
                                  '-ngl', '90',
                                  '-b',  '256'])
    print(f'llama.cpp server started')

else:
    #if 'dolphin' in models[model_number]:
    #    cache = ExLlamaV2Cache(model, max_seq_len=8192)
    config = ExLlamaV2Config()
    config.scale_alpha_value=1.0
    config.model_dir = models_dir+model_name
    #config.max_seq_len=context_size
    #config.max_input_len = min (4096, context_size)
    config.prepare()

    model = ExLlamaV2(config)

    max_seq_len = 4096
    if 'bagel' in model_name.lower():
        print(f' bagel load')
        model.load([20, 23, 23])
    elif 'tulu' in model_name:
        model.load([20, 23, 23])
    elif 'miqu-1-70b' in model_name:
        print('miqu-1-70b load')
        model.load([20, 20, 20])
    elif 'ixtral' in model_name:
        print(f' mixtral load')
        model.load([18, 20, 23])  # leave room on gpu 0 for other stuff, eg embed
    elif 'Qwen1.5' in model_name:
        print(f' QWen1.5 load')
        model.load([15, 16, 23])  #
        context_size=6144
        config.max_seq_len=context_size
    else:
        model.load([22, 22, 22])
    print('model load done..')
    tokenizer = ExLlamaV2Tokenizer(config)

if model_name == 'phi-2':
    pass
else:
    cache = ExLlamaV2Cache(model)
    # Initialize generator
    generator = ExLlamaV2StreamingGenerator(model, cache, tokenizer)

    # Settings
    settings = ExLlamaV2Sampler.Settings()
    settings.temperature = 0.1
    settings.top_k = 50
    settings.top_p = 0.8
    settings.token_repetition_penalty = 1.15
    settings.disallow_tokens(tokenizer, [tokenizer.eos_token_id])

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

stop_gen=False    
async def phi_pseudo_stream(query: Dict[Any, Any], max_new_tokens, temp = .1, stop_on_json=False):
    global stop_gen
    #print(f'phi enter {type(query)}')

    if stop_gen:
        return
    generated_tokens = 0
    # this is a very sloppy heuristic for complete json form, but since chunks are short, maybe ok?
    # or maybe they aren't so short? Think they are, at least for first json form...
    model_inputs = tokenizer(query, return_tensors="pt")
    input_ids = model_inputs.to('cuda')
    
    generated_outputs = model.generate(**input_ids,
                                    do_sample=True,
                                    max_new_tokens=max_new_tokens,
                                    temperature=temp,
                                    num_return_sequences=1)
    
    print(f'out[0] {generated_outputs[0]}')
    yield tokenizer.decode(generated_outputs[0], skip_special_tokens=True)

async def miqu_pseudo_stream(query: Dict[Any, Any], max_new_tokens, temp = .1, stop_on_json=False):
    global stop_gen
    #print(f'miqu enter {type(query)} keys {query.keys()}')

    if stop_gen:
        return
    del query['max_tokens']
    del query['temp']
    del query['top_p']
    query['n_predict']=int(max_new_tokens)
    #print(f'\n\nmiqu query {query}\n')
    response = requests.post('http://127.0.0.1:8080/completion',
                             headers ={"Content-Type": "application/json"},
                             data=json.dumps(query))
    print(f'\nlama.cpp server response {response.json()["content"]}\n')
    yield response.json()['content']

async def llamacpp_pseudo_stream(query: Dict[Any, Any], max_new_tokens, temp = .1, stop_on_json=False):
    global stop_gen
    print(f'miqu enter {type(query)} keys {query.keys()}')

    if stop_gen:
        return
    del query['max_tokens']
    del query['temp']
    del query['top_p']
    query['n_predict']=int(max_new_tokens)
    #print(f'\n\nsmaug query {query}\n')
    response = requests.post('http://127.0.0.1:8080/completion',
                             headers ={"Content-Type": "application/json"},
                             data=json.dumps(query))
    #print(f'\nlama.cpp server response {response.json()["content"]}\n')
    yield response.json()['content']

    
host = socket.gethostname()
host = ''
port = 5004  # use port above 1024

app = FastAPI()
print(f"starting server")
@app.post("/template")
async def template(request: Request):
    
    global model_prompt_template, context_size
    print(f'template: {model_prompt_template}, context: {context_size}')
    return {"template":model_prompt_template, "context_size":context_size}
    
@app.post("/")
async def get_stream(request: Request):
    global stop_gen
    global generator, settings, tokenizer
    query = await request.json()
    print(f'request: {query}')
    message_j = query

    if 'template_query' in message_j.keys():
        return Response(template)
    temp = 0.1
    if 'temp' in message_j.keys():
        temp = message_j['temp']
    
    top_p = 1.0
    if 'top_p' in message_j.keys():
        top_p = message_j['top_p']

    max_tokens = 100
    if 'max_tokens' in message_j.keys():
        max_tokens = message_j['max_tokens']
    stop_conditions = ['###','<|endoftext|>', "Reference(s):"]
    if 'eos' in message_j.keys():
        print(f'\n received eos {message_j["eos"]}')
        stop_conditions = message_j['eos']

    stop_on_json = False
    if 'json' in stop_conditions:
        stop_on_json=True
        stop_conditions = stop_conditions.remove('json') # s/b no exceptions, since we know 'json' is in the list
    if 'stop_on_json' in message_j.keys() and message_j['stop_on_json']==True:
        stop_on_json=True

    prompt = message_j['prompt']
    if model_name == 'phi-2':
        stop_gen=False
        print(f'phi-2! {query}')
        return StreamingResponse(phi_pseudo_stream(prompt, max_new_tokens=max_tokens, temp=temp, stop_on_json=stop_on_json))
    elif model_name.startswith('mixtral-gguf'):
        stop_gen=False
        return StreamingResponse(llamacpp_pseudo_stream(query, max_new_tokens=max_tokens, temp=temp, stop_on_json=stop_on_json))
    else:
        settings.temperature = temp
        settings.top_p = top_p
        input_ids = tokenizer.encode(prompt)
        print(f'input_ids {input_ids.shape}')
        generator.set_stop_conditions(stop_conditions)
        generator.begin_stream(input_ids, settings)
        return StreamingResponse(stream_data(query, max_new_tokens = max_tokens, stop_on_json=stop_on_json))

