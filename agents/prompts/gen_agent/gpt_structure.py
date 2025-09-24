"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import os
import time
import json
from pathlib import Path
from openai import AzureOpenAI, OpenAI
import requests
from PIL import Image
from io import BytesIO
import base64

# from utils import *
from openai_cost_logger import DEFAULT_LOG_PATH
from generative_agents.persona.prompt_template.openai_logger_singleton import OpenAICostLogger_Singleton

# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, BertTokenizer, DistilBertModel

import torch

current_directory = os.getcwd()
# print("Current working directory:", current_directory)

config_path = Path("generative_agents/openai_config.json")
with open(config_path, "r") as f:
    openai_config = json.load(f)

# model_id = "neuralmagic/Meta-Llama-3.1-8B-Instruct-quantized.w8a16"
# number_gpus = 1
# max_model_len = 2048
# llama_3_1_8B_Instruct_quantized_w8a16_tokenizer = AutoTokenizer.from_pretrained(model_id)
# llama_3_1_8B_Instruct_quantized_w8a16_llm = LLM(model=model_id, tensor_parallel_size=number_gpus, max_model_len=max_model_len, gpu_memory_utilization=0.5)

# distilbert_base_tokenizer = BertTokenizer.from_pretrained('distilbert-base-uncased')
# distilbert_base_model = DistilBertModel.from_pretrained('distilbert-base-uncased')

EMBEDDING_MODEL = openai_config["embeddings"]
# EMBEDDING_MODEL = "distilbert"

def setup_client(type: str, config: dict):
    """Setup the OpenAI client.

    Args:
        type (str): the type of client. Either "azure" or "openai".
        config (dict): the configuration for the client.

    Raises:
        ValueError: if the client is invalid.

    Returns:
        The client object created, either AzureOpenAI or OpenAI.
    """
    if type == "azure":
        client = AzureOpenAI(
            azure_endpoint=config["endpoint"],
            api_key=config["key"],
            api_version=config["api-version"],
        )
    elif type == "openai":
        client = OpenAI(
            api_key=config["key"],
        )
    else:
        raise ValueError("Invalid client")
    return client

if openai_config["client"] == "azure":
    client = setup_client("azure", {
        "endpoint": openai_config["model-endpoint"],
        "key": openai_config["model-key"],
        "api-version": openai_config["model-api-version"],
    })
elif openai_config["client"] == "openai":
    client = setup_client("openai", { "key": openai_config["model-key"] })

if openai_config["embeddings-client"] == "azure":
    embeddings_client = setup_client("azure", {
        "endpoint": openai_config["embeddings-endpoint"],
        "key": openai_config["embeddings-key"],
        "api-version": openai_config["embeddings-api-version"],
    })
elif openai_config["embeddings-client"] == "openai":
    embeddings_client = setup_client("openai", { "key": openai_config["embeddings-key"] })
else:
    raise ValueError("Invalid embeddings client")

cost_logger = OpenAICostLogger_Singleton(
  experiment_name = openai_config["experiment-name"],
  log_folder = DEFAULT_LOG_PATH,
  cost_upperbound = openai_config["cost-upperbound"]
)


def temp_sleep(seconds=0.1):
    time.sleep(seconds)


def ChatGPT_single_request(prompt):
    temp_sleep()
    completion = client.chat.completions.create(
      model=openai_config["model"],
      messages=[{"role": "user", "content": prompt}]
    )
    cost_logger.update_cost(completion, input_cost=openai_config["model-costs"]["input"], output_cost=openai_config["model-costs"]["output"])
    return completion.choices[0].message.content


def ChatGPT_request(prompt):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT-3's response.
    """
    # temp_sleep()
    try:
        completion = client.chat.completions.create(
        model=openai_config["model"],
        messages=[{"role": "user", "content": prompt}]
        )
        cost_logger.update_cost(completion, input_cost=openai_config["model-costs"]["input"], output_cost=openai_config["model-costs"]["output"])
        return completion.choices[0].message.content

    except Exception as e:
        print(f"Error: {e}")
        return "ChatGPT ERROR"


def ChatGPT_safe_generate_response(prompt,
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
    # prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    prompt = '"""\n' + prompt + '\n"""\n'
    prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    prompt += "Example output json:\n"
    prompt += '{"output": "' + str(example_output) + '"}'

    if verbose:
        print ("CHAT GPT PROMPT")
        print (prompt)

    for i in range(repeat):

        try:
            curr_gpt_response = ChatGPT_request(prompt).strip()
            end_index = curr_gpt_response.rfind('}') + 1
            curr_gpt_response = curr_gpt_response[:end_index]
            curr_gpt_response = json.loads(curr_gpt_response)["output"]

            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)

            if verbose:
                print ("---- repeat count: \n", i, curr_gpt_response)
                print (curr_gpt_response)
                print ("~~~~")

        except:
            pass

    return False


def ChatGPT_safe_generate_response_OLD(prompt,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False):
    if verbose:
        print ("CHAT GPT PROMPT")
        print (prompt)

    for i in range(repeat):
        try:
            curr_gpt_response = ChatGPT_request(prompt).strip()
            if func_validate(curr_gpt_response, prompt=prompt):
                return func_clean_up(curr_gpt_response, prompt=prompt)
            if verbose:
                print (f"---- repeat count: {i}")
                print (curr_gpt_response)
                print ("~~~~")

        except:
            pass
    print ("FAIL SAFE TRIGGERED")
    return fail_safe_response


def GPT_request(prompt, gpt_parameter):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT's response.
    """
    def return_gpt_parameter_query_text(parameters):
        return '_'.join(map(str, parameters))
    temp_sleep()
    file_path = "cached_gpt_responses.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            responses = json.load(file)
    else:
        responses = {}
    gpt_parameter_query_text = return_gpt_parameter_query_text([gpt_parameter["engine"],
                                                                gpt_parameter["temperature"],
                                                                gpt_parameter["max_tokens"],
                                                                gpt_parameter["top_p"],
                                                                gpt_parameter["frequency_penalty"],
                                                                gpt_parameter["presence_penalty"],
                                                                gpt_parameter["stream"],
                                                                gpt_parameter["stop"]])
    # print(gpt_parameter_query_text)
    if gpt_parameter_query_text not in responses.keys():
        responses[gpt_parameter_query_text] = {}
    if prompt in responses[gpt_parameter_query_text].keys():
        return responses[gpt_parameter_query_text][prompt]
    else:
        # print("running gpt4 request...")
        try:
            messages = [{
              "role": "system", "content": prompt
            }]
            response = client.chat.completions.create(
                        model=gpt_parameter["engine"],
                        messages=messages,
                        temperature=gpt_parameter["temperature"],
                        max_tokens=gpt_parameter["max_tokens"],
                        top_p=gpt_parameter["top_p"],
                        frequency_penalty=gpt_parameter["frequency_penalty"],
                        presence_penalty=gpt_parameter["presence_penalty"],
                        stream=gpt_parameter["stream"],
                        stop=gpt_parameter["stop"],)
            cost_logger.update_cost(response=response, input_cost=openai_config["model-costs"]["input"], output_cost=openai_config["model-costs"]["output"])
            responses[gpt_parameter_query_text][prompt] = response.choices[0].message.content
            with open(file_path, 'w') as f:
                json.dump(responses, f, indent=4)
            return responses[gpt_parameter_query_text][prompt]
        except Exception as e:
            print(f"Error: {e}")
            return "TOKEN LIMIT EXCEEDED OR OTHER EXCEPTIONS"
        
def LLaMA_generate(prompt, llama_parameter, llm, tokenizer):
    def return_gpt_parameter_query_text(parameters):
        return '_'.join(map(str, parameters))
    temp_sleep()
    file_path = "cached_gpt_responses.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            responses = json.load(file)
    else:
        responses = {}
    llama_parameter_query_text = return_gpt_parameter_query_text([llama_parameter["engine"],
                                                                llama_parameter["temperature"],
                                                                llama_parameter["max_tokens"],
                                                                llama_parameter["top_p"],
                                                                llama_parameter["frequency_penalty"],
                                                                llama_parameter["presence_penalty"],
                                                                llama_parameter["stream"],
                                                                llama_parameter["stop"]])
    if llama_parameter_query_text not in responses.keys():
        responses[llama_parameter_query_text] = {}
    if prompt in responses[llama_parameter_query_text].keys():
        return responses[llama_parameter_query_text][prompt]
    else:
        try:
            messages = [{
              "role": "system", "content": prompt
            }]
            prompts = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            sampling_params = SamplingParams(temperature=llama_parameter["temperature"], 
                                             top_p=llama_parameter["top_p"], 
                                             max_tokens=llama_parameter["max_tokens"])
            outputs = llm.generate(prompts, sampling_params)
            responses[llama_parameter_query_text][prompt] = outputs[0].outputs[0].text
            with open(file_path, 'w') as f:
                json.dump(responses, f, indent=4)
            return responses[llama_parameter_query_text][prompt]
        except Exception as e:
            print(f"Error: {e}")
            return "TOKEN LIMIT EXCEEDED OR OTHER EXCEPTIONS"

def GPT_request_all_messages(messages, gpt_parameter):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT's response.
    """
    try:
        response = client.chat.completions.create(
                    model=gpt_parameter["engine"],
                    messages=messages,
                    temperature=gpt_parameter["temperature"],
                    max_tokens=gpt_parameter["max_tokens"],
                    top_p=gpt_parameter["top_p"],
                    frequency_penalty=gpt_parameter["frequency_penalty"],
                    presence_penalty=gpt_parameter["presence_penalty"],
                    stream=gpt_parameter["stream"],
                    stop=gpt_parameter["stop"],)
        cost_logger.update_cost(response=response, input_cost=openai_config["model-costs"]["input"], output_cost=openai_config["model-costs"]["output"])
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return "TOKEN LIMIT EXCEEDED OR OTHER EXCEPTIONS"

def GPT4o_request(image_path, sys_prompt="Describe what you see.", user_prompt=None):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT's response.
    """
    gpt_parameter = {}
    gpt_parameter["engine"] = "gpt-4o"
    gpt_parameter["temperature"] = 0
    gpt_parameter["max_tokens"] = 300
    
    def return_gpt_parameter_query_text(parameters):
        return '_'.join(map(str, parameters))
    
    def encode_image(image):
        if isinstance(image, str):
            with open(image, "rb") as img:
                return base64.b64encode(img.read()).decode('utf-8')
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    temp_sleep()
    file_path = "cached_gpt_responses.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            responses = json.load(file)
    else:
        responses = {}
    gpt_parameter_query_text = return_gpt_parameter_query_text([gpt_parameter["engine"],
                                                                gpt_parameter["temperature"],
                                                                gpt_parameter["max_tokens"]])
    # print(gpt_parameter_query_text)
    if gpt_parameter_query_text not in responses.keys():
        responses[gpt_parameter_query_text] = {}
    if f"{image_path}_{sys_prompt}_{user_prompt}" in responses[gpt_parameter_query_text].keys():
        return responses[gpt_parameter_query_text][f"{image_path}_{sys_prompt}_{user_prompt}"]
    else:
        # print("running gpt-4o request...")
        content = [
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                "detail": "low"
            }
            }
        ]
        if user_prompt:
            content.append({
                "type": "text",
                "text": user_prompt
            })
        try:
            messages = [
                {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": sys_prompt
                    }
                ]
                },
                {
                "role": "user",
                "content": [
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                        "detail": "low"
                    }
                    }
                ]
                }
            ]
            response = client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages,
                        temperature=gpt_parameter["temperature"],
                        max_tokens=gpt_parameter["max_tokens"])
            responses[gpt_parameter_query_text][f"{image_path}_{sys_prompt}_{user_prompt}"] = response.choices[0].message.content
            with open(file_path, 'w') as f:
                json.dump(responses, f, indent=4)
            return responses[gpt_parameter_query_text][f"{image_path}_{sys_prompt}_{user_prompt}"]
        except Exception as e:
            print(f"Error: {e}")
            return "TOKEN LIMIT EXCEEDED OR OTHER EXCEPTIONS"
        
def GPT4o_request_new(image_paths, sys_prompt, user_prompt, gpt_parameter):
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response.
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of
                     the parameter and the values indicating the parameter
                     values.
    RETURNS:
      a str of GPT's response.
    """
    def encode_image(image):
        if isinstance(image, str):
            with open(image, "rb") as img:
                return base64.b64encode(img.read()).decode('utf-8')
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    temp_sleep()

    # print("running gpt-4o request...")
    user_content = []
    for image_path in image_paths:
        user_content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                "detail": "low"}
            })
    if user_prompt:
        user_content.append({
            "type": "text",
            "text": user_prompt
        })
    try:
        messages = [
            {
            "role": "system",
            "content": [
                {
                "type": "text",
                "text": sys_prompt
                }
            ]
            },
            {
            "role": "user",
            "content": user_content
            }
        ]
        response = client.chat.completions.create(
                    model=gpt_parameter["engine"],
                    messages=messages,
                    temperature=gpt_parameter["temperature"],
                    max_tokens=gpt_parameter["max_tokens"],
                    top_p=gpt_parameter["top_p"],
                    frequency_penalty=gpt_parameter["frequency_penalty"],
                    presence_penalty=gpt_parameter["presence_penalty"],
                    stream=gpt_parameter["stream"],
                    stop=gpt_parameter["stop"])
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return "TOKEN LIMIT EXCEEDED OR OTHER EXCEPTIONS"

def generate_prompt(curr_input, prompt_lib_file):
    """
    Takes in the current input (e.g. comment that you want to classifiy) and
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this
    function replaces this substr with the actual curr_input to produce the
    final promopt that will be sent to the GPT3 server.
    ARGS:
      curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                  INPUT, THIS CAN BE A LIST.)
      prompt_lib_file: the path to the promopt file.
    RETURNS:
      a str prompt that will be sent to OpenAI's GPT server.
    """
    if type(curr_input) == type("string"):
        curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]

    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):
        prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt:
        prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()


def safe_generate_response(prompt,
                           parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=True):
    if verbose:
        with open(f"chat_raw.jsonl", 'a') as f:
            f.write(json.dumps({'prompt': prompt}))
            f.write('\n')

    for i in range(repeat):
        if parameter["engine"][:3] == "gpt":
            curr_response = GPT_request(prompt, parameter)
        elif parameter["engine"][:5] == "llama":
            if parameter["engine"] == "llama-3.1-8B-Instruct-quantized.w8a16":
                curr_response = LLaMA_generate(prompt, parameter, llama_3_1_8B_Instruct_quantized_w8a16_llm, llama_3_1_8B_Instruct_quantized_w8a16_tokenizer)
            else:
                NotImplementedError(f"This LLaMA model {parameter['engine']} is not supported yet.")
        else:
            raise NotImplementedError("Only GPT and LLaMA are supported.")
        try:
            # print("debug funcval and clean:")
            # print("func_validate:", func_validate)
            # print("func_validate val:", func_validate(curr_gpt_response, prompt=prompt))
            # print("func_clean_up val:", func_clean_up(curr_gpt_response, prompt=prompt))
            if func_validate(curr_response, prompt=prompt):
                return func_clean_up(curr_response, prompt=prompt)
            # if verbose:
                # print ("---- repeat count: ", i, curr_response)
                # print (curr_response)
                # print ("~~~~")
        except:
            pass
    return fail_safe_response

def GPT4o_safe_generate_response(image_paths, sys_prompt, user_prompt, parameter, repeat=5, fail_safe_response="error", func_validate=None, func_clean_up=None, verbose=False):
    if verbose:
        print (prompt)

    for i in range(repeat):
        if parameter["engine"][:3] == "gpt":
            curr_response = GPT4o_request_new(image_paths, sys_prompt, user_prompt, parameter)
            print(curr_response)
        else:
            raise NotImplementedError("Only GPT and LLaMA are supported.")
        try:
            # print("debug funcval and clean:")
            # print("func_validate:", func_validate)
            # print("func_validate val:", func_validate(curr_gpt_response, prompt=prompt))
            # print("func_clean_up val:", func_clean_up(curr_gpt_response, prompt=prompt))
            if func_validate(curr_response, prompt=prompt):
                return func_clean_up(curr_response, prompt=prompt)
            if verbose:
                print ("---- repeat count: ", i, curr_response)
                print (curr_response)
                print ("~~~~")
        except:
            pass
    return fail_safe_response

def get_embedding(text, model=EMBEDDING_MODEL):
    file_path = "cached_gpt_embeddings.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            embeddings = json.load(file)
    else:
        embeddings = {}
    
    text = text.replace("\n", " ")
    if not text:
        text = "this is blank"

    if text in embeddings.keys():
        return embeddings[text]
    else:
        try:
            if model == openai_config["embeddings"]:
                response = embeddings_client.embeddings.create(input=[text], model=model)
                cost_logger.update_cost(response=response, input_cost=openai_config["embeddings-costs"]["input"], output_cost=openai_config["embeddings-costs"]["output"])
                embeddings[text] = response.data[0].embedding
            elif model == "distilbert":
                inputs = distilbert_base_tokenizer(text, padding=True, truncation=True, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                embeddings = outputs.last_hidden_state
                embeddings[text] = torch.mean(embeddings, dim=1)
            with open(file_path, 'w') as f:
                json.dump(embeddings, f, indent=4)
            return embeddings[text]
        except Exception as e:
            print(f"Error: {e}")
            return "EMBEDDING MODEL EXCEPTIONS"


if __name__ == '__main__':
    gpt_parameter = {"engine": openai_config["model"], "max_tokens": 50,
                     "temperature": 0, "top_p": 1, "stream": False,
                     "frequency_penalty": 0, "presence_penalty": 0,
                     "stop": ['"']}
    curr_input = ["driving to a friend's house"]
    prompt_lib_file = "prompt_template/test_prompt_July5.txt"
    prompt = generate_prompt(curr_input, prompt_lib_file)

    def __func_validate(gpt_response):
        if len(gpt_response.strip()) <= 1:
            return False
        if len(gpt_response.strip().split(" ")) > 1:
            return False
        return True
    def __func_clean_up(gpt_response):
        cleaned_response = gpt_response.strip()
        return cleaned_response

    output = safe_generate_response(prompt,
                                   gpt_parameter,
                                   5,
                                   "rest",
                                   __func_validate,
                                   __func_clean_up,
                                   True)

    print (output)
