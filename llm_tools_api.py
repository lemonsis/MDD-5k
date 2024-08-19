import os
import pandas as pd
import json
from openai import OpenAI
# Set OpenAI's API key and API base to use vLLM's API server.

class DoctorCost:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.input_cost = 15 / 1000000
        self.output_cost = 5 / 1000000
        self.total_cost = 0

    def money_cost(self, prompt_token_num, generate_token_num):
        if self.model_name == 'gpt-4o':
            self.total_cost += prompt_token_num * self.input_cost + generate_token_num * self.output_cost

    def get_cost(self):
        return self.total_cost
    
class PatientCost:
    def __init__(self, model_name) -> None:
        self.model_name = model_name
        self.input_cost = 15 / 1000000
        self.output_cost = 5 / 1000000
        self.total_cost = 0

    def money_cost(self, prompt_token_num, generate_token_num):
        if self.model_name == 'gpt-4o':
            self.total_cost += prompt_token_num * self.input_cost + generate_token_num * self.output_cost
    
    def get_cost(self):
        return self.total_cost


def gpt4_client_init():
    openai_api_key = #enter your apikey here
    client = OpenAI(
        api_key=openai_api_key
    )
    return client

def qwen_client_init():
    openai_api_key = #enter your apikey here
    openai_api_base = #api

    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )
    return client

def tool_client_init(model_name):
    if 'gpt' in model_name:
        client = gpt4_client_init()
    else:
        client = qwen_client_init()
    return client

def doctor_client_init(model_name):
    if 'gpt' in model_name:
        client = gpt4_client_init()
    else:
        client = qwen_client_init()
    return client

def patient_client_init(model_name):
    if 'gpt' in model_name:
        client = gpt4_client_init()
    else:
        client = qwen_client_init()
    return client

def api_load_for_extraction(model_name, input_sentence):    #extract kv pair
    messages = []
    client = tool_client_init(model_name)
    example = {"孕产情况":"足月顺产",
                "发育情况":"正常"}
    prompt = f'提取文本中所有形如A：B的键值对，以json格式输出，不允许输出其他文字！'
    messages.extend([{"role": "system", "content": "你是一个功能强大的助手，可以处理各种文本任务"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages
    )
    response = chat_response.choices[0].message.content
    messages.extend([{"role": "assistant", "content":response},
                    {"role": "user", "content":input_sentence}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.8
    )
    response = chat_response.choices[0].message.content
    return response

def api_load_for_background_gen(model_name, input_sentence):    #background story generation
    messages = []
    client = tool_client_init(model_name)
    prompt = "输入文本是关于精神疾病患者的基本状况和过去经历的关键词，发挥想象力，根据这些信息以第一人称编写一个故事，完整讲述患者过去的经历，这段经历是患者出现精神疾病的主要原因。\n要求1.输出一整段故事，扩充事件的起因、经过、结果，不要使用比喻句，不要使用浮夸的表述。2.不要输出虚拟的患者姓名。3.不允许输出类似“我正在努力走出阴影”，“在医生的指导下”，只需要输出虚构的故事。\n ###输入文本如下：{}".format(input_sentence)
    messages.extend([{"role": "system", "content": "你是一个功能强大，想象力丰富的文本助手，非常善于写故事"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        top_p=0.9
    )
    response = chat_response.choices[0].message.content
    return response

def api_background_exist(model_name, input_sentence):    #check if background already exists
    messages = []
    client = tool_client_init(model_name)
    prompt = "你需要判断输入内容中是否包含了患者过去的经历，这段经历直接或者间接导致了患者出现精神疾病。例如，“”"
    messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，非常善于写故事"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.4
    )
    response = chat_response.choices[0].message.content
    messages.extend([{"role": "assistant", "content":response},
                    {"role": "user", "content":input_sentence}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        top_p=0.95,
        temperature=1
    )
    response = chat_response.choices[0].message.content
    return response

def api_dialogue_state(model_name, input_sentence):
    messages = []
    client = tool_client_init(model_name)
    prompt = input_sentence
    messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，擅长处理各种文本问题"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.5
    )
    response = chat_response.choices[0].message.content
    return response, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]

def api_parse_experience(model_name, input_sentence):
    messages = []
    client = tool_client_init(model_name)
    prompt = "一名精神疾病患者与精神科医生的对话历史为：{}。根据患者对于自身情况的描述，想象作为一名医生会从哪几个角度进行进一步的询问。\n返回格式如下：以python列表的格式'''[]'''仅返回医生可能询问的角度，返回2-3个，并且以精炼简短的,口语化的语言概括。".format(input_sentence)
    messages.extend([{"role": "system", "content": "你是一个专业的精神健康心理科医生，正在与一名精神疾病患者交流"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        # response_format={"type": "json_object"},
        top_p=0.95
    )
    response = chat_response.choices[0].message.content
    return response, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]

def api_topic_detection(model_name, input_sentence):
    messages = []
    client = tool_client_init(model_name)
    prompt = input_sentence
    messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，擅长处理各种文本问题"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=1
    )
    response = chat_response.choices[0].message.content
    return response, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]

def load_background_story(path):
    with open(path, 'r') as f:
        story = f.readlines()
    return story

def api_patient_experience_trigger(model_name, dialogue_history, path):    #返回生成的背景故事
    messages = []
    client = tool_client_init(model_name)
    prompt = "根据患者和医生的对话历史：{}，判断患者现在是否应该说出导致自己出现精神疾病的过去经历，如果应该说出，则输出“True”，否则返回”None“。".format(dialogue_history)
    messages.extend([{"role": "system", "content": "你是一名精神疾病患者，正在与一位专业的精神健康心理科医生交流。"},
                {"role": "user", "content": prompt}])
    chat_response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.8
    )
    response = chat_response.choices[0].message.content
    if 'True' in response:
        response = load_background_story(path)
        return response[0], [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]
    else:
        return None, [chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens]
    
def api_isroleplay_end(model_name, input_sentence):
    if input_sentence == []:
        return False
    elif len(input_sentence) > 22:
        return True
    else:
        messages = []
        client = tool_client_init(model_name)
        prompt = '一段精神科医生与精神疾病患者之间的诊断对话历史如下:{}，请判断诊断是否应该结束，如果应该结束请返回“是”，如果应该继续请返回“否。”'.format(input_sentence)
        messages.extend([{"role": "system", "content": "你是一个功能强大的文本助手，擅长处理各种文本问题"},
                    {"role": "user", "content": prompt}])
        chat_response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.6
        )
        response = chat_response.choices[0].message.content
        if '是' in response:
            return True
        elif '否' in response:
            return False
        else:
            return True