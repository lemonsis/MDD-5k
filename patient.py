import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import llm_tools_api

class Patient(llm_tools_api.PatientCost):
    def __init__(self, patient_template, model_path, use_api, story_path) -> None:
        super().__init__(model_path.split('/')[-1])
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.patient_model = None
        self.patient_tokenizer = None
        self.experience = None
        self.patient_template = patient_template
        self.system_prompt = "你是一名{}岁的{}性{}患者，正在和一位精神科医生交流，使用口语化的表达，输出一整段没有空行的内容。如果医生的问题可以用是/否来回答，你的回复要简短精确。".format(self.patient_template['年龄'], self.patient_template['性别'], self.patient_template['诊断结果'])
        self.messages = []
        self.use_api = use_api
        self.client = None
        self.story_path = story_path
        self.dialbegin = True


    def patientbot_init(self):
        if self.use_api:
            self.client = llm_tools_api.patient_client_init(self.model_name)
        else:
            self.patient_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.patient_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.messages.append({"role": "system", "content": self.system_prompt})


    def patient_response_gen(self, current_topic, dialogue_history):
        # self.messages.append({"role": "user", "content": doctor_response})
        if self.use_api:
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            patient_template = {key:val for key, val in self.patient_template.items() if key != '处理意见'} 
            if self.experience is None:
                self.experience, x = llm_tools_api.api_patient_experience_trigger(self.model_name, dialogue_history, self.story_path)
                super().money_cost(x[0], x[1])
            if self.experience is None:
                patient_prompt = "你是一名{}患者，正在和一位精神卫生中心临床心理科医生进行交流。如果医生的问题可以用是/否来回答，你的回复要简短精确。\n你的病例为“{}”，\n你和医生的对话历史为{}， \
                    \n现在请根据下面要求生成:\n1.使用第一人称口语化的回答，如果不是必要情况，不要生成疑问句，不要总是以”医生，“开头。\n2.回答围绕{}展开，如果医生的问题可以用是/否来回答，你的回复要简短精确。在对话历史中提到过的内容不要重复再提起。\n3.回复内容必须根据病例内容，对话历史。如果出现不在病例内容中的问题，发挥想象力虚构回答。" \
                    .format(self.patient_template['诊断结果'], patient_template, dialogue_history[-8:], current_topic)
                self.messages.append({"role": "user", "content": patient_prompt})
                chat_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.messages,
                        top_p=0.85,
                        frequency_penalty=0.8
                    )
                super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
                patient_response = chat_response.choices[0].message.content
                self.messages.pop()
            else:
                patient_prompt = "你是一名{}患者，正在和一位精神卫生中心临床心理科医生进行交流。 \
                    \n\n现在请根据下面要求生成对医生的回答:\n1.回复内容必须根据：\n  （1）病例：“{}“\n  （2）过去的创伤经历：“{}”\n  （3）对话历史：“{}”。\n2.你当前的回复需要围绕话题“{}”展开，要精炼精确。在对话历史中提到过的内容不要重复再提起。\n3.涉及到过去的创伤经历时，需要详细清晰的阐述。但是如果已经说过，不允许重复生成。 \n4.使用第一人称口语化的回答，不要生成疑问句，不要总是以”医生，“开头。如果遇到不在病例和过去创伤经历中的问题，发挥想象力虚构回答，不要输出类似”一些事情“，要具体细节。" \
                    .format(self.patient_template['诊断结果'], patient_template, self.experience, dialogue_history[-8:], current_topic)
                self.messages.append({"role": "user", "content": patient_prompt})
                chat_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.messages,
                        top_p=0.85,
                        frequency_penalty=0.7
                    )
                super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
                patient_response = chat_response.choices[0].message.content
                self.messages.pop()
                # self.messages.append({"role": "assistant", "content": patient_response})
        else:
            #TODO
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            text = self.patient_tokenizer.apply_chat_template(
                self.messages,
                tokenize=False,
                add_generation_prompt=True
            )
            patient_model_inputs = self.patient_tokenizer([text], return_tensors="pt").to(self.patient_model.device)
            generated_ids = self.patient_model.generate(
                patient_model_inputs.input_ids,
                max_new_tokens=2048
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(patient_model_inputs.input_ids, generated_ids)
            ]
            patient_response = self.patient_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            self.messages.append({"role": "assistant", "content": patient_response})
        
        return patient_response, super().get_cost()
    
class Roleplay_Patient:
    def __init__(self, patient_template, model_path, use_api) -> None:
        self.patient_template = patient_template
        self.use_api = use_api
        self.model_name = model_path.split('/')[-1]
        self.system_prompt = "你是一名{}岁的{}性{}患者，正在和一位精神科医生交流，使用口语化的表达，输出一整段没有空行的内容。如果医生的问题可以用是/否来回答，你的回复要简短精确。".format(self.patient_template['年龄'], self.patient_template['性别'], self.patient_template['诊断结果'])
        self.messages = []
        self.dialbegin = True

    def patientbot_init(self):
        if self.use_api:
            self.client = llm_tools_api.patient_client_init(self.model_name)

    def patient_response_gen(self, dialogue_history):
        if self.use_api:
            if self.dialbegin:
                self.patientbot_init()
                self.dialbegin = False
            patient_template = {key:val for key, val in self.patient_template.items() if key != '处理意见'} 
            patient_prompt = "你是一名{}患者，正在和一位精神卫生中心临床心理科医生进行交流。如果医生的问题可以用是/否来回答，你的回复要简短精确。\n你的病例为“{}”，\n你和医生的对话历史为{}， \
                    \n现在请根据下面要求生成:\n1.使用第一人称口语化的回答，如果不是必要情况，不要生成疑问句，不要总是以”医生，“开头。\n2.如果医生的问题可以用是/否来回答，你的回复要简短精确。在对话历史中提到过的内容不要重复再提起。\n3.回复内容必须根据病例内容，对话历史。如果出现不在病例内容中的问题，发挥想象力虚构回答。" \
                    .format(self.patient_template['诊断结果'], patient_template, dialogue_history[-8:])
            self.messages.append({"role": "user", "content": patient_prompt})
            chat_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    top_p=0.85,
                    frequency_penalty=0.8
                )
            patient_response = chat_response.choices[0].message.content
            self.messages.pop()
        return patient_response