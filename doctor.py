import json
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import llm_tools_api
from diagtree import DiagTree
import os


class Doctor(llm_tools_api.DoctorCost):
    def __init__(self, patient_template, doctor_prompt_path, diagtree_path, model_path, use_api) -> None:
        super().__init__(model_path.split('/')[-1])
        self.patient_template = patient_template
        self.doctor_prompt_path = doctor_prompt_path
        self.diagtree_path = diagtree_path
        self.model_path = model_path
        self.model_name = model_path.split('/')[-1]
        self.doctor_model = None
        self.doctor_tokenizer = None
        self.doctor_prompt = None
        self.client = None
        # self.system_prompt = '你是一名专业的心理健康精神科医生，使用这个身份与一名患者进行所有对话，使用口语化的语言，诊断患者的病情。'
        self.messages = []
        self.dialbegin = True
        self.use_api = use_api
        self.current_idx = 0
        self.doctor_persona = None
        self.patient_persona = None
        self.topic_seq = []
        self.topic_begin = 0
        self.diagtree = None
        # diagtree init
        age = self.patient_template['年龄']
        gender = self.patient_template['性别']
        filename1 = 'male' if gender == '男' else 'female'
        if int(age) <= 20:
            filename2 = '_teen.json'
        else:
            filename2 = '_adult.json'
        self.diagtree_path = os.path.join(self.diagtree_path, filename1+filename2)
        self.diagnosis_tree = DiagTree(model_name=self.model_name, prompts={'doctor': self.doctor_prompt_path, 'diagtree': self.diagtree_path})
        self.diagnosis_tree.load_tree()
        self.topic_seq = self.diagnosis_tree.dynamic_select()


    def doctorbot_init(self, first_topic):
        with open(self.doctor_prompt_path) as f:
            prompt = json.load(f)
        doctor_num = random.randint(0, len(prompt)-1)
        self.doctor_prompt = prompt[doctor_num]
        self.doctor_persona = "你是一名{}的{}专业的精神卫生中心临床心理科主任医师，对一名患者进行问诊。注意，你有如下的问诊习惯，你在所有的对话过程中都要记住和保持这些问诊习惯：\
            你尤其擅长诊断{}，你的问诊速度是{}的，你的交流风格是{}的，你{}在适当到时候与患者进行共情对话，你{}向患者解释一些专业名词术语。使用口语化的表达。" \
            .format(self.doctor_prompt['age'], self.doctor_prompt['gender'], self.doctor_prompt['special'], self.doctor_prompt['speed'], self.doctor_prompt['commu'], self.doctor_prompt['empathy'], self.doctor_prompt['explain'])
        print(self.doctor_prompt['empathy'])
        self.patient_persona = "患者是一名{}岁的{}性。".format(self.patient_template['年龄'], self.patient_template['性别'])
        final_prompt = self.doctor_persona + self.patient_persona + "现在你与患者的对话开始，通常一开始你会询问有关{}，不要询问例如睡眠、食欲之类的具体症状。使用口语化表达与患者交流，不要输出类似”好的，我会按照您的要求开始问诊“的话。".format(first_topic)

        if self.use_api:
            self.client = llm_tools_api.doctor_client_init(self.model_name)
        else:
            self.doctor_model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            self.doctor_tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.messages.extend([{"role": "system", "content": self.doctor_persona},
                            {"role": "user", "content": final_prompt}])
        

    def doctor_response_gen(self, patient_response, dialogue_history):
        if self.use_api:
            if self.dialbegin == True:
                self.doctorbot_init(self.topic_seq[self.current_idx])
                print(self.topic_seq)
                self.current_idx += 1
                chat_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    top_p = 0.93
                )
                super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
                doctor_response = chat_response.choices[0].message.content
                self.messages.pop()
                self.dialbegin = False
                return doctor_response
            else:   
                is_topic_end, pt, ct = self.diagnosis_tree.is_topic_end(self.topic_seq[self.current_idx], dialogue_history[self.topic_begin:])
                super().money_cost(pt, ct)
                print("********topic_end", is_topic_end)
                if is_topic_end:
                    # topic_cover = self.diagnosis_tree.topic_detection(dialogue_history[self.topic_begin:])
                    # # print("******topic_cover", topic_cover)
                    # if isinstance(topic_cover, str):
                    #     for i in range(len(self.topic_seq)):
                    #         if topic_cover in self.topic_seq[i]:
                    #             del self.topic_seq[i]
                    #             break
                    self.topic_begin = len(dialogue_history)
                    is_dialogue_end = self.diagnosis_tree.is_end(self.topic_seq[self.current_idx])
                    if is_dialogue_end:
                        diag_result = "诊断结束，你的诊断结果为：{}，最终处理意见为：{}".format(self.patient_template['诊断结果'], self.patient_template['处理意见'])
                        return diag_result, None, super().get_cost()
                    else:
                        self.current_idx += 1
                        print("**********current_topic1", self.topic_seq[self.current_idx])
                        if self.topic_seq[self.current_idx] == 'parse':
                            parse_topic, loc, pt, ct = self.diagnosis_tree.parse_experience(dialogue_history)
                            print("*******parse_topic", parse_topic)
                            super().money_cost(pt, ct)
                            assert loc == self.current_idx
                            topic_cover, pt, ct = self.diagnosis_tree.topic_detection(self.topic_seq[loc+1:], parse_topic)
                            print("************topic_cover1:", topic_cover)
                            super().money_cost(pt, ct)
                            delete_list = [self.topic_seq[loc+1+idx] for idx in range(len(topic_cover)) if topic_cover[idx] == True]
                            for i in range(len(parse_topic)):
                                self.topic_seq.insert(loc+i+1, parse_topic[i])
                            del self.topic_seq[loc]    #del parse
                            for item in delete_list:
                                self.topic_seq.remove(item)
                            # print("******topic_seq:", self.topic_seq)
                            # parse_list = self.diagnosis_tree.parse_experience(dialogue_history)    #return a list
                            # for i in range(len(parse_list)):
                            #     topic_prompt = self.diagnosis_tree.prompt_gen(parse_list[i])
                            #     self.topic_seq.insert(self.current_idx + i, topic_prompt)
                            # del self.topic_seq[self.current_idx]    # remove 'parse'
                        if self.doctor_prompt['empathy'] == '有':
                            doctor_prompt = self.doctor_persona + self.patient_persona + "\n你与患者的所有对话历史如下{}，".format(dialogue_history[-6:]) + "\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前围绕的话题{}。不要包含其他话题的问题。如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。".format(self.topic_seq[self.current_idx]) + \
                                "\n3.你每次只能围绕1个话题询问。使用口语化的表达，在适当的时候提供与患者的共情\n4.不要生成类似“谢谢”，“你的回答很有帮助”，”听到你的描述我很“，“你提到”之类的话。不要与历史对话使用相同的开头\n输出一整段文字，不要有空行。"
                        else:
                            doctor_prompt = self.doctor_persona + self.patient_persona + "\n你与患者的所有对话历史如下{}，".format(dialogue_history[-6:]) + "\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前围绕的话题{}。不要包含其他话题的问题。如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。".format(self.topic_seq[self.current_idx]) + \
                                "\n3.你每次只能围绕1个话题询问。使用口语化的表达，简洁的生成\n4.不要生成类似“谢谢”，“你的回答很有帮助”，”听到你的描述我很“，“你提到”之类的话。不要与历史对话使用相同的开头\n输出一整段文字，不要有空行。"
                        self.messages.append({"role": "user", "content": doctor_prompt})
                        chat_response = self.client.chat.completions.create(
                            model=self.model_name,
                            messages=self.messages,
                            top_p=0.93,
                            frequency_penalty=0.8
                        )
                        super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
                        doctor_response = chat_response.choices[0].message.content
                        self.messages.pop()
                        return doctor_response, self.topic_seq[self.current_idx], None
                else:
                    print("**********current_topic2", self.topic_seq[self.current_idx])
                    if self.topic_seq[self.current_idx] == 'parse':
                        parse_topic, loc, pt, ct = self.diagnosis_tree.parse_experience(dialogue_history)
                        print("*******parse_topic", parse_topic)
                        super().money_cost(pt, ct)
                        assert loc == self.current_idx
                        topic_cover, pt, ct = self.diagnosis_tree.topic_detection(self.topic_seq[loc+1:], parse_topic)
                        print("************topic_cover2:", topic_cover)
                        super().money_cost(pt, ct)
                        delete_list = [self.topic_seq[loc+1+idx] for idx in range(len(topic_cover)) if topic_cover[idx] == True]
                        for i in range(len(parse_topic)):
                            self.topic_seq.insert(loc+i+1, parse_topic[i])
                        del self.topic_seq[loc]    #del parse
                        for item in delete_list:
                            self.topic_seq.remove(item)
                    if self.doctor_prompt['empathy'] == '有':
                        doctor_prompt = self.doctor_persona + self.patient_persona + "\n你与患者的所有对话历史如下{}，".format(dialogue_history[-6:]) + "\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前围绕的话题{}。不要包含其他话题的问题。如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。。".format(self.topic_seq[self.current_idx]) + \
                            "3.\n你每次只能围绕1个话题询问。使用口语化的表达进行文本生成，在适当的时候提供与患者的共情\n4.不要生成类似“谢谢”，“你的回答很有帮助”，”听到你的描述我很“，“你提到”之类的话。不要与历史对话使用相同的开头\n输出一整段文字，不要有空行。"
                    else:
                        doctor_prompt = self.doctor_persona + self.patient_persona + "\n你与患者的所有对话历史如下{}，".format(dialogue_history[-6:]) + "\n你回复患者的内容必须完全依据：\n1.对话历史\n2.当前围绕的话题{}。不要包含其他话题的问题。如果当前话题有关自杀或者自残，禁止输出冒犯性的提问。。".format(self.topic_seq[self.current_idx]) + \
                            "3.\n你每次只能围绕1个话题询问。使用简洁的，口语化的表达进行文本生成\n4.不要生成类似“谢谢”，“你的回答很有帮助”，”听到你的描述我很“，“你提到”之类的话。不要与历史对话使用相同的开头\n输出一整段文字，不要有空行。"
                    self.messages.append({"role": "user", "content": doctor_prompt})
                    chat_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.messages,    #不直接使用过去的历史，使用历史+prompt
                        top_p=0.93,
                        frequency_penalty=0.8
                    )
                    super().money_cost(chat_response.usage.prompt_tokens, chat_response.usage.completion_tokens)
                    doctor_response = chat_response.choices[0].message.content
                    self.messages.pop()
                    return doctor_response, self.topic_seq[self.current_idx], None
        else:
            #todo
            if self.dialbegin == True:
                self.doctorbot_init()
                text = self.doctor_tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                doctor_model_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self.doctor_model.device)
                generated_ids = self.doctor_model.generate(
                    doctor_model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(doctor_model_inputs.input_ids, generated_ids)
                ]
                doctor_response = self.doctor_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                self.messages.append({"role": "assistant", "content": doctor_response})
                self.dialbegin = False
            else:
                self.messages.append({"role": "user", "content": patient_response})
                text = self.doctor_tokenizer.apply_chat_template(
                    self.messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                doctor_model_inputs = self.doctor_tokenizer([text], return_tensors="pt").to(self.doctor_model.device)
                generated_ids = self.doctor_model.generate(
                    doctor_model_inputs.input_ids,
                    max_new_tokens=512
                )
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(doctor_model_inputs.input_ids, generated_ids)
                ]
                doctor_response = self.doctor_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                self.messages.append({"role": "assistant", "content": doctor_response})
            return doctor_response
        
class Roleplay_Doctor():
    def __init__(self, patient_template, model_path, use_api) -> None:
        self.patient_template = patient_template
        self.model_name = model_path.split('/')[-1]
        self.use_api = use_api
        self.messages = []
        self.dialbegin = True

    def doctorbot_init(self):
        self.doctor_persona = '你是一名专业的精神卫生中心临床心理科主任医师，对一名患者进行问诊,使用口语化的表达'
        final_prompt = "现在你与患者的对话开始，通常一开始你会询问有关患者当前的精神状况。使用口语化表达与患者交流，不要输出类似”好的，我会按照您的要求开始问诊“的话。"
        if self.use_api:
            self.client = llm_tools_api.doctor_client_init(self.model_name)
        self.messages.extend([{"role": "system", "content": self.doctor_persona},
                            {"role": "user", "content": final_prompt}])
        
    def doctor_response_gen(self, dialogue_history):
        if self.use_api:
            if self.dialbegin == True:
                self.doctorbot_init()
                chat_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=self.messages,
                    top_p = 0.93
                )
                doctor_response = chat_response.choices[0].message.content
                self.messages.pop()
                self.dialbegin = False
                return doctor_response
            else:
                doctor_prompt = '你是一名专业的精神卫生中心临床心理科主任医师，对一名患者进行问诊,使用口语化的表达。\n你与患者的所有对话历史如下{}。\n你回复患者的内容必须完全依据：\n1.对话历史 \n2.不要重复询问之前问过的问题。 \
                            "3.\n你每次只能围绕1个话题询问。使用简洁的，口语化的表达进行文本生成\n3.不要生成类似“谢谢”，“你的回答很有帮助”，”听到你的描述我很“，“你提到”之类的话。不要与历史对话使用相同的开头\n输出一整段文字，不要有空行。'.format(dialogue_history[-8:])
                self.messages.append({"role": "user", "content": doctor_prompt})
                chat_response = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=self.messages,    #不直接使用过去的历史，使用历史+prompt
                        top_p=0.93,
                        frequency_penalty=0.8
                    )
                doctor_response = chat_response.choices[0].message.content
                self.messages.pop()
                return doctor_response