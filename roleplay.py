import llm_tools_api
import json
from doctor import Roleplay_Doctor
from patient import Roleplay_Patient
from tqdm import tqdm
import os

NUM = 5
PATIENT_INFO_PATH = './raw_data/pa20.json'
MODEL_NAME = 'gpt-4o'
OUTPUT_DATASYN_PATH = './Roleplay'

with open(PATIENT_INFO_PATH, 'r') as f:
    patient_info = json.load(f)

for patient_template in tqdm(patient_info):
    total_output_list = []
    for i in range(NUM):
        dialogue_history = []
        output_list = []
        output_dict = {}
        doc = Roleplay_Doctor(patient_template, MODEL_NAME, True)
        pat = Roleplay_Patient(patient_template, MODEL_NAME, True)
        
        while not llm_tools_api.api_isroleplay_end(MODEL_NAME, dialogue_history):
            doctor_response = doc.doctor_response_gen(dialogue_history)
            output_dict['doctor'] = doctor_response
            dialogue_history.append('医生：' + doctor_response)
            print("医生：", doctor_response)
            patient_response = pat.patient_response_gen(dialogue_history)
            output_dict['patient'] = patient_response
            dialogue_history.append('患者：' + patient_response)
            output_list.append(output_dict)
            output_dict = {}
            print("患者：", patient_response)
        result = patient_template["处理意见"]
        output_list.append({"doctor":result})
        total_output_list.append({"conversation":output_list})
    with open(os.path.join(OUTPUT_DATASYN_PATH, 'patient_{}.json'.format(patient_template['患者'])), 'w') as f:
        json_data = json.dump(total_output_list, f, indent=2, ensure_ascii=False)