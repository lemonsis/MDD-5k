from doctor import Doctor
from patient import Patient
from diagtree import DiagTree
import json
import os
from tqdm import tqdm

DOCTOR_PROMPT_PATH = './prompts/doctor/doctor_persona.json'
PATIENT_INFO_PATH = './raw_data/pa20.json'
DIAGTREE_PATH = './prompts/diagtree'
MODEL_NAME = 'gpt-4o'    #Qwen2-72B-Instruct
NUM = 5
OUTPUT_DATASYN_PATH = './DataSyn'
OUTPUT_PASTEXP_PATH = './prompts/patient/background_story'

total_cost = 0

with open(PATIENT_INFO_PATH, 'r') as f:
    patient_info = json.load(f)

for patient_template in tqdm(patient_info):
    total_output_list = []
    for i in range(NUM):
        dialogue_history = []
        output_list = []
        output_dict = {}
        story_path = os.path.join(OUTPUT_PASTEXP_PATH, 'patient_{}'.format(patient_template['患者']), 'story_{}.txt'.format(i+1))

        doc = Doctor(patient_template, DOCTOR_PROMPT_PATH, DIAGTREE_PATH, MODEL_NAME, True)
        pat = Patient(patient_template, MODEL_NAME, True, story_path)

        doctor_response = doc.doctor_response_gen(None, None)
        output_dict['doctor'] = doctor_response
        dialogue_history.append('医生：' + doctor_response)
        print("医生：", doctor_response)
        current_topic = '患者的精神状况'
        while True:
            patient_response, patient_cost = pat.patient_response_gen(current_topic, dialogue_history)
            output_dict['patient'] = patient_response
            dialogue_history.append('患者：' + patient_response)
            output_list.append(output_dict)
            output_dict = {}
            print("患者：", patient_response)
            doctor_response, current_topic, doctor_cost = doc.doctor_response_gen(patient_response, dialogue_history)
            if '诊断结束，你的诊断结果' in doctor_response:
                output_dict = {'doctor':doctor_response}
                output_list.append(output_dict)
                print("医生：", doctor_response)
                break
            else:
                dialogue_history.append('医生：' + doctor_response)
                output_dict['doctor'] = doctor_response
                print("医生：", doctor_response)
        total_output_list.append({"conversation":output_list})
        total_cost += doctor_cost+patient_cost

    with open(os.path.join(OUTPUT_DATASYN_PATH, 'patient_{}.json'.format(patient_template['患者'])), 'w') as f:
        json_data = json.dump(total_output_list, f, indent=2, ensure_ascii=False)

print("********总价格*********:", total_cost)