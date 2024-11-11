import os
import pandas as pd
import json
import re
import llm_tools_api
import ast
from tqdm import tqdm
import random

PATIENT_CASES_ORIGIN_PATH = './raw_data/patient_info.xlsx'
PATIENT_CASES_JSON_PATH = './raw_data/pa20.json'
PROMPT_PATH = './prompts'
OUTPUT_PASTEXP_PATH = './prompts/patient/background_story'


class PatientCases():
    def __init__(self, xlsx_path, json_path, prompt_path, use_api):
        self.xlsx_path = xlsx_path
        self.json_path = json_path
        self.use_api = use_api
        self.prompt_path = prompt_path    # root path of prompt
        self.gender_mode = None
        self.age_mode = None

    def patient_cases_json(self):
        # keys = ['患者', '年龄', '性别', 'ICD编码', '诊断结果', '主诉', '现病史', '重要或相关躯体疾病史', '家族史', '个人史', '精神检查', '处理意见']
        output_list = []
        num = 0
        file = pd.read_excel(self.xlsx_path, sheet_name='Sheet1')
        for index, row in file.iterrows():
            if (not pd.isna(row['Diagnosis'])) and (not pd.isna(row['ChiefComplaint'])) and (not pd.isna(row['PresentIllnessHistory'])) and (not pd.isna(row['TreatmentRecommendation'])):   # filter nan diagnosis
                num += 1
                output_dict = {}
                output_dict['患者'] = num
                output_dict['年龄'] = int(row['age'])
                output_dict['性别'] = row['gender']
                output_dict['ICD编码'] = row['DiagnosisCode'] if str(row['DiagnosisCode'])[-1] != ',' else str(row['DiagnosisCode'])[:-1]
                output_dict['诊断结果'] = row['Diagnosis'] if str(row['Diagnosis'])[-1] != ',' else str(row['Diagnosis'])[:-1]
                output_dict['主诉'] = '无' if re.findall(r"主诉：(.+)", str(row['ChiefComplaint'])) == [" "] else re.findall(r"主诉：(.+)", str(row['ChiefComplaint']))[-1]
                output_dict['现病史'] = '无' if re.findall(r"现病史：(.+)", str(row['PresentIllnessHistory'])) == [" "] else re.findall(r"现病史：(.+)", str(row['PresentIllnessHistory']))[-1]
                output_dict['重要或相关躯体疾病史'] = '无' if pd.isna(row['ImportantRelevantPhysicalIllnessHistory']) else row['ImportantRelevantPhysicalIllnessHistory']
                output_dict['家族史'] = '无' if pd.isna(row['FamilyHistory']) else row['FamilyHistory']
                output_dict['个人史'] = '无' if re.findall(r"个人史:(.+)", str(row['PersonalHistory'])) == [" "] else re.findall(r"个人史:(.+)", str(row['PersonalHistory']))[-1]
                output_dict['精神检查'] = '无' if re.findall(r"精神检查描述：(.+)", str(row['PsychiatricExamination'])) == [" "] else re.findall(r"精神检查描述：(.+)", str(row['PsychiatricExamination']))[-1]
                output_dict['处理意见'] = '无' if re.findall(r"处理意见：(.+)", str(row['TreatmentRecommendation'])) == [" "] else re.findall(r"处理意见：(.+)", str(row['TreatmentRecommendation']))[-1]

                output_dict['家族史'] = '无' if output_dict['家族史'] == '家族史：阴性。 ' or output_dict['家族史'] == '家族史：阴性 ' else output_dict['家族史']
                output_dict['家族史'] = re.findall(r"家族史：(.+)", str(output_dict['家族史']))[-1] if output_dict['家族史'] != '无' else output_dict['家族史']
                if re.findall(r"重要或相关躯体疾病史：(.+)", str(output_dict['重要或相关躯体疾病史'])) == []:
                    output_dict['重要或相关躯体疾病史'] = '无'
                elif re.findall(r"重要或相关躯体疾病史：(.+)", str(output_dict['重要或相关躯体疾病史']))[-1][0] == '无':
                    output_dict['重要或相关躯体疾病史'] = '无'
                else:
                    output_dict['重要或相关躯体疾病史'] = re.findall(r"重要或相关躯体疾病史：(.+)", str(output_dict['重要或相关躯体疾病史']))[-1]

                filter = False
                for key in output_dict.keys():    #keep filtering
                    if key in ['主诉', '处理意见', '精神检查', '个人史']:
                        if output_dict[key] == '无':
                            filter = True
                            break
                if filter:
                    continue
                else:
                    if self.use_api:
                        detail_personal = llm_tools_api.api_load_for_extraction("gpt-4o", output_dict['个人史'])
                        detail_mental = llm_tools_api.api_load_for_extraction("gpt-4o", output_dict['精神检查'])
                        detail_mental = ast.literal_eval(detail_mental)
                        detail_personal = ast.literal_eval(detail_personal)
                        output_dict['个人史'] = detail_personal
                        output_dict['精神检查'] = detail_mental
                    else:
                        #todo
                        detail_mental = llm_tools.load_Qwen_for_extraction(output_dict['精神检查'])
                    output_list.append(output_dict)
        with open (self.json_path, 'w') as f:
            json_data = json.dump(output_list, f, indent=2, ensure_ascii=False)


    def key_word_selelction1(self):
        paths = []
        json_data = []
        for mode in self.age_mode:
            path = self.gender_mode + '_' + mode + '.json'
            paths.append(path)
            with open(os.path.join(self.prompt_path, 'patient', path)) as f:
                data = json.load(f)
                json_data.append(data)
        # merge
        dict_for_gen = {}

        for key in json_data[0].keys():
            if isinstance(json_data[0][key], dict):
                temp = {}
                for key1 in json_data[0][key].keys():                 
                    tmp = []
                    for data in json_data:
                        tmp.extend(data[key][key1])
                    temp[key1] = list(set(tmp))
                dict_for_gen[key] = temp
            else:
                tmp = []
                for data in json_data:
                    tmp.extend(data[key])
                tmp = list(set(tmp))
                dict_for_gen[key] = tmp
        return dict_for_gen
        

    def key_word_selelction(self):
        if self.age_mode is not None:
            path = self.gender_mode + '_' + self.age_mode + '.json'
            with open(os.path.join(self.prompt_path, 'patient', path)) as f:
                data = json.load(f)
            return data
        else:
            return None

    def story_gen_for_background(self, patient):
        dict_for_gen = self.key_word_selelction()
        if dict_for_gen is not None:
            first = False
            chosen_key = None
            for key in dict_for_gen.keys():
                if isinstance(dict_for_gen[key], dict) and first == False:
                    chosen_key = random.choices([x for x in dict_for_gen[key].keys()], [len(dict_for_gen[key][x]) for x in dict_for_gen[key].keys()])[0]
                    value1 = random.choice(dict_for_gen[key][chosen_key])
                    dict_for_gen[key] = value1
                    first = True
                elif isinstance(dict_for_gen[key], list):
                    dict_for_gen[key] = random.choice(dict_for_gen[key])
                else:
                    dict_for_gen[key] = random.choice(dict_for_gen[key][chosen_key])
            print(dict_for_gen)
            with open(os.path.join(self.prompt_path, 'patient', 'patient_background.txt')) as f:
                text_prompt = f.readlines()[0]
            text_prompt = text_prompt.format(age=patient['年龄'],gender=patient['性别'],diagnosis=patient['诊断结果'],illness=patient['现病史'],work=patient['个人史']['工作、学习情况'],
                                            time=dict_for_gen['time'],poeple=dict_for_gen['people'],experience=dict_for_gen['experience'])
            response = llm_tools_api.api_load_for_background_gen("gpt-4o", text_prompt)
            return response
        else:
            return ''

    def save_background_story(self, patient, output_path):
        age = patient['年龄']
        gender = patient['性别']
        if gender == "男":
            self.gender_mode = 'male'
        else:
            self.gender_mode = 'female'
        if int(age) <= 50:
            self.age_mode = str(age)
        else:
            self.age_mode = None
        story = self.story_gen_for_background(patient)
        story = story.replace("\n", "")
        with open(output_path, 'w') as f:
            f.write(story)


    def statistics(self):    #statistical calculation
        with open(self.json_path, 'r') as f:
            patient_data = json.load(f)
        total_num = len(patient_data)
        gender = [0, 0]    #male, female
        age = [0, 0, 0, 0, 0, 0, 0, 0]    #10,20,30,40,50,60,70,80
        icd_code = {}
        family = [0, 0]    #family without_disease, with
        personal = [0, 0]    #personal without_disease, with
        for case in patient_data:
            if case['性别'] == '男':
                gender[0] += 1
            else:
                gender[1] += 1
            if case['年龄'] == 10:
                age[0] += 1
            elif case['年龄'] == 20:
                age[1] += 1
            elif case['年龄'] == 30:
                age[2] += 1
            elif case['年龄'] == 40:
                age[3] += 1
            elif case['年龄'] == 50:
                age[4] += 1
            elif case['年龄'] == 60:
                age[5] += 1
            elif case['年龄'] == 70:
                age[6] += 1
            elif case['年龄'] == 80:
                age[7] += 1
            if case['家族史'] == '无':
                family[0] += 1
            else:
                family[1] += 1
            if case['重要或相关躯体疾病史'] == '无':
                personal[0] += 1
            else:
                personal[1] += 1
            icd = case['ICD编码']
            icd = icd.split(',')
            # for i in range(len(icd)):
            #     icd[i] = icd[i].split('.')[0]
            for i in icd:
                if i in icd_code.keys():
                    icd_code[i] += 1
                else:
                    icd_code[i] = 1
        assert gender[0]+gender[1] == total_num == family[0]+family[1] == personal[0]+personal[1] == sum(age)
        print(age, gender, icd_code, family, personal)            

patient = PatientCases(PATIENT_CASES_ORIGIN_PATH, PATIENT_CASES_JSON_PATH, PROMPT_PATH, use_api=True)
patient.statistics()    # output the statistics of the patient cases
NUM = 5    # 1 patient case will be used to generate 5 conversations
with open(PATIENT_CASES_JSON_PATH, 'r') as f:
    patient_info = json.load(f)
# generate patient experiences
for patient_template in tqdm(patient_info):
    for i in range(NUM):
        if not os.path.exists(os.path.join(OUTPUT_PASTEXP_PATH, 'patient_{}'.format(patient_template['患者']))):
            os.mkdir(os.path.join(OUTPUT_PASTEXP_PATH, 'patient_{}'.format(patient_template['患者'])))
        output_path = os.path.join(OUTPUT_PASTEXP_PATH, 'patient_{}'.format(patient_template['患者']), 'story_{}.txt'.format(i+1))
        patient.save_background_story(patient_template, output_path)
