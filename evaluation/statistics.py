import json
import os

def read_file(name):
    total = []
    if 'MDD' in name:
        for i in range(20):
            with open(os.path.join('./DataSyn', 'patient_'+str(i+1)+'.json')) as f:
                file = json.load(f)
            total.extend(file)
    elif 'D4' in name:
        with open('./evaluation/dialog_final_mapped.json') as f:
            file = json.load(f)
        total = file
    elif 'roleplay' in name:
        for i in range(20):
            with open(os.path.join('./Roleplay', 'patient_'+str(i+1)+'.json')) as f:
                file = json.load(f)
            total.extend(file)
    elif 'CPsy' in name:
        with open('./evaluation/CPsyCounD.json') as f:
            file = json.load(f)
        total = file
    return total

def avg_turns(name:str, file:list):
    if 'MDD' in name or 'roleplay' in name:
        total_turns = 0
        for dic in file:
            con = dic['conversation']
            turn = len(con)
            total_turns += turn
    elif 'D4' in name:
        pass
    return total_turns / 100

def avg_words(name, file):
    doc_total_words = 0
    pat_total_words = 0
    doc_total_turns = 0
    pat_total_turns = 0
    if 'MDD' in name or 'roleplay' in name:
        for dic in file:
            doc_words = 0
            pat_words = 0
            con = dic['conversation']
            pat_turns = len(con)
            doc_turns = pat_turns+1
            for i in range(len(con)):
                if i == len(con)-1:
                    doc_words += len(con[i]['doctor'])
                else:
                    doc_words += len(con[i]['doctor'])
                    pat_words += len(con[i]['patient'])
            doc_total_words += doc_words
            pat_total_words += pat_words
            doc_total_turns += doc_turns
            pat_total_turns += pat_turns
        total_words = doc_total_words + pat_total_words
        return total_words / 100, doc_total_words / doc_total_turns, pat_total_words / pat_total_turns
    elif 'CPsy' in name:
        pass
    elif 'D4' in name:
        for dic in file:
            doc_words = 0
            pat_words = 0
            pat_turns = 0
            doc_turns = 0
            dial = dic['dialog']
            for piece in dial:
                if piece['role'] == 'doctor':
                    doc_words += len(piece['content'])
                    doc_turns += 1
                elif piece['role'] == 'user':
                    pat_words += len(piece['content'])
                    pat_turns += 1
                else:
                    raise ValueError
            doc_total_words += doc_words
            pat_total_words += pat_words
            doc_total_turns += doc_turns
            pat_total_turns += pat_turns
        total_words = doc_total_words + pat_total_words
        return total_words / 1340, doc_total_words / doc_total_turns, pat_total_words / pat_total_turns

def extract_baselines(name, file):
    if 'D4' in name:
        total_output_list = []
        total_output_dict = {}
        for i in range(100):
            dial = file[i]['dialog']
            output_list = []
            output_dict = {}
            for piece in dial:
                if piece['role'] == 'doctor':
                    output_dict['doctor'] = piece['content']
                elif piece['role'] == 'user':
                    output_dict['patient'] = piece['content']
                else:
                    raise ValueError
                if len(output_dict) == 2:
                    output_list.append(output_dict)
                    output_dict = {}
                
            total_output_dict["conversation"] = output_list
            total_output_list.append(total_output_dict)
            total_output_dict = {}
            if (i+1) % 5 == 0:
                with open(os.path.join('evaluation/d4', 'patient_'+str(int(i/5)+1)+'.json'), 'w') as f:
                    json_data = json.dump(total_output_list, f, indent=2, ensure_ascii=False)
                    total_output_list = []

    elif 'CPsy' in name:
        total_output_list = []
        total_output_dict = {}
        for i in range (100):
            output_list = []
            output_dict = {}
            hist = file[i]['history']
            hist.append([file[i]['instruction'], file[i]['output']])

            for item in hist:
                output_dict['patient'] = item[0]
                output_dict['doctor'] = item[1]
                output_list.append(output_dict)
                output_dict = {}
            total_output_dict["conversation"] = output_list
            total_output_list.append(total_output_dict)
            total_output_dict = {}
            if (i+1) % 5 == 0:
                with open(os.path.join('evaluation/cpsy', 'patient_'+str(int(i/5)+1)+'.json'), 'w') as f:
                    json_data = json.dump(total_output_list, f, indent=2, ensure_ascii=False)
                    total_output_list = []


ll = read_file('D4')
print(avg_words('D4', ll))
extract_baselines('D4', ll)