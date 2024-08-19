import json
import os
import llm_tools_api
import random
import ast
import re

class Tree:
    def __init__(self, value) -> None:
        self.value = value
        self.children = []

    def add_node(self, child):
        self.children.append(child)

    def traversal(self):
        print(self.value)
        for child in self.children:
            child.traversal()

    def is_leaf(self):
        if self.children == []:
            return True
        else:
            return False
        
    
class DiagTree():
    def __init__(self, model_name, prompts={}) -> None:
        self.doctor_promot_path = prompts['doctor']
        self.diagtree_path = prompts['diagtree']
        self.model_name = model_name
        self.diagtree = None
        self.dialstate = []
        self.topic_end = []
    

    def load_tree(self):    #加载一棵针对x症状的树
        with open(self.diagtree_path, 'r') as f:
            json_tree = json.load(f)
        root_data = list(json_tree.keys())[0]
        self.diagtree = self.jsontree_to_diagtree(json_tree[root_data], root_data)
        # self.diagtree.traversal()
    

    def force_topic_end(self):
        if len(self.topic_end) < 6:
            return self.topic_end[-1]
        # elif len(self.topic_end) == 2:
        #     if self.topic_end[0] == False and random.randint(0, 1) != 1:
        #         self.topic_end[-1] = True
        #         return True
        #     else:
        #         return self.topic_end[-1]
        else:
            if self.topic_end[-2] == False:
                if self.topic_end[-3] == False:
                    self.topic_end[-1] = True
                    return True
                elif random.randint(0, 2) == 1:
                    return self.topic_end[-1]
                else:
                    self.topic_end[-1] = True
                    return True
            else:
                if self.topic_end[-1] == True:
                    return True
                else:
                    if random.randint(0, 2) == 1:
                        self.topic_end[-1] = True
                        return True
                    else:
                        return self.topic_end[-1]


    def jsontree_to_diagtree(self, json_tree, node_data):
        node = Tree(node_data)
        if isinstance(json_tree, dict):
            for key in json_tree.keys():
                child = self.jsontree_to_diagtree(json_tree[key], key)
                node.add_node(child)
        elif json_tree is None:
            node.value = node_data
        else:
            raise TypeError
        return node
        

    def parse_experience(self, input):    #如果遇到parse，解析病人的回答
        ans, x = llm_tools_api.api_parse_experience(self.model_name, input)
        while re.findall(r"\[(.+)\]", ans) == []:    
            ans, x = llm_tools_api.api_parse_experience(self.model_name, input)
        # if re.findall(r"\[(.+)\]", ans) == []:
        #     loc = self.dialstate.index('parse')
        #     del self.dialstate[loc]
        #     return self.dialstate, parse_result
        # else:
        ans = re.findall(r"\[(.+)\]", ans)[0]
        ans = list(ast.literal_eval(ans))
        loc = self.dialstate.index('parse')
        # for i in range(len(ans)):
        #     self.dialstate.insert(loc+i+1, self.prompt_gen(ans[i]))
        #     parse_result.append(ans[i])
        # del self.dialstate[loc]
        for i in range(len(ans)):
            ans[i] = self.prompt_gen(ans[i])
        return ans, loc, x[0], x[1]
                                                                                                                                                                                                                                                                                                                                                                              

    def topic_detection(self, topic_seq, parse_topic):    #检测回答中是否已经包含了问诊树的话题
        result = []
        for topic in topic_seq:
            prompt = "判断在话题集合“{}”中是否有表达意思与“{}”相似或者相同的，如果包含输出”是“，如果不包含输出”否“。".format(topic, parse_topic)
            answer, x = llm_tools_api.api_topic_detection(self.model_name, prompt)
            if '否' in answer:
                result.append(False)
            elif '是' in answer:
                result.append(True)
            else:
                raise ValueError
        return result, x[0], x[1]


    def dynamic_select(self):    #动态的根据LLM判断当前状态访问树中节点,返回一个list
        if self.diagtree.value == '精神状况':
            prompt = self.prompt_gen(self.diagtree.value)
            self.dialstate.append(prompt)

        assert self.diagtree.children[0].value == '事件询问'
        options = self.diagtree.children[0].children
        for i in range(len(options)):
            option = random.sample(options, 1)[0]
            options.remove(option)
            if option.value == 'parse':
                # self.parse_experience(self.model_name, input)
                self.dialstate.append('parse')
            else:
                prompt = self.prompt_gen(option.value)
                self.dialstate.append(prompt)

        assert self.diagtree.children[1].value == '病情判断'
        options = self.diagtree.children[1].children
        for i in range(len(options)):
            option = random.sample(options, 1)[0]
            options.remove(option)
            if option.children == []:
                prompt = self.prompt_gen(option.value)
                self.dialstate.append(prompt)
            elif len(option.children) >= 1:
                opts = option.children
                for j in range(len(opts)):
                    opt = random.sample(opts, 1)[0]
                    opts.remove(opt)
                    prompt = self.prompt_gen(opt.value)
                    self.dialstate.append(prompt)
            else:
                raise TypeError
        
        assert self.diagtree.children[2].value == '个人史'
        assert self.diagtree.children[3].value == '家族史，亲戚中是否有精神疾病患者'

        if random.randint(0,1) == 1:
            options = self.diagtree.children[2].children
            for i in range(len(options)):
                option = random.sample(options, 1)[0]
                if option.children == []:
                    options.remove(option)
                    prompt = self.prompt_gen(option.value)
                    self.dialstate.append(prompt)
                else:
                    raise TypeError
            prompt = self.prompt_gen(self.diagtree.children[3].value)
            self.dialstate.append(prompt)
        else:
            prompt = self.prompt_gen(self.diagtree.children[3].value)
            self.dialstate.append(prompt)
            options = self.diagtree.children[2].children
            for i in range(len(options)):
                option = random.sample(options, 1)[0]
                if option.children == []:
                    options.remove(option)
                    prompt = self.prompt_gen(option.value)
                    self.dialstate.append(prompt)
                else:
                    raise TypeError

        return self.dialstate
                
    
    def is_topic_end(self, current_state, input_history):    #根据随机的话题，并判断话题持续轮数
        prompt = "一段精神科医生和精神疾病患者之间的对话为{}，判断围绕诊断话题”{}“的对话是否应该结束，如果应该结束返回”是“，如果不应该结束返回”否“。倾向于判断话题应该结束".format(input_history, current_state)
        result, x = llm_tools_api.api_dialogue_state(self.model_name, prompt)
        if '是' in result:
            result = True
        elif '否' in result:
            result = False
        else:
            raise ValueError
        self.topic_end.append(result)
        print("**********1", self.topic_end)
        result = self.force_topic_end()
        print("**********2", self.topic_end)
        return self.topic_end[-1], x[0], x[1]


    def is_end(self, current_state):    #判断整个问诊树是否被完全遍历
        if current_state == self.dialstate[-1]:
            return True
        else:
            return False


    def prompt_gen(self, option):    #根据叶子结点问题生成LLM问题
        prompt = "询问患者有关{}，不要包含其他话题和问题".format(option) 
        return prompt
