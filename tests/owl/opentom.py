import json
import os, pathlib, sys
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from OwlCoT import LLM, ListDialog, generate_faiss_id
from OwlCoT import OwlInnerVoice as oiv


def f1(tp, tn, fp, fn):
    return tp/(tp+.5*(fp+fn))

with open('/home/bruce/Downloads/datasets/OpenTom/opentom_data_metadata_long.json', 'r') as f:
    data_long = json.load(f)
with open('/home/bruce/Downloads/datasets/OpenTom/opentom_data_attitude.json', 'r') as f:
    attitude = json.load(f)
with open('/home/bruce/Downloads/datasets/OpenTom/opentom_data_location_cg_fo.json', 'r') as f:
    location_cg_fo = json.load(f)
with open('/home/bruce/Downloads/datasets/OpenTom/opentom_data_location_cg_so.json', 'r') as f:
    location_cg_so = json.load(f)
with open('/home/bruce/Downloads/datasets/OpenTom/opentom_data_location_fg_fo.json', 'r') as f:
    location_fg_fo = json.load(f)
with open('/home/bruce/Downloads/datasets/OpenTom/opentom_data_location_fg_so.json', 'r') as f:
    location_fg_so = json.load(f)
with open('/home/bruce/Downloads/datasets/OpenTom/opentom_data_multihop_fo.json', 'r') as f:
    multihop_fo = json.load(f)
with open('/home/bruce/Downloads/datasets/OpenTom/opentom_data_multihop_so.json', 'r') as f:
    multihop_so = json.load(f)

print(f' long count: {len(data_long.keys())}')
print(f' attitude count: {len(attitude.keys())}')
print(f' cg_fo count: {len(location_cg_fo.keys())}')
print(f' cg_so count: {len(location_cg_so.keys())}')
print(f' fg_fo count: {len(location_fg_fo.keys())}')
print(f' fg_so count: {len(location_fg_so.keys())}')
print(f' mh_fo count: {len(multihop_fo.keys())}')
print(f' mh_so count: {len(multihop_so.keys())}')

class Question:
    def __init__(self, answers):
        self.answers = []
        for answer in answers:
            self.answers.append(answer.strip().lower())
            self.scores = [[0,0,0] for i in range(len(answers))] #tp, fp, fn
        print(f'question scores {self.scores}')

    def incorporate(self, answer, guess):
        answer = answer.strip().lower()
        guess = guess.strip().lower()
        guess_index = -1
        for a, ans in enumerate(self.answers):
            if ans in guess: # check if this choice is in response
                guess_index = a
                break
        if guess_index == -1:
            print(f' unknown guess: {guess} skipping')
            return # not sure what to do with these?\
        answer_index = -1
        for a, ans in enumerate(self.answers):
            if ans == answer: # check if this choice is in response
                answer_index = a
                break
        if answer_index == -1:
            print(f' unknown answer: {answer}, skipping')
            return # not sure what to do with these?\

        if guess_index == answer_index: #tp
            self.scores[answer_index][0] += 1
        elif guess_index > -1:
            self.scores[guess_index][1] += 1
            self.scores[answer_index][2] += 1

    def f1(self):
        macro_sum = 0
        micro_row = [0,0,0]
        for r in range(len(self.scores)):
            row = self.scores[r]
            if row[0]+row[1]+row[2] > 0:
                row_f1 = row[0]/(row[0]+.5*(row[1]+row[2]))
                macro_sum += row_f1
                micro_row[0] += row[0]; micro_row[1] += row[1]; micro_row[2] += row[2]; 
        return macro_sum/len(self.scores), micro_row[0]/(micro_row[0]+.5*(micro_row[1]+micro_row[2]))
    
q = Question(['a','b','c'])
q.incorporate('b','b')
q.incorporate('b','a')
print(q.scores)
print(q.f1())
#sys.exit(0)


# attitude: positive/negative/neutral
# cg_fo: yes/no
# cg_so: yes/no
# fg_fo: short string (just check for containment
# fg_so: short string (just check for containment
# multihop: more/less/equal

cot = oiv(None)


answers = []
for key in data_long.keys():
    queries = location_cg_fo[key]
    for query in queries:
        answer = query['answer']
        if answer not in answers:
            answers.append(answer)
print(f' Answers: {answers}')
scoring = Question(answers)

for k, key in enumerate(data_long.keys()):
    narrative = data_long[key]['long_narrative']
    prompt = [SystemMessage('The User will provide a narrative about a situation and participant actions. The user will follow with a question about the end state of the scenario, typically about participant beliefs or emotions. Reason step by step to determine the answer to the user question based on the provided narrative. Respond with only Yes or No. Do not include any discursive or explanatory text.'),
              UserMessage('{{$narrative}}\n\nQuestion: {{$question}}\n'),
              AssistantMessage('Answer: ')
              ]
    queries = location_cg_fo[key]
    for query in queries:
        question = query['question']
        answer = query['answer']
        response = cot.llm.ask({'narrative':narrative, 'question':question}, prompt, temp=0.1, max_tokens=15)#, template='gpt-4-turbo')
        #print(f' dataset answer: {answer}, {response.strip()}')
        response = response.strip().lower()
        scoring.incorporate(answer, response)
    #if k > 40:
    #    break
print(f'cg_fo {scoring.scores}')
print(f'cg_fo {scoring.f1()}')


answers = []
for key in data_long.keys():
    queries = location_cg_so[key]
    for query in queries:
        answer = query['answer']
        if answer not in answers:
            answers.append(answer)
print(f' Answers: {answers}')
scoring = Question(answers)


for k, key in enumerate(data_long.keys()):
    narrative = data_long[key]['long_narrative']
    prompt = [SystemMessage('The User will provide a narrative about a situation and participant actions. The user will follow with a question about the end state of the scenario, typically about participant beliefs or emotions. Reason step by step to determine the answer to the user question based on the provided narrative. Respond with only Yes or No. Do not include any discursive or explanatory text.'),
              UserMessage('{{$narrative}}\n\nQuestion: {{$question}}\n'),
              AssistantMessage('Answer: ')
              ]
    queries = location_cg_so[key]
    for query in queries:
        question = query['question']
        answer = query['answer']
        response = cot.llm.ask({'narrative':narrative, 'question':question}, prompt, temp=0.1, max_tokens=15)#, template='gpt-4-turbo')
        # print(f' dataset answer: {answer}, {response.strip()}')
        response = response.strip().lower()
        scoring.incorporate(answer, response)
    #if k > 40:
    #    break
        
print(f'cg_so {scoring.f1()}')

answers = []
for key in data_long.keys():
    queries = location_fg_fo[key]
    for query in queries:
        answer = query['answer']
        if answer not in answers:
            answers.append(answer)
print(f' Answers: {answers}')
scoring = Question(answers)

for key in data_long.keys():
    narrative = data_long[key]['long_narrative']
    prompt = [SystemMessage('The User will provide a narrative about a situation and participant actions. The user will follow with a question about the end state of the scenario, typically about participant beliefs or emotions. Reason step by step to determine the answer to the user question based on the provided narrative. Respond only with the choice designating the location in question, choosing from the Choices provided by the user. Do not include any explanatory or discursive text.'),
              UserMessage('{{$narrative}}\n\nQuestion: {{$question}}\n\n Choices: {{$answers}}\n'),
              AssistantMessage('Answer: ')
              ]
    queries = location_fg_fo[key]
    for query in queries:
        question = query['question']
        answer = query['answer']
        response = cot.llm.ask({'narrative':narrative, 'question':question, 'answers': answers}, prompt, temp=0.01, max_tokens=25)#, template='gpt-4-turbo')
        #print(f'\nnarrative\n{narrative}\nQuestion: {question}')
        #print(f' dataset answer: {answer}, {response.strip()}')
        response = response.strip().lower()
        scoring.incorporate(answer, response)

print(f'fg_fo {scoring.f1()}')



answers = []
for key in data_long.keys():
    queries = location_fg_so[key]
    for query in queries:
        answer = query['answer']
        if answer not in answers:
            answers.append(answer)
print(f' Answers: {answers}')
scoring = Question(answers)

for key in data_long.keys():
    narrative = data_long[key]['long_narrative']
    prompt = [SystemMessage('The User will provide a narrative about a situation and participant actions. The user will follow with a question about the end state of the scenario, typically about participant beliefs or emotions. Reason step by step to determine the answer to the user question based on the provided narrative.  Respond only with the choice designating the location in question, choosing from the Choices provided by the user. Do not include any explanatory or discursive text.'),
              UserMessage('{{$narrative}}\n\nQuestion: {{$question}}\n\n Choices: {{$answers}}\n'),
              AssistantMessage('Answer: ')
              ]
    queries = location_fg_so[key]
    answers = []
    for query in queries:
        answer = query['answer']
        if answer not in answers:
            answers.append(answer)
    for query in queries:
        question = query['question']
        answer = query['answer']
        response = cot.llm.ask({'narrative':narrative, 'question':question, "answers":answers}, prompt, temp=0.01, max_tokens=25)#, template='gpt-3.5-turbo')
        #print(f'\nnarrative\n{narrative}\nQuestion: {question}')
        #print(f' dataset answer: {answer}, {response.strip()}')
        response = response.strip().lower()
        scoring.incorporate(answer, response)

print(f'fg_so {scoring.f1()}')


answers = []
for key in data_long.keys():
    queries = multihop_fo[key]
    for query in queries:
        answer = query['answer']
        if answer not in answers:
            answers.append(answer)
print(f' Answers: {answers}')
scoring = Question(answers)

for key in data_long.keys():
    narrative = data_long[key]['long_narrative']
    prompt = [SystemMessage('The User will provide a narrative about a situation and participant actions. The user will follow with a question about the end state of the scenario, typically about participant beliefs or emotions. Reason step by step to determine the answer to the user question based on the provided narrative. Respond only with the choice designating the location in question, choosing from the Choices provided by the user. Do not include any explanatory or discursive text.'),
              UserMessage('{{$narrative}}\n\nQuestion: {{$question}}\n\n Choices: {{$answers}}\n'),
              AssistantMessage('Answer: ')
              ]
    queries = multihop_fo[key]
    for query in queries:
        question = query['question']
        answer = query['answer']
        response = cot.llm.ask({'narrative':narrative, 'question':question, 'answers': answers}, prompt, temp=0.01, max_tokens=25)#, template='gpt-4-turbo')
        #print(f'\nnarrative\n{narrative}\nQuestion: {question}')
        #print(f' dataset answer: {answer}, {response.strip()}')
        response = response.strip().lower()
        scoring.incorporate(answer, response)

print(f'multihop_fo {scoring.f1()}')



answers = []
for key in data_long.keys():
    queries = multihop_so[key]
    for query in queries:
        answer = query['answer']
        if answer not in answers:
            answers.append(answer)
print(f' Answers: {answers}')
scoring = Question(answers)

for key in data_long.keys():
    narrative = data_long[key]['long_narrative']
    prompt = [SystemMessage('The User will provide a narrative about a situation and participant actions. The user will follow with a question about the end state of the scenario, typically about participant beliefs or emotions. Reason step by step to determine the answer to the user question based on the provided narrative.  Respond only with the choice designating the location in question, choosing from the Choices provided by the user. Do not include any explanatory or discursive text.'),
              UserMessage('{{$narrative}}\n\nQuestion: {{$question}}\n\n Choices: {{$answers}}\n'),
              AssistantMessage('Answer: ')
              ]
    queries = multihop_so[key]
    answers = []
    for query in queries:
        answer = query['answer']
        if answer not in answers:
            answers.append(answer)
    for query in queries:
        question = query['question']
        answer = query['answer']
        response = cot.llm.ask({'narrative':narrative, 'question':question, "answers":answers}, prompt, temp=0.01, max_tokens=25)#, template='gpt-3.5-turbo')
        #print(f'\nnarrative\n{narrative}\nQuestion: {question}')
        #print(f' dataset answer: {answer}, {response.strip()}')
        response = response.strip().lower()
        scoring.incorporate(answer, response)

print(f'multihop_so {scoring.f1()}')

