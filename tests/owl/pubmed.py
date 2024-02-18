import json
import os, pathlib, sys, re
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from OwlCoT import LLM, ListDialog, generate_faiss_id
from OwlCoT import OwlInnerVoice as oiv


"""
{
    "12377809": {
        "QUESTION": "Is anorectal endosonography valuable in dyschesia?",
        "CONTEXTS": [
            "Dyschesia can be provoked by inappropriate defecation movements. The aim of this prospective study was to demonstrate dysfunction of the anal sphincter and/or the musculus (m.) puborectalis in patients with dyschesia using anorectal endosonography.",
            "Twenty consecutive patients with a medical history of dyschesia and a control group of 20 healthy subjects underwent linear anorectal endosonography (Toshiba models IUV 5060 and PVL-625 RT). In both groups, the dimensions of the anal sphincter and the m. puborectalis were measured at rest, and during voluntary squeezing and straining. Statistical analysis was performed within and between the two groups.",
            "The anal sphincter became paradoxically shorter and/or thicker during straining (versus the resting state) in 85% of patients but in only 35% of control subjects. Changes in sphincter length were statistically significantly different (p<0.01, chi(2) test) in patients compared with control subjects. The m. puborectalis became paradoxically shorter and/or thicker during straining in 80% of patients but in only 30% of controls. Both the changes in length and thickness of the m. puborectalis were significantly different (p<0.01, chi(2) test) in patients versus control subjects."
        ],
        "LABELS": [
            "AIMS",
            "METHODS",
            "RESULTS"
        ],
        "MESHES": [
            "Adolescent",
            "Adult",
            "Aged",
            "Aged, 80 and over",
            "Anal Canal",
            "Case-Control Studies",
            "Chi-Square Distribution",
            "Constipation",
            "Defecation",
            "Endosonography",
            "Female",
            "Humans",
            "Male",
            "Middle Aged",
            "Pelvic Floor",
            "Rectum"
        ],
        "YEAR": "2002",
        "reasoning_required_pred": "yes",
        "reasoning_free_pred": "yes",
        "final_decision": "yes",
        "LONG_ANSWER": "Linear anorectal endosonography demonstrated incomplete or even absent relaxation of the anal sphincter and the m. puborectalis during a defecation movement in the majority of our patients with dyschesia. This study highlights the value of this elegant ultrasonographic technique in the diagnosis of \"pelvic floor dyssynergia\" or \"anismus\"."
    },
    "26163474": {

"""
with open('/home/bruce/Downloads/pubmedqa/data/test_set.json') as f:
    test_set = json.load(f)

print(f' test_set: {len(test_set.keys())}')

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
            print(f' unknown guess: {guess}, assume maybe')
            guess_index = 2
            #return # not sure what to do with these?\
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
    
cot = oiv(None)

answers = ['yes','no','maybe']
print(f' Answers: {answers}')
scoring = Question(answers)
with open('test_data_predictions.json', 'w') as f:
    f.write('[\n')
    for k, key in enumerate(test_set.keys()):
        contexts= test_set[key]['CONTEXTS']
        labels = test_set[key]['LABELS']
        formatted_context = ''
        for label, context in zip(labels, contexts):
            formatted_context += label+':\n'+context+'\n'
            
        question = test_set[key]['QUESTION']
        answer = test_set[key]['final_decision']

        prompt = [SystemMessage('The User will provide information about a research study. The user will follow with a question about the study, typically about an inference that might be drawn from the information provided. Reason step by step to determine the answer to the user question based on the provided information. Respond with only with one of {{$answers}} Do not include any discursive or explanatory text.'),
                  UserMessage('{{$context}}\n\nQuestion: {{$question}}\n'),
                  AssistantMessage('Answer: ')
                  ]
        response = cot.llm.ask({'context':formatted_context, 'question':question}, prompt, temp=0.1, max_tokens=5)#, template='gpt-4-turbo')
        #response = 'yes'
        response = response.strip()
        response = re.split('[., ]', response)[0].lower()
        if response == 'not':
            response = 'maybe'
        print(f' dataset answer: {answer}, {response}')
        scoring.incorporate(answer, response)
        #if k > 250:
        #    break
        f.write('{'+key+': '+response+'}\n')
    f.write(']')
print(f'cg_fo {scoring.scores}')
print(f'cg_fo {scoring.f1()}')

