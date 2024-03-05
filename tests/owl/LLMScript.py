from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
import semanticScholar3 as s2
import rewrite as rw

class LLMScript:
    def __init__ (self, interpreter, cot):
        self.interpreter = interpreter
        self.cot = cot
        self.s2 = s2
        s2.cot = cot
        rw.cot = cot
        self.wm = interpreter.wm
        print(f'\ncreated script runner!')
        
    def fetch(self, uri, instruction, dest, max_tokens):
        paper_id = s2.index_url(uri)
        if paper_id < 0:
            return False
        texts = s2.get_paper_sections(paper_id = paper_id)
        resource = rw.shorten(texts, instruction, max_tokens)
        self.wm.assign(dest, resource)

    #process $wm1, 'extract key themes and topics of $wm1 in a form useful as a search query.', $wm2'
    def process1(self, arg1, instruction, dest, max_tokens):
        resolved_arg= self.interpreter.resolve_arg(arg1)
        resolved_inst = instruction.replace(arg1, 'the user input provided below')
        prompt = [SystemMessage("""{{$resolved_inst}}
Do not include any discursive or explanatory text. End your response with </Response>"""),
                  UserMessage('\n<INPUT>\n{{$input}}\n</INPUT>\n\n'),
                  AssistantMessage('<Response>\n')
                  ]
        query = self.cot.llm.ask({"resolved_inst":resolved_inst, "input":resolved_arg, "length":int(max_tokens)},
                            prompt, max_tokens=int(1.2*max_tokens), eos='</Query>')
        self.wm.assign(dest, query)
        return query

    #process $wm1, 'extract key themes and topics of $wm1 in a form useful as a search query.', $wm2'
    def process2(self, arg1, arg2, instruction, dest, max_tokens):
        resolved_arg1 = self.interpreter.resolve_arg(arg1)
        resolved_arg2 = self.interpreter.resolve_arg(arg2)
        resolved_inst = instruction.replace(arg1, 'the user Input provided below')
        resolved_inst = instruction.replace(arg2, 'the user SecondInput provided below')
        prompt = [SystemMessage("""{{$resolved_inst}}
Do not include any discursive or explanatory text. End your response with </Query>"""),
                  UserMessage('\n<Input>\n{{$text1}}\n</Input>\n<SecondInput>\n{{$text2}}\n</SecondInput>\n\n'),
                  AssistantMessage('<Response>\n')
                  ]
        query = self.cot.llm.ask({"resolved_inst":resolved_inst, "text1":resolved_arg1, "text2":resolved_arg2,
                                  "length":int(max_tokens)}, prompt, max_tokens=int(1.2*max_tokens), eos='</Response>')
        self.wm.assign(dest, query)
        return query
    
