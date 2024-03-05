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
        
    def fetch(self, uri, focus, max_tokens, wm_name):
        paper_id = s2.index_url(uri)
        if paper_id < 0:
            return False
        texts = s2.get_paper_sections(paper_id = paper_id)
        resource = rw.shorten(texts, focus, max_tokens)
        self.wm.assign(wm_name, resource)

    #process $wm1, 'extract key themes and topics of $wm1 in a form useful as a search query.', $wm2'
    def process(self, arg, instruction, dest, max_tokens):
        resolved_arg= self.interpreter.resolve_arg(arg)
        resolved_inst = instruction.replace(arg, 'the user input provided below')
        prompt = [SystemMessage("""{{$resolved_inst}}
Do not include any discursive or explanatory text. End your response with </Query>"""),
                  UserMessage('\n<INPUT>\n{{$input}}\n</INPUT>\n\n'),
                  AssistantMessage('<Query>\n')
                  ]
        query = self.cot.llm.ask({"resolved_inst":resolved_inst, "input":resolved_arg, "length":int(max_tokens)},
                            prompt, max_tokens=int(1.2*max_tokens), eos='</Query>')
        self.wm.assign(dest, query)
        return query
