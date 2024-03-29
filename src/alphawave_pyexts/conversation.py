"""
Conversation prompt templates.
"""

import dataclasses
from enum import auto, Enum
from typing import List, Any, Dict
from transformers import AutoTokenizer
from typing import List, Dict
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")


class SeparatorStyle(Enum):
    """Separator styles."""

    ADD_COLON_SINGLE = auto()
    ADD_COLON_TWO = auto()
    ADD_COLON_SPACE_SINGLE = auto()
    NO_COLON_SINGLE = auto()
    ADD_NEW_LINE_SINGLE = auto()
    DOLLY = auto()
    RWKV = auto()
    PHOENIX = auto()
    ROBIN = auto()
    SINGLE = auto()
    OPENORCA = auto()

@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates """

    # The name of this template
    name: str
    # The system prompt
    system: str
    # Two roles
    roles: List[str]
    # All messages. Each item is (role, message).
    messages: List[List[str]]
    # The number of few shot examples
    offset: int
    # Separators
    sep_style: SeparatorStyle
    sep: str
    sep2: str = None
    # Stop criteria (the default one is EOS token)
    stop_str: str = None
    # Stops generation if meeting any token in this list
    stop_token_ids: List[int] = None
    # prepends first msg text to front of prompt
    first_msg_no_role:bool = False
    # response_prime - should we end with 'Assistant'?
    response_prime:bool = False

    def get_mistral_prompt(messages: List[Dict[str, str]], tokenizer: AutoTokenizer):
        prompt = ""
        for i, msg in enumerate(messages):
            is_user = {"user": True, "assistant": False}[msg["role"]]
            assert (i % 2 == 0) == is_user
            content = msg["content"]
            assert content == content.strip()
            if is_user:
                prompt += f"[INST] {content} [/INST]"
            else:
                prompt += f" {content}</s>"
        tokens_ids = tokenizer.encode(prompt)
        token_str = tokenizer.convert_ids_to_tokens(tokens_ids)
        return tokens_ids, token_str

    # mistral prompt use example
    """
    messages = [
        {"role": "user", "content": "2+2"},
        {"role": "assistant", "content": "4!"},
        {"role": "user", "content": "+2"},
        {"role": "assistant", "content": "6!"},
        {"role": "user", "content": "+4"},
    ]
    tokens_ids, token_str = build_prompt(messages, tok)
    """

    def get_llama_2_prompt(self):
        #
        ## Note a problem here - conversation history can get confused, may not alternate! this code assumes it does.
        ## also, CHSam now is a single message, but what if we have Sys, CHSam, User?
        ## looks like CHSam get's incorporated as part of sys/init-user, which seems ok,
        ## but that means first user msg gets tacked on as asst.
        ## maybe change CHSam to return 2 msgs to keep order straight?
        ## ah, blank second, so asst replies '' to init message. that seems ok?
        ## or, below, check msg 1, and if from asst, insert '' user msg instead of using first
        B_INST, E_INST = "\n<s>[INST]", "[/INST]\n"
        B_SYS, E_SYS = "<<SYS>>", "<</SYS>>"
        DEFAULT_SYSTEM_PROMPT = """You are a helpful, respectful and truthful assistant.""" 
        messages_out = []
        prompt_str=''
        #print(f'Input: {messages}')
        sysMsg = None
        if self.messages[0][0] == "system": # we have an initial system message
            sysMsg = self.messages[0][1]
            if len(self.messages) > 1:
                if self.messages[1][0] == "user": # 
                    # sys + user, combine
                    prompt_str = B_INST+B_SYS+sysMsg+E_SYS+self.messages[1][1]+E_INST
                    self.messages = self.messages[2:] #drop first two messages
                else: #next msg is assistant
                    prompt_str = B_INST+B_SYS+sysMsg+E_SYS+E_INST
                    self.messages = self.messages[1:] #drop first message
            else: # just sys message
                # just a single msg, the system msg
                prompt_str = B_INST+B_SYS+sysMsg+E_SYS+E_INST
                return prompt_str
        for m, message in enumerate(self.messages):
            # we now expect alternating asst/user, since we already inserted first user msg above
            # note MistralAIClient will handle its dislike of final asst msg internally.
            if message[0] == 'user':
                prompt_str += B_INST+message[1]+E_INST
            else:
                prompt_str += message[1]
                if m < len(self.messages)-1:
                    # if not last message, assumes last assistant message is a prime for asst response
                    prompt_str+'</s>'
        
        #if not prompt+str.endswith('</s>'):
        #    prompt_str += '</s>'

        return prompt_str

        
    def get_prompt(self, include_system=False) -> str:
        """Get the prompt for generation."""
        if self.name == 'llama-2':
            return self.get_llama_2_prompt()
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = ''
            if len(self.system) > 0 and include_system:
                ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>\n"
                else:
                    ret += role + ": " + "<s>\n"
            return ret
        if self.sep_style == SeparatorStyle.ADD_COLON_SINGLE:
            ret = ''
            if len(self.system) > 0 and include_system:
                ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_TWO:
            seps = [self.sep, self.sep2]
            ret = ''
            if len(self.system) > 0 and include_system:
                ret = self.system + self.sep
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.ADD_COLON_SPACE_SINGLE:
            ret = ''
            if len(self.system) > 0 and include_system:
                ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ": "  # must be end with a space
            return ret
        elif self.sep_style == SeparatorStyle.ADD_NEW_LINE_SINGLE:
            ret = ''
            if len(self.system) > 0 and include_system:
                ret = self.system + self.sep
            for n, (role, message) in enumerate(self.messages):
                if message:
                    # should add check to make sure role is asst!
                    if n < len(self.messages)-1: # text in last message is prompt, not complete message!
                        ret += role + "\n" + message + self.sep
                    else:
                        ret += role + "\n" + message
                else:
                    ret += role + "\n"
            return ret
        elif self.sep_style == SeparatorStyle.NO_COLON_SINGLE:
            ret = ''
            if len(self.system) > 0 and include_system:
                ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            return ret
        elif self.sep_style == SeparatorStyle.RWKV:
            ret = ''
            if len(self.system) > 0 and include_system:
                ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += (
                        role
                        + ": "
                        + message.replace("\r\n", "\n").replace("\n\n", "\n")
                    )
                    ret += "\n\n"
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.DOLLY:
            seps = [self.sep, self.sep2]
            ret = ''
            if len(self.system) > 0 and include_system:
                ret = self.system
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ":\n" + message + seps[i % 2]
                    if i % 2 == 1:
                        ret += "\n\n"
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.PHOENIX:
            ret = ''
            if len(self.system) > 0 and include_system:
                ret = self.system
            for role, message in self.messages:
                if message:
                    ret += role + ": " + "<s>" + message + "</s>"
                else:
                    ret += role + ": " + "<s>"
            return ret
        elif self.sep_style == SeparatorStyle.ROBIN:
            ret = ''
            if len(self.system) > 0 and include_system:
                ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ":\n" + message + self.sep
                else:
                    ret += role + ":\n"
            return ret
        elif self.sep_style == SeparatorStyle.OPENORCA:
            ret = ''
            if len(self.system) > 0 and include_system:
                ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + message + self.sep
                else:
                    ret += role
            #ret += self.sep
            return ret
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: str):
        """Update the last output.

        The last message is typically set to be None when constructing the prompt,
        so we need to update it in-place after getting the response from a model.
        """
        self.messages[-1][1] = message

    def to_gradio_chatbot(self):
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self):
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
        return ret

    def copy(self):
        return Conversation(
            name=self.name,
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
            first_msg_no_role=self.first_msg_no_role,
            response_prime = self.response_prime,
        )

    def dict(self):
        return {
            "template_name": self.name,
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


# A global registry for all conversation templates
conv_templates: Dict[str, Conversation] = {}


def register_conv_template(template: Conversation, override: bool = False):
    """Register a new conversation template."""
    if not override:
        assert template.name not in conv_templates, f"{name} has been registered."
    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    if name.startswith('gpt'):
        return conv_templates['gpt-4'].copy()
    else:
        return conv_templates[name].copy()


# A template with a one-shot conversation example
register_conv_template(
    Conversation(
        name="one_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(
            (
                "Human",
                "Got any creative ideas for a 10 year old’s birthday?",
            ),
            (
                "Assistant",
                """Of course! Here are some creative ideas for a 10-year-old's birthday party:
1. Treasure Hunt: Organize a treasure hunt in your backyard or nearby park. Create clues and riddles for the kids to solve, leading them to hidden treasures and surprises.
2. Science Party: Plan a science-themed party where kids can engage in fun and interactive experiments. You can set up different stations with activities like making slime, erupting volcanoes, or creating simple chemical reactions.
3. Outdoor Movie Night: Set up a backyard movie night with a projector and a large screen or white sheet. Create a cozy seating area with blankets and pillows, and serve popcorn and snacks while the kids enjoy a favorite movie under the stars.
4. DIY Crafts Party: Arrange a craft party where kids can unleash their creativity. Provide a variety of craft supplies like beads, paints, and fabrics, and let them create their own unique masterpieces to take home as party favors.
5. Sports Olympics: Host a mini Olympics event with various sports and games. Set up different stations for activities like sack races, relay races, basketball shooting, and obstacle courses. Give out medals or certificates to the participants.
6. Cooking Party: Have a cooking-themed party where the kids can prepare their own mini pizzas, cupcakes, or cookies. Provide toppings, frosting, and decorating supplies, and let them get hands-on in the kitchen.
7. Superhero Training Camp: Create a superhero-themed party where the kids can engage in fun training activities. Set up an obstacle course, have them design their own superhero capes or masks, and organize superhero-themed games and challenges.
8. Outdoor Adventure: Plan an outdoor adventure party at a local park or nature reserve. Arrange activities like hiking, nature scavenger hunts, or a picnic with games. Encourage exploration and appreciation for the outdoors.
Remember to tailor the activities to the birthday child's interests and preferences. Have a great celebration!""",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# raw, just your text with \n between
register_conv_template(
    Conversation(
        name="raw",
        system="",
        roles=("", ""),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep=""
    )
)

# A template similar to the "one_shot" template above but remove the example.
register_conv_template(
    Conversation(
        name="zero_shot",
        system="A chat between a curious human and an artificial intelligence assistant. "
        "The assistant gives helpful, detailed, and polite answers to the human's questions.",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n### ",
        stop_str="###",
    )
)

# Vicuna v1.1 template
register_conv_template(
    Conversation(
        name="vicuna",
        system="",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep=" ",
        sep2="</s>",
        response_prime=True,
    )
)

# Koala default template
register_conv_template(
    Conversation(
        name="koala_v1",
        system="BEGINNING OF CONVERSATION:",
        roles=("USER", "GPT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep=" ",
        sep2="</s>",
    )
)

# Alpaca default template
register_conv_template(
    Conversation(
        name="alpaca",
        system="",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n\n",
        response_prime=True,
    )
)

# Phi-2 default template
register_conv_template(
    Conversation(
        name="phi-2",
        system="",
        roles=("Instruct", "Output"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n",
        response_prime=True,
    )
)

# OpenLLama default template (copy of alpaca)
register_conv_template(
    Conversation(
        name="openllama",
        system="",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n\n",
    )
)

# Dolly V2 default template
register_conv_template(
    Conversation(
        name="dolly_v2",
        system="",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="### End",
    )
)

# ChatGLM2 default template
register_conv_template(
    Conversation(
        name="chatglm2",
        system="",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.DOLLY,
        sep="\n\n",
        sep2="### End",
    )
)

# OpenAssistant Pythia default template
register_conv_template(
    Conversation(
        name="oasst_pythia",
        system="",
        roles=("<|prompter|>", "<|assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="<|endoftext|>",
        response_prime=True,
    )
)

# StableLM Alpha default template
register_conv_template(
    Conversation(
        name="stablelm",
        system="",
        roles=("<|USER|>", "<|ASSISTANT|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="",
        stop_token_ids=[50278, 50279, 50277, 1, 0],
    )
)
# Zephyr Beta template
register_conv_template(
    Conversation(
        name="zephyr",
        system="",
        roles=("\n<|user|>", "\n<|assistant|>", "<|system|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="</s>",
        stop_str="<|user|>",
        stop_token_ids=[2],
    )
)

# Baize default template
register_conv_template(
    Conversation(
        name="baize",
        system="",
        roles=("[|Human|]", "[|AI|]"),
        messages=(
            ("[|Human|]", "Hello!"),
            ("[|AI|]", "Hi!"),
        ),
        offset=2,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="\n",
        stop_str="[|Human|]",
    )
)

# RWKV-4-Raven default template
register_conv_template(
    Conversation(
        name="rwkv",
        system="",
        roles=("Bob", "Alice"),
        messages=(
            ("Bob", "hi"),
            (
                "Alice",
                "Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.",
            ),
        ),
        offset=2,
        sep_style=SeparatorStyle.RWKV,
        sep="",
        stop_str="\n\n",
    )
)

# Buddy default template
register_conv_template(
    Conversation(
        name="openbuddy",
        system="",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)
# FreeWilly default template
register_conv_template(
    Conversation(
        name="freewilly",
        system="",
        roles=("### User", "### Assistant", "### System"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>"
    )
)

# Phoenix default template
register_conv_template(
    Conversation(
        name="phoenix",
        system="",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.PHOENIX,
        sep="</s>",
    )
)

# Yi default template
register_conv_template(
    Conversation(
        name="yi",
        system="",
        roles=("<|Human|>", "<|Assistant|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="</s>",
    )
)

# ChatGPT default template
register_conv_template(
    Conversation(
        name="gpt-3.5-turbo",
        system="",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep=None,
    )
)

# ChatGPT default template
register_conv_template(
    Conversation(
        name="gpt-3.5-turbo-16k",
        system="",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep=None,
    )
)

# ChatGPT default template
register_conv_template(
    Conversation(
        name="gpt-4",
        system="",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep=None,
    )
)

# Llama-2 default template
register_conv_template(
    Conversation(
        name="llama-2",
        system="",
        roles=("user", "assistant", "system"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep='</s>',
    )
)

# mistral default template
register_conv_template(
    Conversation(
        name="mistral-tiny",
        system="",
        roles=("user", "assistant", "system"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep='</s>',
    )
)

# mistral default template
register_conv_template(
    Conversation(
        name="mistral-small",
        system="",
        roles=("user", "assistant", "system"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep='</s>',
    )
)

# mistral default template
register_conv_template(
    Conversation(
        name="mistral-medium",
        system="",
        roles=("user", "assistant", "system"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep='</s>',
    )
)

# mistral default template
register_conv_template(
    Conversation(
        name="mistral-large",
        system="",
        roles=("user", "assistant", "system"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep='</s>',
    )
)

# Claude default template
register_conv_template(
    Conversation(
        name="claude",
        system="",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)
# Claude default template
register_conv_template(
    Conversation(
        name="claude-sonnet",
        system="",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)
# Claude default template
register_conv_template(
    Conversation(
        name="claude-opus",
        system="",
        roles=("user", "assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
    )
)

# MPT default template
register_conv_template(
    Conversation(
        name="mpt",
        system="",
        roles=("<|im_start|>user", "<|im_start|>assistant", "<|im_start|>system"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="<|im_end|>",
        sep2='',
        stop_token_ids=[50278, 0],
        response_prime=True,
    )
)
# chatml default template
register_conv_template(
    Conversation(
        name="chatml",
        system="",
        roles=("<|im_start|>user", "<|im_start|>assistant", "<|im_start|>system"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="<|im_end|>\n",
        sep2='',
        stop_token_ids=[2],
        stop_str= "<|im_end|>",
        response_prime=False,
    )
)
# MPT instruct template as per HG MPT 30b instruct model page
register_conv_template(
    Conversation(
        name="mpt_instruct",
        system="",
        roles=("### Instruction", "### Response",""),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_NEW_LINE_SINGLE,
        sep="\n",
        sep2='',
        stop_token_ids=[50278, 0],
        response_prime=True,
    )
)

# Bard default template
# Reference: https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L150
#            https://github.com/google/generative-ai-python/blob/9c99bcb474a991a97a2e7d62fcdb52db7ce40729/google/generativeai/discuss.py#L40
register_conv_template(
    Conversation(
        name="bard",
        system="",
        roles=("0", "1"),
        messages=(),
        offset=0,
        sep_style=None,
        sep=None,
    )
)

# BiLLa default template
register_conv_template(
    Conversation(
        name="billa",
        system="",
        roles=("Human", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n",
        stop_str="Human:",
    )
)

# RedPajama INCITE default template
register_conv_template(
    Conversation(
        name="redpajama-incite",
        system="",
        roles=("<human>", "<bot>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="<human>",
    )
)

# h2oGPT default template
register_conv_template(
    Conversation(
        name="h2ogpt",
        system="",
        roles=("<|prompt|>", "<|answer|>"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.NO_COLON_SINGLE,
        sep="</s>",
    )
)

# Robin default template
register_conv_template(
    Conversation(
        name="Robin",
        system="",
        roles=("###Human", "###Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n",
        stop_token_ids=[2, 396],
        stop_str="###",
    )
)

# Snoozy default template
# Reference: https://github.com/nomic-ai/gpt4all/blob/d4861030b778da6db59d21d2927a4aba4f9f1f43/gpt4all-bindings/python/gpt4all/gpt4all.py#L232
register_conv_template(
    Conversation(
        name="snoozy",
        system="",
        roles=("### Prompt", "### Response", "### Instruction:\n"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        stop_str="###",
    )
)

# manticore default template
register_conv_template(
    Conversation(
        name="manticore",
        system="",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="</s>",
    )
)

# Falcon default template
register_conv_template(
    Conversation(
        name="falcon",
        system="",
        roles=("User", "Assistant"),
        messages=[],
        offset=0,
        sep_style=SeparatorStyle.RWKV,
        sep="\n",
        sep2="<|endoftext|>",
        stop_str="\nUser", 
        stop_token_ids=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
        ],  # it better only put special tokens here, because tokenizer only remove special tokens
        response_prime=True,
    )
)

register_conv_template(
    Conversation(
        name="falcon_instruct",
        system="",
        roles=("User", "Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        stop_str="User:",
        stop_token_ids=[11],
        sep="\n",
        sep2="<|endoftext|>",
        first_msg_no_role=False,
        response_prime=True,
    )
)
register_conv_template(
    Conversation(
        name="falcon_GPTQ",
        system="",
        roles=("### Human", "### Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        stop_str="### Human",
        stop_token_ids=[11],
        sep="\n",
        sep2="<|endoftext|>",
        first_msg_no_role=False,
    )
)
register_conv_template(Conversation(
        name="galactica",
        system="",
        roles=("Question", "Answer", ""), # third entry is system prefix. if absent defaults to user prefix.
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="",
        response_prime=True,
    )
)
register_conv_template(Conversation(
        name="guanaco",
        system="",
        roles=("### Human", "### Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="",
        response_prime=True,
    )
)
register_conv_template(Conversation(
        name="wizardLM2",
        system="",
        roles=("### Input", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="\n",
        response_prime=True,
    )
)
register_conv_template(Conversation(
        name="wizardCoder",
        system="Below is an instruction that describes a task. Write a response that appropriately completes the request",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="<|end|>",
        response_prime=False,
    )
)
register_conv_template(Conversation(
        name="wizardLM",
        system="",
        roles=("USER", "ASSISTANT"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_TWO,
        sep="\n",
        sep2="\n",
        response_prime=True,
    )
)

register_conv_template(Conversation(
        name="synthia",
        system="",
        roles=("USER", "ASSISTANT", "SYSTEM"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SPACE_SINGLE,
        sep="\n",
        sep2="\n",
        response_prime=False,
    )
)

# ChagGPT default template
register_conv_template(
    Conversation(
        name="polyglot_changgpt",
        system="",
        roles=("B", "A"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
    )
)

# Pygmalion template
register_conv_template(
    Conversation(
        name="pygmalion",
        system="",
        roles=("You", "Samanath"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n",
        response_prime=True,
    )
)

# tigerbot template
register_conv_template(
    Conversation(
        name="tigerbot",
        system="",
        roles=("### Instruction", "### Response"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ROBIN,
        sep="\n\n",
        stop_str="###",
    )
)
# tigerbot template
register_conv_template(
    Conversation(
        name="xgen",
        system="",
        roles=("### Human", "### Assistant"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.ADD_COLON_SINGLE,
        sep="\n\n",
        stop_str="###",
    )
)

register_conv_template(
    Conversation(
        name="openorca",
        system="",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n", "<|im_start|>system\n"),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.OPENORCA,
        sep="<|im_end|>",
    )
)

register_conv_template(
    Conversation(
        name="openchat",
        system="",
        roles=("GPT4 Correct User: ", "GPT4 Correct Assistant: ", "GPT4 CorrectUser: "),
        messages=(),
        offset=0,
        sep_style=SeparatorStyle.OPENORCA,
        sep="<|end_of_turn|>",
    )
)

if __name__ == "__main__":
    conv = get_conv_template("llama-2")
    conv.append_message('system', "You are a friendly AI")
    conv.append_message(conv.roles[0], "Hi!")
    conv.append_message(conv.roles[1], "How are you?")
    conv.append_message(conv.roles[0], "What is your name")
    print(conv.get_prompt())
