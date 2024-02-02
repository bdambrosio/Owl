from jsonschema import validate, ValidationError
from promptrix.promptrixTypes import Message, PromptFunctions, PromptMemory, Tokenizer
from alphawave.alphawaveTypes import PromptResponse, Validation, PromptResponseValidator
from alphawave.Response import Response
from pyee import AsyncIOEventEmitter
import json
import ast
import traceback
import re

def preprocess_json_string(s):
    # Regular expression to find unescaped quotes
    pattern = r'(?<!\\)"'
    
    # Split the string on unescaped quotes and process in pairs
    parts = re.split(pattern, s)
    for i in range(1, len(parts), 2):
        # Escape backslashes and other special characters in the string value
        parts[i] = parts[i].replace('\\', '\\\\').replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t').replace('\b', '\\b').replace('\f', '\\f')
    
    # Join the parts back together
    return '"'.join(parts)

def extract_json_template(schema, p=True):
    template = {}
    if schema is not None and "properties" in schema:
        for key, value in schema["properties"].items():
            if "type" in value:
                if value["type"] == "object":
                    template[key] = extract_json_template(value, p=False)
                elif value["type"] == "array":
                    if 'description' in value:
                        template[key] = '['+value['description']+']'
                    elif "items" in value:
                        template[key] = [extract_json_template(value["items"], p=False)]
                    else:
                        template[key] = []
                else:   # plain old string value
                    if 'description' in value:
                        template[key] = value['description']
                    else: template[key] = '<'+key+'>'
            elif "$ref" in value:
                ref = value["$ref"]
                ref_schema = schema
                for part in ref.split("/")[1:]:
                    ref_schema = ref_schema[part]
                template[key] = extract_json_template(ref_schema, p=False)

    if p:
        pass #print(json.dumps(template))
    return json.dumps(template)

class JSONResponseValidator(PromptResponseValidator):
    def __init__(self, schema=None, missing_json_feedback='Invalid JSON. return valid JSON.'):
        self.schema = schema
        self.missing_json_feedback = missing_json_feedback

    def parse_dict(self, s):
        if s is None:
            return s
        # Try to parse as JSON
        # Try to repair common errors and parse again
        s = s.strip()
        s = s.replace("```\njson", "")  # get rid of markdown prefix
        s = s.replace("```", "")  # get rid of markdown suffix
        if not (s.startswith('{') and s.endswith('}')):
            s = '{' + s + '}'
        s = re.sub(r"'([^\"']+)':", r'"\1":', s) # keys as doublequote
        s = s.replace('\\*', '*')
        s = s.replace('\\+', '+')
        s = s.replace('\\-', '-')
        s = s.replace('\\/', '/')
        s = s.replace('\n', '')
        try:
            s = json.loads(s)
            #print(f'***** JSONReponseValidator parse_dict json.loads success')
            return s
        except json.JSONDecodeError as e:
           pass
        
        # Try to parse as a Python literal
        try:
            #sq = re.sub(r"'(.*?)':", r'"\1"', s) # keys must be double quoted
            sast = ast.literal_eval(sq)
            #print(f'***** JSONReponseValidator parse_dict sast {s}')
            sastj = json.loads(sast)
            #print(f'***** JSONReponseValidator parse_dict ast then re then loads success')
            return sast
        except (SyntaxError, ValueError) as e:
            pass


        # Try to parse the repaired string
        try:
            y = json.loads(s)
            #print(f'***** JSONReponseValidator parse_dict final loads success {y}')
            return y
        except json.JSONDecodeError:
            #print(f'***** JSONReponseValidator parse_dict final return {s}')
            return s

    def validate_response(self, memory: PromptMemory, functions: PromptFunctions, tokenizer: Tokenizer, response: PromptResponse, remaining_attempts: int) -> Validation:
        message = response['message']

        template = {}
        template_suffix = ''
        if self.schema is not None:
            template = extract_json_template(self.schema)
        if len(str(template)) > 4:
            template_suffix = f' Respond using this template:\n{template}\n'
        
        text = message if isinstance(message, str) else message.get('content', '')
        print(f'***** JSONResponseValidator input {type(text)}, {text}')
        # Parse the response text
        #text = re.sub('\n+', '\n', text)
        cleaned_text = ""
        for char in text:
            if ord(char) >= 10:
                cleaned_text += char
        text = cleaned_text.strip()
        start = text.find('{')
        end = text.rfind('}')
        if start >= 0 and end > 0 and  end > start:
            text = text[start:end+1]
        parsed=[]
        print(f'***** JSONResponseValidator cleaned \n{text}\n')
        try:
            text = preprocess_json_string(text)
            print(f'***** JSONResponseValidator preprocessed \n{text}\n')
            parsed = Response.parse_all_objects(text)
            print(f'***** JSONResponseValidator Response parse \n{parsed}\n')
        except Exception as e:
            raise e
        if len(parsed) == 0:
            print(f'***** JSONResponseValidator failure len(parsed) == 0')
            return {
                'type': 'Validation',
                'valid': False,
                'feedback': self.missing_json_feedback+template_suffix,
                'value':raw_text
            }

        # Validate the response against the schema
        errors = None
        for i in range(len(parsed)):  # return first one that passes
            obj = parsed[i]
            #print(f'***** JSONResponseValidator obj {type(obj)} \n{obj}\n')
            try:
                try:
                    print(f'***** JSONResponseValidator before parse_dict {type(obj)}\n{obj}\n')
                    obj = self.parse_dict(obj) if type(obj) == str else obj
                    print(f'***** JSONResponseValidator after parse_dict {type(obj)} \n{parsed}\n')
                except Exception as e:
                    pass
                if self.schema is not None:
                    validate(obj, self.schema)
                    return {
                        'type': 'Validation',
                        'valid': True,
                        'value': obj
                    }
                else:
                    if type(obj) == dict:
                        return {'type': 'Validation', 'valid': True, 'value': obj}
                    str_obj = str(obj).strip()
                    if str_obj.startswith('{') and not str_obj.endswith('}'):
                        str_obj += "'}"
                        try:
                            objj = json.loads(str_obj)
                            return {'type': 'Validation', 'valid': True, 'value': objj}
                        except:
                            return {'type': 'Validation', 'valid': false, 'value': str_obj}
            except ValidationError as e:
                path = str(list(e.relative_schema_path)[1:-1]).replace('[','').replace(']',"").replace(', ', ':')
                if not errors:
                    errors = e
                    print(f'***** JSONResponseValidator ValidationError exception {str(e)}\n{self.schema}\n')
                    return {
                        'type': 'Validation',
                        'valid': False,
                        'feedback': f'The JSON returned had errors. Apply these fixes:\n{self.get_error_fix(errors)}.'+template_suffix
                    }
            except Exception as e:
                print(f'***** JSONResponseValidator validator generic exception {str(e)}\n{self.schema}')
                return {
                    'type': 'Validation',
                    'valid': False,
                    'feedback': f'The JSON returned had errors. Apply these fixes:\n{self.get_error_fix(e)}.'+template_suffix
                }      
    

    def get_error_fix(self, error: ValidationError) -> str:
        # Get argument as a string
        try:
            arg = error.validator
        except:
            return error
        path = str(list(error.relative_schema_path)[1:-1]).replace('[','').replace(']',"").replace(', ', ':')
        
        switcher = {
            'type': f'convert "{path}" value to a {error.validator_value}' if len(path)> 0 else '',
            'anyOf': f'convert "{path}" to one of the allowed types: {error.validator_value}',
            'additionalProperties': f'remove the "{arg}" field from "{path}"',
            'required': f"add the {error.validator_value} fields to {path if len(path)>0 else 'response'}",
            'format': f'change the "{path}" field to be a {error.validator_value}',
            'uniqueItems': f'remove all duplicate items from "{error.path}"',
            'enum': f'change the "{path}" value to be one of these values: {arg}',
            'const': f'change the "{path}" value to be {arg}',
        }

        return switcher[arg] if arg in switcher else error.message

