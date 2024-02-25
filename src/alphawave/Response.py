import json
import re

class Response:
    @staticmethod
    def parse_all_objects(text):
        objects = []
        if text is None:
            return objects
        obj = Response.parse_json(text)
        if obj is not None:
            objects.append(obj)
        """lines = text
        #lines = text.split('\n')
        print(f'Response len(lines): {len(lines)}')
        if len(lines) > 1:
            for line in lines:
                obj = Response.parse_json(line)
                if obj:
                    objects.append(obj)

        if len(objects) == 0:
            obj = Response.parse_json(text)
            if obj:
                objects.append(obj)

        """
        #print(f'***** Response return \n{objects}\n')
        return objects

    @staticmethod
    def parse_json(text):
        text = ''.join(c for c in text if c.isprintable())
        text = text.replace('\n', '')
        #text = text.replace('}\n', '}')
        #text = re.sub(r"'([^\"']+)'", r'"\1"', text) # all pairs as doublequote
        #text = re.sub(r"'([^\"']+)':", r'"\1":', text) # keys as doublequote
        #text = re.sub(r'"([^\'"]+)":', r"'\1':", text) # keys as singlequote
        #text = text.replace("'", '"')
        #text = text.replace("\'", '"')
        #print(f'parse_json post re {text}')
        start_brace = text.find('{')
        if start_brace >= 0:
            obj_text = text[start_brace:]
            nesting = ['}']
            cleaned = '{'
            in_string = False
            i = 1
            #print(f'cleaning {obj_text}')
            while i < len(obj_text) and len(nesting) > 0:
                ch = obj_text[i]
                if in_string:
                    #print(f'in string, char {ch}')
                    cleaned += ch
                    if ch == '\\': # ignore escape within a string
                        #print(f'parse_json backslash {ch}')
                        i += 1
                        if i < len(obj_text):
                            cleaned += obj_text[i]
                        else: # string can't end with backslash
                            return None
                    elif ch == '"': # string must end with ", 
                        # skip spaces!
                        next = obj_text[i+1:].lstrip()
                        skipped = len(obj_text[i+1:])-len(next)
                        if len(next) > 0 and (next[0] ==':' or next[0] ==',' or next[0] =='}' or next[0] ==']'):
                            in_string = False
                            i = i+skipped
                            #print(f'string end! remainder:{obj_text[i+1:]}')
                        elif skipped > 0 and next[0] == '"':
                            # quote was followed by space and next char was quote, assume missing ','
                            in_string = False
                            i = i+skipped
                            cleaned += ','+obj_text[i]
                        else: # " is within a string, escape it
                            cleaned += '\\'+obj_text[i]
                            
                else: # not in string
                    #print(f'not in string, char {ch}')
                    if ch == '"': # start of string
                        #print(f'parse_json string start {ch}')
                        in_string = True
                        cleaned += ch
                    elif ch == '{': # start of nested dict
                        #print(f'parse_json start of nested dict {ch}')
                        nesting.append('}')
                        cleaned += ch
                    elif ch == '[': #start of array
                        #print(f'parse_json start of array {ch}')
                        nesting.append(']')
                        cleaned += ch
                    elif ch == '}': # end of dict
                        #print(f'parse_json end of nested dict {ch}')
                        close_object = nesting.pop()
                        cleaned += ch
                        if close_object != '}':
                            #print(f'parse_json mismatched dict/array {ch}')
                            return None
                    elif ch == ']': #end of array
                        close_array = nesting.pop()
                        cleaned += ch
                        #print(f'parse_json start of nested dict {ch}')
                        if close_array != ']':
                            return None
                    elif ch == ' ': # whitespace in dict
                        pass
                    #elif ch == '<': # assuming < or > need to be surrounded by ", this seems wrong for python code!
                    #    ch = '"<'
                    #elif ch == '>':
                    #    ch = '>"'
                    else:
                        cleaned += ch
                i += 1

            if len(nesting) > 0:
                cleaned += ''.join(reversed(nesting))

            try:
                #print(f'trying load of {cleaned}')
                if type(cleaned) == str:
                    obj = json.loads(cleaned)
                    #print(f'parse_json return dict {obj}')
                    return obj
                else:
                    print(f'parse_json fail {type(cleaned)}')
                    return cleaned
                return obj if len(obj.keys()) > 0 else None
            except json.JSONDecodeError as e:
                print(f'Response failed to load scrubbed str {str(e)}\n {cleaned}')
                return cleaned
        else:
            return None

        
