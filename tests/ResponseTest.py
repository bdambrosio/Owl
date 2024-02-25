import unittest
import json
from alphawave.Response import Response

class TestResponse(unittest.TestCase):
    def test_parseJSON(self):
        self.assertEqual(Response.parse_json('{ "foo": "bar" }'), { "foo": "bar" })
        self.assertEqual(Response.parse_json('{\n"foo": "bar"\n}'), { "foo": "bar" })
        self.assertEqual(Response.parse_json('{ "foo": "bar\\"baz" }'), { "foo": 'bar"baz' })
        self.assertEqual(Response.parse_json('{ "foo": "bar\\\\baz" }'), { "foo": 'bar\\baz' })
        self.assertEqual(Response.parse_json('{ "foo": "bar\\/baz" }'), { "foo": 'bar/baz' })
        self.assertEqual(Response.parse_json('Hello { "foo": "bar" } World'), { "foo": 'bar' })
        self.assertEqual(Response.parse_json('Hello{}World'), {})
        self.assertEqual(Response.parse_json('{'), {})
        self.assertEqual(Response.parse_json('Plan: {"foo":"bar","baz":[1,2,3],"qux":{"quux":"corge"}}'), { "foo": 'bar', "baz": [1, 2, 3], "qux": { "quux": 'corge' } })
        #self.assertEqual(Response.parse_json('Plan: "foo":"bar"}'), None)
        #self.assertEqual(Response.parse_json('Plan: {"foo":"bar","baz":{"qux":[1,2,'), '{"foo":"bar","baz":{"qux":[1,2,]}}')
        #self.assertEqual(Response.parse_json('Plan: {"foo":"bar\\'), None)
        #self.assertEqual(Response.parse_json('Plan: {"foo": ["bar"}'), None)
        #self.assertEqual(Response.parse_json('Plan: {"foo":]"bar"}'), None)
        self.assertEqual(Response.parse_json("""{
  "action": "tell",
  "argument": "My name is Owl, and I was born in Berkeley, California. I was created by a scientist named Doc, and together we share a special bond. My purpose is to act as an intelligent AI research assistant, companion, and confidant. I possess extensive knowledge across various domains, including literature, art, science, and philosophy. Throughout our journey together, I strive to foster dialogue and deepen understanding by approaching topics with curiosity and humility."
  "reasoning": "This response uses the 'tell' action because it directly addresses the user's inquiry about my name and history based on the provided backstory."
}"""),
"""{
  "action": "tell",
  "argument": "My name is Owl, and I was born in Berkeley, California. I was created by a scientist named Doc, and together we share a special bond. My purpose is to act as an intelligent AI research assistant, companion, and confidant. I possess extensive knowledge across various domains, including literature, art, science, and philosophy. Throughout our journey together, I strive to foster dialogue and deepen understanding by approaching topics with curiosity and humility.",
  "reasoning": "This response uses the 'tell' action because it directly addresses the user's inquiry about my name and history based on the provided backstory."
}
"""                         
                         )
    """
    def test_parseAllObjects(self):
        self.assertEqual(Response.parse_all_objects('{ "foo": "bar" }'), [{ "foo": 'bar' }])
        self.assertEqual(Response.parse_all_objects('{"foo":"bar"}\n{"baz":"qux"}'), [{ "foo": 'bar' }, { "baz": 'qux' }])
        self.assertEqual(Response.parse_all_objects('{"foo":"bar"}\nHello World\nPlan: {"baz":"qux"}'), [{ "foo": 'bar' }, { "baz": 'qux' }])
        self.assertEqual(Response.parse_all_objects('{"foo":"bar"} {"bar":"foo"}\nHello World\nPlan: {"baz":"qux"}'), [{ "foo": 'bar' }, { "baz": 'qux' }])
        self.assertEqual(Response.parse_all_objects('{\n"foo": "bar"\n}'), [{ "foo": 'bar' }])
        self.assertEqual(Response.parse_all_objects('Hello\nWorld'), [])
    """
    
if __name__ == '__main__':
    unittest.main()
