from pathlib import Path
import OwlCoT as oiv
cot = oiv.OwlInnerVoice(None)
print (cot.template)
# set cot for rewrite so it can access llm
import semanticScholar3 as s2

s2.cot = cot
s2.rw.cot = cot
def process_file(filepath):
    # Example function that does something with a filepath
    print(f"Processing file: {filepath}")
    s2.queue_url_for_indexing('file://'+filepath)

def scan_directory(directory):
    path = Path(directory)
    for filepath in path.rglob('*'):
        if filepath.is_file():
            process_file(directory+filepath.name)
            #return # just process one to text
        
# Example usage
directory_path = '/home/bruce/Downloads/owl/tests/owl/arxiv/papers/'
scan_directory(directory_path)
