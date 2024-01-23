#
## Working Memory for plan execution: a simple key-addressable dict for now, no persistance
#

class WorkingMemory:
    def __init__(self):
        self.wm = {}
        
    def save(self, filename):
        # note we need to update wm when awm changes! tbd
        with open(filename+'.pkl', 'wb') as f:
          pickle.dump(self.wm, f)
   
    def load(self, filename):
       try:
          with open(filename+'.pkl', 'rb') as f:
             self.wm = pickle.load(f)
             print(f'loaded {len(self.wm.keys())} items from {filename}.pkl')
       except Exception as e:
           print(f'load failed {str(e)}')
           self.wm = {}
      
    def has(self, label):
        return label in self.wm
        
    def get(self, label):
        if self.has(label):
            return self.wm[label]
        else:
            return None
        
    def assign(self, label, item, type=str, notes=''):
        print(f'assign {label}, {type}, {str(item)[:32]}')
        if type not in [str, int, dict, list, 'action', 'plan']:
            print (f'unknown type for wm item {type}')
            raise TypeException(f'bad wm type {type}')
        elif ((type(item) in [str, int] and type(item) != type)
              or (type(item) is dict and type not in [dict, 'plan', 'action'])
              or (type in [dict, 'action'] and type(item) is not dict)
              or (type is list and type(item) is not list)):
            print (f'type mismatch, declared: {type}, actual: {type(item)}')
            raise TypeException(f'bad wm type {type}')

        self.wm[label] = {"label":label, "item":item, "type":type, "notes":notes}
        return self.wm[label]

    def delete (self, label):
        if self.has(label):
            del self.wm[label]
            return True
        return False
