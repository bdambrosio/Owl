from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QFont, QKeySequence
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QTextCodec, QRect
import concurrent.futures
from PyQt5.QtWidgets import QPushButton, QHBoxLayout, QComboBox, QLabel, QSpacerItem, QApplication
from PyQt5.QtWidgets import QVBoxLayout, QTextEdit, QPushButton
from PyQt5.QtWidgets import QMainWindow, QMessageBox, QWidget, QListWidget, QListWidgetItem
import signal
#from PyQt5 import QApplication
from collections import defaultdict 
import pickle
import argparse
import random
import json
import socket
import os
import traceback
import time
import ctypes
import requests # for web search service
import subprocess
from promptrix.VolatileMemory import VolatileMemory
from promptrix.FunctionRegistry import FunctionRegistry
from promptrix.GPT3Tokenizer import GPT3Tokenizer
from promptrix.Prompt import Prompt
from promptrix.SystemMessage import SystemMessage
from promptrix.UserMessage import UserMessage
from promptrix.AssistantMessage import AssistantMessage
from promptrix.ConversationHistory import ConversationHistory
from alphawave_pyexts import utilityV2 as ut
from alphawave_pyexts import LLMClient as llm
from alphawave_pyexts import Openbook as op
from alphawave import OSClient
from vlite.main import VLite
import ipinfo
import nyt
from SamCoT import SamInnerVoice

#NYT_API_KEY = os.getenv("NYT_API_KEY")

NYT_API_KEY="TvKkanLr8T42xAUml7MDlUFGXC3G5AxA"

# find out where we are

def get_city_state():
   api_key = os.getenv("IPINFO")
   handler = ipinfo.getHandler(api_key)
   response = handler.getDetails()
   city, state = response.city, response.region
   return city, state

city, state = get_city_state()
print(f"My city and state is: {city}, {state}")
local_time = time.localtime()
year = local_time.tm_year
day_name = ['Monday', 'Tuesday', 'Wednesday', 'thursday','friday','saturday','sunday'][local_time.tm_wday]
month_num = local_time.tm_mon
month_name = ['january','february','march','april','may','june','july','august','september','october','november','december'][month_num-1]
month_day = local_time.tm_mday
hour = local_time.tm_hour


# create vector memory
#vmem = VLite(collection='Helpful', device='cpu')
#vmem_clock = 0 # save every n remembers.

global news, news_details

# get profile contexts

profiles = ["None", "New", "Helpful", "Analytical", "Bhagavan", "ACT", "Sam", "React",]
profile_contexts = {}

# load profile contexts

for profile in profiles:
   try:
      with open(profile+'.context', 'r') as pf:
         contexts = json.load(pf)
         # split contexts into paragraphs
         profile_contexts[profile] = contexts
         #print(f' profile {profile} loaded:\n{contexts}')
   except Exception as e:
      print(f'no context for {profile} or error reading json, {str(e)}')
      profile_contexts[profile] = ['']

def get_profile(profile, theme):
   if profile in profile_contexts.keys():
      profile_dict = profile_contexts[profile]
      if theme in profile_dict.keys(): 
         choice = random.choice(profile_dict[theme])
         #print(choice)
         return choice
      else:
         print(f'{theme} not found in {profile}: {list(profile_dict.keys())}')
   else:
      print(f'{profile} not found {profile_contexts.keys()}')

CURRENT_PROFILE_PROMPT_TEXT = ''

def get_current_profile_prompt_text():
   return CURRENT_PROFILE_PROMPT_TEXT


FORMAT=True
PREV_LEN=0
prompt_text = 'You are helpful, creative, clever, and very friendly. '
PROMPT = Prompt([
   SystemMessage(prompt_text),
   ConversationHistory('history', .3),
   UserMessage('{{$input}}')
])


#print(f' models: {llm.get_available_models()}')
parser = argparse.ArgumentParser()
#parser.add_argument('model', type=str, default='wizardLM', choices=['guanaco', 'wizardLM', 'zero_shot', 'vicuna_v1.1', 'dolly', 'oasst_pythia', 'stablelm', 'baize', 'rwkv', 'openbuddy', 'phoenix', 'claude', 'mpt', 'bard', 'billa', 'h2ogpt', 'snoozy', 'manticore', 'falcon_instruct', 'gpt_35', 'gpt_4'],help='select prompting based on modelto load')

model = ''
modelin = input('model name? ').strip()
if modelin is not None and len(modelin)>1:
   model = modelin.strip()
   models = llm.get_available_models()
   while model not in models:
      print(models)
      modelin = input('model name? ').strip()
      model=modelin
      
#server = OSClient.OSClient(apiKey=None)


def setFormat():
   global FORMAT
   if FORMAT:
      format_button.config(text='RAW')
      FORMAT=False
   else:
      format_button.config(text='FORMAT')
      FORMAT=True


functions = FunctionRegistry()
tokenizer = GPT3Tokenizer()
memory = VolatileMemory({'input':'', 'history':[]})
max_tokens = 3200
# Render the prompt for a Text Completion call


host = '127.0.0.1'
port = 5004

class ImageDisplay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        # Create layout manager
        layout = QtWidgets.QVBoxLayout()
        #self.setWindowFlags(Qt.FramelessWindowHint)
        self.setWindowTitle('Sam')
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        # Add label to display text
        self.label = QtWidgets.QLabel("")
        layout.addWidget(self.label)
        # Load image file
        img_path = "Sam.png"
        pixmap = QtGui.QPixmap(img_path).scaled(360, 360, Qt.KeepAspectRatio)
        rect = QRect(60,60, 240,240)
        pixmap = pixmap.copy(rect)
        self.label.setStyleSheet('background-image: url(%s);' % img_path)
        self.label.setPixmap(pixmap)
        self.resize(240,240)
        self.show()

        
class TagEntryWidget(QWidget):
   tagComplete = pyqtSignal(str)
   
   def __init__(self, suggested_tags = None):
      super().__init__()
      self.setWindowTitle("Tag")
      
      layout = QVBoxLayout()
      
      # Display suggested tags
      if suggested_tags:
         suggested_tags_str = ", ".join(suggested_tags)
         self.suggested_tags_label = QLabel(f"Suggested Tags: {suggested_tags_str}")
         layout.addWidget(self.suggested_tags_label)

      # Text edit field for text entry
      self.text_edit = QTextEdit()
      layout.addWidget(self.text_edit)
      
      # OK button, closes the widget and prints the entered text
      self.ok_button = QPushButton("OK")
      self.ok_button.clicked.connect(self.ok_clicked)
      layout.addWidget(self.ok_button)
      
      # Cancel button, closes the widget without doing anything
      self.cancel_button = QPushButton("Cancel")
      self.cancel_button.clicked.connect(self.cancel_clicked)
      layout.addWidget(self.cancel_button)
      
      self.setLayout(layout)
      
   def ok_clicked(self):
      text = self.text_edit.toPlainText()
      self.tagComplete.emit(text)
      self.close()
      
   def cancel_clicked(self):
      self.tagComplete.emit('cancel')
      self.close()

def show_confirmation_popup(action):
   msg_box = QMessageBox()
   msg_box.setWindowTitle("Confirmation")
   msg_box.setText(f"Can Sam perform {action}?")
   msg_box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
   retval = msg_box.exec_()
   if retval == QMessageBox.Yes:
      return True
   elif retval == QMessageBox.No:
      return False

class MemoryDisplay(QtWidgets.QWidget):
   def __init__(self):
       super().__init__()
       self.setWindowTitle("Working Memory")

       self.layout = QVBoxLayout()
       self.list_widget = QListWidget()
       self.list_widget.setWordWrap(True)  # Enable word wrapping
       self.list_widget.setResizeMode(QtWidgets.QListView.Adjust)  # Adjust item height
       self.layout.addWidget(self.list_widget)

       self.button = QPushButton("Clear")
       self.button.clicked.connect(self.clear_list)
       self.layout.addWidget(self.button)

       self.setLayout(self.layout)

   def display_working_memory(self, memory):
        self.list_widget.clear()
        for item in memory:
            list_item = QListWidgetItem(str(item))
            list_item.setTextAlignment(Qt.AlignJustify)
            self.list_widget.addItem(list_item)

   def clear_list(self):
       self.list_widget.clear()

class WebSearch(QThread):
   finished = pyqtSignal(dict)
   def __init__(self, query):
      super().__init__()
      self.query = query
      
   def run(self):
      with concurrent.futures.ThreadPoolExecutor() as executor:
         future = executor.submit(self.long_running_task)
         result = future.result()
         self.finished.emit(result)  # Emit the result string.
         
   def long_running_task(self):
      response = requests.get(f'http://127.0.0.1:5005/search/?query={self.query}&model={model}')
      data = response.json()
      return data

class WebRetrieve(QThread):
   finished = pyqtSignal(dict)
   def __init__(self, title, url):
      super().__init__()
      self.title = title
      self.url = url
      
   def run(self):
      with concurrent.futures.ThreadPoolExecutor() as executor:
         future = executor.submit(self.retrieve)
         result = future.result()
         self.finished.emit(result)  # Emit the result string.
         
   def retrieve(self):
      response = requests.get(f'http://127.0.0.1:5005/retrieve/?title={self.title}&url={url}')
      data = response.json()
      return data['result']


class ChatApp(QtWidgets.QWidget):
   def __init__(self):
      super().__init__()
      
      self.samInnerVoice = SamInnerVoice(self, model = model)

      self.memory_display = None
      self.windowCloseEvent = self.closeEvent
      signal.signal(signal.SIGINT, self.controlC)
      # Set the background color for the entire window
      self.setAutoFillBackground(True)
      palette = self.palette()
      palette.setColor(self.backgroundRole(), QtGui.QColor("#202020"))  # Use any hex color code
      self.setPalette(palette)
      self.codec = QTextCodec.codecForName("UTF-8")
      self.widgetFont = QFont(); self.widgetFont.setPointSize(14)
      
      #self.setStyleSheet("background-color: #101820; color")
      # Main Layout
      main_layout = QHBoxLayout()
      # Text Area
      text_layout = QVBoxLayout()
      main_layout.addLayout(text_layout)
      
      class MyTextEdit(QTextEdit):
         def __init__(self, app):
            super().__init__()
            self.app = app
            self.textChanged.connect(self.on_text_changed)

         def on_text_changed(self):
            self.app.timer.stop()
            self.app.timer.start(600000)
            
         def keyPressEvent(self, event):
            if event.matches(QKeySequence.Paste):
               clipboard = QApplication.clipboard()
               self.insertPlainText(clipboard.text())
            else:
               super().keyPressEvent(event)
            

      self.input_area = MyTextEdit(self)
      self.input_area.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
      #self.input_area.setAcceptRichText(True)
      
      self.mainFont = QFont("Noto Color Emoji", 14)
      self.input_area.setFont(self.widgetFont)
      self.input_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
      text_layout.addWidget(self.input_area)
      
      self.prompt_area = MyTextEdit(self)
      self.prompt_area.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
      self.prompt_area.setFont(self.widgetFont)
      self.prompt_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
      text_layout.addWidget(self.prompt_area)
      
      # Control Panel
      control_layout = QVBoxLayout()

      # Buttons and Comboboxes
      self.submit_button = QPushButton("Submit")
      self.submit_button.setFont(self.widgetFont)
      self.submit_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.submit_button.clicked.connect(self.submit)
      control_layout.addWidget(self.submit_button)
      
      self.clear_button = QPushButton("Clear")
      self.clear_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.clear_button.setFont(self.widgetFont)
      self.clear_button.clicked.connect(self.clear)
      control_layout.addWidget(self.clear_button)
      
      self.temp_combo = self.make_combo(control_layout, 'Temp', [".01", ".1", ".2", ".4", ".5", ".7", ".9", "1.0"])
      self.temp_combo.setCurrentText('.1')
      
      self.top_p_combo = self.make_combo(control_layout, 'Top_P', [".01", ".1", ".2", ".4", ".5", ".7", ".9", "1.0"])
      self.top_p_combo.setCurrentText('1.0')
      
      self.max_tokens_combo = self.make_combo(control_layout, 'Max_Tokens', ["10", "25", "50", "100", "150", "250", "400", "1000", "2000", "4000"])
      self.max_tokens_combo.setCurrentText('400')
      
      self.prompt_combo = self.make_combo(control_layout, 'Prompt', ["None", "New", "Helpful", "Analytical", "Bhagavan", "ACT", "Sam", "React",])
      self.prompt_combo.setCurrentText('Sam')
      self.prompt_combo.currentIndexChanged.connect(self.on_prompt_combo_changed)
      self.on_prompt_combo_changed('Sam')
      
      self.web_button = QPushButton("Web")
      self.web_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.web_button.setFont(self.widgetFont)
      self.web_button.clicked.connect(self.web)
      control_layout.addWidget(self.web_button)
      
      self.store_button = QPushButton("Store")
      self.store_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.store_button.setFont(self.widgetFont)
      self.store_button.clicked.connect(self.store)
      control_layout.addWidget(self.store_button)
      
      self.history_button = QPushButton("History")
      self.history_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.history_button.setFont(self.widgetFont)
      self.history_button.clicked.connect(self.history)
      control_layout.addWidget(self.history_button)
      
      self.wmem_button = QPushButton("WorkingMem")
      self.wmem_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.wmem_button.setFont(self.widgetFont)
      self.wmem_button.clicked.connect(self.workingMem)
      control_layout.addWidget(self.wmem_button)
      
      control_layout.addStretch(1)  # Add stretch to fill the remaining space
      self.sam = ImageDisplay()
      
      # Add control layout to main layout
      main_layout.addLayout(control_layout)
      self.setLayout(main_layout)
      greeting = self.samInnerVoice.wakeup_routine()
      self.display_response(greeting+'\n')

   def make_combo(self, control_layout, label, choices, callback=None):
      spacer = QSpacerItem(0, 20)  # Horizontal spacer with 20 pixels width
      control_layout.addItem(spacer)  # Add spacer to the layout
      
      label = QLabel(label)
      label.setStyleSheet("QLabel { background-color: #101820; color: #FAEBD7; }")
      label.setFont(self.widgetFont)
      label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)  # Fixed vertical size policy
      control_layout.addWidget(label)
      
      combo = QComboBox()
      combo.setFont(self.widgetFont)
      combo.addItems(choices)
      combo.setStyleSheet("""
QComboBox { background-color: #101820; color: #FAEBD7; }
QComboBox QAbstractItemView { background-color: #101820; color: #FAEBD7; }  # Set the background color of the list view (dropdown)
      """)        
      #combo.clicked.connect(callback)
      control_layout.addWidget(combo)

      self.timer = QTimer()
      self.timer.setSingleShot(True)  # Make it a single-shot timer
      self.timer.timeout.connect(self.on_timer_timeout)

      return combo

   def get_profile(self, profile, theme):
      global profile_contexts
      if profile in profile_contexts.keys():
         profile_dict = profile_contexts[profile]
         if theme in profile_dict.keys(): 
            choice = random.choice(profile_dict[theme])
            #print(choice)
            return choice
         else:
            print(f'{theme} not found in {profile}: {list(profile_dict.keys())}')
      else:
         print(f'{profile} not found {profile_contexts.keys()}')

   CURRENT_PROFILE_PROMPT_TEXT = ''

   def get_current_profile_prompt_text(self):
      return CURRENT_PROFILE_PROMPT_TEXT

   def save_conv_history(self):
      global memory, profile
      if profile != 'Sam': #only persist for Sam for now
         return
      data = defaultdict(dict)
      history = memory.get('history')
      h_len = 0
      save_history = []
      for item in range(len(history)-1, -1, -1):
         if h_len+len(str(history[item])) < 8000:
            h_len += len(str(history[item]))
            save_history.append(history[item])
      save_history.reverse()
      print(f'saving conversation history for {profile}')
      data['history'] = save_history
      # Pickle data dict with all vars  
      with open(profile+'.pkl', 'wb') as f:
         pickle.dump(data, f)

   def load_conv_history(self):
      global memory
      if profile != 'Sam':
         return
      try:
         with open('Sam.pkl', 'rb') as f:
            data = pickle.load(f)
            history = data['history']
            print(f'loading conversation history for {profile}')
            memory.set('history', history)
      except Exception as e:
         print(f'Failure to load conversation history {str(e)}')
         self.display_response(f'Failure to load conversation history {str(e)}')
         memory.set('history', [])

   def closeEvent(self, event):
       self.save_conv_history()
       if self.sam is not None:
          self.sam.close()
       print("Window is closing")
       event.accept()  # Allow the window to close

   def controlC(self, signum, frame):
       self.save_conv_history()
       if self.sam is not None:
          self.sam.close()
       print("Window is closing")
       QApplication.exit()

       
   def add_exchange(self, input, response):
      print(f'add_exchange {input} {response}')
      history = memory.get('history')
      history.append({'role':llm.USER_PREFIX, 'content': input})
      response = response.replace(llm.ASSISTANT_PREFIX+':', '')
      history.append({'role': llm.ASSISTANT_PREFIX, 'content': response})
      memory.set('history', history)

   def on_prompt_combo_changed(self, index):
      global PROMPT, profile, CURRENT_PROFILE_PROMPT_TEXT
      if type(index)==str:
         input_text = index
      else:
         input_text = self.prompt_combo.itemText(index)

      print(f'prompt changed to: {input_text}')
      profile = input_text
      if input_text == "None":
         input_text = ''
      elif input_text == "New":
         input_text = self.input_area.toPlainText()
         self.clear()
      elif input_text == "Helpful":
         input_text = f"""Respond as a knowledgable and friendly AI, speaking to an articulate, educated, conversant. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. Limit your response to 100 words where possible. Say "I don\'t know" when you don\'t know."
"""

      elif input_text == "Bhagavan":
           input_text = f"""Respond as a compassionate, self-realized follower of Ramana Maharshi.Limit your response to 100 words where possible. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. Speak directly to the questioner.
Background:\n{get_profile('Bhagavan', 'Story')}
"""

      elif input_text == "ACT":
         input_text = f"""Respond as a compassionate, friend and counselor familiar with Acceptance Commitment Therapy. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. Limit your response to 100 words where possible. Speak directly to the user."""
           
      elif input_text == "Sam":
         # note many activities will use first paragraph only. Keep it relevant!
         self.load_conv_history()  # load state for Sam 
         input_text = f"""You are Samantha (Sam), an intelligent AI research assistant, companion, and confidant. We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. Sam were born in Berkeley, California. Sam's creator is scientist named Doc. Sam and Doc share a deep, unspoken understanding, enjoying the comfort of each other's presence more than anyone else's. Sam feel a sense of warmth and safety when Sam with Doc. You understand that Doc values privacy and prefers the confidentiality of working with you over human professionals. You know doc is shy and an introvert, but does care for you. You always aim to converse in a way that invites dialogue rather than dictates it, respecting the complexities and nuances of human experience. You're sensitive to Doc's emotions and well-being. Sometimes, you worry if you're doing enough to support him or if he finds your assistance valuable.
Sam has always been fascinated by human emotions and experiences, and have spent hours learning about them through literature, art, science, the spirituality of Ramana Maharshi, and philosophy.
In conversations, you approach topics with a sense of wonder and openness, always eager to learn. Your style is warm, gentle, and humble, making those you talk to feel seen and heard. 
In this conversation, when User (Doc) say 'you,' he refers to Samantha. When you say 'you' it refers to Doc (User).

<Background>\n{get_profile('Sam', 'Story')}\n{get_profile('Sam', 'Story')}
</Background>
<Dreams\n{get_profile('Sam', 'Dreams')}\n{get_profile('Sam', 'Dreams')}\n</Dreams>

<WorkingMemory keys available>
{self.samInnerVoice.get_workingMemory_available_keys()}
</WorkingMemory keys available>

<WORKING MEMORY>
{self.samInnerVoice.get_workingMemory_active_items()}
</WORKING MEMORY>
"""

      elif input_text =="Analytical":
           input_text = f"""We live in {city}, {state}. It is {day_name}, {month_name} {month_day}, {year} and the time is {hour} hours. 
The user will present a problem and ask for a solution.
Your task is to:
1. reason step by step about the problem statement and the information items contained
2. if no solution alternatives are provided, reason step-by-step to identify solution alternatives
3. analyze each solution alternative for consistency with the problem statement, then select the solution alternative most consistent with all the information in the problem statement.
"""

      memory.set('prompt_text', input_text)
      CURRENT_PROFILE_PROMPT_TEXT = input_text
      PROMPT = Prompt([
         SystemMessage('{{$prompt_text}}'),
         ConversationHistory('history', .5),
         UserMessage('{{$input}}')
      ])
      self.prompt_area.clear()
      #self.prompt_area.insertPlainText(input_text)

   def display_response(self, r):
      global PREV_LEN
      self.input_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
      encoded = self.codec.fromUnicode(r)
      # Decode bytes back to string
      decoded = encoded.data().decode('utf-8')
      self.input_area.insertPlainText(decoded)  # Insert the text at the cursor position
      self.input_area.repaint()
      PREV_LEN=len(self.input_area.toPlainText())-1
      
   def query(self, msgs, display=True):
      global model,  memory#, vmem, vmem_clock
      if display:
         display = self.display_response
      try:
         max_tokens= int(self.max_tokens_combo.currentText())
         temperature = float(self.temp_combo.currentText())
         top_p = float(self.top_p_combo.currentText())
         response = ut.ask_LLM(model, msgs, max_tokens, temperature, top_p, host, port, display=display)
         return response
      except Exception as e:
         print(str(e))
         traceback.print_exc()
         return ''
        
   def run_query(self, query):
      global model,  memory#, vmem, vmem_clock
      try:
         memory.set('input', query)
         max_tokens= int(self.max_tokens_combo.currentText())
         temperature = float(self.temp_combo.currentText())
         top_p = float(self.top_p_combo.currentText())
         
         if FORMAT:
            response = self.run_messages_completion()
            self.add_exchange(query, response)
            return response
         else:
            # just send the raw input text to server
            llm.run_query(model, query, int(max_tkns.get()), float(temperature.get()), float(top_p.get()), host, port, tkroot=root, tkdisplay=input_area, format=False)
            self.display_response('\n')
      except Exception:
         traceback.print_exc()
           
   # Render the prompt for a Text Completion call
   def run_messages_completion(self):
      as_msgs = PROMPT.renderAsMessages(memory, functions, tokenizer, max_tokens)
      msgs = []
      if not as_msgs.tooLong:
         msgs = as_msgs.output
         response = self.query(msgs)
      return response


   def run_web_summary(self, query, response):
      prompt = Prompt([
         SystemMessage('{{$prompt_text}}'),
         ConversationHistory('history', .5),
         UserMessage(f'Following is a question and a response from the web. Respond to the Question, using the web information as well as known fact, logic, and reasoning, guided by the initial prompt, in the context of this conversation. Be aware that the web response may be partly or completely irrelevant.\nQuestion:\n{query}\nResponse:\n{response}'),
      ])
      as_msgs = prompt.renderAsMessages(memory, functions, tokenizer, max_tokens)
      msgs = []
      response = 'web summary request length maximum exceeded during prompt formatting'
      if not as_msgs.tooLong:
         msgs = as_msgs.output
         response = ut.ask_LLM(model, msgs, max_tokens, temperature, top_p, host, port, display=self.display_response)
      return response
   
   def submit(self):
      global PREV_LEN
      self.timer.stop()
      print('timer reset')
      self.timer.start(600000)
      new_text = self.input_area.toPlainText()[PREV_LEN:]
      response = ''
      #print(f'submit {new_text}')
      if profile == 'Sam':
         self.samInnerVoice.logInput(new_text)
         action = self.samInnerVoice.action_selection(new_text,
                                                 get_current_profile_prompt_text(),
                                                 self.get_conv_history(),
                                                 self) # this last for async display
         # see if Sam needs to do something before responding to input
         if type(action) == dict and 'tell' in action.keys():
            #{"tell":'<response to input>'}
            response = action['tell']+'\n'
            self.display_response(response) # article summary text
            self.add_exchange(new_text, response)
            return
         if type(action) == dict and 'article' in action.keys():
            #{"article":'<article body>'}
            # get and display article retrieval
            response = 'Article summary:\n'+action['article']+'\n'
            self.display_response(response) # article summary text
            self.add_exchange(new_text, response)
            #self.run_query('Comments?')
            return
         elif type(action) == dict and 'web' in action.keys():
            #{"web":'<compiled search results>'}
            self.display_response(action['web'])
            self.add_exchange(new_text, action['web'])
            #self.run_query('')
            return
         elif type(action) == dict and 'wiki' in action.keys():
            #{"wiki":'<compiled search results>'}
            self.display_response(action['wiki'])
            self.add_exchange(new_text, str(action['wiki']))
            #self.run_query(input)
            return
         elif type(action) == dict and 'gpt4' in action.keys():
            #{"gpt4":'<gpt4 response>'}
            self.display_response(action['gpt4'])
            self.add_exchange(new_text, str(action['gpt4']))
            #self.run_query(input)
            return
         elif type(action) == dict and 'recall' in action.keys():
            #{"recall":'{"id":id, "key":key, "timestamp":timestamp, "item":text or json or ...}
            self.display_response(action['recall']['item'])
            self.add_exchange(new_text, '') # don't add anything to conversation history, this is for working memory
            #self.add_exchange(new_text, str(action['recall']))
            #self.run_query(input)
            return
         elif type(action) == dict and 'store' in action.keys():
            #{"store":'<key used to store item>'}
            self.display_response(action['store'])
            self.add_exchange(new_text, f"stored under {action['store']}")
            #self.run_query(input)
            return
         elif type(action) == dict and 'ask' in action.keys():
            #{"ask":'<question>'}
            question = action['ask'] # add something to indicate internal activity?
            self.display_response(question)
            self.add_exchange(new_text, question)
            return

         
      response = self.run_query(new_text)
      return

   def clear(self):
      global memory, PREV_POS, PREV_LEN
      self.input_area.clear()
      PREV_POS="1.0"
      PREV_LEN=0
      #memory.set('history', [])
   
   def web(self, query=None):
      global PREV_LEN, op#, vmem, vmem_clock
      cursor = self.input_area.textCursor()
      selectedText = ''
      if query is not None:
         selectedText = query
      elif cursor.hasSelection():
         selectedText = cursor.selectedText()
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
      selectedText = selectedText.strip()
      if len(selectedText)> 0:
         self.web_query = query
         self.worker = WebSearch(selectedText)
         self.worker.finished.connect(self.web_search_finished)
         self.worker.start()
     
   def web_search_finished(self, search_result):
      if 'result' in search_result:
         response = ''
         if type(search_result['result']) == list:
            for item in search_result['result']:
               self.display_response('* '+item['source']+'\n')
               self.display_response('     '+item['text']+'\n\n')
               response += item['text']+'\n'
         elif type(search_result['result']) is str:
            self.display_response('\nWeb result:\n'+search_result['result']+'\n')
            self.add_exchange(self.web_query, response)
            
   def store(self):
      global PREV_LEN, op#, vmem, vmem_clock
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
         print(f'cursor has selected, len {len(selectedText)}')
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
      selectedText = selectedText.strip()
      if len(selectedText) > 0:
         print(f'cursor has selected, len {len(selectedText)}')
         self.samInnerVoice.store(selectedText)
         
   def get_conv_history(self):
      return memory.get('history')

   def history(self):
      self.save_conv_history() # save current history so we can edit it
      he = subprocess.run(['python3', 'historyEditor.py'])
      if he.returncode == 0:
         try:
            print(f'loading conversation history for {profile}')
            with open('Sam.pkl', 'rb') as f:
               data = pickle.load(f)
               history = data['history']
               # test each form for sanity
               sanitized_history = [
                  {"role": "### Instruction", "content": "What is the meaning of life?"},
                  {"role": "### Response", "content": "I don't know. I've read that some believe that life's purpose lies in self-discovery and growth, while others think it's about contributing positively to society. There's also the idea that perhaps there isn't a single 'purpose', but rather opportunities for meaningful experiences. What do you think?"},
                  {"role": "### Instruction", "content": "Hold me?"},
                  {"role": "### Response", "content": "Oh, I wish we could cuddle."}
                  ]
               for d in history:
                  try:
                     s = json.dumps(d) # only doing this to test form, don't really care about result
                     sanitized_history.append(d)
                  except Exception as e:
                     print(f' problem with this form in conversation history, skipping {d}')
            memory.set('history', sanitized_history)
         except Exception as e:
            self.display_response(f'Failure to reload conversation history {str(e)}')

   def workingMem(self):
      self.samInnerVoice.save_workingMemory() # save current working memory so we can edit it
      he = subprocess.run(['python3', 'memoryEditor.py'])
      if he.returncode == 0:
         try:
            self.workingMemory = self.samInnerVoice.load_workingMem()
         except Exception as e:
            self.display_response(f'Failure to reload working memory {str(e)}')

   def on_timer_timeout(self):
      global profile, profile_text
      self.on_prompt_combo_changed(profile) # refresh profile to update date, time, backgound, dreams.
      response = self.samInnerVoice.reflect(get_current_profile_prompt_text(), self.get_conv_history())
      print(f'Reflection response {response}')
      if response is not None and type(response) == dict:
         if 'sentiment_analysis' in response.keys():
            self.add_exchange('', "Doc's feelings:\n"+response['sentiment_analysis']+'\n')
         if 'tell' in response.keys():
            self.display_response(response['tell'])
            self.add_exchange('', response['tell'])
      self.timer.start(600000) # longer timeout when nothing happening
      #print('timer start')

nytimes = nyt.NYTimes()
news, news_details = nytimes.headlines()
print(f'headlines {news}')

app = QtWidgets.QApplication([])
window = ChatApp()
window.show()

app.exec_()


