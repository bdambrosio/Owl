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
import ipinfo
import nyt
from owlCoT import OwlInnerVoice
from Planner import Planner, PlanInterpreter

NYT_API_KEY = os.getenv("NYT_API_KEY")

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

global news, news_details

# get profile contexts

profiles = ["None", "New", "Helpful", "Analytical", "Bhagavan", "ACT", "Owl", "React",]
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


FORMAT=True
PREV_LEN=0
#print(f' template: {llm.get_available_models()}')
parser = argparse.ArgumentParser()
#parser.add_argument('model', type=str, default='wizardLM', choices=['guanaco', 'wizardLM', 'zero_shot', 'vicuna_v1.1', 'dolly', 'oasst_pythia', 'stablelm', 'baize', 'rwkv', 'openbuddy', 'phoenix', 'claude', 'mpt', 'bard', 'billa', 'h2ogpt', 'snoozy', 'manticore', 'falcon_instruct', 'gpt_35', 'gpt_4'],help='select prompting based on modelto load')

template = 'bad'
models = llm.get_available_models()
while template not in models:
   if template.startswith('gpt'):
      break
   template = input('template name? ').strip()
      
#server = OSClient.OSClient(apiKey=None)

def setFormat():
   global FORMAT
   if FORMAT:
      format_button.config(text='RAW')
      FORMAT=False
   else:
      format_button.config(text='FORMAT')
      FORMAT=True


max_tokens = 7144

host = '127.0.0.1'
port = 5004

class ImageDisplay(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_files = ["images/Owl.png", "images/Owl2.png", "images/Owl_as_human.png","images/Owl_as_human2.png",
                            "images/yiCoach.png", "images/jnani2.png"]
        self.current_image_index=0
        # Create layout manager
        layout = QtWidgets.QVBoxLayout()
        self.setWindowTitle('Owl')
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)
        self.label = QtWidgets.QLabel("")
        layout.addWidget(self.label)
        # Load image file
        self.update_image()
        self.resize(240,240)
        self.show()

    def update_image(self):
        """Updates the displayed image."""
        img_path = self.image_files[self.current_image_index]
        pixmap = QtGui.QPixmap(img_path).scaled(360, 360, Qt.KeepAspectRatio)
        rect = QRect(60, 60, 240, 240)
        pixmap = pixmap.copy(rect)
        self.label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        """Handle mouse press events to change the image on left click."""
        if event.button() == Qt.LeftButton:
            # Increment the index. If at the end of the list, go back to the start.
            self.current_image_index = (self.current_image_index + 1) % len(self.image_files)
            # Update the image displayed
            self.update_image()

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


class ChatApp(QtWidgets.QWidget):
   def __init__(self):
      super().__init__()
      global model, template
      self.tts = False
      self.template = template
      self.wolCoT = OwlInnerVoice(self, template = template)
      self.memory_display = None
      self.planner = Planner(self, self.owlCoT)
      #self.interpreter = self.planner.interpreter
      self.windowCloseEvent = self.closeEvent
      signal.signal(signal.SIGINT, self.controlC)
      # Set the background color for the entire window
      self.setAutoFillBackground(True)
      palette = self.palette()
      palette.setColor(self.backgroundRole(), QtGui.QColor("#202020"))  # Use any hex color code
      self.setPalette(palette)
      self.codec = QTextCodec.codecForName("UTF-8")
      self.widgetFont = QFont(); self.widgetFont.setPointSize(14)
      self.reflect = True
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
      control_layout2 = QVBoxLayout()

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
      
      self.prompt_combo = self.make_combo(control_layout, 'Prompt', ["None", "New", "Helpful", "Analytical", "Bhagavan", "ACT", "Owl", "React",])
      self.prompt_combo.setCurrentText('Owl')
      self.prompt_combo.currentIndexChanged.connect(self.on_prompt_combo_changed)
      self.on_prompt_combo_changed('Owl')
      
      
      self.history_button = QPushButton("History") # launch Conversation History editor
      self.history_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.history_button.setFont(self.widgetFont)
      self.history_button.clicked.connect(self.history)
      control_layout.addWidget(self.history_button)
      
      self.wmem_button = QPushButton("WorkingMem") # launch working memory editor
      self.wmem_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.wmem_button.setFont(self.widgetFont)
      self.wmem_button.clicked.connect(self.workingMem)
      control_layout.addWidget(self.wmem_button)
      
      self.tts_button = QPushButton("Speak") # launch working memory editor
      self.tts_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.tts_button.setFont(self.widgetFont)
      self.tts_button.clicked.connect(self.speak)
      control_layout.addWidget(self.tts_button)


      self.create_AWM_button = QPushButton("Create AWM")
      self.create_AWM_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.create_AWM_button.setFont(self.widgetFont)
      self.create_AWM_button.clicked.connect(self.create_AWM)
      control_layout2.addWidget(self.create_AWM_button)
      
      self.recall_WM_button = QPushButton("Recall WM")
      self.recall_WM_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.recall_WM_button.setFont(self.widgetFont)
      self.recall_WM_button.clicked.connect(self.recall_WM)
      control_layout2.addWidget(self.recall_WM_button)
      
      self.edit_AWM_button = QPushButton("Edit AWM")
      self.edit_AWM_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.edit_AWM_button.setFont(self.widgetFont)
      self.edit_AWM_button.clicked.connect(self.edit_AWM)
      control_layout2.addWidget(self.edit_AWM_button)
      
      self.eval_AWM_button = QPushButton("Eval AWM")
      self.eval_AWM_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.eval_AWM_button.setFont(self.widgetFont)
      self.eval_AWM_button.clicked.connect(self.eval_AWM)
      control_layout2.addWidget(self.eval_AWM_button)
      
      self.gc_AWM_button = QPushButton("gc AWM")
      self.gc_AWM_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.gc_AWM_button.setFont(self.widgetFont)
      self.gc_AWM_button.clicked.connect(self.gc_AWM)
      control_layout2.addWidget(self.gc_AWM_button)
      
      self.save_AWM_button = QPushButton("Save AWM")
      self.save_AWM_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.save_AWM_button.setFont(self.widgetFont)
      self.save_AWM_button.clicked.connect(self.save_AWM)
      control_layout2.addWidget(self.save_AWM_button)
      
      spacer = QSpacerItem(0, 20)  # vertical spacer with 20 pixels height
      control_layout2.addItem(spacer)  # Add spacer to the layout

      self.plan_button = QPushButton("Plan")
      self.plan_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.plan_button.setFont(self.widgetFont)
      self.plan_button.clicked.connect(self.plan)
      control_layout2.addWidget(self.plan_button)
      
      self.run_plan_button = QPushButton("Run Plan")
      self.run_plan_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.run_plan_button.setFont(self.widgetFont)
      self.run_plan_button.clicked.connect(self.run_plan)
      control_layout2.addWidget(self.run_plan_button)
      
      self.step_plan_button = QPushButton("Step Plan")
      self.step_plan_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.step_plan_button.setFont(self.widgetFont)
      self.step_plan_button.clicked.connect(self.step_plan)
      control_layout2.addWidget(self.step_plan_button)

      
      control_layout.addStretch(1)  # Add stretch to fill the remaining space
      control_layout2.addStretch(1)  # Add stretch to fill the remaining space
      self.owl = ImageDisplay()
      
      # Add control layout to main layout
      main_layout.addLayout(control_layout)
      main_layout.addLayout(control_layout2)
      self.setLayout(main_layout)
      greeting = self.owlCoT.wakeup_routine()
      self.display_response(greeting+'\n')

   def make_combo(self, control_layout, label, choices, callback=None):
      spacer = QSpacerItem(0, 10)  # Vertical spacer with 20 pixels width
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


   def closeEvent(self, event):
      self.owlCoT.save_conv_history()
      if self.owl is not None:
          self.owl.close()
      event.accept()  # Allow the window to close

   def controlC(self, signum, frame):
      self.owlCoT.save_conv_history()
      if self.owl is not None:
          self.owl.close()
      QApplication.exit()

       
   def on_prompt_combo_changed(self, index):
      global profile, CURRENT_PROFILE_PROMPT_TEXT
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
      self.prompt_area.clear()

   def display_response(self, r):
      global PREV_LEN
      self.input_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
      r = str(r)
      encoded = self.codec.fromUnicode(r)
      # Decode bytes back to string
      decoded = encoded.data().decode('utf-8')
      self.input_area.insertPlainText(decoded)  # Insert the text at the cursor position
      if self.tts:
         try:
            self.speech_service(decoded)
         except:
            traceback.print_exc()
      self.input_area.repaint()
      PREV_LEN=len(self.input_area.toPlainText())-1
      
   def submit(self):
      global PREV_LEN
      self.timer.stop()
      print('timer reset')
      self.timer.start(600000)
      new_text = self.input_area.toPlainText()[PREV_LEN:]
      response = ''
      #print(f'submit {new_text}')
      if profile == 'Owl':
         self.owlCoT.logInput(new_text)
         action = self.owlCoT.action_selection(new_text, self) # this last for async display
         # see if Owl needs to do something before responding to input
         if type(action) == dict and 'tell' in action.keys():
            response = action['tell']+'\n'
            self.display_response(response) 
            return
         if type(action) == dict and 'article' in action.keys():
            #{"article":'<article body>'}
            # get and display article retrieval
            response = '\nArticle summary:\n'+action['article']+'\n'
            self.display_response(response) # article summary text
            #self.run_query('Comments?')
            return
         elif type(action) == dict and 'web' in action.keys():
            #{"web":'<compiled search results>'}
            self.display_response(action['web'])
            #self.run_query('')
            return
         elif type(action) == dict and 'wiki' in action.keys():
            #{"wiki":'<compiled search results>'}
            self.display_response(action['wiki'])
            #self.run_query(input)
            return
         elif type(action) == dict and 'gpt4' in action.keys():
            #{"gpt4":'<gpt4 response>'}
            self.display_response(action['gpt4'])
            #self.run_query(input)
            return
         elif type(action) == dict and 'recall' in action.keys():
            #{"recall":'{"id":id, "key":key, "timestamp":timestamp, "item":text or json or ...}
            self.display_response(action['recall'])
            return
         elif type(action) == dict and 'store' in action.keys():
            #{"store":'<key used to store item>'}
            self.display_response(action['store'])
            #self.run_query(input)
            return
         elif type(action) == dict and 'question' in action.keys():
            #{"ask":'<question>'}
            question = action['question'] # add something to indicate internal activity?
            self.display_response(question)
            return
         else:
            self.display_response(str(action))
            return
         
      # response = self.run_query(new_text)
      # no need to display, query does in while streaming.
      return

   def clear(self):
      global PREV_POS, PREV_LEN
      self.input_area.clear()
      PREV_POS="1.0"
      PREV_LEN=0
   
   def create_AWM(self): # create a new working memory item and put it in active memory
      global PREV_LEN, op#, vmem, vmem_clock
      selectedText = ''
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
         print(f'cursor has selected, len {len(selectedText)}')
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
         selectedText = selectedText.strip()
      print(f'cursor has selected, len {len(selectedText)}')
      self.owlCoT.create_awm(selectedText)
         
   def recall_WM(self): # create a new working memory item and put it in active memory
      selectedText = ''
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
         selectedText = selectedText.strip()
      self.owlCoT.recall_wm(selectedText)
         
   def edit_AWM(self): # edit an active working memory item
      self.owlCoT.edit_awm()
         
   def eval_AWM(self): # edit an active working memory item
      self.planner.interpreter.eval_AWM()
         
   def gc_AWM(self): # release a working memory item from active memory
      self.owlCoT.gc_awm()
         
   def save_AWM(self): # save active working memory items
      self.owlCoT.save_awm()


   def plan(self): # select or create a plan
      self.planner.select_plan()
         
   def run_plan(self): # ask planner to run a plan
      self.planner.run_plan()
         
   def step_plan(self): # release a working memory item from active memory
      self.planner.step_plan()

         
   def workingMem(self): # lauching working memory editor
      self.owlCoT.save_workingMemory() # save current working memory so we can edit it
      he = subprocess.run(['python3', 'memoryEditor.py'])
      if he.returncode == 0:
         try:
            self.workingMemory = self.owlCoT.load_workingMemory()
         except Exception as e:
            self.display_response(f'Failure to reload working memory {str(e)}')

   def speak(self): # lauching working memory editor
      if self.tts:
         self.tts = False
      else:
         self.tts = True

   def speech_service(self, text):
      print("trying to speak")
      r = requests.post("http://bruce-linux:5004/", json={"text":text})

   def history(self):
      self.owlCoT.historyEditor() # save and display Conversation history for editting

   def on_timer_timeout(self):
      global profile, profile_text
      if not self.reflect:
         return
      self.on_prompt_combo_changed(profile) # refresh profile to update date, time, backgound, dreams.
      response = self.owlCoT.reflect()
      #print(f'Reflection response {response}')
      if response is not None and type(response) == dict:
         if 'tell' in response.keys():
            self.display_response(response['tell'])
            self.owlCoT.add_exchange('reflect', response['tell'])
      self.timer.start(600000) # longer timeout when nothing happening
      #print('timer start')

nytimes = nyt.NYTimes()
news, news_details = nytimes.headlines()
print(f'headlines {news}')
app = QtWidgets.QApplication([])
window = ChatApp()
window.show()

if __name__== '__main__':
   app.exec_()


