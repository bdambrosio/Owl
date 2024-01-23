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
from OwlCoT import OwlInnerVoice
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

profiles = ["Owl"] # there used to be others...
profile_contexts = {}
profile = "Owl"
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

def setFormat():
   global FORMAT
   if FORMAT:
      format_button.config(text='RAW')
      FORMAT=False
   else:
      format_button.config(text='FORMAT')
      FORMAT=True


max_tokens = 7144

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

class ChatApp(QtWidgets.QWidget):
   def __init__(self):
      super().__init__()
      self.tts = False
      self.owlCoT = OwlInnerVoice(self)
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
      
      self.msg_area = MyTextEdit(self)
      self.msg_area.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
      self.msg_area.setFont(self.widgetFont)
      self.msg_area.setStyleSheet("QTextEdit { background-color: #101820; color: #FAEBD7; }")
      text_layout.addWidget(self.msg_area)
      
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
      
      self.temp_combo = self.make_combo(control_layout, '   Temp', [".01", ".1", ".2", ".4", ".5", ".7", ".9", "1.0"])
      self.temp_combo.setCurrentText('.1')
      
      self.top_p_combo = self.make_combo(control_layout, '  Top_P', [".01", ".1", ".2", ".4", ".5", ".7", ".9", "1.0"])
      self.top_p_combo.setCurrentText('1.0')
      
      self.max_tokens_combo = self.make_combo(control_layout, 'MaxTkns', ["25", "50", "100", "150", "250", "400", "600", "1000", "2000", "4000"])
      self.max_tokens_combo.setCurrentText('600')
      
      #self.prompt_combo = self.make_combo(control_layout, 'Prompt', ["None", "New", "Helpful", "Analytical", "Bhagavan", "ACT", "Owl", "React",])
      #self.prompt_combo.setCurrentText('Owl')
      #self.prompt_combo.currentIndexChanged.connect(self.on_prompt_combo_changed)
      #self.on_prompt_combo_changed('Owl')
      
      
      self.history_button = QPushButton("History") # launch Conversation History editor
      self.history_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.history_button.setFont(self.widgetFont)
      self.history_button.clicked.connect(self.history)
      control_layout.addWidget(self.history_button)
      
      self.wmem_button = QPushButton("WrkMem") # launch working memory editor
      self.wmem_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.wmem_button.setFont(self.widgetFont)
      self.wmem_button.clicked.connect(self.workingMem)
      control_layout.addWidget(self.wmem_button)
      
      self.tts_button = QPushButton("Speak") # launch working memory editor
      self.tts_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.tts_button.setFont(self.widgetFont)
      self.tts_button.clicked.connect(self.speak)
      control_layout.addWidget(self.tts_button)


      label = QLabel("    AWM")
      label.setStyleSheet("QLabel {background-color: #202020; color: #AAAAAA; }")
      label.setFont(self.widgetFont)
      label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)  # Fixed vertical size policy
      control_layout2.addWidget(label)
      
      self.create_awm_button = QPushButton("Create")
      self.create_awm_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.create_awm_button.setFont(self.widgetFont)
      self.create_awm_button.clicked.connect(self.create_awm)
      control_layout2.addWidget(self.create_awm_button)
      
      self.recall_wm_button = QPushButton("Recall")
      self.recall_wm_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.recall_wm_button.setFont(self.widgetFont)
      self.recall_wm_button.clicked.connect(self.recall_wm)
      control_layout2.addWidget(self.recall_wm_button)
      
      self.edit_awm_button = QPushButton("Edit")
      self.edit_awm_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.edit_awm_button.setFont(self.widgetFont)
      self.edit_awm_button.clicked.connect(self.edit_awm)
      control_layout2.addWidget(self.edit_awm_button)
      
      self.save_awm_button = QPushButton("Save")
      self.save_awm_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.save_awm_button.setFont(self.widgetFont)
      self.save_awm_button.clicked.connect(self.save_awm)
      control_layout2.addWidget(self.save_awm_button)
      
      self.eval_awm_button = QPushButton("Eval")
      self.eval_awm_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.eval_awm_button.setFont(self.widgetFont)
      self.eval_awm_button.clicked.connect(self.eval_awm)
      control_layout2.addWidget(self.eval_awm_button)
      
      self.gc_awm_button = QPushButton("GC")
      self.gc_awm_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.gc_awm_button.setFont(self.widgetFont)
      self.gc_awm_button.clicked.connect(self.gc_awm)
      control_layout2.addWidget(self.gc_awm_button)
      
      spacer = QSpacerItem(0, 20)  # vertical spacer with 20 pixels height
      control_layout2.addItem(spacer)  # Add spacer to the layout

      label = QLabel(" Planner")
      label.setStyleSheet("QLabel {background-color: #202020; color: #AAAAAA; }")
      label.setFont(self.widgetFont)
      label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)  # Fixed vertical size policy
      control_layout2.addWidget(label)
      
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

      spacer = QSpacerItem(0, 20)  # vertical spacer with 20 pixels height
      control_layout2.addItem(spacer)  # Add spacer to the layout

      label = QLabel("Library")
      label.setStyleSheet("QLabel {background-color: #202020; color: #AAAAAA; }")
      label.setFont(self.widgetFont)
      label.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)  # Fixed vertical size policy
      control_layout2.addWidget(label)
      
      self.arxiv_button = QPushButton("Search")
      self.arxiv_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.arxiv_button.setFont(self.widgetFont)
      self.arxiv_button.clicked.connect(self.search_s2)
      control_layout2.addWidget(self.arxiv_button)
      
      self.index_button = QPushButton("Index")
      self.index_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.index_button.setFont(self.widgetFont)
      self.index_button.clicked.connect(self.index_url)
      control_layout2.addWidget(self.index_button)
      
      self.report_button = QPushButton("Report")
      self.report_button.setStyleSheet("QPushButton { background-color: #101820; color: #FAEBD7; }")
      self.report_button.setFont(self.widgetFont)
      self.report_button.clicked.connect(self.generate_report)
      control_layout2.addWidget(self.report_button)
      
      
      control_layout.addStretch(1)  # Add stretch to fill the remaining space
      control_layout2.addStretch(1)  # Add stretch to fill the remaining space
      self.owl = ImageDisplay()
      
      # Add control layout to main layout
      main_layout.addLayout(control_layout)
      main_layout.addLayout(control_layout2)
      self.setLayout(main_layout)
      greeting = self.owlCoT.wakeup_routine()
      self.display_response(greeting)

   def make_combo(self, control_layout, label, choices, callback=None):
      spacer = QSpacerItem(0, 10)  # Vertical spacer with 20 pixels width
      control_layout.addItem(spacer)  # Add spacer to the layout
      
      label = QLabel(label)
      label.setStyleSheet("QLabel {background-color: #202020; color: #AAAAAA; }")
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

       
   # do not delete, need this for prompt dynamic content refresh during reflection
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
      #self.prompt_area.clear()

   def display_response(self, r):
      global PREV_LEN
      self.input_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
      r = str(r)
      encoded = self.codec.fromUnicode(r)
      # Decode bytes back to string
      decoded = encoded.data().decode('utf-8')
      if not decoded.endswith('\n'):
         decoded += '\n'
      self.input_area.insertPlainText(decoded)  # Insert the text at the cursor position
      if self.tts:
         try:
            self.speech_service(decoded)
         except:
            traceback.print_exc()
      self.input_area.repaint()
      PREV_LEN=len(self.input_area.toPlainText())-1
      
   def display_msg(self, r):
      self.msg_area.moveCursor(QtGui.QTextCursor.End)  # Move the cursor to the end of the text
      r = str(r)
      # presumably helps handle extended chars
      encoded = self.codec.fromUnicode(r)
      decoded = encoded.data().decode('utf-8')+'\n'
      self.msg_area.insertPlainText(decoded)  # Insert the text at the cursor position
      if self.tts:
         try:
            self.speech_service(decoded)
         except:
            traceback.print_exc()
      self.msg_area.repaint()
      
   def submit(self):
      global PREV_LEN
      self.timer.stop()
      print('timer reset')
      self.timer.start(600000)
      new_text = self.input_area.toPlainText()[PREV_LEN:]
      response = ''
      print(f'submit {new_text}')
      self.owlCoT.logInput(new_text)
      action = self.owlCoT.action_selection(new_text, self) # this last for async display
      # see if Owl needs to do something before responding to input
      if type(action) == dict and 'tell' in action.keys():
         response = str(action['tell'])+'\n'
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
   
   #
   ## Working memory interface
   #
   
   def create_awm(self): # create a new working memory item and put it in active memory
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
         
   def recall_wm(self): # create a new working memory item and put it in active memory
      selectedText = ''
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
         selectedText = selectedText.strip()
      self.owlCoT.recall_wm(selectedText)
         
   def edit_awm(self): # edit an active working memory item
      self.owlCoT.edit_awm()
         
   def eval_awm(self): # edit an active working memory item
      self.planner.interpreter.eval_awm()
         
   def gc_awm(self): # release a working memory item from active memory
      self.owlCoT.gc_awm()
         
   def save_awm(self): # save active working memory items
      self.owlCoT.save_awm()

      #
      ## Planner interface
      #

   def plan(self): # select or create a plan
      self.planner.select_plan()
         
   def run_plan(self): # ask planner to run a plan
      self.planner.run_plan()
         
   def step_plan(self): # release a working memory item from active memory
      self.planner.step_plan()

   #
   ## Semantic memory interface
   #

   def search_s2(self): # release a working memory item from active memory
      selectedText = ''
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
         selectedText = selectedText.strip()
      response,titles = self.owlCoT.s2_search(selectedText)
      self.display_response('\n'+response)
      self.display_msg('\nRefs:\n'+'\n'.join(titles))
      
   def index_url(self): # index a url in S2 faiss
      global PREV_LEN, op#, vmem, vmem_clock
      selectedText = ''
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
         print(f'cursor has selected, len {len(selectedText)}')
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
         selectedText = selectedText.strip()
         print(f'cursor has selected {len(selectedText)} chars')
      start = selectedText.find('http')
      if start < 0:
         self.display_msg(f'not url: {selectedText}')
         return
      if start > 0:
         selectedText = selectedText[start:]
      self.s2.queue_url_for_indexing(selectedText)
      self.display_msg("Indexing request submitted.")

   def generate_report(self): # index a url in S2 faiss
      global PREV_LEN, op#, vmem, vmem_clock
      selectedText = ''
      cursor = self.input_area.textCursor()
      if cursor.hasSelection():
         selectedText = cursor.selectedText()
      elif PREV_LEN < len(self.input_area.toPlainText())+2:
         selectedText = self.input_area.toPlainText()[PREV_LEN:]
         selectedText = selectedText.strip()
      rr = subprocess.Popen(['python3', 'paper_writer.py', '-report', 't'])
      self.display_msg("report writer spawned.")
         
   def workingMem(self): # lauching working memory editor
      self.owlCoT.save_workingMemory() # save current working memory so we can edit it
      he = subprocess.run(['python3', 'memoryEditor.py'])
      if he.returncode == 0:
         try:
            self.workingMemory = self.owlCoT.load_workingMemory()
         except Exception as e:
            self.display_msg(f'Failure to reload working memory\n  {str(e)}')

   def speak(self): # lauching working memory editor
      if self.tts:
         self.tts = False
         self.display_msg('Speech off')
      else:
         self.display_msg('Speech on')
         self.tts = True

   def speech_service(self, text):
      #self.display_msg('speaking...')
      try:
         r = requests.post("http://bruce-linux:5004/", json={"text":text})
      except Exception as e:
         print('\nspeech attempt failed {str(e)}\n')

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
   import semanticScholar2 as s2
   window.s2=s2
   s2.ui=window
   s2.cot=window.owlCoT
   app.exec_()


