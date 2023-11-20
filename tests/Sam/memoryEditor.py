import sys
import pickle
import json
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QPushButton, QWidget, QMessageBox
from PyQt5.QtCore import Qt

def strj(item):
    # convert item to string.
    # if item is dict, use json.dumps
    if type(item) == dict:
        return json.dumps(item)
    else:
        return str(item)
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            self.docHash = {}
            self.metaData = {}
            with open('SamDocHash.pkl', 'rb') as f:
                data = pickle.load(f)
                self.docHash = data['docHash']
        except Exception as e:
            print(f'Failure to load memory {str(e)}')
            sys.exit(-1)

        self.setWindowTitle("Document Display")
        self.setGeometry(300, 300, 800, 600)

        self.tableWidget = QTableWidget(self)
        self.tableWidget.setColumnCount(8)
        self.tableWidget.setHorizontalHeaderLabels(["name", "item", "type", "notes", "key", "id", "timestamp", "Delete"])
        self.tableWidget.setRowCount(len(self.docHash))

        idx = 0
        for id in self.docHash.keys():
            try:
                entry = self.docHash[id]
                #print(f'type {type(entry)}, keys {entry.keys()}')
                if 'name' not in entry:
                    entry['name'] = ''
                self.tableWidget.setItem(idx, 0, QTableWidgetItem(strj(entry['name'])))
                self.tableWidget.setItem(idx, 1, QTableWidgetItem(strj(entry['item'])))
                self.tableWidget.setItem(idx, 2, QTableWidgetItem(strj(entry["type"])))
                if 'notes' not in entry:
                    entry['notes'] = ''
                self.tableWidget.setItem(idx, 3, QTableWidgetItem(strj(entry["notes"])))
                self.tableWidget.setItem(idx, 4, QTableWidgetItem(strj(entry["key"])))
                self.tableWidget.setItem(idx, 5, QTableWidgetItem(strj(entry['id'])))
                self.tableWidget.setItem(idx, 6, QTableWidgetItem(strj(entry['timestamp'])))
                deleteButton = QPushButton("Delete")
                deleteButton.clicked.connect(lambda _, row=idx: self.deleteRow(row))
                self.tableWidget.setCellWidget(idx, 7, deleteButton)
            except Exception as e:
                print(f'{str(e)}\n pblm formatting entry\n {self.docHash[id]}')
            idx += 1

        # Save button
        self.saveButton = QPushButton('Save Document')
        self.saveButton.clicked.connect(self.saveDocument)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.tableWidget)
        layout.addWidget(self.saveButton)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.tableWidget.itemChanged.connect(self.onItemChanged)

    def onItemChanged(self, item):
        row = item.row()
        column = item.column()
        new_value = item.text()
        print(f'onItemChanged {row}, {column}, {new_value}')
        try:
            id = int(self.tableWidget.item(row, 5).text())  # column 5 holds the ID
            #print(id)
            if id is None:
                print (f'cant get id')
                return
            key = self.tableWidget.horizontalHeaderItem(column).text()  # Get the column header to know which field to update
            item = self.docHash[id]
            # Update the dictionary
            print(f'key updated {key}')

            if key == 'item':
                try:
                    new_value = json.loads(new_value)
                except Exception as e:
                    if  item['type'] == 'dict':
                        print(f'item is dict, json parse failed! {str(e)}')
                        return
                    else:
                        item['item'] = new_value
                        return
                item['type'] = 'dict'
                item[key] = new_value
                
            elif key in self.docHash[id]:
                self.docHash[id][key] = new_value
            else:
                QMessageBox.warning(self, 'Warning', f"Key '{key}' not found in docHash")
        except Exception as e:
            QMessageBox.warning(self, 'Warning', f'Error updating item: {str(e)}')

    def deleteRow(self, row):
        buttonReply = QMessageBox.question(self, 'Delete Confirmation', "Are you sure you want to delete this row?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            try:
                id = int(self.tableWidget.item(row, 5).text())
                del self.docHash[id]
            except Exception as e:
                print(f' failed to get id or id not int {str(e)}')
            self.tableWidget.removeRow(row)
           
    def saveDocument(self):
        buttonReply = QMessageBox.question(self, 'Save Confirmation', "Are you sure you want to save the entire document?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if buttonReply == QMessageBox.Yes:
            try:
                with open('SamDocHash.pkl', 'wb') as f:
                    pickle.dump({'docHash': self.docHash}, f)
                QMessageBox.information(self, 'Success', 'Document successfully saved.')
            except Exception as e:
                QMessageBox.warning(self, 'Failure', f'Could not save document: {str(e)}')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
