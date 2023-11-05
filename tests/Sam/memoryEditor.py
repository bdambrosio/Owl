import sys
import pickle
from PyQt5.QtWidgets import QApplication, QMainWindow, QTableWidget, QTableWidgetItem, QVBoxLayout, QPushButton, QWidget, QMessageBox
from PyQt5.QtCore import Qt

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
        self.tableWidget.setColumnCount(6)
        self.tableWidget.setHorizontalHeaderLabels(["ID", "Key", "Item", "Notes", "Timestamp", "Delete"])
        self.tableWidget.setRowCount(len(self.docHash))

        idx = 0
        for id in self.docHash.keys():
           entry = self.docHash[id]
           self.tableWidget.setItem(idx, 0, QTableWidgetItem(str(entry['id'])))
           self.tableWidget.setItem(idx, 1, QTableWidgetItem(str(entry["key"])))
           self.tableWidget.setItem(idx, 2, QTableWidgetItem(str(entry['item'])))
           if 'notes' in entry:
               self.tableWidget.setItem(idx, 3, QTableWidgetItem(str(entry["notes"])))
           else:
               entry['notes'] = ''
               self.tableWidget.setItem(idx, 3, QTableWidgetItem(""))
           self.tableWidget.setItem(idx, 4, QTableWidgetItem(str(entry['timestamp'])))
           deleteButton = QPushButton("Delete")
           deleteButton.clicked.connect(lambda _, row=idx: self.deleteRow(row))
           self.tableWidget.setCellWidget(idx, 5, deleteButton)
           
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

    def deleteRow(self, row):
        buttonReply = QMessageBox.question(self, 'Delete Confirmation', "Are you sure you want to delete this row?", QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        id = int(self.tableWidget.item(row, 0).text())
        if buttonReply == QMessageBox.Yes:
           self.tableWidget.removeRow(row)
           del self.docHash[id]
           
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
