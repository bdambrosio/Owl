import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QTableView, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import QAbstractTableModel, Qt

# Reading the parquet file into a pandas DataFrame
# Replace 'your_parquet_file.parquet' with the path to your parquet file
df = pd.read_parquet('arxiv/section_library.parquet')

# Exclude the 'synopsis' column
#df = df.drop(columns=['synopsis'])

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]

    def data(self, index, role=Qt.DisplayRole):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self._data.iloc[index.row(), index.column()])
        return None

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            return self._data.columns[section]
        return None

# Create the application
app = QApplication(sys.argv)
main_widget = QWidget()
main_layout = QVBoxLayout(main_widget)

# Create and configure the table view
table_view = QTableView()
model = PandasModel(df)
table_view.setModel(model)
main_layout.addWidget(table_view)

# Button to remove selected row
remove_row_button = QPushButton("Remove Selected Row")
def remove_selected_row():
    index_list = table_view.selectionModel().selectedRows()
    for index in index_list:
        model.removeRow(index.row())
remove_row_button.clicked.connect(remove_selected_row)
main_layout.addWidget(remove_row_button)

# Button to save the DataFrame
save_button = QPushButton("Save DataFrame")
def save_dataframe():
    model._data.to_parquet('updated_parquet_file.parquet')
save_button.clicked.connect(save_dataframe)
main_layout.addWidget(save_button)

# Set main layout and show the main widget
main_widget.setLayout(main_layout)
main_widget.show()

# Execute the application
sys.exit(app.exec_())
