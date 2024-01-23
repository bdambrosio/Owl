import sys
import pandas as pd
from PyQt5.QtWidgets import QApplication, QTableView, QVBoxLayout, QPushButton, QWidget
from PyQt5.QtCore import QAbstractTableModel, Qt, QModelIndex

paper_library_filepath = 'arxiv/paper_library.parquet'
# Read the parquet file into a pandas DataFrame
df = pd.read_parquet(paper_library_filepath)
#df = df.drop(columns=['synopsis'])

class PandasModel(QAbstractTableModel):
    def __init__(self, data):
        QAbstractTableModel.__init__(self)
        self._data = data
        self.hidden_columns = []

    def rowCount(self, parent=None):
        return self._data.shape[0]

    def columnCount(self, parent=None):
        return self._data.shape[1]-len(self.hidden_columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role != Qt.DisplayRole:
            return None
        # Adjust column index to account for hidden columns
        column = self.visibleColumn(index.column())
        return str(self._data.iloc[index.row(), column])

    def headerData(self, section, orientation, role):
        if orientation == Qt.Horizontal and role == Qt.DisplayRole:
            # Adjust section index to account for hidden columns
            section = self.visibleColumn(section)
            return self._data.columns[section]
        return None

    def visibleColumn(self, section):
        # Adjust the column index to skip hidden columns
        for hidden_col in self.hidden_columns:
            hidden_index = self._data.columns.get_loc(hidden_col)
            if section >= hidden_index:
                section += 1
        return section

    def removeRow(self, row, parent=None):
        self.beginRemoveRows(QModelIndex(), row, row)
        self._data = self._data.drop(self._data.index[row])
        self.endRemoveRows()
        return True

# Create the application
app = QApplication(sys.argv)
main_widget = QWidget()
main_layout = QVBoxLayout(main_widget)

table_view = QTableView()
model = PandasModel(df)
table_view.setModel(model)
main_layout.addWidget(table_view)
# Set the selection behavior to select entire rows
table_view.setSelectionBehavior(QTableView.SelectRows)

remove_row_button = QPushButton("Remove Selected Row")
def remove_selected_row():
    model.layoutAboutToBeChanged.emit()  # Notify the view to prepare for layout changes
    index_list = table_view.selectionModel().selectedRows()
    rows = sorted(set(index.row() for index in index_list), reverse=True)
    for row in rows:
        model.removeRow(row)
    model.layoutChanged.emit()  # Notify the view that layout changes are done
remove_row_button.clicked.connect(remove_selected_row)
main_layout.addWidget(remove_row_button)

save_button = QPushButton("Save DataFrame")
def save_dataframe():
    model._data.to_parquet(paper_library_filepath)
save_button.clicked.connect(save_dataframe)
main_layout.addWidget(save_button)

main_widget.setLayout(main_layout)
main_widget.show()
sys.exit(app.exec_())
