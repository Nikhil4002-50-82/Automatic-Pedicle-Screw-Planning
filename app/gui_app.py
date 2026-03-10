import sys
import os
import time

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QTextEdit, QHeaderView, QProgressBar, QSplashScreen
)

from PyQt5.QtCore import Qt, QThread, pyqtSignal, QObject
from PyQt5.QtGui import QFont, QPixmap

from run_totalseg import run_totalseg
from mesh_builder import build_vertebra_mesh
from geometryV2 import run_planner
from visualizerV2 import visualize_surgical_plan


# ---------- REAL TIME TERMINAL STREAM ----------

class LogStream(QObject):

    newText = pyqtSignal(str)

    def write(self, text):
        if text.strip():
            self.newText.emit(str(text))

    def flush(self):
        pass


# ---------- WORKER THREAD ----------

class Worker(QThread):

    finished = pyqtSignal(object, object, object)

    def __init__(self, ct):
        super().__init__()
        self.ct = ct

    def run(self):

        segData = run_totalseg(self.ct)
        segFolder = segData["seg_folder"]

        vertsWorld, faces = build_vertebra_mesh(segFolder)

        resultsList = run_planner(segData["combined_seg_path"])

        self.finished.emit(vertsWorld, faces, resultsList)


# ---------- MAIN GUI ----------

class GUI(QWidget):

    def __init__(self):

        super().__init__()

        self.ct = None
        self.verts = None
        self.faces = None
        self.results = None

        self.initUI()

        # Create log stream
        self.stream = LogStream()
        self.stream.newText.connect(self.write)

        # Redirect terminal output
        sys.stdout = self.stream
        sys.stderr = self.stream


    def initUI(self):

        self.setWindowTitle("Pedicle Screw Planning System")
        self.setGeometry(200, 100, 1000, 700)

        layout = QVBoxLayout()

        title = QLabel("Pedicle Screw Planning System")
        title.setFont(QFont("Arial", 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)

        layout.addWidget(title)

        fileLayout = QHBoxLayout()

        self.fileLabel = QLabel("No CT Scan Selected")

        btn = QPushButton("Select CT Scan")
        btn.clicked.connect(self.selectCT)

        fileLayout.addWidget(btn)
        fileLayout.addWidget(self.fileLabel)

        layout.addLayout(fileLayout)

        self.runBtn = QPushButton("Run Full Planning Pipeline")
        self.runBtn.clicked.connect(self.runPipeline)

        layout.addWidget(self.runBtn)

        self.progress = QProgressBar()
        self.progress.setValue(0)

        layout.addWidget(self.progress)

        # ---------- RESULTS TABLE ----------

        self.table = QTableWidget()

        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(
            ["Vertebra", "Side", "Diameter", "Entry", "Tip"]
        )

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        layout.addWidget(self.table)

        # ---------- VISUALIZE BUTTON ----------

        self.visualBtn = QPushButton("Visualize 3D")
        self.visualBtn.clicked.connect(self.visualize)

        layout.addWidget(self.visualBtn)

        # ---------- LOG BOX ----------

        self.logBox = QTextEdit()
        self.logBox.setReadOnly(True)

        layout.addWidget(self.logBox)

        self.setLayout(layout)

        # ---------- STYLE ----------

        self.setStyleSheet("""
        QWidget{
            background:#f6f8fa;
            font-size:14px;
        }

        QPushButton{
            background:#2d89ef;
            color:white;
            border-radius:6px;
            padding:8px;
            font-weight:bold;
        }

        QPushButton:hover{
            background:#1b5fbf;
        }

        QTableWidget{
            background:white;
        }

        QTextEdit{
            background:black;
            color:#00ff9c;
            font-family:Consolas;
        }
        """)


    # ---------- LOG WRITER ----------

    def write(self, text):

        cursor = self.logBox.textCursor()
        cursor.movePosition(cursor.End)
        cursor.insertText(text)
        self.logBox.setTextCursor(cursor)
        self.logBox.ensureCursorVisible()


    # ---------- SELECT CT ----------

    def selectCT(self):

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select CT",
            "",
            "NIfTI Files (*.nii *.nii.gz)"
        )

        if path:

            self.ct = path
            self.fileLabel.setText(os.path.basename(path))

            print("CT selected:", path)


    # ---------- RUN PIPELINE ----------

    def runPipeline(self):

        if not self.ct:

            print("Select CT first")
            return

        self.progress.setValue(0)

        self.worker = Worker(self.ct)

        self.worker.finished.connect(self.finishPipeline)

        self.worker.start()


    # ---------- FINISH PIPELINE ----------

    def finishPipeline(self, v, f, r):

        self.verts = v
        self.faces = f
        self.results = r

        self.table.setRowCount(len(r))

        for i, x in enumerate(r):

            self.table.setItem(i, 0, QTableWidgetItem(x["vertebra"]))
            self.table.setItem(i, 1, QTableWidgetItem(x["side"]))
            self.table.setItem(i, 2, QTableWidgetItem(str(x["diameter"])))
            self.table.setItem(i, 3, QTableWidgetItem(str(x["entry"])))
            self.table.setItem(i, 4, QTableWidgetItem(str(x["tip"])))

        print("Planning completed")


    # ---------- VISUALIZATION ----------

    def visualize(self):

        if self.results:

            visualize_surgical_plan(
                self.verts,
                self.faces,
                self.results
            )


# ---------- MAIN ----------

if __name__ == "__main__":

    app = QApplication(sys.argv)

    pix = QPixmap(400, 200)
    pix.fill(Qt.black)

    splash = QSplashScreen(pix)

    splash.showMessage(
        "Launching Pedicle Screw Planner...",
        Qt.AlignCenter,
        Qt.white
    )

    splash.show()

    time.sleep(2)

    window = GUI()
    window.show()

    splash.finish(window)

    sys.exit(app.exec_())