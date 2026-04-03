import sys
import os
import time
from datetime import datetime

# 🔥 CRITICAL: Must come BEFORE any QApplication usage
from PyQt6.QtCore import Qt
from PyQt6 import QtCore

QtCore.QCoreApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts)

# 🔥 Force WebEngine initialization early
from PyQt6.QtWebEngineWidgets import QWebEngineView

# ---------------- UI Imports ----------------
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QTextEdit, QHeaderView, QSplashScreen
)

from PyQt6.QtCore import QThread, pyqtSignal, QObject
from PyQt6.QtGui import QFont, QPixmap, QColor

# Importing your existing modules (Unchanged)
from run_totalseg import run_totalseg
from mesh_builder import build_vertebra_mesh
from geometryV4 import run_planner, loadNifti, getValidLabels, computeStableFrame, computeDistance, pedicleCenters, optimize
from visualizerV5 import visualize_surgical_plan

# ---------- REAL TIME TERMINAL STREAM ----------
class LogStream(QObject):
    newText = pyqtSignal(str)

    def write(self, text):
        if text.strip():
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_text = f"[{timestamp}] {text.strip()}"
            self.newText.emit(formatted_text)

    def flush(self):
        pass

# ---------- PIPELINE WORKER THREAD ----------
class Worker(QThread):
    screw_found = pyqtSignal(dict)
    finished = pyqtSignal(object, object, list)

    def __init__(self, ct_path):
        super().__init__()
        self.ct_path = ct_path

    def run(self):
        print("INITIATING: Segmentation Pipeline...")
        segData = run_totalseg(self.ct_path)
        segFolder = segData["seg_folder"]
        combined_path = segData["combined_seg_path"]

        print("PROCESSING: Generating 3D Mesh Surfaces...")
        vertsWorld, faces = build_vertebra_mesh(segFolder)

        print("PLANNING: Calculating Optimal Trajectories...")
        resultsList = []
        labelMap = {5:"L1", 4:"L2", 3:"L3", 2:"L4", 1:"L5"}
        
        seg, spacing, affine = loadNifti(combined_path)
        validSegments = getValidLabels(seg)
        
        for labelVal, mask in sorted(validSegments, reverse=True):
            name = labelMap.get(labelVal, str(labelVal))
            centroid, axes, totalDepth = computeStableFrame(mask, affine)
            dist = computeDistance(mask, spacing)
            maskFloat = mask.astype(float)
            lCenter, rCenter = pedicleCenters(mask, dist, centroid, axes, affine)
            
            for side, center in [("Left", lCenter), ("Right", rCenter)]:
                res = optimize(center, axes, side, maskFloat, dist, affine, centroid, totalDepth, name)
                if res:
                    score, entry, tip, length, minDT, lrAng, siAng, diam = res
                    screw_data = {
                        "vertebra": name, "side": side, "entry": entry, "tip": tip,
                        "diameter": diam, "length": length, "axial": lrAng, "sagittal": siAng
                    }
                    resultsList.append(screw_data)
                    self.screw_found.emit(screw_data)
                    print(f"SUCCESS: Trajectory Found for {name} ({side})")

        self.finished.emit(vertsWorld, faces, resultsList)

# ---------- MAIN GUI V2 (PyQt6 Port) ----------
class GUI(QWidget):
    def __init__(self):
        super().__init__()
        self.ct = None
        self.verts = None
        self.faces = None
        self.results = []
        self.viz_thread = None 
        self.initUI()

        self.stream = LogStream()
        self.stream.newText.connect(self.updateLog)
        sys.stdout = self.stream
        sys.stderr = self.stream

    def initUI(self):
        self.setWindowTitle("Automatic Pedicle Screw Planning System - V5.0")
        self.setGeometry(100, 100, 1200, 800)

        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        title = QLabel("AUTOMATIC PEDICLE SCREW PLANNING SYSTEM")
        title.setStyleSheet("font-size: 26px; font-weight: bold; color: #2c3e50; letter-spacing: 2px;")
        # Qt.AlignCenter -> Qt.AlignmentFlag.AlignCenter
        title.setAlignment(Qt.AlignmentFlag.AlignCenter) 
        main_layout.addWidget(title)

        ctrl_layout = QHBoxLayout()
        self.fileLabel = QLabel("DATASET: NOT SELECTED")
        self.fileLabel.setStyleSheet("color: #7f8c8d; font-family: 'Segoe UI';")
        
        select_btn = QPushButton("SELECT CT SCAN")
        select_btn.clicked.connect(self.selectCT)
        
        self.runBtn = QPushButton("EXECUTE PLANNING PIPELINE")
        self.runBtn.clicked.connect(self.runPipeline)
        self.runBtn.setStyleSheet("background-color: #27ae60;")

        ctrl_layout.addWidget(select_btn)
        ctrl_layout.addWidget(self.fileLabel)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(self.runBtn)
        main_layout.addLayout(ctrl_layout)

        self.table = QTableWidget()
        headers = ["VERTEBRA", "SIDE", "DIAMETER (mm)", "LENGTH (mm)", "AXIAL ∠", "SAGITTAL ∠", "ENTRY POINT"]
        self.table.setColumnCount(len(headers))
        self.table.setHorizontalHeaderLabels(headers)
        # QHeaderView.Stretch -> QHeaderView.ResizeMode.Stretch
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setAlternatingRowColors(True)
        main_layout.addWidget(self.table)

        self.visualBtn = QPushButton("LAUNCH 3D SURGICAL VIEW")
        self.visualBtn.clicked.connect(self.visualize)
        self.visualBtn.setEnabled(False)
        main_layout.addWidget(self.visualBtn)

        console_label = QLabel("SYSTEM CONSOLE")
        console_label.setStyleSheet("font-weight: bold; color: #34495e; font-size: 12px;")
        main_layout.addWidget(console_label)
        
        self.logBox = QTextEdit()
        self.logBox.setReadOnly(True)
        self.logBox.setFixedHeight(180)
        self.logBox.setStyleSheet("""
            background-color: #1e1e1e;
            color: #dcdcdc;
            font-family: 'Consolas';
            font-size: 12px;
            border-radius: 4px;
            padding: 10px;
        """)
        main_layout.addWidget(self.logBox)

        self.setLayout(main_layout)

        self.setStyleSheet("""
            QWidget { 
                background-color: #f5f6fa; 
                font-family: 'Segoe UI'; 
            }
            
            /* Default Table Style (Black text on White/Light background) */
            QTableWidget { 
                background-color: #ffffff; 
                color: #000000; 
                gridline-color: #dcdde1;
                selection-background-color: #2980b9; /* Blue background on select/hover */
                selection-color: #ffffff;            /* White text on select/hover */
                border: 1px solid #dcdde1;
            }

            /* Ensure the headers stay dark with white text */
            QHeaderView::section { 
                background-color: #34495e; 
                color: white; 
                padding: 8px; 
                font-weight: bold; 
                border: none;
            }

            /* Keep your buttons consistent */
            QPushButton { 
                background-color: #2980b9; 
                color: white; 
                border: none; 
                padding: 12px 24px; 
                border-radius: 4px; 
                font-weight: bold; 
            }
            QPushButton:hover { background-color: #3498db; }
            QPushButton:disabled { background-color: #bdc3c7; }
        """)

    def updateLog(self, text):
        self.logBox.append(text)
        self.logBox.ensureCursorVisible()

    def selectCT(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select CT NIfTI", "", "NIfTI Files (*.nii *.nii.gz)")
        if path:
            self.ct = path
            self.fileLabel.setText(f"DATASET: {os.path.basename(path)}")
            print(f"SYSTEM: Loaded source {os.path.basename(path)}")

    def runPipeline(self):
        if not self.ct:
            print("ERROR: No CT scan selected for processing.")
            return

        self.table.setRowCount(0)
        self.results = []
        self.runBtn.setEnabled(False)
        self.visualBtn.setEnabled(False)

        self.worker = Worker(self.ct)
        self.worker.screw_found.connect(self.addTableRow)
        self.worker.finished.connect(self.finishPipeline)
        self.worker.start()

    def addTableRow(self, data):
        row = self.table.rowCount()
        self.table.insertRow(row)
        entry_str = f"[{data['entry'][0]:.1f}, {data['entry'][1]:.1f}, {data['entry'][2]:.1f}]"
        
        self.table.setItem(row, 0, QTableWidgetItem(data["vertebra"]))
        self.table.setItem(row, 1, QTableWidgetItem(data["side"]))
        self.table.setItem(row, 2, QTableWidgetItem(f"{data['diameter']} mm"))
        self.table.setItem(row, 3, QTableWidgetItem(f"{data['length']:.1f} mm"))
        self.table.setItem(row, 4, QTableWidgetItem(f"{data['axial']:.1f}°"))
        self.table.setItem(row, 5, QTableWidgetItem(f"{data['sagittal']:.1f}°"))
        self.table.setItem(row, 6, QTableWidgetItem(entry_str))

        for col in range(7):
            # Qt.AlignCenter -> Qt.AlignmentFlag.AlignCenter
            self.table.item(row, col).setTextAlignment(Qt.AlignmentFlag.AlignCenter)

    def finishPipeline(self, v, f, r):
        self.verts = v
        self.faces = f
        self.results = r
        self.runBtn.setEnabled(True)
        self.visualBtn.setEnabled(True)
        print("COMPLETED: All anatomical planning tasks finished successfully.")

    def visualize(self):
        if self.results:
            print("SYSTEM: Initializing 3D Surgical Visualization Engine...")

            try:
                fig, show = visualize_surgical_plan(
                    self.verts,
                    self.faces,
                    self.results
                )
                show()   # 🔥 IMPORTANT
            except Exception as e:
                print(f"VISUALIZER ERROR: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    pix = QPixmap(400, 200)
    pix.fill(QColor("#2c3e50"))
    splash = QSplashScreen(pix)
    splash.show()
    
    window = GUI()
    time.sleep(1) 
    window.show()
    splash.finish(window)
    # .exec_() -> .exec()
    sys.exit(app.exec())