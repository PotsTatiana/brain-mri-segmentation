import sys
import torch
from PyQt5.QtWidgets import QApplication
from PyQt5.QtGui import QPalette, QColor
from PyQt5.QtCore import Qt
from model_loader import load_model
from predictor import Predictor
from gui import MRIViewer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = load_model("best_model.pth", device)
predictor = Predictor(model, device)

app = QApplication(sys.argv)

app.setStyleSheet("""
QPushButton { padding:8px; font-size:14px; }
QLabel { font-size:13px; }
""")
app.setStyle("Fusion")
dark_palette = QPalette()
dark_palette.setColor(QPalette.Window, QColor(45,45,45))
dark_palette.setColor(QPalette.WindowText, Qt.white)
dark_palette.setColor(QPalette.Base, QColor(30,30,30))
dark_palette.setColor(QPalette.Text, Qt.white)
dark_palette.setColor(QPalette.Button, QColor(50,50,50))
dark_palette.setColor(QPalette.ButtonText, Qt.white)
app.setPalette(dark_palette)

window = MRIViewer(predictor)
window.resize(1200, 900)
window.show()

sys.exit(app.exec_())