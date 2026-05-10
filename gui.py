import torch
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import Qt, pyqtSignal
import numpy as np
import cv2
import os
import tempfile
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from datetime import datetime

from hdf5_loader import load_hdf5
from graphics_view import MedicalImageView

# ---------- 3D визуализация ----------
try:
    from visualization_3d import visualize_tumor_3d
    VEDO_AVAILABLE = True
except ImportError:
    VEDO_AVAILABLE = False
    print("Библиотека vedo не найдена. 3D визуализация недоступна.")

# ---------- Постобработка маски (удаление мелких шумов) ----------
def light_postprocess_mask(mask: np.ndarray, min_area=30) -> np.ndarray:
    if mask.sum() == 0:
        return mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            mask[labels == i] = 0
    return mask

# ---------- Данные пациента ----------
class PatientData:
    def __init__(self, filepath, images, masks_gt=None):
        self.filepath = filepath
        self.filename = os.path.basename(filepath)
        self.images = images
        self.masks_gt = masks_gt[:, 0, :, :] if masks_gt is not None else None
        self.masks_pred = None
        self.masks_edited = None
        self.history = []
        self.history_index = -1
        self.current_slice = 0
        self.spacing = (3.0, 1.0, 1.0)

    def reset_history(self):
        self.history = []
        self.history_index = -1
        if self.masks_edited is not None:
            self.push_to_history()

    def push_to_history(self):
        if self.masks_edited is None:
            return
        self.history = self.history[:self.history_index+1]
        self.history.append(self.masks_edited.copy())
        self.history_index = len(self.history) - 1

    def undo(self):
        if self.history_index > 0:
            self.history_index -= 1
            self.masks_edited = self.history[self.history_index].copy()
            return True
        return False

    def redo(self):
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
            self.masks_edited = self.history[self.history_index].copy()
            return True
        return False

# ---------- Главное окно приложения ----------
class MRIViewer(QWidget):
    def __init__(self, predictor):
        super().__init__()
        self.predictor = predictor
        self.device = predictor.device

        self.patients = []
        self.current_patient_index = -1
        self.threshold = 0.65
        self.idx = 0

        # Флаг активности рисования (включено только когда нажата «Рисовать» или «Стирать»)
        self.drawing_active = False

        self.setWindowTitle("ИИ сегментация МРТ мозга - несколько пациентов")
        self.setup_ui()

    # ----------------------- Инициализация интерфейса -----------------------
    def setup_ui(self):
        # Виджеты визуализации
        self.view_mri = MedicalImageView()
        self.view_gt = MedicalImageView()
        self.view_pred = MedicalImageView()
        self.view_overlay = MedicalImageView()

        self.view_mri.mouse_coords_signal.connect(self.update_coords_status)
        self.view_gt.mouse_coords_signal.connect(self.update_coords_status)
        self.view_pred.mouse_coords_signal.connect(self.update_coords_status)
        self.view_overlay.mouse_coords_signal.connect(self.update_coords_status)

        # Список пациентов
        self.patient_list = QListWidget()
        self.patient_list.setMaximumWidth(200)
        self.patient_list.itemClicked.connect(self.on_patient_selected)

        btn_add = QPushButton("+ Добавить HDF5")
        btn_add.clicked.connect(self.add_patient)
        btn_remove = QPushButton("Удалить текущего")
        btn_remove.clicked.connect(self.remove_current_patient)

        patient_layout = QVBoxLayout()
        patient_layout.addWidget(QLabel("Пациенты / исследования"))
        patient_layout.addWidget(self.patient_list)
        patient_layout.addWidget(btn_add)
        patient_layout.addWidget(btn_remove)

        # Элементы навигации
        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.update_slice)
        self.channel_selector = QComboBox()
        self.channel_selector.addItems(["MRI Channel 0", "MRI Channel 1"])
        self.channel_selector.currentIndexChanged.connect(self.update_slice)

        # Порог
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(0)
        self.threshold_slider.setMaximum(100)
        self.threshold_slider.setValue(65)
        self.threshold_slider.valueChanged.connect(self.on_threshold_changed)

        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.00, 1.00)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setValue(0.65)
        self.threshold_spin.valueChanged.connect(self.on_threshold_spin_changed)

        # Панель рисования
        self.btn_cursor = QPushButton("Курсор")
        self.btn_cursor.setCheckable(True)
        self.btn_cursor.setChecked(True)
        self.btn_cursor.clicked.connect(lambda: self.set_active_mode(None))
        self.btn_draw = QPushButton("Рисовать")
        self.btn_draw.setCheckable(True)
        self.btn_draw.clicked.connect(lambda: self.set_active_mode('draw'))
        self.btn_erase = QPushButton("Стирать")
        self.btn_erase.setCheckable(True)
        self.btn_erase.clicked.connect(lambda: self.set_active_mode('erase'))

        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(20)
        self.brush_slider.setValue(1)
        self.brush_slider.setToolTip("Размер кисти (пикселей)")
        self.brush_slider.valueChanged.connect(self.set_brush_size)

        self.btn_export = QPushButton("Экспортировать маски для обучения")
        self.btn_export.clicked.connect(self.export_masks_for_training)

        # Информационные метки
        self.dice_label = QLabel("Коэффициент Dice: -")
        self.tumor_label = QLabel("Размер патологического процесса: -")
        self.coords_label = QLabel("x: -  y: -  intensity: -")

        # Кнопки действий
        run_btn = QPushButton("Применить сегментацию (текущий срез)")
        run_btn.clicked.connect(self.run_ai_current_slice)
        report_btn = QPushButton("Сохранить PDF отчет")
        report_btn.clicked.connect(self.save_report)
        undo_btn = QPushButton("Отменить (Undo)")
        undo_btn.clicked.connect(self.undo)
        redo_btn = QPushButton("Вернуть (Redo)")
        redo_btn.clicked.connect(self.redo)

        self.btn_3d = QPushButton("3D визуализация патологического процесса")
        self.btn_3d.clicked.connect(self.show_3d_tumor)
        if not VEDO_AVAILABLE:
            self.btn_3d.setEnabled(False)
            self.btn_3d.setToolTip("Установите vedo: pip install vedo")

        # Layout левой панели
        left_panel = QVBoxLayout()
        left_panel.addLayout(patient_layout)
        left_panel.addWidget(QLabel("MRI канал"))
        left_panel.addWidget(self.channel_selector)
        left_panel.addWidget(QLabel("Выбор среза"))
        left_panel.addWidget(self.slider)
        left_panel.addWidget(QLabel("Порог сегментации"))
        thr_layout = QHBoxLayout()
        thr_layout.addWidget(self.threshold_slider)
        thr_layout.addWidget(self.threshold_spin)
        left_panel.addLayout(thr_layout)

        left_panel.addWidget(QLabel("Инструменты:"))
        tool_layout = QHBoxLayout()
        tool_layout.addWidget(self.btn_cursor)
        tool_layout.addWidget(self.btn_draw)
        tool_layout.addWidget(self.btn_erase)
        left_panel.addLayout(tool_layout)
        left_panel.addWidget(QLabel("Размер кисти (квадратная):"))
        left_panel.addWidget(self.brush_slider)
        left_panel.addWidget(self.btn_export)

        left_panel.addWidget(run_btn)
        left_panel.addWidget(report_btn)
        left_panel.addWidget(undo_btn)
        left_panel.addWidget(redo_btn)
        left_panel.addWidget(self.btn_3d)
        left_panel.addStretch()
        left_panel.addWidget(self.dice_label)
        left_panel.addWidget(self.tumor_label)
        left_panel.addWidget(self.coords_label)

        # Сетка изображений
        grid = QGridLayout()
        grid.addWidget(QLabel("MRI"), 0, 0)
        grid.addWidget(QLabel("Истинная маска"), 0, 1)
        grid.addWidget(QLabel("Предсказание ИИ"), 2, 0)
        grid.addWidget(QLabel("Overlay патологического процесса"), 2, 1)
        grid.addWidget(self.view_mri, 1, 0)
        grid.addWidget(self.view_gt, 1, 1)
        grid.addWidget(self.view_pred, 3, 0)
        grid.addWidget(self.view_overlay, 3, 1)

        main_layout = QHBoxLayout()
        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(grid, 3)
        self.setLayout(main_layout)

        # Инициализация
        self.set_brush_size(self.brush_slider.value())
        self.set_active_mode(None)  # режим курсора (без рисования)
        self.view_overlay.drawing_started.connect(self.on_drawing_started)
        self.view_overlay.drawing_moved.connect(self.on_drawing_moved)

    # ----------------------- Управление режимами рисования -----------------------
    def set_active_mode(self, mode):
        # Обновляем активную кнопку
        self.btn_cursor.setChecked(mode is None)
        self.btn_draw.setChecked(mode == 'draw')
        self.btn_erase.setChecked(mode == 'erase')
        # Включаем/выключаем возможность рисования
        self.drawing_active = (mode is not None)
        # Устанавливаем режим для вьюх (для корректной отрисовки кисти)
        if mode == 'draw':
            self.view_overlay.set_brush_mode('draw')
            self.view_pred.set_brush_mode('draw')
        elif mode == 'erase':
            self.view_overlay.set_brush_mode('erase')
            self.view_pred.set_brush_mode('erase')
        else:
            self.view_overlay.set_brush_mode('none')
            self.view_pred.set_brush_mode('none')

    # ----------------------- Квадратная кисть -----------------------
    def set_brush_size(self, size):
        self.view_overlay.set_brush_size(size)
        self.view_pred.set_brush_size(size)

    def on_drawing_started(self, pos):
        if not self.drawing_active:
            return
        self.on_drawing_moved(pos, pos)

    def on_drawing_moved(self, last_pos, new_pos):
        if not self.drawing_active:
            return
        if self.current_patient_index < 0:
            return
        patient = self.patients[self.current_patient_index]
        if patient.masks_edited is None:
            return

        mask = patient.masks_edited[self.idx].copy()
        h, w = mask.shape
        view = self.view_overlay

        def scene_to_pixel(pos):
            rect = view.pixmap_item.boundingRect()
            if not rect.contains(pos):
                return None
            x_ratio = (pos.x() - rect.x()) / rect.width()
            y_ratio = (pos.y() - rect.y()) / rect.height()
            px = int(x_ratio * w)
            py = int(y_ratio * h)
            if 0 <= px < w and 0 <= py < h:
                return (px, py)
            return None

        p1 = scene_to_pixel(last_pos)
        p2 = scene_to_pixel(new_pos)
        if p1 is None or p2 is None:
            return

        brush_size = view.brush_size
        mode = view.brush_mode

        points = self.bresenham_line(p1[0], p1[1], p2[0], p2[1])
        half = brush_size // 2
        for (cx, cy) in points:
            for dx in range(-half, half + 1):
                for dy in range(-half, half + 1):
                    x = cx + dx
                    y = cy + dy
                    if 0 <= x < w and 0 <= y < h:
                        if mode == 'draw':
                            mask[y, x] = 1
                        elif mode == 'erase':
                            mask[y, x] = 0

        patient.masks_edited[self.idx] = mask
        patient.push_to_history()
        self.refresh_display_only()

    @staticmethod
    def bresenham_line(x0, y0, x1, y1):
        points = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            points.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return points

    def export_masks_for_training(self):
        if self.current_patient_index < 0:
            QMessageBox.warning(self, "Ошибка", "Нет выбранного пациента")
            return
        patient = self.patients[self.current_patient_index]
        if patient.masks_edited is None:
            QMessageBox.warning(self, "Ошибка", "Нет отредактированных масок. Выполните сегментацию.")
            return
        folder = QFileDialog.getExistingDirectory(self, "Выберите папку для сохранения")
        if not folder:
            return
        try:
            count = 0
            for i in range(patient.images.shape[0]):
                if patient.masks_edited[i].sum() > 0:
                    np.save(os.path.join(folder, f"img_{i}_ch0.npy"), patient.images[i,0])
                    np.save(os.path.join(folder, f"img_{i}_ch1.npy"), patient.images[i,1])
                    np.save(os.path.join(folder, f"mask_{i}.npy"), patient.masks_edited[i])
                    count += 1
            QMessageBox.information(self, "Экспорт", f"Сохранено {count} срезов (только с патологией) в папку\n{folder}")
        except Exception as e:
            QMessageBox.critical(self, "Ошибка", str(e))

    # ----------------------- Загрузка и управление пациентами -----------------------
    def add_patient(self):
        path, _ = QFileDialog.getOpenFileName(self, "Выберите HDF5")
        if not path:
            return
        try:
            images, masks = load_hdf5(path)
            if images.ndim != 4 or images.shape[1] != 2:
                QMessageBox.warning(self, "Ошибка", "Файл должен иметь форму (N,2,H,W)")
                return
            patient = PatientData(path, images, masks)
            self.patients.append(patient)
            self.patient_list.addItem(patient.filename)
            self.patient_list.setCurrentRow(len(self.patients)-1)
            self.switch_to_patient(len(self.patients)-1)
            self.slider.setEnabled(True)
            self.channel_selector.setEnabled(True)
            self.threshold_slider.setEnabled(True)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка загрузки", str(e))

    def remove_current_patient(self):
        if self.current_patient_index < 0:
            return
        reply = QMessageBox.question(self, "Удаление", "Удалить этого пациента? Данные будут потеряны.",
                                     QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.patients.pop(self.current_patient_index)
            self.patient_list.takeItem(self.current_patient_index)
            if self.patients:
                new_idx = min(self.current_patient_index, len(self.patients)-1)
                self.patient_list.setCurrentRow(new_idx)
                self.switch_to_patient(new_idx)
            else:
                self.current_patient_index = -1
                self.clear_display()
                self.slider.setEnabled(False)
                self.channel_selector.setEnabled(False)
                self.threshold_slider.setEnabled(False)

    def on_patient_selected(self, item):
        idx = self.patient_list.row(item)
        self.switch_to_patient(idx)

    def switch_to_patient(self, idx):
        if idx < 0 or idx >= len(self.patients):
            return
        if self.current_patient_index >= 0:
            old_patient = self.patients[self.current_patient_index]
            old_patient.current_slice = self.slider.value()
        self.current_patient_index = idx
        patient = self.patients[idx]
        self.slider.setMaximum(patient.images.shape[0] - 1)
        self.slider.setValue(patient.current_slice)
        self.idx = patient.current_slice
        self.update_slice()

    def clear_display(self):
        blank = np.zeros((256, 256), dtype=np.uint8)
        self.view_mri.set_image(blank)
        self.view_gt.set_image(blank)
        self.view_pred.set_image(blank)
        self.view_overlay.set_image(blank)
        self.dice_label.setText("Коэффициент Dice: -")
        self.tumor_label.setText("Размер патологического процесса: -")

    # ----------------------- Сегментация текущего среза -----------------------
    def run_ai_current_slice(self):
        if self.current_patient_index < 0:
            QMessageBox.warning(self, "Ошибка", "Сначала добавьте пациента")
            return
        patient = self.patients[self.current_patient_index]
        if patient.images is None:
            return

        img_ch0 = patient.images[self.idx, 0]
        img_ch1 = patient.images[self.idx, 1]
        img_tensor = torch.from_numpy(np.stack([img_ch0, img_ch1], axis=0)).float()
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.predictor.model(img_tensor)
            pred = torch.sigmoid(output).cpu().numpy()[0, 0]

        max_prob = pred.max()
        print(f"Срез {self.idx}, max вероятность: {max_prob:.4f}, порог: {self.threshold}")

        binary = (pred > self.threshold).astype(np.uint8)
        print(f"  Пикселей до постобработки: {binary.sum()}")
        binary = light_postprocess_mask(binary, min_area=30)
        print(f"  Пикселей после постобработки: {binary.sum()}")

        if patient.masks_pred is None:
            patient.masks_pred = np.zeros((patient.images.shape[0], *binary.shape), dtype=np.uint8)
        patient.masks_pred[self.idx] = binary
        if patient.masks_edited is None:
            patient.masks_edited = patient.masks_pred.copy()
        else:
            patient.masks_edited[self.idx] = binary
        patient.push_to_history()
        self.update_slice()

    # ----------------------- Обновление отображения -----------------------
    def update_slice(self):
        if self.current_patient_index < 0:
            return
        patient = self.patients[self.current_patient_index]
        if patient.images is None:
            return

        self.idx = self.slider.value()
        patient.current_slice = self.idx

        ch = self.channel_selector.currentIndex()
        img = patient.images[self.idx, ch]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        self.view_mri.set_image(img_norm)
        self.view_gt.set_image(patient.masks_gt[self.idx] if patient.masks_gt is not None else np.zeros_like(img_norm))

        if patient.masks_edited is not None:
            pred = patient.masks_edited[self.idx]
            overlay = self.create_overlay(img_norm, pred)
            self.view_pred.set_image(pred)
            self.view_overlay.set_image(overlay)

            tumor_pixels = pred.sum()
            if tumor_pixels == 0:
                self.tumor_label.setText("Патологический процесс не обнаружен")
                self.dice_label.setText("Коэффициент Dice: -")
            else:
                voxel_vol = patient.spacing[0] * patient.spacing[1] * patient.spacing[2]
                volume = tumor_pixels * voxel_vol
                self.tumor_label.setText(f"Размер патологического процесса: {tumor_pixels} пикселей | {volume:.2f} мм³")
                if patient.masks_gt is not None:
                    dice = self.compute_dice(patient.masks_gt[self.idx], pred)
                    self.dice_label.setText(f"Коэффициент Dice: {dice:.3f}")
                else:
                    self.dice_label.setText("Коэффициент Dice: нет эталона")
        else:
            self.view_pred.set_image(np.zeros_like(img_norm))
            self.view_overlay.set_image(np.zeros_like(img_norm))

    def refresh_display_only(self):
        if self.current_patient_index < 0:
            return
        patient = self.patients[self.current_patient_index]
        if patient.masks_edited is None:
            return
        ch = self.channel_selector.currentIndex()
        img = patient.images[self.idx, ch]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        pred = patient.masks_edited[self.idx]
        overlay = self.create_overlay(img_norm, pred)
        self.view_pred.set_image_keep_transform(pred)
        self.view_overlay.set_image_keep_transform(overlay)

        tumor_pixels = pred.sum()
        if tumor_pixels == 0:
            self.tumor_label.setText("Патологический процесс не обнаружен")
            self.dice_label.setText("Коэффициент Dice: -")
        else:
            voxel_vol = patient.spacing[0] * patient.spacing[1] * patient.spacing[2]
            volume = tumor_pixels * voxel_vol
            self.tumor_label.setText(f"Размер патологического процесса: {tumor_pixels} пикселей | {volume:.2f} мм³")
            if patient.masks_gt is not None:
                dice = self.compute_dice(patient.masks_gt[self.idx], pred)
                self.dice_label.setText(f"Коэффициент Dice: {dice:.3f}")
            else:
                self.dice_label.setText("Коэффициент Dice: нет эталона")

    def create_overlay(self, img_norm, mask):
        overlay = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        red = np.zeros_like(overlay)
        red[:, :, 2] = (mask * 255).astype(np.uint8)
        return cv2.addWeighted(overlay, 1.0, red, 0.5, 0)

    # ----------------------- Вспомогательные методы -----------------------
    def on_threshold_changed(self, val):
        self.threshold = val / 100.0
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setValue(self.threshold)
        self.threshold_spin.blockSignals(False)

    def on_threshold_spin_changed(self, val):
        self.threshold = val
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(int(val * 100))
        self.threshold_slider.blockSignals(False)

    def undo(self):
        if self.current_patient_index < 0:
            return
        patient = self.patients[self.current_patient_index]
        if patient.undo():
            self.update_slice()

    def redo(self):
        if self.current_patient_index < 0:
            return
        patient = self.patients[self.current_patient_index]
        if patient.redo():
            self.update_slice()

    def show_3d_tumor(self):
        if not VEDO_AVAILABLE:
            QMessageBox.warning(self, "Ошибка", "Установите vedo: pip install vedo")
            return
        if self.current_patient_index < 0:
            QMessageBox.warning(self, "Ошибка", "Выберите пациента")
            return
        patient = self.patients[self.current_patient_index]
        if patient.masks_edited is None or patient.masks_edited.sum() == 0:
            QMessageBox.warning(self, "Ошибка", "Нет данных для 3D")
            return
        try:
            visualize_tumor_3d(patient.masks_edited, spacing=patient.spacing, threshold=0.5)
        except Exception as e:
            QMessageBox.critical(self, "Ошибка 3D", str(e))

    @staticmethod
    def compute_dice(gt, pred):
        inter = (gt * pred).sum()
        union = gt.sum() + pred.sum()
        return (2 * inter + 1e-6) / (union + 1e-6)

    def update_coords_status(self, x, y, intensity):
        self.coords_label.setText(f"x: {x}  y: {y}  intensity: {intensity:.2f}")

    # ----------------------- PDF отчёт (одна страница) -----------------------
    def save_report(self):
        if self.current_patient_index < 0:
            QMessageBox.warning(self, "Ошибка", "Нет выбранного пациента")
            return
        patient = self.patients[self.current_patient_index]
        file_path, _ = QFileDialog.getSaveFileName(self, "Сохранить PDF", f"{patient.filename}_report.pdf", "PDF Files (*.pdf)")
        if not file_path:
            return

        try:
            pdfmetrics.registerFont(TTFont("Arial", "arial.ttf"))
            font_name = "Arial"
        except:
            font_name = "Helvetica"

        c = canvas.Canvas(file_path)
        width, height = 595, 842
        margin = 50
        tmp = tempfile.gettempdir()

        ch_idx = self.channel_selector.currentIndex()
        channel_name = "DWI" if ch_idx == 0 else "ADC"

        c.setFont(font_name, 14)
        c.drawString(margin, height - 40, "Результаты сегментации")
        c.setFont(font_name, 12)
        c.drawString(margin, height - 60, f"Файл: {patient.filename}")
        c.drawString(margin, height - 75, f"Дата: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

        c.drawString(margin, height - 105, "Технические параметры:")
        c.drawString(margin + 20, height - 120, f"Порог: {self.threshold:.2f}")
        c.drawString(margin + 20, height - 135, f"Канал: {channel_name}")
        c.drawString(margin + 20, height - 150, "Постобработка: удаление объектов <30 пикселей")

        ch = self.channel_selector.currentIndex()
        img = patient.images[self.idx, ch]
        img_norm = (img - img.min()) / (img.max() - img.min() + 1e-6)
        img_uint8 = (img_norm * 255).astype(np.uint8)
        temp_img = os.path.join(tmp, "mri.png")
        cv2.imwrite(temp_img, img_uint8)

        c.drawString(margin, height - 180, f"Срез {self.idx+1} из {patient.images.shape[0]}")
        c.drawImage(temp_img, margin, height - 360, 200, 200)

        if patient.masks_edited is not None:
            pred = patient.masks_edited[self.idx]
            overlay_img = self.create_overlay(img_norm, pred)
            temp_over = os.path.join(tmp, "overlay.png")
            cv2.imwrite(temp_over, overlay_img)
            c.drawImage(temp_over, margin + 250, height - 360, 200, 200)

            pix = pred.sum()
            if pix > 0:
                c.drawString(margin, height - 380, f"Площадь участка: {pix} пикселей")
                if patient.masks_gt is not None:
                    dice = self.compute_dice(patient.masks_gt[self.idx], pred)
                    c.drawString(margin, height - 395, f"Индекс совпадения (Dice): {dice:.3f}")
            else:
                c.drawString(margin, height - 380, "Патологический участок не обнаружен")

        c.save()
        QMessageBox.information(self, "Готово", "PDF отчёт сохранён")