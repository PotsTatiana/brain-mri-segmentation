# graphics_view.py
from PyQt5.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QWheelEvent, QMouseEvent, QPixmap, QImage, QPainter
import numpy as np

class MedicalImageView(QGraphicsView):
    mouse_coords_signal = pyqtSignal(float, float, float)
    drawing_started = pyqtSignal(object)
    drawing_moved = pyqtSignal(object, object)
    drawing_ended = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.current_image = None
        self.zoom_factor = 1.15
        self.setMouseTracking(True)

        # Рисование
        self.drawing = False
        self.last_pos = None
        self.brush_size = 5
        self.brush_mode = 'draw'

        # Панорамирование правой кнопкой
        self.panning = False
        self.pan_start = None

    # ---------------------- Обновление изображения ----------------------
    def _update_pixmap_from_array(self, image_np):
        if image_np is None:
            return
        if image_np.ndim == 2:
            h, w = image_np.shape
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
            qimage = QImage(image_np.data, w, h, w, QImage.Format_Grayscale8)
        else:
            h, w, ch = image_np.shape
            if ch == 3:
                image_np = image_np.astype(np.uint8)
                qimage = QImage(image_np.data, w, h, w * 3, QImage.Format_RGB888)
            else:
                raise ValueError("Only 2D or RGB images supported")
        pixmap = QPixmap.fromImage(qimage)
        self.pixmap_item.setPixmap(pixmap)

    def set_image(self, image_np):
        """Полная установка – сброс масштаба и подгон под размер окна."""
        self.current_image = image_np
        self._update_pixmap_from_array(image_np)
        self.setSceneRect(self.scene.itemsBoundingRect())
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def set_image_keep_transform(self, image_np):
        """Обновляет изображение, сохраняя текущие трансформации (зум, панораму)."""
        if image_np is None:
            return
        # Сохраняем трансформацию
        transform = self.transform()
        self.current_image = image_np
        self._update_pixmap_from_array(image_np)
        self.setTransform(transform)

    # ---------------------- События мыши ----------------------
    def wheelEvent(self, event: QWheelEvent):
        zoom_in = event.angleDelta().y() > 0
        factor = self.zoom_factor if zoom_in else 1 / self.zoom_factor
        self.scale(factor, factor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.last_pos = self.mapToScene(event.pos())
            self.drawing_started.emit(self.last_pos)
        elif event.button() == Qt.RightButton:
            self.panning = True
            self.pan_start = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Сначала координаты (всегда, даже если рисуем)
        scene_pos = self.mapToScene(event.pos())
        if self.current_image is not None and self.pixmap_item.isVisible():
            pixmap_rect = self.pixmap_item.boundingRect()
            if pixmap_rect.contains(scene_pos):
                x_ratio = (scene_pos.x() - pixmap_rect.x()) / pixmap_rect.width()
                y_ratio = (scene_pos.y() - pixmap_rect.y()) / pixmap_rect.height()
                h, w = self.current_image.shape[:2]
                px = int(x_ratio * w)
                py = int(y_ratio * h)
                if 0 <= px < w and 0 <= py < h:
                    intensity = self.current_image[py, px]
                    if isinstance(intensity, np.ndarray):
                        # Берём среднюю яркость, если цветное
                        intensity = intensity.mean()
                    self.mouse_coords_signal.emit(float(px), float(py), float(intensity))

        # Затем рисование или панорамирование
        if self.drawing and (event.buttons() & Qt.LeftButton):
            new_pos = self.mapToScene(event.pos())
            self.drawing_moved.emit(self.last_pos, new_pos)
            self.last_pos = new_pos
        elif self.panning and (event.buttons() & Qt.RightButton):
            delta = event.pos() - self.pan_start
            self.pan_start = event.pos()
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.drawing = False
            self.drawing_ended.emit()
        elif event.button() == Qt.RightButton and self.panning:
            self.panning = False
            self.setCursor(Qt.ArrowCursor)
        super().mouseReleaseEvent(event)

    # ---------------------- Утилиты рисования ----------------------
    def set_brush_size(self, size):
        self.brush_size = max(1, size)

    def set_brush_mode(self, mode):
        self.brush_mode = mode

    def reset_view(self):
        self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)