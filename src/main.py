import matplotlib

matplotlib.use("Qt5Agg")

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, QRectF
from PyQt5.QtGui import QImage, QPainter, QBrush, QPen, QColor, QLinearGradient
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtWidgets import (
    QGraphicsScene,
    QGraphicsView,
    QGraphicsItem,
    QMenu,
    QWidget,
    QGraphicsObject,
)
import sip

from PIL import Image
import numpy as np
import skimage
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from numba import njit

import threading
import json
import math

# import UI
from ui.app import Ui_MainWindow
import ui.image_histogram as image_histogram_ui
import ui.marge_images as marge_images_ui
import ui.image_settings as image_settings_ui
import ui.chrom_abers as chrom_abers_ui
import ui.effect_base as effect_base_ui
import ui.effect_manager as effect_manager_ui
import ui.gamma_correction as gamma_correction_ui
import ui.filter_creator as filter_creator_ui
import ui.mask_editor as mask_editor_ui
import ui.mask_editor_menu as mask_editor_menu_ui

# import effects for image
from effects import *

# import validate_edit for validate inputs values
from widgets.validate_edit import VelidateLineEdit


from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f"Function {func.__name__} Took {total_time:.4f} seconds")
        return result

    return timeit_wrapper


def qt_image_to_array(img, share_memory=False):
    """Creates a numpy array from a QImage.

    If share_memory is True, the numpy array and the QImage is shared.
    Be careful: make sure the numpy array is destroyed before the image,
    otherwise the array will point to unreserved memory!!
    """
    assert isinstance(img, QtGui.QImage), "img must be a QtGui.QImage object"
    assert (
        img.format() == QtGui.QImage.Format.Format_RGB32
    ), "img format must be QImage.Format.Format_RGB32, got: {}".format(img.format())

    img_size = img.size()
    buffer = img.constBits()
    print(buffer)

    # Sanity check
    n_bits_buffer = len(buffer)  # * 8
    n_bits_image = img_size.width() * img_size.height() * img.depth()
    assert n_bits_buffer == n_bits_image, "size mismatch: {} != {}".format(
        n_bits_buffer, n_bits_image
    )

    assert img.depth() == 32, "unexpected image depth: {}".format(img.depth())

    # Note the different width height parameter order!
    arr = np.ndarray(
        shape=(img_size.height(), img_size.width(), img.depth() // 8),
        buffer=buffer,
        dtype=np.uint8,
    )

    if share_memory:
        return arr
    else:
        return copy.deepcopy(arr)


class EffectsContainer(QtCore.QObject):
    undo_signal = QtCore.pyqtSignal()
    redo_signal = QtCore.pyqtSignal()
    update_effects = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.effects = []
        self.current_effects_idx = -1

    def add(self, new_effect):
        self.effects = self.effects[: self.current_effects_idx + 1]
        self.effects.append(new_effect)
        self.current_effects_idx += 1
        self.update_effects.emit()

    def remove(self, idx):
        self.effects.pop(idx)
        self.current_effects_idx = len(self.effects) - 1
        self.update_effects.emit()

    def undo(self):
        if self.current_effects_idx < 0:
            return False

        self.current_effects_idx -= 1
        self.undo_signal.emit()
        self.update_effects.emit()
        return True

    def redo(self):
        if self.current_effects_idx >= len(self.effects) - 1:
            return False

        self.current_effects_idx += 1
        self.redo_signal.emit()
        self.update_effects.emit()
        return True

    def is_undo_available(self):
        if self.current_effects_idx < 0:
            return False
        return True

    def is_redo_available(self):
        if self.current_effects_idx >= len(self.effects) - 1:
            return False
        return True

    def get_current_effects(self):
        return self.effects[: self.current_effects_idx + 1]


class ImageItem(QGraphicsObject):
    count = 0
    on_image_change = QtCore.pyqtSignal()
    on_delete = QtCore.pyqtSignal()

    on_update_masks = QtCore.pyqtSignal()

    def __init__(self, image, position=(0, 0)):
        QGraphicsObject.__init__(self)
        self.__init_context_menubar()
        self.__init_pens_and_brushes()
        self.setFlag(QGraphicsItem.ItemIsMovable, True)

        self.image = image
        self.position = position
        self.setPos(*self.position)
        self.is_blur = False
        self.is_select = False

        self.name = f"image_{self.get_new_id()}"
        self.image_effected = self.image

        self.effects = EffectsContainer()
        self.effects.update_effects.connect(self.paint_with_update)

        self.is_image_changed = False

        self.rander_to_QImage()

        # передаелать по примеру effects
        # протопит из-за нехватри времени
        self.masks = {}
        self.current_mask = None
        self.use_mask = True

    @classmethod
    def get_new_id(cls):
        result = cls.count
        cls.count += 1
        return result

    def __init_pens_and_brushes(self):
        self.pen_w = 4
        self.pen_w_2 = math.ceil(self.pen_w / 2)
        self.blur_pen = QPen(QColor(20, 20, 100, 255), self.pen_w)
        self.blur_pen.setJoinStyle(Qt.MiterJoin)
        self.blur_brush = QBrush(QColor(20, 20, 100, 127), Qt.SolidPattern)
        self.select_brush = QBrush(QColor(0, 0, 0, 255), Qt.NoBrush)
        self.select_pen = QPen(QColor(255, 0, 0, 255), self.pen_w)
        self.select_pen.setJoinStyle(Qt.MiterJoin)

    def __init_context_menubar(self):
        self.delete_act = QtWidgets.QAction("Delete")
        self.delete_act.triggered.connect(self.delete)
        self.save = QtWidgets.QAction("Save image")
        self.save.triggered.connect(self.save_image_ui)

    def contextMenuEvent(self, event):
        menu = QMenu()
        menu.addAction(self.delete_act)
        menu.addAction(self.save)
        menu.exec(event.screenPos())

    def delete(self):
        self.on_delete.emit()
        sip.delete(self)

    def get_masks(self):
        return self.masks

    def set_masks(self, masks):
        self.masks = masks
        self.on_update_masks.emit()

    def get_current_mask(self):
        if self.current_mask is not None and self.use_mask:
            return self.masks[self.current_mask].get_bitmask()
        return None

    def set_select(self, val):
        self.is_select = val
        self.update()

    def set_blur(self, val):
        self.is_blur = val
        self.update()

    def get_size(self):
        h, w, _ = self.image.shape
        return (w, h)

    def set_name(self, new_name: str):
        self.name = new_name

    def get_name(self):
        return self.name

    def save_image_ui(self):
        filename, _ = QFileDialog.getSaveFileName(
            None, "Save image file", f"{self.name}.jpg", "Image (*.jpg)"
        )
        if filename:
            self.save_image(filename)

    def use_effects(self):
        image = self.image.copy()
        for effect in self.effects.get_current_effects():
            image = effect(image)

        self.image_effected = image

        return image

    def get_raw_image(self):
        return self.image

    def get_image(self):
        if not self.is_image_changed:
            return self.image_effected

        self.is_image_changed = False
        return self.use_effects()

    def rander_to_QImage(self):
        self.is_image_changed = True
        image = self.get_image()

        h, w, _ = image.shape
        bytes_per_line = w * 3
        self.paint_img = QImage(image, w, h, bytes_per_line, QImage.Format_RGB888)
        self.update()
        self.on_image_change.emit()

    def save_image(self, fname):
        im = Image.fromarray(self.get_image())
        im.save(fname)

    def boundingRect(self):
        h, w, _ = self.image.shape
        return QRectF(0, 0, w, h)

    def paint_with_update(self):
        self.rander_to_QImage()
        self.update()

    def paint(self, painter, option, widget):
        pos = list(map(int, self.position))
        painter.drawImage(0, 0, self.paint_img)
        w, h = self.get_size()
        if self.is_blur:
            painter.setBrush(self.blur_brush)
            painter.setPen(self.blur_pen)
            painter.drawRect(self.pen_w_2, self.pen_w_2, w - self.pen_w, h - self.pen_w)
        if self.is_select:
            painter.setBrush(self.select_brush)
            painter.setPen(self.select_pen)
            painter.drawRect(self.pen_w_2, self.pen_w_2, w - self.pen_w, h - self.pen_w)


class ImageMaskItem(QGraphicsObject):
    background_color = QColor(0, 0, 0, 0)  # 255
    mask_color = QColor(250, 100, 120, 255)
    tools = ["CircleBrush", "RectBrush", "Ellipse", "Rect"]

    def __init__(self, image):
        QGraphicsObject.__init__(self)
        self.parent_image = image
        w, h = image.get_size()
        self.mask = QtGui.QPixmap(w, h)
        self.tools_canvas = QtGui.QPixmap(w, h)
        self.tools_canvas.fill(QColor(0, 0, 0, 0))

        self.mask_pen = Qt.NoPen
        self.mask_brush = QBrush(self.mask_color, Qt.SolidPattern)
        self.erase_pen = Qt.NoPen
        self.erase_brush = QBrush(self.background_color, Qt.SolidPattern)
        self.clear()
        self.setFlag(QGraphicsItem.ItemIsMovable, True)
        self.setAcceptHoverEvents(True)
        self.is_paint = True
        self.tool_type = self.tools[3]
        self.draw_or_erase = 0
        self.tool_size = 20

        self.pres_pos = (0, 0)
        self.prev_pos = (0, 0)
        # createMaskFromColor

    def clear(self):
        self.mask.fill(self.background_color)

    def boundingRect(self):
        return self.parent_image.boundingRect()

    def set_tool_type(self, tool_type):
        self.tool_type = tool_type

    def set_tool_size(self, tool_size):
        self.tool_size = tool_size
        self.draw_tools(*self.prev_pos, [Qt.LeftButton])
        self.update()

    def set_draw_or_erase(self, draw_or_erase):
        self.draw_or_erase = draw_or_erase

    def draw_tools(self, x, y, buttons=0):
        self.tools_canvas.fill(QColor(0, 0, 0, 0))
        painter_tools = QPainter(self.tools_canvas)
        painter_tools.setPen(QPen(QColor(255, 0, 0), 1))

        d = self.tool_size
        d_2 = d // 2
        if self.tool_type == self.tools[0]:
            painter_tools.drawEllipse(x - d_2, y - d_2, d, d)
        elif self.tool_type == self.tools[1]:
            painter_tools.drawRect(x - d_2, y - d_2, d, d)
        elif self.tool_type == self.tools[2]:
            if buttons == Qt.LeftButton:
                x_p, y_p = self.pres_pos
                w, h = x - x_p, y - y_p
                painter_tools.drawEllipse(x_p, y_p, w, h)
        elif self.tool_type == self.tools[3]:
            if buttons == Qt.LeftButton:
                x_p, y_p = self.pres_pos
                w, h = x - x_p, y - y_p
                painter_tools.drawRect(x_p, y_p, w, h)

    def mouseMoveEvent(self, event):
        x, y = int(event.scenePos().x()), int(event.scenePos().y())
        self.draw_tools(x, y, event.buttons())

        if event.buttons() == Qt.LeftButton:
            d = self.tool_size
            d_2 = d // 2
            painter = QPainter(self.mask)

            if self.draw_or_erase == 0:
                painter.setPen(self.mask_pen)
                painter.setBrush(self.mask_brush)
            else:
                painter.setPen(self.erase_pen)
                painter.setBrush(self.erase_brush)

            if self.tool_type == self.tools[0]:
                painter.drawEllipse(x - d_2, y - d_2, d, d)
            elif self.tool_type == self.tools[1]:
                d = self.tool_size
                painter.drawRect(x - d_2, y - d_2, d, d)

        self.prev_pos = (x, y)
        self.update()
        if not self.is_paint:
            super().mouseMoveEvent(self.tools_canvas)

    def mousePressEvent(self, event):
        self.pres_pos = (int(event.scenePos().x()), int(event.scenePos().y()))

        if not self.is_paint:
            super().mouseMoveEvent(self.tools_canvas)

    def mouseReleaseEvent(self, event):
        painter = QPainter(self.mask)
        x, y = int(event.scenePos().x()), int(event.scenePos().y())

        if self.draw_or_erase == 0:
            painter.setPen(self.mask_pen)
            painter.setBrush(self.mask_brush)
        else:
            painter.setPen(self.erase_pen)
            painter.setBrush(self.erase_brush)

        if self.tool_type == self.tools[2]:
            x_p, y_p = self.pres_pos
            w, h = x - x_p, y - y_p
            painter.drawEllipse(x_p, y_p, w, h)
        elif self.tool_type == self.tools[3]:
            x_p, y_p = self.pres_pos
            w, h = x - x_p, y - y_p
            painter.drawRect(x_p, y_p, w, h)

        if not self.is_paint:
            super().mouseMoveEvent(self.tools_canvas)

    def get_bitmask(self):
        mk = self.mask.toImage()
        w, h = self.parent_image.get_size()
        c = 4
        s = mk.bits().asstring(h * w * c)
        result = np.fromstring(s, dtype=np.uint8).reshape((h, w, c))

        result = np.sum(result, axis=2)
        result = result / np.max(result)
        fin = np.empty((h, w, 3))
        fin[:, :, 0] = result
        fin[:, :, 1] = result
        fin[:, :, 2] = result
        return fin

    def get_mask_pixmap(self):
        return self.mask

    def paint(self, painter, option, widget):
        painter.drawImage(0, 0, self.parent_image.paint_img)
        # mk = self.mask.createMaskFromColor(self.mask_color).toImage()
        # mk = mk.convertToFormat(QImage.Format.Format_RGB32)
        # print(qt_image_to_array(mk))
        painter.drawImage(0, 0, self.mask.toImage())
        painter.drawImage(0, 0, self.tools_canvas.toImage())


class EditorScene(QGraphicsScene):
    update_info = QtCore.pyqtSignal(str)

    def __init__(self, editor_menu, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        self.editor_menu = editor_menu
        self.mouse_pos = (0, 0)

    def contextMenuEvent(self, event):
        self.menu_mouse_pos = (event.scenePos().x(), event.scenePos().y())
        item = self.itemAt(event.scenePos().toPoint(), QtGui.QTransform())
        if item:
            item.contextMenuEvent(event)
            return

        menu = QMenu()
        self.load_image_action = QtWidgets.QAction("Open image", self)
        self.load_image_action.triggered.connect(self.open_image)
        menu.addAction(self.load_image_action)
        menu.exec(event.screenPos())

    def get_info(self):
        x, y = map(int, self.mouse_pos)
        item = self.itemAt(x, y, QtGui.QTransform())
        if item is None:
            return f"mouse pos: (x:{x}, y:{y})"
        if not isinstance(item, ImageItem):
            return f"mouse pos: (x:{x}, y:{y})"
        name = item.get_name()
        rel_x = int(x - item.x())
        rel_y = int(y - item.y())
        w, h = item.get_size()
        return f"mouse pos: (x:{x}, y:{y}) | {name} {w}X{h} mouse relpos: (x:{rel_x}, y:{rel_y})"

    def mouseMoveEvent(self, event):
        self.mouse_pos = (event.scenePos().x(), event.scenePos().y())
        self.update_info.emit(self.get_info())
        super().mouseMoveEvent(event)

    def my_leaveEvent(self):
        self.update_info.emit("")

    def mouseDoubleClickEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            item = self.itemAt(event.scenePos().toPoint(), QtGui.QTransform())
            press_out_of_item_flag = True
            if item:
                if isinstance(item, ImageItem):
                    self.editor_menu.mouse_double_click_item(item)
                    press_out_of_item_flag = False

            if press_out_of_item_flag:
                self.editor_menu.mouse_double_click_out()

        super().mouseDoubleClickEvent(event)

    def mousePressEvent(self, event):
        if event.buttons() == QtCore.Qt.LeftButton:
            item = self.itemAt(event.scenePos().toPoint(), QtGui.QTransform())
            press_out_of_items_flag = True
            if item:
                if isinstance(item, ImageItem):
                    self.editor_menu.mouse_click_item(item)
                    press_out_of_item_flag = False

            if press_out_of_items_flag:
                self.editor_menu.mouse_click_out()

        super().mousePressEvent(event)

    def open_image(self):
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open image file", ".", "Image Files (*.png *.jpg *.bmp)"
        )
        if file_name:
            self.load_image(file_name)

    def load_image(self, file_name):
        image = skimage.io.imread(file_name)
        self.addItem(ImageItem(image, self.menu_mouse_pos))


class EditorView(QGraphicsView):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._zoom = 1
        self.setMouseTracking(True)

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if bool(modifiers == QtCore.Qt.ControlModifier):
            factor = 1.2

            if event.angleDelta().y() > 0 and self._zoom <= 100:
                self._zoom *= factor
                self.scale(factor, factor)
            elif self._zoom >= 0.05:
                self._zoom *= 1 / factor
                self.scale(1 / factor, 1 / factor)

        super().wheelEvent(event)

    def leaveEvent(self, event):
        self.scene().my_leaveEvent()
        super().leaveEvent(event)


class ImageEditorMenu(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.editing_image = None

        self.editor_layout = QtWidgets.QVBoxLayout(self)
        self.editor_layout.setContentsMargins(0, 0, 0, 0)

        self.editor_scrollArea = QtWidgets.QScrollArea(self)
        self.editor_scrollArea.setWidgetResizable(True)
        scrollAreaWidgetContents = QtWidgets.QWidget()
        self.editor_scrollArea_layout = QtWidgets.QVBoxLayout(scrollAreaWidgetContents)
        self.editor_scrollArea_layout.setContentsMargins(6, 3, 6, 3)
        self.editor_scrollArea.setWidget(scrollAreaWidgetContents)

        label_title = QtWidgets.QLabel(self)
        label_title.setText("Image editor menu")
        self.editor_layout.addWidget(label_title)
        self.editor_layout.addWidget(self.editor_scrollArea)

        # add ImageSettingsWidget
        self.settings = ImageSettingsWidget()
        self.editor_scrollArea_layout.addWidget(self.settings)

        # add ImageHistogramWidget
        self.histogram = ImageHistogramWidget()
        self.editor_scrollArea_layout.addWidget(self.histogram)

        # add GammaCorrectionWidget
        self.gamma_correction = GammaCorrectionWidget()
        self.editor_scrollArea_layout.addWidget(self.gamma_correction)

        # add mask_editor_menu
        self.mask_editor_menu = MaskEditorMenuWidget()
        self.editor_scrollArea_layout.addWidget(self.mask_editor_menu)

        # add MrageImagesWidget
        self.marge = MrageImagesWidget()
        self.editor_scrollArea_layout.addWidget(self.marge)

        # add ChromaticAberrationWidget
        self.chrom = ChromaticAberrationWidget()
        self.editor_scrollArea_layout.addWidget(self.chrom)

        # add effect menu
        self.effect_manager = EffectManagerWidget()
        self.editor_scrollArea_layout.addWidget(self.effect_manager)

        spacer_item = QtWidgets.QSpacerItem(
            20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding
        )
        self.editor_scrollArea_layout.addItem(spacer_item)

        self.set_image(None)
        self.setEnabled(False)

    def mouse_double_click_item(self, item):
        self.set_image(item)

    def mouse_double_click_out(self):
        self.unset_image()

    def mouse_click_item(self, item):
        self.marge.merge_images(item)

    def mouse_click_out(self):
        self.marge.cancel_merging()

    def set_image(self, image):
        if self.editing_image == image:
            return

        if self.editing_image is not None:
            self.editing_image.set_select(False)
            self.editing_image.on_delete.disconnect(self.unset_image)
            self.editing_image.update()

        self.editing_image = image
        if image is None:
            self.histogram.set_image()
            self.settings.set_image()
            self.marge.set_image()
            self.chrom.set_image()
            self.effect_manager.set_image()
            self.gamma_correction.set_image()
            self.mask_editor_menu.set_image()

            self.setEnabled(False)
        else:
            self.histogram.set_image(self.editing_image)
            self.settings.set_image(self.editing_image)
            self.marge.set_image(self.editing_image)
            self.chrom.set_image(self.editing_image)
            self.effect_manager.set_image(self.editing_image)
            self.gamma_correction.set_image(self.editing_image)
            self.mask_editor_menu.set_image(self.editing_image)

            self.setEnabled(True)
            self.editing_image.on_delete.connect(self.unset_image)
            self.editing_image.set_select(True)

    def unset_image(self):
        self.set_image(None)

    def get_image(self):
        return self.editing_image


class ImageSettingsWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = image_settings_ui.Ui_Form()
        self.ui.setupUi(self)

        self.ui.undoBtn.clicked.connect(self.undo_image)
        self.ui.redoBtn.clicked.connect(self.redo_image)
        self.ui.imageName.editingFinished.connect(self.update_image_name)

        self.image = None

    def set_image(self, image=None):
        if self.image is not None:
            self.image.effects.update_effects.disconnect(self.update_undo_redo_btn)

        self.image = image
        text = "NO IMAGE"
        if self.image is not None:
            text = self.image.get_name()
            self.image.effects.update_effects.connect(self.update_undo_redo_btn)

        self.ui.imageName.setText(text)
        self.update_undo_redo_btn()

    def update_image_name(self):
        if self.image is None:
            return
        name = self.ui.imageName.text()
        self.image.set_name(name)

    def undo_image(self):
        if self.image is None:
            return
        self.image.effects.undo()
        self.update_undo_redo_btn()

    def redo_image(self):
        if self.image is None:
            return
        self.image.effects.redo()
        self.update_undo_redo_btn()

    def update_undo_redo_btn(self):
        if self.image is None:
            return
        is_undo = self.image.effects.is_undo_available()
        is_redo = self.image.effects.is_redo_available()
        self.ui.undoBtn.setEnabled(is_undo)
        self.ui.redoBtn.setEnabled(is_redo)


class ChromaticAberrationWidget(QWidget):
    update_params = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = chrom_abers_ui.Ui_Form()
        self.ui.setupUi(self)

        self.ui.chromBtn.clicked.connect(self.chrom)

        self.set_image()

    def set_image(self, image=None):
        self.image = image

    def chrom(self):
        if self.image is None:
            return
        dx = self.ui.dxSb.value()
        dy = self.ui.dySb.value()
        mask = self.image.get_current_mask()
        caex = ChromaticAberrationEffect(dx, dy, mask=mask)
        self.image.effects.add(caex)
        self.update_params.emit()


class MrageImagesWidget(QWidget):
    update_params = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = marge_images_ui.Ui_Form()
        self.ui.setupUi(self)

        self.ui.margeImagesBtn.clicked.connect(self.select_merge_image)

        self.set_image()

    def set_image(self, image=None):
        self.image = image
        self.is_merging = False

    def select_merge_image(self):
        if self.image is None:
            return

        self.image.set_blur(True)
        self.is_merging = True

    def merge_images(self, ref_img):
        if (self.image is None) or (not self.is_merging):
            return

        ct = self.ui.margeTypeSelect.currentText()
        mask = self.image.get_current_mask()
        mex = MargeEffect(ref_img.get_image(), ct, mask=mask)
        self.image.effects.add(mex)
        self.update_params.emit()

        self.cancel_merging()

    def cancel_merging(self):
        self.is_merging = False
        if self.image is not None:
            self.image.set_blur(False)


class SCWidget(QWidget):
    showed = QtCore.pyqtSignal()
    closed = QtCore.pyqtSignal()

    def showEvent(self, event):
        self.showed.emit()

    def closeEvent(self, event):
        self.closed.emit()
        super().closeEvent(event)


class FigureCanvasMouseDoubleClickEvent(FigureCanvas):
    mouse_double_click = QtCore.pyqtSignal()
    lg_dark = QColor(0, 0, 0, 100)
    lg_light = QColor(255, 255, 255, 30)

    def mouseDoubleClickEvent(self, event):
        self.mouse_double_click.emit()

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setPen(Qt.NoPen)
        lg = QLinearGradient(0, 0, self.width(), 0)
        lg.setColorAt(0, self.lg_dark)
        lg.setColorAt(1, self.lg_light)
        painter.setBrush(QBrush(lg))
        painter.drawRect(0, 0, self.width(), self.height())


class ImageHistogramWidget(QWidget):
    hist_type = 0

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = image_histogram_ui.Ui_Form()
        self.ui.setupUi(self)

        self.image = None

        self.imageHist_figure = Figure(facecolor="#20242d")
        self.imageHist_canvas = FigureCanvasMouseDoubleClickEvent(self.imageHist_figure)
        self.imageHist_axis = self.imageHist_figure.add_axes(
            [0, 0, 1, 1], frameon=False
        )

        self.imageHist_axis.get_xaxis().set_visible(False)
        self.imageHist_axis.get_yaxis().set_visible(False)
        self.imageHist_canvas.draw()

        self.ui.histogram_layout.addWidget(self.imageHist_canvas)

        self.dialog = SCWidget()
        self.set_is_dialog(False)
        self.dialog.showed.connect(self.dialog_open_event)
        self.dialog.closed.connect(self.dialog_close_event)
        self.imageHist_canvas.mouse_double_click.connect(self.dialog.show)

        dialog_layout = QtWidgets.QHBoxLayout(self.dialog)
        self.dialog_figure = Figure(facecolor="#20242d")
        self.dialog_canvas = FigureCanvasMouseDoubleClickEvent(self.dialog_figure)
        self.dialog_axis = self.dialog_figure.add_axes([0, 0, 1, 1], frameon=False)
        dialog_layout.addWidget(self.dialog_canvas)

        self.thread_it = False
        if self.thread_it:
            self.update_hist_thread = threading.Thread(target=self.update_hist_func)
            self.update_hist_thread.start()
            self.update_hist_thread.join()

    def dialog_close_event(self):
        self.set_is_dialog(False)
        self.update_hist_func()

    def dialog_open_event(self):
        self.set_is_dialog(True)
        self.update_hist_func()

    def open_dialog(self):
        pass

    def set_is_dialog(self, val=False):
        self.is_dialog = val

    def set_image(self, image=None):
        if self.image is not None:
            self.image.on_image_change.disconnect(self.update_hist)

        self.image = image
        if self.image is not None:
            self.image.on_image_change.connect(self.update_hist)
        else:
            self.clear_axes()

        self.update_hist()

    def update_hist(self):
        if not self.thread_it:
            self.update_hist_func()
        else:
            if self.update_hist_thread.is_alive():
                self.update_hist_thread.join()

            del self.update_hist_thread
            self.update_hist_thread = threading.Thread(target=self.update_hist_func)
            self.update_hist_thread.start()

    def clear_axes(self):
        self.imageHist_axis.clear()
        self.dialog_axis.clear()
        self.imageHist_canvas.draw()
        self.dialog_canvas.draw()

    @staticmethod
    @njit
    def count_image_hists(image, bins):
        R, _ = np.histogram(image[:, :, 0].ravel(), bins=bins)
        G, _ = np.histogram(image[:, :, 1].ravel(), bins=bins)
        B, _ = np.histogram(image[:, :, 2].ravel(), bins=bins)
        RG = np.fmin(R, G)
        RB = np.fmin(R, B)
        GB = np.fmin(G, B)
        W = np.fmin(RG, B)

        hists = np.zeros((7, R.shape[0]), dtype=np.int64)
        hists[0] = R
        hists[1] = G
        hists[2] = B
        hists[3] = RG
        hists[4] = RB
        hists[5] = GB
        hists[6] = W
        return hists

    def update_hist_func(self):
        if self.image is None:
            return

        image = self.image.get_image()
        if self.is_dialog:
            hits_ax = self.dialog_axis
            self.imageHist_axis.clear()
        else:
            hits_ax = self.imageHist_axis
        hits_ax.clear()

        colors = [
            "#f05050",
            "#50f050",
            "#5050f0",
            "#f0f050",
            "#f050f0",
            "#50f0f0",
            "#ccc",
        ]
        bins = np.arange(0, 256)

        hists = self.count_image_hists(image, bins)

        for i, hist in enumerate(hists):
            hits_ax.stairs(hist, bins, color="#222", linewidth=2, alpha=0.3)
            hits_ax.stairs(hist, bins, color=colors[i], fill=True, linewidth=0)

        hits_ax.autoscale()
        if self.is_dialog:
            self.imageHist_canvas.draw()
            self.dialog_canvas.draw()
            self.dialog_canvas.flush_events()
        else:
            self.imageHist_canvas.draw()
            self.imageHist_canvas.flush_events()

    def mouseDoubleClickEvent(self, event):
        self.dialog.show()


class CurvePoint(QGraphicsObject):
    moved = QtCore.pyqtSignal()

    def __init__(self, x, y, d):
        QGraphicsObject.__init__(self)
        self.setPos(x, y)
        self.d = d
        self.prev_pos = self.pos()
        self.setFlag(QGraphicsItem.ItemIsMovable, True)

    def boundingRect(self):
        r = self.d // 2
        return QRectF(-r, -r, self.d, self.d)

    def set_parent_curve(self, curve):
        self.parent_curve = curve

    def paint(self, painter, option, widget):
        r = self.d // 2
        painter.setBrush(QBrush(QColor(255, 77, 77, 120), Qt.SolidPattern))
        painter.setPen(QPen(QColor(255, 77, 77, 255), 1))
        painter.drawEllipse(-r, -r, self.d, self.d)

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.LeftButton:
            rect = self.parent_curve.sceneBoundingRect()
            x, y = event.scenePos().x(), event.scenePos().y()

            if rect.contains(x, y):
                self.setPos(event.scenePos())
                self.moved.emit()


class BezierCurve(QGraphicsObject):
    def __init__(self):
        QGraphicsObject.__init__(self)

    def set_points(self, points):
        self.points = points
        for point in points:
            point.set_parent_curve(self)
            point.moved.connect(self.update)
        steps = 2 * len(points) * int(max(self.w, self.h))
        self.update_polynomial(len(points), steps)

    def set_size(self, size):
        self.w, self.h = size

    def get_croped_rect(self):
        x_p = []
        y_p = []
        for point in self.points:
            x_p.append(point.x())
            y_p.append(point.y())

        x, y, e_x, e_y = min(x_p), min(y_p), max(x_p), max(y_p)
        w, h = e_x - x, e_y - y
        return (x, y, w, h)

    def boundingRect(self):
        return QRectF(0, 0, self.w, self.h)

    def update_polynomial(self, n, steps):
        @njit(fastmath=True)
        def binomial(n, k):
            if not 0 <= k <= n:
                return 0
            b = 1
            for t in range(min(k, n - k)):
                b *= n
                b /= t + 1
                n -= 1
            return int(b)

        @njit(fastmath=True)
        def bernstein_poly(i, n, t):
            return binomial(n, i) * (t ** (n - i)) * (1 - t) ** i

        t = np.linspace(0.0, 1.0, steps)
        self.polynomial_array = np.array(
            [bernstein_poly(i, n - 1, t) for i in range(0, n)]
        )

    @staticmethod
    @njit
    def get_curve(points, polynomial_array):
        x = np.dot(points[:, 0], polynomial_array)
        y = np.dot(points[:, 1], polynomial_array)
        return x, y

    def get_current_curve(self):
        points = []
        x, y = self.x(), self.y()
        for point in self.points:
            points.append(np.array([point.x() - x, point.y() - y]))
        x, y = self.get_curve(np.array(points), self.polynomial_array)
        return x, y

    def paint(self, painter, option, widget):
        x, y = self.get_current_curve()
        for i in range(x.shape[0]):
            painter.drawPoint(int(x[i]), int(y[i]))


class ControlLine(QGraphicsObject):
    def __init__(self, p1, p2):
        QGraphicsObject.__init__(self)
        self.p1 = p1
        self.p2 = p2

    def get_rect(self):
        x1, y1 = self.p1.x(), self.p1.y()
        x2, y2 = self.p2.x(), self.p2.y()
        x, y = min(x1, x2), min(y1, y2)
        w, h = max(x1, x2) - x, max(y1, y2) - y
        return (x, y, w, h)

    def boundingRect(self):
        return QRectF(*self.get_rect())

    def paint(self, painter, option, widget):
        painter.setPen(QPen(QColor(255, 77, 77, 120), 1))
        painter.drawLine(self.p1.pos(), self.p2.pos())


class Spline(BezierCurve):
    def set_points(self, points):
        super().set_points(points)
        cl1 = ControlLine(self.points[0], self.points[1])
        cl1.setZValue(100)
        cl2 = ControlLine(self.points[2], self.points[3])
        cl2.setZValue(100)
        scene = self.scene()
        scene.addItem(cl1)
        scene.addItem(cl2)


class GammaCurve(Spline):
    def paint(self, painter, option, widget):
        super().paint(painter, option, widget)
        painter.drawRect(0, 0, self.w, self.h)


class GammaScene(QGraphicsScene):
    def __init__(self, parent=None):
        QGraphicsScene.__init__(self, parent)
        self.__init_items()

    def __init_items(self):
        self.spline = GammaCurve()
        self.setFocusItem(self.spline)
        self.spline.set_size((100, 100))
        self.spline.setPos(0, 0)

        d = 6

        start_p = CurvePoint(0, 100, d)
        start_p.setVisible(False)
        self.addItem(start_p)

        end_p = CurvePoint(100, 0, d)
        end_p.setVisible(False)
        self.addItem(end_p)

        c1_p = CurvePoint(0, 50, d)
        c1_p.moved.connect(lambda: self.setFocusItem(self.spline))
        self.addItem(c1_p)

        c2_p = CurvePoint(100, 50, d)
        c2_p.moved.connect(lambda: self.setFocusItem(self.spline))
        self.addItem(c2_p)

        self.addItem(self.spline)
        self.spline.set_points([start_p, c1_p, c2_p, end_p])

    def get_spline(self):
        return self.spline


class GammaView(QGraphicsView):
    def wheelEvent(self, event):
        pass


class GammaCorrectionWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = gamma_correction_ui.Ui_Form()
        self.ui.setupUi(self)

        self.gamma_view = QGraphicsView(self)
        self.gamma_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gamma_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.gamma_view.setMinimumSize(QtCore.QSize(120, 120))

        self.gamma_scene = GammaScene()
        self.gamma_view.setScene(self.gamma_scene)

        self.use_correction_btn = QtWidgets.QPushButton()
        self.use_correction_btn.setText("Use correction")
        self.use_correction_btn.clicked.connect(self.use_correction)

        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.ui.gamma_layout.addItem(spacerItem)

        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.gamma_view)
        layout.addWidget(self.use_correction_btn)
        self.ui.gamma_layout.addLayout(layout)

        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        self.ui.gamma_layout.addItem(spacerItem)

        self.set_image()

    def set_image(self, image=None):
        self.image = image

    def use_correction(self):
        if self.image is None:
            return

        x, y = self.gamma_scene.get_spline().get_current_curve()
        y = np.max(y) - y
        y = y - np.min(y)
        x = x - np.min(x)
        y = y / np.max(y) * 255
        x = x / np.max(x) * 255

        res = np.full((256), -1)
        for i in range(x.shape[0]):
            if res[int(x[i])] == -1:
                res[int(x[i])] = y[i]
            elif res[int(x[i])] > y[i]:
                res[int(x[i])] = y[i]

        mask = self.image.get_current_mask()
        ef = GammaCorrectionEffect(res, mask)
        self.image.effects.add(ef)


# effects widgets
class EffectWidget(QWidget):
    update_params = QtCore.pyqtSignal()
    effect_cls = FilterEffect
    channels = ["RGB", "R", "G", "B", "RG", "GB", "RB"]
    params = {}

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = effect_base_ui.Ui_Form()
        self.ui.setupUi(self)

        self.name = "effect"

        self.ui.useEffectBtn.clicked.connect(self.use_effect)
        self.ui.channelSelector.addItems(self.channels)

        self.__init_params()
        self.set_image()
        self.update_effect()

    def __init_params(self):
        def get_valid_func(func, name):
            return lambda: func(name)

        for name in self.params:
            if "resource_model" in self.params[name]:
                self.params[name]["resource"] = self.params[name]["resource_model"](
                    self.ui.effectGroup
                )
                self.ui.effectLayout.insertWidget(0, self.params[name]["resource"])
                label = QtWidgets.QLabel(self.ui.effectGroup)
                label.setText(name)
                self.ui.effectLayout.insertWidget(0, label)
                self.set_param_default(name)
                if "validate" in self.params[name]:
                    self.params[name]["validate_trigger"](
                        self.params[name]["resource"],
                        get_valid_func(self.param_validate, name),
                    )

    def set_name(self, name):
        self.name = name
        self.ui.effectGroup.setTitle(self.name)

    def set_image(self, image=None):
        self.image = image

    def collect_params(self):
        params = {}
        for name in self.params:
            if "resource_model" in self.params[name]:
                raw = self.params[name]["get_value_func"](self.params[name]["resource"])
                params[name] = self.params[name]["type"](raw)
            else:
                params[name] = self.params[name]["value"]

        return params

    def param_validate(self, name):
        text = self.params[name]["get_value_func"](self.params[name]["resource"])
        try:
            res = self.params[name]["type"](text)
            a, b = self.params[name]["validate"]
            if a is None and b is None:
                return True
            elif a is None:
                if res < b:
                    return True
            elif b is None:
                if res > a:
                    return True
            else:
                if a < res < b:
                    return True
        except:
            pass

        self.set_param_default(name)
        return False

    def set_param_default(self, name):
        self.params[name]["set_value_func"](
            self.params[name]["resource"], self.params[name]["default_value"]
        )

    def update_effect(self):
        channels = self.ui.channelSelector.currentText()
        params = self.collect_params()

        self.effect = self.effect_cls(**params, channels=channels)
        return self.effect

    def use_effect(self):
        if self.image is None:
            return
        self.update_effect()
        mask = self.image.get_current_mask()
        self.effect.set_mask(mask)
        self.image.effects.add(self.effect)
        self.update_params.emit()


class BlurEffectWidget(EffectWidget):
    effect_cls = BlurEffect
    params = {
        "a": {
            "default_value": 1,
            "type": int,
            "validate": (0, None),
            "validate_trigger": lambda x, f: x.editingFinished.connect(f),
            "resource_model": QtWidgets.QLineEdit,
            "set_value_func": lambda x, s: x.setText(str(s)),
            "get_value_func": lambda x: x.text(),
        }
    }


class SharpeningEffectWidget(EffectWidget):
    effect_cls = SharpeningEffect
    params = {
        "a": {
            "default_value": 1,
            "type": int,
            "validate": (0, None),
            "validate_trigger": lambda x, f: x.editingFinished.connect(f),
            "resource_model": QtWidgets.QLineEdit,
            "set_value_func": lambda x, s: x.setText(str(s)),
            "get_value_func": lambda x: x.text(),
        },
        "b": {
            "default_value": 1,
            "type": int,
            "validate": (0, None),
            "validate_trigger": lambda x, f: x.editingFinished.connect(f),
            "resource_model": QtWidgets.QLineEdit,
            "set_value_func": lambda x, s: x.setText(str(s)),
            "get_value_func": lambda x: x.text(),
        },
    }


class MedianEffectWidget(EffectWidget):
    effect_cls = MedianEffect
    params = {
        "a": {
            "default_value": 1,
            "type": int,
            "validate": (0, None),
            "validate_trigger": lambda x, f: x.editingFinished.connect(f),
            "resource_model": QtWidgets.QLineEdit,
            "set_value_func": lambda x, s: x.setText(str(s)),
            "get_value_func": lambda x: x.text(),
        }
    }


class EdgeDetectionCentralEffectWidget(EffectWidget):
    effect_cls = EdgeDetectionCentralEffect


class EdgeDetectionVerticalEffectWidget(EffectWidget):
    effect_cls = EdgeDetectionVerticalEffect


class EdgeDetectionHorizontalEffectWidget(EffectWidget):
    effect_cls = EdgeDetectionHorizontalEffect


class FirstEmbossingEffectWidget(EffectWidget):
    effect_cls = FirstEmbossingEffect


class SecondEmbossingEffectWidget(EffectWidget):
    effect_cls = SecondEmbossingEffect


class FilterEffectWidget(EffectWidget):
    def __init__(self, filter, parent=None):
        self.params = {
            "kernel": {
                "value": filter["kernel"],
            },
            "color_shift": {
                "value": filter["color_shift"],
            },
            "divider": {
                "value": filter["divider"],
            },
        }
        EffectWidget.__init__(self, parent)


class FilterEditor(SCWidget):
    grid_h = 11
    grid_w = 11
    on_add_filter = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter editor")
        self.ui = filter_creator_ui.Ui_Form()
        self.ui.setupUi(self)

        self.ui.filter_h.valueChanged[int].connect(self.update_grid)
        self.ui.filter_w.valueChanged[int].connect(self.update_grid)
        self.ui.clear_filter.clicked.connect(self.clear_all)
        self.ui.load_filter.clicked.connect(self.open_filter)

        self.ui.divider.set_default(1)
        self.ui.divider.set_validate([0, None])
        self.ui.color_shift.set_try_type(int)

        self.ui.save_filter.clicked.connect(self.save_filter)
        self.ui.add_filter.clicked.connect(self.on_add_filter.emit)

        self.__init_input_grid()

    def __init_input_grid(self):
        self.ui.filter_h.setMaximum(self.grid_h)
        self.ui.filter_w.setMaximum(self.grid_w)

        self.grid_input = []

        for i in range(self.grid_h):
            self.grid_input.append([])
            for j in range(self.grid_w):
                self.grid_input[i].append(VelidateLineEdit())
                self.ui.filterLayout.addWidget(self.grid_input[i][j], i, j)

        self.update_grid()

    def hide_all(self):
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                self.grid_input[i][j].hide()

    def clear_all(self):
        for i in range(self.grid_h):
            for j in range(self.grid_w):
                self.grid_input[i][j].setText("0")

    def update_grid(self):
        h = self.ui.filter_h.value()
        w = self.ui.filter_w.value()
        self.hide_all()
        self.clear_all()
        for i in range(h):
            for j in range(w):
                self.grid_input[i][j].show()

    def collect_filter(self):
        h = self.ui.filter_h.value()
        w = self.ui.filter_w.value()
        kernel = np.zeros((h, w))
        for i in range(h):
            for j in range(w):
                kernel[i][j] = float(self.grid_input[i][j].text())

        filter = {
            "kernel": kernel,
            "color_shift": int(self.ui.color_shift.text()),
            "divider": float(self.ui.divider.text()),
            "name": self.ui.filter_name.text(),
        }

        return filter

    def unpack_filter(self, filter):
        h, w = filter["kernel"].shape
        for i in range(h):
            for j in range(w):
                self.grid_input[i][j].setText(str(filter["kernel"][i][j]))

        self.ui.color_shift.setText(str(filter["color_shift"]))
        self.ui.divider.setText(str(filter["divider"]))
        self.ui.filter_name.setText(str(filter["name"]))

        self.ui.filter_h.setValue(h)
        self.ui.filter_w.setValue(w)

    def open_filter(self):
        filename, _ = QFileDialog.getOpenFileName(
            None, "Open filter", ".json", "JSON (*.json)"
        )
        if not filename:
            return

        with open(filename, "r") as file:
            filter = json.loads(file.read())

        filter["kernel"] = np.array(filter["kernel"])

        self.unpack_filter(filter)

    def save_filter(self):
        filter = self.collect_filter()
        filename, _ = QFileDialog.getSaveFileName(
            None, "Save filter", f'{filter["name"]}.json', "JSON (*.json)"
        )
        if not filename:
            return

        # print(filter)
        filter["kernel"] = filter["kernel"].tolist()

        with open(filename, "w") as file:
            file.write(json.dumps(filter))

    def get_filter(self):
        filter = self.collect_filter()
        return filter


class MaskEditorScene(QGraphicsScene):
    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

    def mouseMoveEvent(self, event):
        item = self.itemAt(event.scenePos().toPoint(), QtGui.QTransform())
        if isinstance(item, ImageMaskItem):
            item.mouseMoveEvent(event)
        super().mouseMoveEvent(event)


class MaskEditorView(QGraphicsView):
    tool_size_changed = QtCore.pyqtSignal(int)

    def __init__(self, parent=None, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)
        self._zoom = 1
        self.setMouseTracking(True)
        self.tool_size = 5

    def wheelEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        if bool(modifiers == QtCore.Qt.ControlModifier):
            factor = 1.2

            if event.angleDelta().y() > 0 and self._zoom <= 100:
                self._zoom *= factor
                self.scale(factor, factor)
            elif self._zoom >= 0.05:
                self._zoom *= 1 / factor
                self.scale(1 / factor, 1 / factor)
        else:
            new_size = self.tool_size + event.angleDelta().y() // 10
            self.tool_size = max(5, min(100, new_size))
            self.tool_size_changed.emit(self.tool_size)

        super().wheelEvent(event)


class MaskEditor(SCWidget):
    LIGHT_RED = "#DB4544"
    LIGHT_GREEN = "#44DB71"
    RED = "#8F1110"
    GREEN = "#259b48"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter editor")
        self.ui = mask_editor_ui.Ui_Form()
        self.ui.setupUi(self)
        self.current_mask = None
        self.ui.maskList.currentTextChanged.connect(self.select_mask)
        self.ui.nameEditor.editingFinished.connect(self.rename_validate)
        self.ui.saveMaskBtn.clicked.connect(self.save_mask)
        self.ui.newMaskBtn.clicked.connect(self.new_mask)

        self.ui.RectBtn.clicked.connect(lambda: self.update_tool("Rect"))
        self.ui.ElipseBtn.clicked.connect(lambda: self.update_tool("Ellipse"))
        self.ui.RectBrushBtn.clicked.connect(lambda: self.update_tool("RectBrush"))
        self.ui.CircleBrushBtn.clicked.connect(lambda: self.update_tool("CircleBrush"))
        self.ui.DrawOrErase.stateChanged.connect(
            lambda x: self.update_draw_or_erase(not x)
        )

        self.editor_view = MaskEditorView()
        sip.delete(self.ui.editorView)
        self.editor_scene = MaskEditorScene()
        self.editor_view.setScene(self.editor_scene)
        self.ui.view_layout.addWidget(self.editor_view)

    def set_image(self, image=None):
        self.image = image
        if self.image is not None:
            self.update_masks()

    def update_tool(self, tool):
        if self.image is None:
            return
        self.masks[self.current_mask].set_tool_type(tool)

    def update_draw_or_erase(self, val):
        if self.image is None:
            return
        self.masks[self.current_mask].set_draw_or_erase(val)

    def update_masks(self):
        self.masks = self.image.get_masks()
        self.update_masks_list()

    def update_masks_list(self):
        self.ui.maskList.clear()
        for mask_name in self.masks:
            self.ui.maskList.addItem(mask_name)
            self.editor_view.tool_size_changed.connect(
                self.masks[mask_name].set_tool_size
            )

    def update_image_masks(self):
        self.image.set_masks(self.masks)

    def get_free_name(self, name):
        raw_name = name
        i = 0
        while name in self.masks:
            name = f"{raw_name} ({i})"
            i += 1
        return name

    def new_mask(self):
        name = self.get_free_name("new mask")
        self.ui.nameEditor.setText(name)
        self.ui.maskList.addItem(name)
        self.masks[name] = ImageMaskItem(self.image)
        self.editor_view.tool_size_changed.connect(self.masks[name].set_tool_size)
        self.select_mask(name)

    def rename_validate(self, name=None):
        name = name or self.ui.nameEditor.text()
        if name != self.current_mask:
            if name not in self.masks:
                self.ui.nameEditor.setStyleSheet(f"background-color: {self.GREEN};")
                return True
            else:
                self.ui.nameEditor.setStyleSheet(f"background-color: {self.RED};")
                return False

        return True

    def rename_mask(self, name=None):
        name = name or self.ui.nameEditor.text()
        if name != self.current_mask:
            if name not in self.masks:
                val = self.masks[self.current_mask]
                del self.masks[self.current_mask]
                self.masks[name] = val
                self.current_mask = name
                self.ui.nameEditor.setStyleSheet(f"background-color: {self.GREEN};")
                self.update_image_masks()
                self.update_masks_list()
                return True
            else:
                self.ui.nameEditor.setStyleSheet(f"background-color: {self.RED};")
                return False

        return True

    def select_mask(self, name):
        if self.current_mask is not None and self.current_mask in self.masks:
            self.editor_scene.removeItem(self.masks[self.current_mask])

        self.current_mask = name
        if self.current_mask is not None and self.current_mask in self.masks:
            self.ui.nameEditor.setText(name)
            self.ui.DrawOrErase.setCheckState(
                not self.masks[self.current_mask].draw_or_erase
            )
            self.editor_scene.addItem(self.masks[self.current_mask])

    def save_mask(self):
        save = self.rename_mask()
        if save:
            self.update_image_masks()

    def open(self):
        self.editor_scene.clear()
        self.current_mask = None
        self.show()


class MaskEditorMenuWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = mask_editor_menu_ui.Ui_Form()
        self.ui.setupUi(self)

        self.editor = MaskEditor()
        self.ui.openEditor.clicked.connect(self.editor.open)
        self.ui.useMask.stateChanged.connect(self.update_mask_activiti)
        self.ui.maskSelect.currentTextChanged.connect(self.update_current_mask)
        self.image = None

    def update_masks(self):
        masks_names = list(self.image.get_masks().keys())
        self.ui.maskSelect.clear()
        self.ui.maskSelect.addItems(masks_names)

    def update_current_mask(self, current_mask_name):
        if self.image is None:
            return
        self.image.current_mask = current_mask_name

    def update_mask_activiti(self, val):
        if self.image is None:
            return
        self.image.use_mask = val

    def set_image(self, image=None):
        if self.image is not None:
            self.image.on_update_masks.disconnect(self.update_masks)

        self.image = image
        if self.image is not None:
            self.image.on_update_masks.connect(self.update_masks)
            self.ui.useMask.setTristate(self.image.use_mask)
        self.editor.set_image(image)


class EffectSettings(QWidget):
    def __init__(self, effect_manager, parent=None):
        QWidget.__init__(self, parent)

        self.effect_manager = effect_manager

        self.filter_editor = FilterEditor()
        self.filter_editor.on_add_filter.connect(self.add_filter)

        self.verticalLayout = QtWidgets.QVBoxLayout(self)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.effectGroup = QtWidgets.QGroupBox(self)
        self.effectGroup.setTitle("Effect settings")
        self.verticalLayout.addWidget(self.effectGroup)

        layout = QtWidgets.QHBoxLayout(self.effectGroup)

        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        layout.addItem(spacerItem)

        self.effectLayout = QtWidgets.QVBoxLayout(self.effectGroup)
        layout.addLayout(self.effectLayout)

        spacerItem = QtWidgets.QSpacerItem(
            40, 20, QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Minimum
        )
        layout.addItem(spacerItem)

        self.effectGroup.setLayout(layout)

        self.new_filter = QtWidgets.QPushButton(self.effectGroup)
        self.new_filter.setText("New filter")
        self.new_filter.clicked.connect(self.open_editor)

        self.load_filter = QtWidgets.QPushButton(self.effectGroup)
        self.load_filter.setText("Load filter")
        self.load_filter.clicked.connect(self.load_filter_func)

        self.effectLayout.addWidget(self.new_filter)
        self.effectLayout.addWidget(self.load_filter)

    def open_editor(self):
        self.filter_editor.show()

    def add_filter(self):
        filter = self.filter_editor.get_filter()
        self.effect_manager.add_effect(filter["name"], FilterEffectWidget(filter))

    def load_filter_func(self):
        self.filter_editor.open_filter()
        self.add_filter()

    def set_image(self, image=None):
        pass


class EffectManagerWidget(QWidget):
    default_effects = {
        "Blur": BlurEffectWidget,
        "Sharpening": SharpeningEffectWidget,
        "Median": MedianEffectWidget,
        "Edge detection central": EdgeDetectionCentralEffectWidget,
        "Edge detection vertical": EdgeDetectionVerticalEffectWidget,
        "Edge detection horizontal": EdgeDetectionHorizontalEffectWidget,
        "First embossing": FirstEmbossingEffectWidget,
        "Second embossing": SecondEmbossingEffectWidget,
    }

    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.ui = effect_manager_ui.Ui_Form()
        self.ui.setupUi(self)

        self.effects = {}
        self.image = None

        self.add_effect("Effect settings", EffectSettings(self), idx=0, set_name=False)

        for ef in reversed(list(self.default_effects.keys())):
            self.add_effect(ef, self.default_effects[ef]())

        self.ui.selectEffect.currentTextChanged.connect(self.change_effect)

        self.set_image()
        self.change_effect()

    def hide_all_effects(self):
        for ef in self.effects:
            self.effects[ef].hide()

    def change_effect(self, effect_name=None):
        if effect_name is None:
            effect_name = self.ui.selectEffect.currentText()

        self.hide_all_effects()
        self.effects[effect_name].show()

    def set_image(self, image=None):
        self.image = image
        for ef in self.effects:
            self.effects[ef].set_image(image)

    def add_effect(self, name, effect, *, idx=-2, set_name=True):
        raw_name = name
        i = 0
        while name in self.effects:
            name = f"{raw_name} ({i})"
            i += 1

        self.effects[name] = effect
        self.ui.effectContainer.insertWidget(idx, self.effects[name])
        self.ui.selectEffect.insertItem(idx, name)
        if set_name:
            self.effects[name].set_name(name)
        self.effects[name].hide()
        self.effects[name].set_image(self.image)


class MyUi_MainWindow(Ui_MainWindow):
    def setupUi(self, MainWindow, *args, **kwargs):
        super().setupUi(MainWindow, *args, **kwargs)

        MainWindow.setWindowTitle("PyMage")
        parent = self.editorMenu.parent()
        sip.delete(self.editorView)
        self.editorView = EditorView(parent)
        self.editorView.setMinimumSize(QtCore.QSize(400, 300))
        self.editorView.show()

        parent = self.editorMenu.parent()
        sip.delete(self.editorMenu)
        self.editorMenu = ImageEditorMenu(parent)
        self.editorMenu.setMinimumSize(QtCore.QSize(205, 0))
        self.editorMenu.setMaximumSize(QtCore.QSize(300, 16777215))
        self.editorMenu.setBaseSize(QtCore.QSize(205, 0))
        self.editorMenu.show()

        self.editor_scene = EditorScene(self.editorMenu, MainWindow)
        self.editor_scene.update_info.connect(self.statusbar.showMessage)
        self.editorView.setScene(self.editor_scene)


class MyQMainWindow(QMainWindow):
    closed = QtCore.pyqtSignal()

    def closeEvent(self, event):
        self.closed.emit()


def main():
    import sys

    app = QApplication(sys.argv)

    mainWindow = MyQMainWindow()
    mainWindow.closed.connect(app.quit)

    ui = MyUi_MainWindow()
    ui.setupUi(mainWindow)

    mainWindow.show()

    app.exec()


if __name__ == "__main__":
    main()
