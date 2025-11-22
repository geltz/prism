import sys
import os
import tempfile
import math
import time
import numpy as np
import soundfile as sf
from scipy import signal

from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QUrl, QTimer, QRectF, QPointF,
                          QPropertyAnimation, QEasingCurve)
from PyQt6.QtGui import (QColor, QPainter, QLinearGradient, QPen, QPainterPath, 
                         QRadialGradient, QBrush, QFont, QPixmap, QCursor)
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QSlider, QPushButton, 
                             QFileDialog, QFrame, QGraphicsDropShadowEffect,
                             QMessageBox, QSizePolicy, QCheckBox,
                             QGraphicsOpacityEffect)
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

BPM = 120.0
SR_DEFAULT = 44100

class AudioEngine:
    @staticmethod
    def load_file(path):
        data, sr = sf.read(path, always_2d=False)
        # work in mono initially for slicing ease
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data.astype(np.float32), sr

    @staticmethod
    def save_file(path, data, sr):
        sf.write(path, data, sr)

    @staticmethod
    def simple_reverb(x, sr, mix=0.3, room_size=0.8, damp=0.5):
        if mix <= 0: return x
        if x.ndim == 1:
            x = np.column_stack((x, x))
        mono_x = x.mean(axis=1)
        tail_len = int(sr * (1.5 + room_size * 2.0))
        padded_x = np.pad(mono_x, (0, tail_len))
        noise = np.random.randn(tail_len)
        t = np.linspace(0, 1, tail_len)
        env = np.exp(-t * (4.0 + (1.0-room_size)*10))
        ir = signal.lfilter(*signal.butter(1, 0.2 * (1-damp), btype='low'), noise * env)
        wet_sig = signal.fftconvolve(padded_x, ir, mode='full')[:len(x)]
        wet_sig = wet_sig / (np.max(np.abs(wet_sig)) + 1e-9)
        wet_stereo = np.column_stack((wet_sig, wet_sig))
        out = (1 - mix) * x + mix * wet_stereo
        return out

    @staticmethod
    def make_slice_library(data, sr, num_slices=64):
        slices = []
        min_len = int(0.05 * sr)
        max_len = int(0.4 * sr)
        n_samples = len(data)
        rng = np.random.default_rng()
        for _ in range(num_slices):
            l = rng.integers(min_len, max_len)
            start = rng.integers(0, max(1, n_samples - l))
            segment = data[start:start+l]
            fade = int(0.005 * sr) 
            if len(segment) > fade * 2:
                env = np.ones_like(segment)
                env[:fade] = np.linspace(0, 1, fade)
                env[-fade:] = np.linspace(1, 0, fade)
                segment = segment * env
            slices.append(segment)
        if not slices: slices = [data]
        return slices

    @staticmethod
    def generate_2bar_loop(slice_lib, sr, density):
        rng = np.random.default_rng()
        beat_dur = 60.0 / BPM
        samples_per_beat = int(beat_dur * sr)
        bar_samples = samples_per_beat * 4
        loop_samples = bar_samples * 2
        out = np.zeros(loop_samples, dtype=np.float32)
        cursor = 0
        samples_16th = samples_per_beat // 4
        
        while cursor < loop_samples:
            r = rng.random()
            if r < (0.2 + 0.6 * density): grid_mult = 1 
            elif r < (0.6 + 0.3 * density): grid_mult = 2 
            else: grid_mult = 4 
            dur = samples_16th * grid_mult
            if cursor + dur > loop_samples: dur = loop_samples - cursor
            slc = slice_lib[rng.integers(len(slice_lib))]
            if len(slc) >= dur: chunk = slc[:dur]
            else:
                repeats = int(np.ceil(dur / len(slc)))
                chunk = np.tile(slc, repeats)[:dur]
            out[cursor:cursor+dur] = chunk
            cursor += dur
        return out

    @staticmethod
    def apply_rand_filter(data, sr):
        rng = np.random.default_rng()
        step = int((60/BPM/4) * sr)
        y = data.copy()
        total_len = len(y)
        for i in range(0, total_len, step):
            if rng.random() > 0.25: continue
            end = min(i + step, total_len)
            chunk = y[i:end]
            ftype = rng.choice(['lp', 'hp', 'bp'])
            if ftype == 'lp':
                freq = rng.uniform(300, 1200)
                sos = signal.butter(2, freq, 'low', fs=sr, output='sos')
            elif ftype == 'hp':
                freq = rng.uniform(2000, 5000)
                sos = signal.butter(2, freq, 'high', fs=sr, output='sos')
            else: 
                center = rng.uniform(400, 3000)
                sos = signal.butter(2, [center*0.8, center*1.2], 'band', fs=sr, output='sos')
            if chunk.ndim == 2:
                chunk[:, 0] = signal.sosfilt(sos, chunk[:, 0])
                chunk[:, 1] = signal.sosfilt(sos, chunk[:, 1])
            else:
                chunk = signal.sosfilt(sos, chunk)
            y[i:end] = chunk
        return y

    @staticmethod
    def apply_vol_pan(data, sr):
        rng = np.random.default_rng()
        step = int((60/BPM/4) * sr)
        if data.ndim == 1:
            left = data.copy()
            right = data.copy()
        else:
            left = data[:, 0].copy()
            right = data[:, 1].copy()
        total_len = len(left)
        for i in range(0, total_len, step):
            end = min(i + step, total_len)
            vol = rng.uniform(0.4, 1.0)
            pan = rng.uniform(-0.8, 0.8)
            p_ang = (pan + 1) * (np.pi / 4)
            g_left = np.cos(p_ang) * vol
            g_right = np.sin(p_ang) * vol
            left[i:end] *= g_left
            right[i:end] *= g_right
        return np.column_stack((left, right))

    @staticmethod
    def bitcrush(data, sr, depth=0.0):
        y = data.copy()
        if depth > 0:
            quant = 2 ** (16 - (depth * 12))
            y = np.round(y * quant) / quant
            rate_div = depth 
            if rate_div > 0:
                step = int(1 + (rate_div * 20))
                if y.ndim == 2:
                    for ch in range(2):
                        y[:, ch] = np.repeat(y[::step, ch], step)[:len(y)]
                else:
                    y = np.repeat(y[::step], step)[:len(y)]
        return y

    @staticmethod
    def apply_samplerate(data, original_sr, target_idx):
        rates = [8000, 11025, 16000, 32000, 44100]
        target_sr = rates[int(target_idx)]
        if target_sr >= original_sr: return data
        num_samples_low = int(len(data) * (target_sr / original_sr))
        if data.ndim == 2:
            l_lo = signal.resample(data[:, 0], num_samples_low)
            l_hi = signal.resample(l_lo, len(data))
            r_lo = signal.resample(data[:, 1], num_samples_low)
            r_hi = signal.resample(r_lo, len(data))
            return np.column_stack((l_hi, r_hi))
        else:
            lo_fi = signal.resample(data, num_samples_low)
            restored = signal.resample(lo_fi, len(data))
            return restored

    @staticmethod
    def process(audio, sr, params):
        slices = AudioEngine.make_slice_library(audio, sr)
        loop_data = AudioEngine.generate_2bar_loop(slices, sr, density=params['glitch'])
        target_len = max(len(audio), sr * 10) 
        repeats = int(np.ceil(target_len / len(loop_data)))
        y = np.tile(loop_data, repeats)[:target_len]
        
        if params['rand_filter']: y = AudioEngine.apply_rand_filter(y, sr)
        if params['vol_pan']: y = AudioEngine.apply_vol_pan(y, sr)
        else:
            if y.ndim == 1: y = np.column_stack((y, y))

        y = AudioEngine.bitcrush(y, sr, depth=params['crush'])
        y = AudioEngine.apply_samplerate(y, sr, params['sr_select'])

        rate = params.get('rate', 1.0)
        if rate != 1.0 and rate > 0:
            new_len = int(len(y) / rate)
            y = signal.resample(y, new_len)

        if params['wash'] > 0:
            y = AudioEngine.simple_reverb(y, sr, mix=0.5*params['wash'], room_size=0.9, damp=0.1)
        y = AudioEngine.simple_reverb(y, sr, mix=params['reverb'], room_size=0.85)
        
        peak = np.max(np.abs(y))
        if peak > 1.0: y = y / peak
        return y.astype(np.float32)

class ProcessThread(QThread):
    finished_ok = pyqtSignal(object, int) 
    error = pyqtSignal(str)

    def __init__(self, file_path, params):
        super().__init__()
        self.path, self.params = file_path, params

    def run(self):
        try:
            audio, sr = AudioEngine.load_file(self.path)
            processed = AudioEngine.process(audio, sr, self.params)
            self.finished_ok.emit(processed, sr)
        except Exception as e:
            self.error.emit(str(e))

STYLES = """
    QMainWindow { background-color: #f6f9fc; }
    QLabel { color: #405165; font-family: 'Segoe UI', sans-serif; }
    QLabel#SubHeader { color: #7aa6d4; font-size: 12px; font-weight: bold; letter-spacing: 1px; }
    
    /* File Label Enhanced */
    QLabel#FileLabel { 
        color: #3f6c9b; 
        font-size: 14px; 
        font-weight: bold; 
    }

    QSlider::groove:horizontal { border: 1px solid #bbb; background: rgba(63, 108, 155, 0.15); height: 6px; border-radius: 3px; }
    QSlider::sub-page:horizontal { background: #7aa6d4; border-radius: 3px; }
    QSlider::handle:horizontal { background: #fff; border: 2px solid #7aa6d4; width: 14px; height: 14px; margin: -5px 0; border-radius: 9px; }
    QSlider::handle:horizontal:hover { border-color: #3f6c9b; background: #f0f4f8; }
    
    QPushButton { background-color: rgba(255, 255, 255, 0.6); border: 1px solid #7aa6d4; color: #3f6c9b; border-radius: 6px; padding: 6px 12px; font-weight: bold; }
    QPushButton:hover { background-color: #3f6c9b; color: white; }
    QPushButton:pressed { background-color: #2c4e70; }
    
    QPushButton#ProcessBtn, QPushButton#SaveBtn { 
        background-color: #3f6c9b; 
        color: white; 
        font-size: 13px; 
        padding: 10px; 
        border: none;
    }
    QPushButton#ProcessBtn:hover, QPushButton#SaveBtn:hover { background-color: #5a8fbe; }
    QPushButton#ProcessBtn:disabled, QPushButton#SaveBtn:disabled { background-color: #d0dbe5; color: #f0f0f0; }

    /* Revised Clear Button */
    QPushButton#ClearBtn { 
        background-color: rgba(217, 83, 79, 0.1); 
        border: 1px solid rgba(217, 83, 79, 0.3); 
        color: #d9534f; 
        border-radius: 10px;
        padding: 2px 10px; 
        font-size: 11px; 
        font-weight: bold; 
    }
    QPushButton#ClearBtn:hover { background-color: #d9534f; color: white; }

    QCheckBox { color: #405165; font-weight: bold; font-family: 'Segoe UI'; spacing: 8px; }
    QCheckBox::indicator { width: 16px; height: 16px; border: 1px solid #7aa6d4; border-radius: 4px; background: white; }
    QCheckBox::indicator:checked { background: #3f6c9b; border: 1px solid #3f6c9b; }
"""

class AnimatedTitle(QLabel):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.phase = 0.0
        font = self.font()
        font.setFamily("Segoe UI")
        font.setPixelSize(22)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 4.0)
        self.setFont(font)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)
        # Reduced Opacity for subtle look
        self.color_base = QColor(63, 108, 155, 160)
        self.color_highlight = QColor(209, 227, 246, 200)

    def animate(self):
        self.phase = (self.phase + 0.01) % (2 * math.pi)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        t = (self.phase / (2 * math.pi)) * 2.0 
        w, bw = self.width(), 150.0
        center_x = (t * w * 1.5) - (w * 0.25)
        grad = QLinearGradient(center_x - bw, 0, center_x + bw, 0)
        grad.setColorAt(0.0, self.color_base)
        grad.setColorAt(0.5, self.color_highlight)
        grad.setColorAt(1.0, self.color_base)
        painter.setPen(QPen(QBrush(grad), 0))
        painter.setFont(self.font())
        painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self.text())

class GlassFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("GlassFrame { background-color: rgba(255, 255, 255, 0.65); border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 16px; }")
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setColor(QColor(63, 108, 155, 15))
        shadow.setOffset(0, 4)
        self.setGraphicsEffect(shadow)

class WaveformWidget(QWidget):
    seek_requested = pyqtSignal(float)
    import_clicked = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.data = None
        self.play_head_pos = 0.0
        self.setMinimumHeight(160)
        self.setMouseTracking(True) 
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        # performance: cache the static visuals
        self._static_pixmap = None

    def set_data(self, data):
        if data.ndim > 1: d_mono = data.mean(axis=1)
        else: d_mono = data  
        # aggressive downsample for ui performance
        target_points = 1000
        step = max(1, len(d_mono) // target_points)
        self.data = d_mono[::step]
        self.play_head_pos = 0.0
        self.update_static_waveform()
        self.update()

    def set_play_head(self, pos):
        self.play_head_pos = max(0.0, min(1.0, pos))
        self.update() # Triggers paintEvent

    def resizeEvent(self, event):
        self.update_static_waveform()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self.data is None: self.import_clicked.emit()
        else: self.handle_input(event.pos().x())

    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.data is not None:
            self.handle_input(event.pos().x())

    def handle_input(self, x):
        w = self.width()
        if w > 0:
            pos_norm = max(0, min(w, x)) / w
            self.set_play_head(pos_norm)
            self.seek_requested.emit(pos_norm)

    def update_static_waveform(self):
        """Draws the expensive waveform/gradient to a Pixmap once."""
        w, h = self.width(), self.height()
        if w == 0 or h == 0: return

        self._static_pixmap = QPixmap(w, h)
        self._static_pixmap.fill(Qt.GlobalColor.transparent)

        if self.data is None:
            painter = QPainter(self._static_pixmap)
            painter.setPen(QColor("#6c8dab"))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(self._static_pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "drop file or click to import")
            painter.end()
            return

        # prepare paint on pixmap
        painter = QPainter(self._static_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # build path
        path = QPainterPath()
        cy = h / 2
        path.moveTo(0, cy)
        x_step = w / len(self.data)
        amp_scale = h * 0.45
        
        for i, val in enumerate(self.data):
            path.lineTo(i * x_step, cy - (val * amp_scale))
        
        path.lineTo(w, cy)
        path.lineTo(0, cy)
        
        # Gradient
        grad = QLinearGradient(0, 0, 0, h)
        base_color = QColor(63, 108, 155)
        base_color.setAlpha(0)
        grad.setColorAt(0.0, base_color)
        base_color.setAlpha(40)
        grad.setColorAt(0.2, base_color)
        base_color.setAlpha(180)
        grad.setColorAt(0.5, base_color)
        base_color.setAlpha(40)
        grad.setColorAt(0.8, base_color)
        base_color.setAlpha(0)
        grad.setColorAt(1.0, base_color)

        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)
        painter.end()

    def paintEvent(self, event):
        painter = QPainter(self)
        # subtle container
        painter.setBrush(QColor(255, 255, 255, 50))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 4, 4)

        # blit cached visuals (fast)
        if self._static_pixmap:
            painter.drawPixmap(0, 0, self._static_pixmap)

        # draw dynamic playhead
        if self.data is not None and self.play_head_pos >= 0:
            px = int(self.play_head_pos * self.width())
            painter.setPen(QPen(QColor(63, 108, 155, 200), 1.5)) 
            painter.drawLine(px, 10, px, self.height() - 10)

def setup_row_layout(widget):
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 2, 0, 2) 
    layout.setSpacing(2)
    return layout

class ControlRow(QWidget):
    def __init__(self, label, key, parent_data):
        super().__init__()
        self.key, self.parent_data = key, parent_data
        layout = setup_row_layout(self)
        header = QHBoxLayout()
        self.lbl_name = QLabel(label.lower())
        self.lbl_val = QLabel("0")
        self.lbl_val.setStyleSheet("color: #3f6c9b; font-weight: bold;")
        header.addWidget(self.lbl_name)
        header.addStretch()
        header.addWidget(self.lbl_val)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.valueChanged.connect(self.update_val)
        layout.addLayout(header)
        layout.addWidget(self.slider)
        
    def update_val(self, val):
        self.lbl_val.setText(f"{val}")
        self.parent_data[self.key] = val / 100.0

class RateControlRow(QWidget):
    def __init__(self, label, key, parent_data):
        super().__init__()
        self.key, self.parent_data = key, parent_data
        layout = setup_row_layout(self)
        header = QHBoxLayout()
        self.lbl_name = QLabel(label.lower())
        self.lbl_val = QLabel("1.0x")
        self.lbl_val.setStyleSheet("color: #3f6c9b; font-weight: bold;")
        header.addWidget(self.lbl_name)
        header.addStretch()
        header.addWidget(self.lbl_val)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(5, 20)
        self.slider.setValue(10)
        self.slider.valueChanged.connect(self.update_val)
        layout.addLayout(header)
        layout.addWidget(self.slider)
        
    def update_val(self, val):
        real_val = val / 10.0
        self.lbl_val.setText(f"{real_val:.1f}x")
        self.parent_data[self.key] = real_val

class DiscreteControlRow(QWidget):
    def __init__(self, label, key, parent_data, values):
        super().__init__()
        self.key = key
        self.parent_data = parent_data
        self.values = values
        layout = setup_row_layout(self)
        header = QHBoxLayout()
        self.lbl_name = QLabel(label.lower())
        self.lbl_val = QLabel(f"{values[-1]}")
        self.lbl_val.setStyleSheet("color: #3f6c9b; font-weight: bold;")
        header.addWidget(self.lbl_name)
        header.addStretch()
        header.addWidget(self.lbl_val)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, len(values) - 1)
        self.slider.setValue(len(values) - 1)
        self.slider.valueChanged.connect(self.update_val)
        layout.addLayout(header)
        layout.addWidget(self.slider)
        
    def update_val(self, index):
        real_val = self.values[index]
        self.lbl_val.setText(f"{real_val}")
        self.parent_data[self.key] = float(index)

class ToggleRow(QWidget):
    def __init__(self, label_1, key_1, label_2, key_2, parent_data):
        super().__init__()
        self.parent_data = parent_data
        self.key1, self.key2 = key_1, key_2
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 5, 0, 5)
        self.cb1 = QCheckBox(label_1.lower())
        self.cb1.stateChanged.connect(self.update_1)
        self.cb2 = QCheckBox(label_2.lower())
        self.cb2.stateChanged.connect(self.update_2)
        layout.addWidget(self.cb1)
        layout.addStretch()
        layout.addWidget(self.cb2)
        
    def update_1(self, state): self.parent_data[self.key1] = (state == 2)
    def update_2(self, state): self.parent_data[self.key2] = (state == 2)

class MediaButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(50, 50) 
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("border: none; background: transparent;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg_color = QColor(255, 255, 255, 200)
        if self.underMouse(): bg_color = QColor(255, 255, 255, 255)
        if self.isDown(): bg_color = QColor(200, 220, 240)
        if not self.isEnabled(): bg_color = QColor(255, 255, 255, 100)
        
        # Draw larger rounded rect
        painter.setBrush(bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 25, 25) 

        icon_color = QColor("#3f6c9b") if self.isEnabled() else QColor("#a0b0c0")
        painter.setBrush(icon_color)
        cx, cy, txt = self.width() / 2, self.height() / 2, self.text()
        
        # Scaled up drawing coordinates
        if "▶" in txt:
            path = QPainterPath()
            path.moveTo(cx - 6, cy - 9) 
            path.lineTo(cx - 6, cy + 9)
            path.lineTo(cx + 12, cy)
            path.closeSubpath()
            painter.drawPath(path)
        elif "||" in txt:
            painter.drawRoundedRect(QRectF(cx - 9, cy - 9, 6, 18), 2, 2)
            painter.drawRoundedRect(QRectF(cx + 3, cy - 9, 6, 18), 2, 2)
        elif "■" in txt:
            painter.drawRoundedRect(QRectF(cx - 8, cy - 8, 16, 16), 3, 3)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("prism")
        self.resize(900, 600)
        self.setAcceptDrops(True)
        self.file_path = None
        self.processed_audio = None
        self.sr = 44100
        self.temp_file = None
        
        self.params = {
            'glitch': 0.5, 'wash': 0.0, 'crush': 0.0, 'reverb': 0.0, 
            'sr_select': 4.0, 'rate': 1.0, 'rand_filter': False, 'vol_pan': False
        }

        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self.update_playhead)
        self.player.mediaStatusChanged.connect(self.media_status_changed)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(15)

        self.viewport = QVBoxLayout()
        self.wave_view = WaveformWidget()
        self.wave_view.seek_requested.connect(self.seek_audio)
        self.wave_view.import_clicked.connect(self.open_file_dialog)

        info_row = QHBoxLayout()
        info_row.addStretch()
        
        self.lbl_file = QLabel("no file loaded")
        self.lbl_file.setObjectName("FileLabel")
        self.lbl_file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_clear = QPushButton("clear")
        self.btn_clear.setObjectName("ClearBtn")
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_clear.setVisible(False)
        self.btn_clear.clicked.connect(self.clear_state)
        
        info_row.addWidget(self.lbl_file)
        info_row.addSpacing(10)
        info_row.addWidget(self.btn_clear)
        info_row.addStretch()

        self.viewport.addWidget(self.wave_view, 1)
        self.viewport.addSpacing(10)
        self.viewport.addLayout(info_row)
        self.viewport.addStretch()
        
        # Right Column
        self.sidebar = GlassFrame()
        self.sidebar.setFixedWidth(300)
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(15, 20, 15, 20)
        side_layout.setSpacing(6) 

        side_layout.addWidget(AnimatedTitle("prism"))
        self.lbl_status = QLabel("status: idle")
        self.lbl_status.setObjectName("SubHeader")
        side_layout.addWidget(self.lbl_status)
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: rgba(63, 108, 155, 0.2);")
        side_layout.addWidget(line)
        side_layout.addSpacing(5)

        side_layout.addWidget(ControlRow("grid density", "glitch", self.params))
        side_layout.addWidget(ControlRow("spectral wash", "wash", self.params))
        side_layout.addWidget(ControlRow("bit crush", "crush", self.params))
        side_layout.addWidget(RateControlRow("rate", "rate", self.params))
        sr_opts = [8000, 11025, 16000, 32000, 44100]
        side_layout.addWidget(DiscreteControlRow("samplerate", "sr_select", self.params, sr_opts))
        side_layout.addWidget(ControlRow("glue reverb", "reverb", self.params))
        side_layout.addWidget(ToggleRow("rand filter", "rand_filter", "vol/pan mod", "vol_pan", self.params))

        side_layout.addStretch()

        self.transport_frame = GlassFrame()
        transport_layout = QHBoxLayout(self.transport_frame)
        transport_layout.setContentsMargins(10, 20, 10, 20) 
        
        self.btn_play = MediaButton("▶")
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_play.setEnabled(False)
        self.btn_stop = MediaButton("■")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        
        transport_layout.addStretch()
        transport_layout.addWidget(self.btn_stop)
        transport_layout.addWidget(self.btn_play)
        transport_layout.addStretch()
        side_layout.addWidget(self.transport_frame)

        action_layout = QHBoxLayout()
        action_layout.setSpacing(10)
        
        self.btn_process = QPushButton("process")
        self.btn_process.setObjectName("ProcessBtn")
        self.btn_process.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.start_processing)
        
        self.btn_save = QPushButton("export")
        self.btn_save.setObjectName("SaveBtn")
        self.btn_save.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.quick_export)

        action_layout.addWidget(self.btn_process)
        action_layout.addWidget(self.btn_save)
        side_layout.addLayout(action_layout)

        self.lbl_saved_msg = QLabel("")
        self.lbl_saved_msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_saved_msg.setStyleSheet("color: #3f6c9b; font-size: 11px; font-weight: bold;")
        
        self.fade_effect = QGraphicsOpacityEffect(self.lbl_saved_msg)
        self.lbl_saved_msg.setGraphicsEffect(self.fade_effect)
        
        self.fade_anim = QPropertyAnimation(self.fade_effect, b"opacity")
        self.fade_anim.setDuration(1000)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        self.fade_anim.finished.connect(lambda: self.lbl_saved_msg.setText(""))
        
        side_layout.addWidget(self.lbl_saved_msg)

        main_layout.addLayout(self.viewport, 1)
        main_layout.addWidget(self.sidebar)

    def paintEvent(self, event):
        painter = QPainter(self)
        grad = QRadialGradient(self.width()/2, 0, self.width())
        grad.setColorAt(0, QColor("#eef4f9"))
        grad.setColorAt(1, QColor("#f6f9fc"))
        painter.fillRect(self.rect(), grad)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files: self.load_file(files[0])

    def open_file_dialog(self):
        fname, _ = QFileDialog.getOpenFileName(self, 'open audio', os.path.expanduser("~"), "audio files (*.wav *.mp3 *.flac *.ogg)")
        if fname: self.load_file(fname)

    def clear_state(self):
        self.stop_playback()
        self.player.setSource(QUrl())
        self.file_path = None
        self.processed_audio = None
        self.wave_view.data = None
        self.wave_view.update_static_waveform()
        self.wave_view.update()
        
        self.lbl_file.setText("no file loaded")
        self.btn_clear.setVisible(False)
        self.lbl_status.setText("status: idle")
        
        self.btn_process.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)

    def load_file(self, path):
        self.stop_playback()
        self.file_path = path
        self.lbl_file.setText(os.path.basename(path).lower())
        self.btn_clear.setVisible(True)
        try:
            data, sr = AudioEngine.load_file(path)
            self.sr = sr
            self.player.setSource(QUrl.fromLocalFile(path))
            display_data = data[:sr*30] if len(data) > sr*30 else data
            self.wave_view.set_data(display_data)
            
            self.btn_process.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.lbl_status.setText("status: ready")
        except Exception as e:
            QMessageBox.critical(self, "error", f"could not load file: {e}")

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.btn_play.setText("▶")
        else:
            self.player.play()
            self.btn_play.setText("||")

    def stop_playback(self):
        self.player.stop()
        self.btn_play.setText("▶")
        self.wave_view.set_play_head(0)

    def update_playhead(self, position_ms):
        duration = self.player.duration()
        if duration > 0: self.wave_view.set_play_head(position_ms / duration)

    def seek_audio(self, pos_norm):
        duration = self.player.duration()
        if duration > 0: self.player.setPosition(int(pos_norm * duration))

    def media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.btn_play.setText("▶")
            self.wave_view.set_play_head(0)

    def start_processing(self):
        if not self.file_path: return
        self.stop_playback()
        self.lbl_status.setText("status: processing...")
        self.lbl_status.setStyleSheet("color: #7aa6d4; font-weight: bold;")
        self.set_ui_locked(True)
        self.thread = ProcessThread(self.file_path, self.params.copy())
        self.thread.finished_ok.connect(self.processing_done)
        self.thread.error.connect(self.processing_error)
        self.thread.start()

    def processing_done(self, data, sr):
        self.processed_audio = data
        self.wave_view.set_data(data)
        self.lbl_status.setText("status: done")
        self.lbl_status.setStyleSheet("color: #64b5f6; font-weight: bold;")
        self.set_ui_locked(False)
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            AudioEngine.save_file(temp_path, data, sr)
            self.temp_file = temp_path
            self.player.setSource(QUrl.fromLocalFile(temp_path))
        except Exception as e: print(f"Temp file error: {e}")

    def processing_error(self, msg):
        QMessageBox.critical(self, "processing error", msg)
        self.lbl_status.setText("status: error")
        self.set_ui_locked(False)

    def set_ui_locked(self, locked):
        self.btn_process.setEnabled(not locked)
        self.sidebar.setEnabled(not locked)
        if not locked: self.btn_save.setEnabled(True)

    def quick_export(self):
        if self.processed_audio is None: return
        
        timestamp = int(time.time())
        filename = f"prism_export_{timestamp}.wav"
        save_path = os.path.join(os.getcwd(), filename)
        
        try:
            AudioEngine.save_file(save_path, self.processed_audio, self.sr)
            self.lbl_status.setText("status: exported")
            self.lbl_saved_msg.setText(f"saved: {filename}")
            self.fade_effect.setOpacity(1.0) 
            QTimer.singleShot(2000, self.start_fade_out)
        except Exception as e: 
            QMessageBox.critical(self, "error", f"could not save: {e}")

    def start_fade_out(self):
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.start()
    
    def closeEvent(self, event):
        if self.temp_file and os.path.exists(self.temp_file):
            try: os.remove(self.temp_file)
            except: pass
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    app.setStyleSheet(STYLES)
    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())