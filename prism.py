import ctypes
import sys
import os
import tempfile
import math
import time
import numpy as np
import soundfile as sf
from scipy import signal

from PyQt6.QtCore import (Qt, QThread, pyqtSignal, QUrl, QTimer, QRectF, QPointF,
                          QPropertyAnimation, QEasingCurve, pyqtProperty)
from PyQt6.QtGui import (QColor, QPainter, QLinearGradient, QPen, QPainterPath, 
                         QRadialGradient, QBrush, QFont, QPixmap, QCursor, 
                         QPolygonF, QIcon)
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
    def generate_4bar_loop(slice_lib, sr, density, bpm):
        rng = np.random.default_rng()
        beat_dur = 60.0 / bpm
        samples_per_beat = int(beat_dur * sr)
        bar_samples = samples_per_beat * 4

        loop_samples = bar_samples * 4 
        
        out = np.zeros(loop_samples, dtype=np.float32)
        cursor = 0
        samples_16th = samples_per_beat // 4
        
        while cursor < loop_samples:
            r = rng.random()
            if r < (0.1 + 0.4 * density): grid_mult = 1 
            elif r < (0.4 + 0.4 * density): grid_mult = 2 
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
    def apply_rand_filter(data, sr, intensity, bpm):
        if intensity <= 0.01: return data
        rng = np.random.default_rng()
        step = int((60/bpm/4) * sr)
        y = data.copy()
        total_len = len(y)
        
        for i in range(0, total_len, step):
            if rng.random() > intensity: continue
            end = min(i + step, total_len)
            chunk = y[i:end]
            ftype = rng.choice(['lp', 'hp', 'bp'])
            q_factor = 0.5 + (intensity * 1.5) 
            
            if ftype == 'lp':
                freq = rng.uniform(300, 1200)
                sos = signal.butter(2, freq, 'low', fs=sr, output='sos')
            elif ftype == 'hp':
                freq = rng.uniform(2000, 5000)
                sos = signal.butter(2, freq, 'high', fs=sr, output='sos')
            else: 
                center = rng.uniform(400, 3000)
                width = 0.4 if intensity > 0.8 else 0.8
                sos = signal.butter(2, [center*(1-width/2), center*(1+width/2)], 'band', fs=sr, output='sos')
                
            if chunk.ndim == 2:
                chunk[:, 0] = signal.sosfilt(sos, chunk[:, 0])
                chunk[:, 1] = signal.sosfilt(sos, chunk[:, 1])
            else:
                chunk = signal.sosfilt(sos, chunk)
            y[i:end] = chunk
        return y

    @staticmethod
    def apply_vol_pan(data, sr, intensity, bpm):
        if intensity <= 0.01: return data
        rng = np.random.default_rng()
        step = int((60/bpm/4) * sr)
        if data.ndim == 1:
            left = data.copy()
            right = data.copy()
        else:
            left = data[:, 0].copy()
            right = data[:, 1].copy()
        total_len = len(left)
        
        for i in range(0, total_len, step):
            end = min(i + step, total_len)
            vol_drop = rng.uniform(0.0, 0.6) * intensity
            vol = 1.0 - vol_drop
            pan_width = 0.9 * intensity
            pan = rng.uniform(-pan_width, pan_width)
            p_ang = (pan + 1) * (np.pi / 4)
            g_left = np.cos(p_ang) * vol
            g_right = np.sin(p_ang) * vol
            left[i:end] *= g_left
            right[i:end] *= g_right
            
        return np.column_stack((left, right))

    @staticmethod
    def bitcrush(data, sr, depth=0.0):
        """
        Made smoother: 
        1. Reduced max sample reduction (step) from 20 to 6.
        2. Reduced max bit depth reduction.
        """
        y = data.copy()
        if depth > 0:
            # Gentler Quantization: Max depth drops to ~8 bits instead of ~4 bits
            # (16 - 8) = 8 bits at max intensity
            quant = 2 ** (16 - (depth * 8)) 
            y = np.round(y * quant) / quant
            
            rate_div = depth 
            if rate_div > 0:
                # Gentler Downsample: Max step is 6 instead of 21
                step = int(1 + (rate_div * 5))
                if y.ndim == 2:
                    for ch in range(2):
                        # Simple hold
                        y[:, ch] = np.repeat(y[::step, ch], step)[:len(y)]
                else:
                    y = np.repeat(y[::step], step)[:len(y)]
        return y

    @staticmethod
    def apply_samplerate(data, original_sr, target_sr):
        if target_sr >= original_sr: return data
        target_sr = max(1000, target_sr)
        num_samples_low = int(len(data) * (target_sr / original_sr))
        if num_samples_low < 2: return data 

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
        bpm = params.get('bpm', 120.0)
        slices = AudioEngine.make_slice_library(audio, sr)
        y = AudioEngine.generate_4bar_loop(slices, sr, density=params['glitch'], bpm=bpm)
        
        y = AudioEngine.apply_rand_filter(y, sr, intensity=params['filter_amt'], bpm=bpm)
        y = AudioEngine.apply_vol_pan(y, sr, intensity=params['vol_pan_amt'], bpm=bpm)
        
        if y.ndim == 1: y = np.column_stack((y, y))

        y = AudioEngine.bitcrush(y, sr, depth=params['crush'])
        y = AudioEngine.apply_samplerate(y, sr, params['sr_select'])

        rate = params.get('rate', 1.0)
        if rate != 1.0 and rate > 0:
            new_len = int(len(y) / rate)
            y = signal.resample(y, new_len)

        if params['wash'] > 0:
            y = AudioEngine.simple_reverb(y, sr, mix=0.5*params['wash'], room_size=0.9, damp=0.1)
        y = AudioEngine.simple_reverb(y, sr, mix=params['reverb'] * 0.7, room_size=0.85)
        
        peak = np.max(np.abs(y))
        if peak > 1.0: y = y / peak
        return y.astype(np.float32)

class ProcessThread(QThread):
    finished_ok = pyqtSignal(object, int) 
    error = pyqtSignal(str)

    # CHANGED: Receive raw audio data instead of a file path
    def __init__(self, audio_data, sr, params):
        super().__init__()
        self.audio = audio_data
        self.sr = sr
        self.params = params

    def run(self):
        try:
            # CHANGED: Process the passed audio data directly
            # We copy to ensure the original data in memory is never touched by reference
            processed = AudioEngine.process(self.audio.copy(), self.sr, self.params)
            self.finished_ok.emit(processed, self.sr)
        except Exception as e:
            self.error.emit(str(e))

STYLES = """
    QMainWindow { background-color: #f6f9fc; }
    QLabel { color: #405165; font-family: 'Segoe UI', sans-serif; font-size: 11px; }
    QLabel#SubHeader { color: #7aa6d4; font-size: 11px; font-weight: bold; letter-spacing: 0.5px; }
    
    QLabel#FileLabel { 
        color: #3f6c9b; 
        font-size: 12px; 
        font-weight: bold; 
    }
    
    QPushButton { 
        background-color: rgba(255, 255, 255, 0.6); 
        border: 1px solid #7aa6d4; 
        color: #3f6c9b; 
        border-radius: 5px; 
        padding: 4px 10px; 
        font-weight: bold; 
        font-size: 11px;
    }
    QPushButton:hover { background-color: #3f6c9b; color: white; }
    QPushButton:pressed { background-color: #2c4e70; }
    
    QPushButton#ProcessBtn, QPushButton#SaveBtn { 
        background-color: #3f6c9b; 
        color: white; 
        font-size: 12px; 
        padding: 8px; 
        border: none;
    }
    
    QPushButton#ProcessBtn:hover, QPushButton#SaveBtn:hover { background-color: #5a8fbe; }
    QPushButton#ProcessBtn:disabled, QPushButton#SaveBtn:disabled { background-color: #d0dbe5; color: #f0f0f0; }

    QPushButton#ClearBtn { 
        background-color: rgba(217, 83, 79, 0.1); 
        border: 1px solid rgba(217, 83, 79, 0.3); 
        color: #d9534f; 
        border-radius: 8px;
        padding: 2px 8px; 
        font-size: 11px; 
        font-weight: bold; 
    }
    QPushButton#ClearBtn:hover { background-color: #d9534f; color: white; }

    QCheckBox { color: #405165; font-weight: bold; font-family: 'Segoe UI'; spacing: 8px; }
    QCheckBox::indicator { width: 14px; height: 14px; border: 1px solid #7aa6d4; border-radius: 4px; background: white; }
    QCheckBox::indicator:checked { background: #3f6c9b; border: 1px solid #3f6c9b; }
"""

class PrismLogo(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(45)
        self.phase = 0.0
        self.base_speed = 0.005
        self.current_speed = self.base_speed
        self.target_speed = self.base_speed
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

    def trigger_excitement(self):
        """Accelerate the rainbow effect then slowly drift back."""
        self.current_speed = 0.05  # Higher initial burst for visibility
        self.target_speed = self.base_speed

    def animate(self):
        # Smoothly interpolate speed back to normal (slower decay for smoothness)
        self.current_speed = self.current_speed * 0.96 + self.target_speed * 0.04
        self.phase = (self.phase + self.current_speed) % 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        tri_size = 20
        p1 = QPointF(cx, cy - tri_size)
        p2 = QPointF(cx + tri_size * 0.866, cy + tri_size * 0.5)
        p3 = QPointF(cx - tri_size * 0.866, cy + tri_size * 0.5)
        triangle = QPolygonF([p1, p2, p3])

        painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
        painter.drawLine(QPointF(0, cy + 5), QPointF(cx - 8, cy + 2))

        for i in range(7):
            # Hue now shifts with phase to visualize acceleration
            hue = (i / 7.0 - self.phase) % 1.0
            
            # Alpha pulse still adds to the effect
            pulse = (math.sin(self.phase * 6.28 + i) + 1) / 2
            alpha = int(100 + 155 * pulse)
            
            col = QColor.fromHslF(hue, 0.7, 0.7, alpha/255.0)
            painter.setPen(QPen(col, 1.5))
            
            origin = QPointF(cx + 6, cy)
            angle_deg = -25 + (i * 8) 
            angle_rad = math.radians(angle_deg)
            dest_x = w
            dest_y = cy + math.tan(angle_rad) * (w - cx)
            painter.drawLine(origin, QPointF(dest_x, dest_y))

        grad = QLinearGradient(p1, p3)
        grad.setColorAt(0.0, QColor(255, 255, 255, 100))
        grad.setColorAt(1.0, QColor(255, 255, 255, 10))
        painter.setBrush(grad)
        painter.setPen(QPen(QColor(240, 240, 255), 1.5))
        painter.drawPolygon(triangle)

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
    scrub_started = pyqtSignal()
    scrub_ended = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.data = None
        self.play_head_pos = 0.0
        self.setMinimumHeight(160)
        self.setMouseTracking(True) 
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._static_pixmap = None
        self.is_scrubbing = False 

    def set_data(self, data):
        if data.ndim > 1: d_mono = data.mean(axis=1)
        else: d_mono = data  
        target_points = 2000 # Increased resolution for smoother look
        step = max(1, len(d_mono) // target_points)
        self.data = d_mono[::step]
        self.play_head_pos = 0.0
        self.update_static_waveform()
        self.update()

    def set_play_head(self, pos):
        if not self.is_scrubbing:
            self.play_head_pos = max(0.0, min(1.0, pos))
            self.update()

    def resizeEvent(self, event):
        self.update_static_waveform()
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self.data is None: 
            self.import_clicked.emit()
        elif event.button() == Qt.MouseButton.LeftButton:
            self.is_scrubbing = True
            self.scrub_started.emit()
            self.handle_input(event.pos().x())

    def mouseMoveEvent(self, event):
        # Only update visuals while dragging, don't seek audio yet
        if self.data is not None and self.is_scrubbing:
            w = self.width()
            if w > 0:
                x = event.pos().x()
                pos_norm = max(0, min(w, x)) / w
                self.play_head_pos = pos_norm
                self.update()

    def mouseReleaseEvent(self, event):
        if self.is_scrubbing:
            self.is_scrubbing = False
            # Perform the final seek here when the user lets go
            self.seek_requested.emit(self.play_head_pos)
            self.scrub_ended.emit()

    def handle_input(self, x):
        w = self.width()
        if w > 0:
            pos_norm = max(0, min(w, x)) / w
            self.play_head_pos = pos_norm
            self.seek_requested.emit(pos_norm)
            self.update()

    def update_static_waveform(self):
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

        painter = QPainter(self._static_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        cy = h / 2
        path.moveTo(0, cy)
        if len(self.data) > 0:
            x_step = w / len(self.data)
            amp_scale = h * 0.45
            for i, val in enumerate(self.data):
                path.lineTo(i * x_step, cy - (val * amp_scale))
        
        path.lineTo(w, cy)
        path.lineTo(0, cy)
        
        # Draw purely in white with alpha variation. 
        # This acts as a "mask" for the dynamic gradient in paintEvent.
        grad = QLinearGradient(0, 0, 0, h)
        c = QColor(255, 255, 255)
        c.setAlpha(0); grad.setColorAt(0.0, c)
        c.setAlpha(40); grad.setColorAt(0.2, c)
        c.setAlpha(200); grad.setColorAt(0.5, c)
        c.setAlpha(40); grad.setColorAt(0.8, c)
        c.setAlpha(0); grad.setColorAt(1.0, c)

        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawPath(path)
        painter.end()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        # --- 1. Glass Background (Matching the Sidebar) ---
        # Background: 65% opaque white
        painter.setBrush(QColor(255, 255, 255, 166)) 
        # Border: 80% opaque white
        painter.setPen(QPen(QColor(255, 255, 255, 204), 1)) 
        # Radius: 16px (Matching GlassFrame)
        painter.drawRoundedRect(rect, 16, 16)

        # --- 2. Set Clipping for Waveform Content ---
        # This ensures the waveform doesn't draw outside the rounded corners
        path = QPainterPath()
        path.addRoundedRect(QRectF(rect), 16, 16)
        painter.setClipPath(path)

        # --- 3. Dynamic Pastel Waveform (Base) ---
        if self._static_pixmap:
            colored_wave = QPixmap(self.size())
            colored_wave.fill(Qt.GlobalColor.transparent)
            cw_painter = QPainter(colored_wave)
            cw_painter.drawPixmap(0, 0, self._static_pixmap)
            cw_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            
            t = time.time() * 0.08 
            wave_grad = QLinearGradient(0, 0, self.width(), self.height())
            c1 = QColor.fromHslF((t) % 1.0, 0.6, 0.80, 1.0) 
            c2 = QColor.fromHslF((t + 0.2) % 1.0, 0.6, 0.80, 1.0)
            c3 = QColor.fromHslF((t + 0.4) % 1.0, 0.6, 0.80, 1.0)
            wave_grad.setColorAt(0.0, c1); wave_grad.setColorAt(0.5, c2); wave_grad.setColorAt(1.0, c3)
            
            cw_painter.fillRect(colored_wave.rect(), wave_grad)
            cw_painter.end()
            painter.drawPixmap(0, 0, colored_wave)

        # --- 4. The Highlight Effect ---
        if self.data is not None and self.play_head_pos >= 0 and self._static_pixmap:
            px = int(self.play_head_pos * self.width())
            h = self.height()
            ripple_w = 140 
            rect_h = QRectF(px - ripple_w, 0, ripple_w * 2, h)
            source_rect = rect_h.toRect().intersected(self.rect())
            
            if not source_rect.isEmpty():
                wave_slice = self._static_pixmap.copy(source_rect)
                slice_painter = QPainter(wave_slice)
                slice_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                
                grad_center = px - source_rect.x()
                r_grad = QLinearGradient(grad_center - ripple_w, 0, grad_center + ripple_w, 0)
                hue_now = (t + 0.1) % 1.0
                c_edge = QColor.fromHslF(hue_now, 0.8, 0.5, 0.0)
                c_center = QColor.fromHslF(hue_now, 0.85, 0.45, 0.9)
                r_grad.setColorAt(0.0, c_edge); r_grad.setColorAt(0.5, c_center); r_grad.setColorAt(1.0, c_edge)
                
                slice_painter.fillRect(wave_slice.rect(), r_grad)
                slice_painter.end()
                painter.drawPixmap(source_rect.topLeft(), wave_slice)

            # --- 5. The Playhead Line ---
            line_grad = QLinearGradient(px, 0, px, h)
            hue_shift = self.play_head_pos * 2.5 
            for i in range(5):
                t_g = i / 4.0
                h_val = (t_g * 0.25 + hue_shift) % 1.0
                col = QColor.fromHslF(h_val, 0.7, 0.85, 0.95)
                line_grad.setColorAt(t_g, col)

            painter.setBrush(line_grad)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(QRectF(px - 1, 0, 2, h))

def setup_row_layout(widget):
    layout = QVBoxLayout(widget)
    layout.setContentsMargins(0, 0, 0, 0) 
    layout.setSpacing(0)
    return layout

class PrismSlider(QSlider):
    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(24)
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)

    def animate(self):
        self.phase = (self.phase + 0.02) % 1.0
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            val = self.pixel_to_value(event.pos().x())
            self.setValue(val)
            event.accept()
            
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            val = self.pixel_to_value(event.pos().x())
            self.setValue(val)
            event.accept()

    def pixel_to_value(self, x):
        w = self.width()
        if w == 0: return 0
        norm = max(0.0, min(1.0, x / w))
        return int(self.minimum() + norm * (self.maximum() - self.minimum()))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        groove_h = 4
        groove_y = rect.height() / 2 - groove_h / 2
        groove_rect = QRectF(rect.x(), groove_y, rect.width(), groove_h)
        
        # Draw faint background groove
        painter.setBrush(QColor(63, 108, 155, 30))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(groove_rect, 2, 2)
        
        val_norm = (self.value() - self.minimum()) / (self.maximum() - self.minimum()) if (self.maximum() > self.minimum()) else 0
        
        # Draw Rainbow Fill
        if val_norm > 0.01:
            fill_width = rect.width() * val_norm
            fill_rect = QRectF(rect.x(), groove_y, fill_width, groove_h)
            
            grad = QLinearGradient(rect.x(), 0, rect.x() + fill_width, 0)
            
            stops = 8 # Increased stops for smoother pastel transitions
            for i in range(stops):
                t = i / (stops - 1)
                # Hue mapping stays the same
                hue = t * (val_norm * 0.85)

                color = QColor.fromHslF(hue, 0.75, 0.75, 1.0)
                grad.setColorAt(t, color)
                
            painter.setBrush(grad)
            painter.drawRoundedRect(fill_rect, 2, 2)

        # Draw Handle
        handle_size = 16
        cx = rect.x() + val_norm * (rect.width() - handle_size) + handle_size / 2
        cy = rect.height() / 2
        h_half = handle_size / 2
        p1 = QPointF(cx, cy - h_half)
        p2 = QPointF(cx - h_half, cy + h_half)
        p3 = QPointF(cx + h_half, cy + h_half)
        
        handle_hue = (val_norm * 0.85)
        grad_handle = QLinearGradient(p2, p3)
        
        # Handle is slightly more saturated than the bar (0.55 Sat, 0.80 Lightness)
        # so it stands out just a little bit while remaining pastel.
        c1 = QColor.fromHslF(handle_hue, 0.55, 0.80, 1.0)
        c2 = QColor.fromHslF((handle_hue + 0.1)%1.0, 0.55, 0.80, 1.0)
        
        grad_handle.setColorAt(0.0, c1)
        grad_handle.setColorAt(1.0, c2)
        
        painter.setBrush(grad_handle)
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))
        painter.drawPolygon(QPolygonF([p1, p2, p3]))

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
        self.slider = PrismSlider(Qt.Orientation.Horizontal)
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
        self.lbl_val = QLabel("1.00x")
        self.lbl_val.setStyleSheet("color: #3f6c9b; font-weight: bold;")
        header.addWidget(self.lbl_name)
        header.addStretch()
        header.addWidget(self.lbl_val)
        self.slider = PrismSlider(Qt.Orientation.Horizontal)
        # Free rate range: 500 (0.5x) to 2000 (2.0x)
        self.slider.setRange(500, 2000)
        self.slider.setValue(1000)
        self.slider.valueChanged.connect(self.update_val)
        layout.addLayout(header)
        layout.addWidget(self.slider)
        
    def update_val(self, val):
        real_val = val / 1000.0
        self.lbl_val.setText(f"{real_val:.2f}x")
        self.parent_data[self.key] = real_val

class MediaButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(50, 50) 
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setStyleSheet("border: none; background: transparent;")

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background logic
        bg_color = QColor(255, 255, 255, 0)
        if self.underMouse(): bg_color = QColor(245, 248, 250)
        if self.isDown(): bg_color = QColor(235, 240, 245)
        
        painter.setBrush(bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 25, 25) 

        # Create Pastel Gradient for the Icon
        if self.isEnabled():
            grad = QLinearGradient(0, 0, self.width(), self.height())
            # Darker, high-contrast blue for visibility
            grad.setColorAt(0.0, QColor("#2c4e70")) 
            grad.setColorAt(1.0, QColor("#3f6c9b")) 
            painter.setBrush(grad)
        else:
            painter.setBrush(QColor("#e0e0e0"))

        cx, cy, txt = self.width() / 2, self.height() / 2, self.text()
        
        # Draw Symbols
        if "▶" in txt:
            path = QPainterPath()
            path.moveTo(cx - 5, cy - 8) 
            path.lineTo(cx - 5, cy + 8)
            path.lineTo(cx + 10, cy)
            path.closeSubpath()
            painter.drawPath(path)
        elif "||" in txt:
            painter.drawRoundedRect(QRectF(cx - 8, cy - 8, 5, 16), 2, 2)
            painter.drawRoundedRect(QRectF(cx + 3, cy - 8, 5, 16), 2, 2)
        elif "■" in txt:
            painter.drawRoundedRect(QRectF(cx - 7, cy - 7, 14, 14), 3, 3)

class RainbowButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setMouseTracking(True)
        self.is_hovering = False
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(30) 
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)

    def animate(self):
        self.phase = (self.phase + 0.005) % 1.0
        self.update()

    def enterEvent(self, event):
        self.is_hovering = True
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.is_hovering = False
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Text antialiasing is crucial for sharpness
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing) 
        rect = self.rect()

        # --- 1. Draw Button Background (Keep existing gradient) ---
        if self.isEnabled():
            # Base white layer
            painter.setBrush(QColor("white"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 6, 6)

            # Animated Pastel Fill
            grad_bg = QLinearGradient(0, 0, rect.width(), 0)
            for i in range(4):
                t = i / 3.0
                hue = (self.phase + (t * 0.5)) % 1.0
                opacity = 200 if self.is_hovering else 150
                # Very light pastel background
                col = QColor.fromHslF(hue, 0.6, 0.92, opacity/255.0)
                grad_bg.setColorAt(t, col)

            painter.setBrush(grad_bg)
            painter.drawRoundedRect(rect, 6, 6)

            # Animated Border
            border_col = QColor.fromHslF(self.phase, 0.5, 0.8, 1.0)
            painter.setPen(QPen(border_col, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(rect, 6, 6)

        else:
            # Disabled Background
            painter.setBrush(QColor("#e0e6ed"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(rect, 6, 6)

        # --- 2. Draw Button Text (Fixed for Sharpness) ---
        font = QFont("Segoe UI", 9)
        font.setBold(True)
        font.setCapitalization(QFont.Capitalization.AllLowercase)
        painter.setFont(font)

        if self.isEnabled():
            # Hue moves with phase but remains subtle.
            text_col = QColor.fromHslF(self.phase, 0.25, 0.50, 1.0)
            
            painter.setPen(text_col)
            # Standard drawText is much sharper than fillPath
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())

        else:
            # Disabled Text
            painter.setPen(QColor("#a0b0c0"))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())

class RainbowLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(30)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Ensure it doesn't take up too much vertical space, matching original label behavior
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def animate(self):
        self.phase = (self.phase + 0.005) % 1.0
        self.update()

    def paintEvent(self, event):
        if not self.text(): return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        rect = self.rect()
        
        # Same gradient logic as buttons, but darker Lightness (0.55) for text readability
        grad = QLinearGradient(0, 0, rect.width(), 0)
        for i in range(4):
            t = i / 3.0
            hue = (self.phase + (t * 0.5)) % 1.0
            col = QColor.fromHslF(hue, 0.75, 0.55, 1.0) 
            grad.setColorAt(t, col)

        font = QFont("Segoe UI")
        font.setPixelSize(11) # Explicitly 11px to match "font-size: 11px"
        font.setBold(True)
        
        painter.setFont(font)
        
        # Center text manually to align with gradient brush
        fm = self.fontMetrics()
        txt_w = fm.horizontalAdvance(self.text())
        
        # precise vertical centering
        x = (rect.width() - txt_w) / 2
        y = (rect.height() + fm.capHeight()) / 2 
        
        path = QPainterPath()
        path.addText(x, y, font, self.text())
        painter.fillPath(path, QBrush(grad))


class SampleRateRow(QWidget):
    def __init__(self, label, key, parent_data):
        super().__init__()
        self.key, self.parent_data = key, parent_data
        layout = setup_row_layout(self)
        header = QHBoxLayout()
        self.lbl_name = QLabel(label.lower())
        self.lbl_val = QLabel("44100 Hz")
        self.lbl_val.setStyleSheet("color: #3f6c9b; font-weight: bold;")
        header.addWidget(self.lbl_name)
        header.addStretch()
        header.addWidget(self.lbl_val)
        self.slider = PrismSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(8000, 44100)
        self.slider.setValue(44100)
        self.slider.valueChanged.connect(self.update_val)
        layout.addLayout(header)
        layout.addWidget(self.slider)
        
    def update_val(self, val):
        self.lbl_val.setText(f"{val} Hz")
        self.parent_data[self.key] = float(val)

class BpmControlRow(QWidget):
    def __init__(self, label, key, parent_data):
        super().__init__()
        self.key, self.parent_data = key, parent_data
        layout = setup_row_layout(self)
        header = QHBoxLayout()
        self.lbl_name = QLabel(label.lower())
        self.lbl_val = QLabel("120")
        self.lbl_val.setStyleSheet("color: #3f6c9b; font-weight: bold;")
        header.addWidget(self.lbl_name)
        header.addStretch()
        header.addWidget(self.lbl_val)
        self.slider = PrismSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(60, 200)
        self.slider.setValue(120)
        self.slider.valueChanged.connect(self.update_val)
        layout.addLayout(header)
        layout.addWidget(self.slider)
        
    def update_val(self, val):
        self.lbl_val.setText(f"{val}")
        self.parent_data[self.key] = float(val)

class StatusRainbowLabel(QLabel):
    def __init__(self, text="status: idle", parent=None):
        super().__init__(text, parent)
        self.setObjectName("SubHeader")
        self.phase = 0.0
        self.base_speed = 0.01
        self.current_speed = self.base_speed
        self.target_speed = self.base_speed
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

    def trigger_excitement(self):
        """Syncs with the logo acceleration."""
        self.current_speed = 0.15
        self.target_speed = self.base_speed

    def animate(self):
        # Decelerate logic
        self.current_speed = self.current_speed * 0.92 + self.target_speed * 0.08
        self.phase = (self.phase + self.current_speed) % 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        
        # Create a subtle moving gradient
        grad = QLinearGradient(0, 0, rect.width(), 0)
        # We create a window of rainbow that moves across the text
        shift = self.phase
        
        # Define gradient stops to cycle smoothly
        c1 = QColor("#7aa6d4") # Original blue
        c2 = QColor.fromHslF(shift, 0.6, 0.6, 1.0) # Moving hue
        c3 = QColor("#7aa6d4") 

        grad.setColorAt(0.0, c1)
        grad.setColorAt(0.5, c2)
        grad.setColorAt(1.0, c3)

        painter.setPen(QPen(QBrush(grad), 0))
        painter.setFont(self.font())
        painter.drawText(rect, self.alignment(), self.text())

class PastelFileLabel(QLabel):
    def __init__(self, text="no file loaded", parent=None):
        super().__init__(text, parent)
        self.setObjectName("FileLabel")
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def animate(self):
        self.phase = (self.phase + 0.002) % 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        
        # Slow moving pastel gradient
        grad = QLinearGradient(0, 0, rect.width(), 0)
        for i in range(4):
            t = i / 3.0
            # Hue shifts slowly with phase
            hue = (self.phase + t * 0.3) % 1.0
            # High lightness/saturation for "Pastel" look
            col = QColor.fromHslF(hue, 0.35, 0.6, 1.0)
            grad.setColorAt(t, col)

        painter.setPen(QPen(QBrush(grad), 0))
        # Use the font defined in stylesheet (bold, size 12)
        painter.setFont(self.font())
        painter.drawText(rect, self.alignment(), self.text())

class ExportMessageLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._opacity = 0.0 # Start invisible
        self.phase = 0.0
        
        # Match "Status" font settings exactly
        font = QFont("Segoe UI", 11)
        font.setBold(True)
        font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, 0.5)
        self.setFont(font)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

    def animate(self):
        self.phase = (self.phase + 0.01) % 1.0
        if self._opacity > 0.01:
            self.update()

    def get_opacity(self): 
        return self._opacity

    def set_opacity(self, o): 
        self._opacity = max(0.0, min(1.0, o))
        self.update()

    # Expose opacity as a Qt Property for QPropertyAnimation
    opacity = pyqtProperty(float, get_opacity, set_opacity)

    def paintEvent(self, event):
        if self._opacity <= 0: return
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        
        # Use the same subtle gradient logic as StatusRainbowLabel
        grad = QLinearGradient(0, 0, rect.width(), 0)
        shift = self.phase
        
        c1 = QColor("#7aa6d4")
        c2 = QColor.fromHslF(shift, 0.6, 0.6, 1.0)
        c3 = QColor("#7aa6d4")
        
        # Apply the opacity to the alpha channel of the colors
        c1.setAlphaF(self._opacity)
        c2.setAlphaF(self._opacity)
        c3.setAlphaF(self._opacity)

        grad.setColorAt(0.0, c1)
        grad.setColorAt(0.5, c2)
        grad.setColorAt(1.0, c3)

        painter.setPen(QPen(QBrush(grad), 0))
        painter.setFont(self.font())
        painter.drawText(rect, self.alignment(), self.text())

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("prism")
        self.resize(900, 480) # Reduced height
        self.setAcceptDrops(True)
        self.file_path = None
        self.original_audio = None
        self.processed_audio = None
        self.sr = 44100
        self.temp_file = None
        
        # Smooth playback variables
        self.last_wall_clock = 0.0
        self.last_media_pos = 0
        
        self.params = {
            'glitch': 0.5, 'wash': 0.0, 'crush': 0.0, 'reverb': 0.0, 
            'sr_select': 44100.0, 'rate': 1.0, 
            'filter_amt': 0.0, 'vol_pan_amt': 0.0, 'bpm': 120.0
        }
        
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)

        self.player.mediaStatusChanged.connect(self.media_status_changed)

        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(7)
        self.anim_timer.timeout.connect(self.high_freq_update)

        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(10, 10, 10, 10) # Tighter margins
        main_layout.setSpacing(10) # Tighter spacing

        self.viewport = QVBoxLayout()
        self.wave_view = WaveformWidget()
        self.wave_view.setMinimumHeight(120) # Reduced from 160
        self.wave_view.seek_requested.connect(self.seek_audio)
        self.wave_view.import_clicked.connect(self.open_file_dialog)

        self.wave_view.scrub_started.connect(self.anim_timer.stop)
        self.wave_view.scrub_ended.connect(self.resume_sync)

        info_row = QHBoxLayout()
        info_row.addStretch()
        
        self.lbl_file = PastelFileLabel("no file loaded")
        self.lbl_file.setObjectName("FileLabel")
        self.lbl_file.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.btn_clear = QPushButton("clear")
        self.btn_clear.setObjectName("ClearBtn")
        self.btn_clear.setFixedSize(100, 20)
        self.btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        
        # Reserve layout space even when hidden to prevent resizing/jumping
        sp = self.btn_clear.sizePolicy()
        sp.setRetainSizeWhenHidden(True)
        self.btn_clear.setSizePolicy(sp)
        
        self.btn_clear.setVisible(False)
        self.btn_clear.clicked.connect(self.clear_state)
        
        info_row.addWidget(self.lbl_file)
        info_row.addSpacing(10)
        info_row.addWidget(self.btn_clear)
        info_row.addStretch()

        self.viewport.addWidget(self.wave_view, 1)
        self.viewport.addSpacing(5)
        self.viewport.addLayout(info_row)
        self.viewport.addStretch()

        self.sidebar = GlassFrame()
        self.sidebar.setFixedWidth(280) # Slightly narrower
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(12, 15, 12, 15) # Tighter padding inside glass
        side_layout.setSpacing(2)

        self.logo = PrismLogo()
        side_layout.addWidget(self.logo)
        
        # CHANGED: Use StatusRainbowLabel
        self.lbl_status = StatusRainbowLabel("status: idle")
        side_layout.addWidget(self.lbl_status)
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: rgba(63, 108, 155, 0.2);")
        side_layout.addWidget(line)
        side_layout.addSpacing(5)

        side_layout.addWidget(ControlRow("grid density", "glitch", self.params))
        side_layout.addWidget(BpmControlRow("bpm", "bpm", self.params))
        
        side_layout.addWidget(ControlRow("bit crush", "crush", self.params))
        side_layout.addWidget(RateControlRow("playback rate", "rate", self.params))

        side_layout.addWidget(SampleRateRow("sample rate", "sr_select", self.params))

        side_layout.addWidget(ControlRow("spectral wash", "wash", self.params))
        side_layout.addWidget(ControlRow("glue reverb", "reverb", self.params))

        side_layout.addWidget(ControlRow("filter mod", "filter_amt", self.params))
        side_layout.addWidget(ControlRow("vol/pan mod", "vol_pan_amt", self.params))

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
        action_layout.setContentsMargins(0, 8, 0, 0)
        action_layout.setSpacing(10)
        
        self.btn_process = RainbowButton("process")
        self.btn_process.setObjectName("ProcessBtn")
        self.btn_process.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_process.setEnabled(False)
        self.btn_process.clicked.connect(self.start_processing)
        
        self.btn_save = RainbowButton("export")
        self.btn_save.setObjectName("SaveBtn")
        self.btn_save.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_save.setEnabled(False)
        self.btn_save.clicked.connect(self.quick_export)

        action_layout.addWidget(self.btn_process)
        action_layout.addWidget(self.btn_save)
        side_layout.addLayout(action_layout)

        self.lbl_saved_msg = ExportMessageLabel("")
        
        # Animate the custom 'opacity' property directly on the label
        self.fade_anim = QPropertyAnimation(self.lbl_saved_msg, b"opacity")
        self.fade_anim.setDuration(1000)
        self.fade_anim.setEasingCurve(QEasingCurve.Type.OutQuad)
        # When fade finishes, clear text (optional, but good practice)
        self.fade_anim.finished.connect(lambda: self.lbl_saved_msg.setText(""))
        
        side_layout.addWidget(self.lbl_saved_msg)

        main_layout.addLayout(self.viewport, 1)
        main_layout.addWidget(self.sidebar)

    def paintEvent(self, event):
        painter = QPainter(self)
        # Restore the subtle blue radial gradient
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
        fname, _ = QFileDialog.getOpenFileName(self, 'open', os.path.expanduser("~"), "audio files (*.wav *.mp3 *.flac *.ogg)")
        if fname: self.load_file(fname)

    def clear_state(self):
        self.stop_playback()
        self.player.setSource(QUrl())
        self.file_path = None
        self.original_audio = None
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
            self.original_audio = data
            
            self.player.setSource(QUrl.fromLocalFile(path))
            display_data = data[:sr*30] if len(data) > sr*30 else data
            self.wave_view.set_data(display_data)
            
            self.btn_process.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.lbl_status.setText("status: ready")
        except Exception as e:
            QMessageBox.critical(self, "error", f"could not load file: {e}")

    def resume_sync(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.last_wall_clock = time.perf_counter()
            self.last_media_pos = self.player.position()
            self.anim_timer.start()

    def high_freq_update(self):
        """Calculates interpolated playhead position for buttery smooth 144hz rendering."""
        duration = self.player.duration()
        if duration > 0 and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            
            # Interpolation logic
            now = time.perf_counter()
            delta = now - self.last_wall_clock
            
            # Must account for playback rate for correct speed
            rate = self.params.get('rate', 1.0)
            
            interpolated_ms = self.last_media_pos + (delta * 1000.0 * rate)
            
            # Drift correction: If actual QMediaPlayer pos drifts significantly, resync.
            # (QMediaPlayer updates roughly every 20-50ms)
            actual_pos = self.player.position()
            if abs(interpolated_ms - actual_pos) > 75: # 75ms tolerance
                self.last_media_pos = actual_pos
                self.last_wall_clock = now
                interpolated_ms = actual_pos

            self.wave_view.set_play_head(interpolated_ms / duration)
            
            # Handle Loop/End edge case visually
            if interpolated_ms >= duration:
                self.last_media_pos = 0
                self.last_wall_clock = now

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.anim_timer.stop() 
            self.btn_play.setText("▶")
        else:
            # Init sync variables before playing
            self.last_wall_clock = time.perf_counter()
            self.last_media_pos = self.player.position()
            self.player.play()
            self.anim_timer.start()
            self.btn_play.setText("||")

    def stop_playback(self):
        self.player.stop()
        self.anim_timer.stop() 
        self.btn_play.setText("▶")
        self.wave_view.set_play_head(0)

    def seek_audio(self, pos_norm):
        duration = self.player.duration()
        if duration > 0: 
            ms = int(pos_norm * duration)
            self.player.setPosition(ms)
            # Resync interpolation
            self.last_media_pos = ms
            self.last_wall_clock = time.perf_counter()

    def media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.btn_play.setText("▶")
            self.wave_view.set_play_head(0)
            self.anim_timer.stop()

    def start_processing(self):
        if self.original_audio is None: return
        
        self.stop_playback()
        self.lbl_status.setText("status: processing...")
        # Do not need manual stylesheet color, StatusRainbowLabel handles it
        self.set_ui_locked(True)
        
        self.thread = ProcessThread(self.original_audio, self.sr, self.params.copy())
        self.thread.finished_ok.connect(self.processing_done)
        self.thread.error.connect(self.processing_error)
        self.thread.start()

    def processing_done(self, data, sr):
        self.processed_audio = data
        self.wave_view.set_data(data)
        self.lbl_status.setText("status: done")
        self.set_ui_locked(False)
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            AudioEngine.save_file(temp_path, data, sr)
            self.temp_file = temp_path
            self.player.setSource(QUrl.fromLocalFile(temp_path))
            
            # Auto-play with sync reset
            self.last_media_pos = 0
            self.last_wall_clock = time.perf_counter()
            self.player.play()
            self.btn_play.setText("||")
            self.anim_timer.start()
            
        except Exception as e: print(f"Temp file error: {e}")

    def processing_error(self, msg):
        QMessageBox.critical(self, "processing error", msg)
        self.lbl_status.setText("status: error")
        self.set_ui_locked(False)

    def set_ui_locked(self, locked):
        """
        Prevent flicker:
        Instead of disabling the whole sidebar (which grays out everything),
        we only disable the specific button and perhaps specific inputs if necessary.
        Visual stability is preferred over aggressive disabling.
        """
        self.btn_process.setEnabled(not locked)
        # We purposely leave the sliders and other buttons visually enabled
        # but inactive logic could be applied if strictly needed. 
        # For this app, adjusting sliders during processing is harmless (won't affect running thread).
        
        # Only disabling the Save button if locked
        if locked:
             self.btn_save.setEnabled(False)
        else:
             # Enable save only if we have processed audio
             self.btn_save.setEnabled(self.processed_audio is not None)

    def start_fade_out(self):
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        self.fade_anim.start()
    
    def closeEvent(self, event):
        if self.temp_file and os.path.exists(self.temp_file):
            try: os.remove(self.temp_file)
            except: pass
        event.accept()

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
        fname, _ = QFileDialog.getOpenFileName(self, 'open', os.path.expanduser("~"), "audio files (*.wav *.mp3 *.flac *.ogg)")
        if fname: self.load_file(fname)

    def clear_state(self):
        self.stop_playback()
        self.player.setSource(QUrl())
        self.file_path = None
        self.original_audio = None
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
            self.original_audio = data
            
            self.player.setSource(QUrl.fromLocalFile(path))
            display_data = data[:sr*30] if len(data) > sr*30 else data
            self.wave_view.set_data(display_data)
            
            self.btn_process.setEnabled(True)
            self.btn_play.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.lbl_status.setText("status: ready")
        except Exception as e:
            QMessageBox.critical(self, "error", f"could not load file: {e}")

    def resume_sync(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.anim_timer.start()

    def high_freq_update(self):
        duration = self.player.duration()
        if duration > 0: 
            pos = self.player.position()
            self.wave_view.set_play_head(pos / duration)

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.anim_timer.stop() 
            self.btn_play.setText("▶")
        else:
            self.player.play()
            self.anim_timer.start()
            self.btn_play.setText("||")

    def stop_playback(self):
        self.player.stop()
        self.anim_timer.stop() 
        self.btn_play.setText("▶")
        self.wave_view.set_play_head(0)

    def seek_audio(self, pos_norm):
        duration = self.player.duration()
        if duration > 0: 
            self.player.setPosition(int(pos_norm * duration))

    def media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.btn_play.setText("▶")
            self.wave_view.set_play_head(0)

    def start_processing(self):
        # Check for original_audio instead of file_path
        if self.original_audio is None: return
        
        self.stop_playback()
        self.lbl_status.setText("status: processing...")
        self.lbl_status.setStyleSheet("color: #7aa6d4; font-weight: bold;")
        self.set_ui_locked(True)
        
        # Pass the stored original_audio and current SR to the thread
        self.thread = ProcessThread(self.original_audio, self.sr, self.params.copy())
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
            
            # Auto-play logic
            self.player.play()
            self.btn_play.setText("||")
            self.anim_timer.start()
            
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
        
        # Trigger excitement
        self.logo.trigger_excitement()
        self.lbl_status.trigger_excitement()
        
        # 1. Define the path: C:\Users\[Name]\Music\prism
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, "Music", "prism")
        
        # 2. Create the folder automatically if it doesn't exist
        if not os.path.exists(save_dir):
            try:
                os.makedirs(save_dir)
            except Exception as e:
                QMessageBox.critical(self, "error", f"could not create folder: {e}")
                return

        # 3. Generate filename and full path
        timestamp = int(time.time())
        filename = f"prism_export_{timestamp}.wav"
        save_path = os.path.join(save_dir, filename)
        
        try:
            AudioEngine.save_file(save_path, self.processed_audio, self.sr)
            self.lbl_status.setText("status: exported")
            
            # 4. Tell the user where it went
            self.lbl_saved_msg.setText("saved to: Music/prism")
            
            self.lbl_saved_msg.set_opacity(1.0)
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
    # 1. Fix Taskbar Icon for Windows (Wrapped in try/except to prevent crashes)
    try:
        myappid = 'prism.audio.tool.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass # Ignore if this fails (e.g. on non-Windows or permissions issues)

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLES)
    
    # 2. Set the Application Icon
    # We resolve the absolute path to ensure the EXE finds the icon reliably
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prism.ico")
    
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    elif os.path.exists("prism.ico"):
        # Fallback for some dev environments
        app.setWindowIcon(QIcon("prism.ico"))

    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        
    window = MainWindow()
    window.show()
    sys.exit(app.exec())