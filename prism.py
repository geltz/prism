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
                         QPolygonF, QIcon, QTransform, QFontMetrics)
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
        
        # --- Highpass wet signal (~300Hz) to prevent low-end mud ---
        sos_hp = signal.butter(1, 300, 'hp', fs=sr, output='sos')
        wet_sig = signal.sosfilt(sos_hp, wet_sig)
        # ----------------------------------------------------------------
        
        wet_sig = wet_sig / (np.max(np.abs(wet_sig)) + 1e-9)
        wet_stereo = np.column_stack((wet_sig, wet_sig))
        out = (1 - mix) * x + mix * wet_stereo
        return out

    @staticmethod
    def make_slice_library(data, sr, num_slices=64):
        slices = []
        # UPDATED: Increased lengths for "drawn out" feel
        min_len = int(0.1 * sr)   # Was 0.05
        max_len = int(1.5 * sr)   # Was 0.4 (This prevents micro-looping)
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
    def generate_4bar_loop(slice_lib, sr, density, bpm, swing=0.0, stutter=0.0):
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
            # Old logic: simple probability thresholds
            if r < (0.1 + 0.4 * density): grid_mult = 1 
            elif r < (0.4 + 0.4 * density): grid_mult = 2 
            else: grid_mult = 4 
            
            dur = samples_16th * grid_mult
            
            # Clamp duration to not exceed loop end
            if cursor + dur > loop_samples: dur = loop_samples - cursor
            
            # Pick a random slice
            slc = slice_lib[rng.integers(len(slice_lib))]
            
            # Simple tiling/looping if slice is too short
            if len(slc) >= dur: 
                chunk = slc[:dur]
            else:
                repeats = int(np.ceil(dur / len(slc)))
                chunk = np.tile(slc, repeats)[:dur]
                
            out[cursor:cursor+dur] = chunk
            cursor += dur
            
        return out
    
    @staticmethod
    def apply_tone(data, sr, val):
        # val is 0..1 (0=Lowpass, 0.5=Neutral, 1.0=Highpass)
        if 0.48 < val < 0.52: return data
        
        y = data.copy()
        sos = None
        
        # DJ Filter Curve
        if val <= 0.5:
            # Lowpass: 0.0 -> 100Hz, 0.5 -> 20000Hz
            norm = val * 2.0 
            freq = 100.0 * (200.0 ** norm) # Exponential mapping
            freq = min(freq, sr/2 - 100)
            sos = signal.butter(2, freq, 'low', fs=sr, output='sos')
        else:
            # Highpass: 0.5 -> 20Hz, 1.0 -> 8000Hz
            norm = (val - 0.5) * 2.0
            freq = 20.0 * (400.0 ** norm)
            freq = min(freq, sr/2 - 100)
            sos = signal.butter(2, freq, 'high', fs=sr, output='sos')

        if sos is not None:
            if y.ndim == 2:
                y[:, 0] = signal.sosfilt(sos, y[:, 0])
                y[:, 1] = signal.sosfilt(sos, y[:, 1])
            else:
                y = signal.sosfilt(sos, y)
        return y
    
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
            left[i:end] *= np.cos(p_ang) * vol
            right[i:end] *= np.sin(p_ang) * vol
        return np.column_stack((left, right))

    @staticmethod
    def bitcrush(data, sr, depth=0.0):
        y = data.copy()
        if depth > 0:
            quant = 2 ** (16 - (depth * 8)) 
            y = np.round(y * quant) / quant
            rate_div = depth 
            if rate_div > 0:
                step = int(1 + (rate_div * 5))
                if y.ndim == 2:
                    for ch in range(2):
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
    def process(audio, sr, params):
        bpm = params.get('bpm', 120.0)

        if not params.get('chops_enabled', True):
            # Bypass slicing/looping, use full original audio
            y = audio.copy()
        else:
            # Original Slicing Logic
            slices = AudioEngine.make_slice_library(audio, sr)
            
            # Hardcoded stutter to 0.15 for sparse, internal rhythmic texture
            y = AudioEngine.generate_4bar_loop(
                slices, sr, 
                density=params['glitch'], 
                bpm=bpm,
                swing=params.get('swing', 0.0),
                stutter=0.15 
            )
        
        y = AudioEngine.apply_rand_filter(y, sr, intensity=params['filter_amt'], bpm=bpm)
        
        # Replaced original vol_pan call with the new specific logic
        y = AudioEngine.apply_vol_pan(y, sr, intensity=params['vol_pan_amt'], bpm=bpm)
        
        if y.ndim == 1: y = np.column_stack((y, y))
        y = AudioEngine.bitcrush(y, sr, depth=params['crush'])
        y = AudioEngine.apply_tone(y, sr, params.get('tone', 0.5))
        y = AudioEngine.apply_samplerate(y, sr, params['sr_select'])
        
        rate = params.get('rate', 1.0)
        if rate != 1.0 and rate > 0:
            new_len = int(len(y) / rate)
            y = signal.resample(y, new_len)
            
        if params['reverb'] > 0:
            y = AudioEngine.simple_reverb(y, sr, mix=params['reverb'] * 0.35, room_size=0.6, damp=0.6)
            
        peak = np.max(np.abs(y))
        if peak > 1.0: y = y / peak
        return y.astype(np.float32)

class ProcessThread(QThread):
    finished_ok = pyqtSignal(object, int) 
    error = pyqtSignal(str)
    def __init__(self, audio_data, sr, params):
        super().__init__()
        self.audio = audio_data
        self.sr = sr
        self.params = params
    def run(self):
        try:
            processed = AudioEngine.process(self.audio.copy(), self.sr, self.params)
            self.finished_ok.emit(processed, self.sr)
        except Exception as e:
            self.error.emit(str(e))

STYLES = """
    QMainWindow { background-color: #f6f9fc; }
    QLabel { color: #405165; font-family: 'Segoe UI', sans-serif; font-size: 11px; }
    QLabel#SubHeader { color: #7aa6d4; font-size: 11px; font-weight: bold; letter-spacing: 0.5px; }
    QLabel#FileLabel { color: #3f6c9b; font-size: 12px; font-weight: bold; }
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
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(45)
        self.phase = 0.0
        self.base_speed = 0.005
        self.current_speed = self.base_speed
        self.target_speed = self.base_speed
        self.active = True
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        # REDUCED CPU: 40ms (~25fps) is enough for a slow logo
        self.timer.start(40)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.active = not self.active
            self.clicked.emit()
            
            # Adjust speed based on state
            self.target_speed = 0.005 if self.active else 0.0
            self.update()

    def trigger_excitement(self):
        if self.active:
            self.current_speed = 0.1
            self.target_speed = self.base_speed

    def animate(self):
        # If inactive, slow down to a halt
        target = self.target_speed if self.active else 0.0
        self.current_speed = self.current_speed * 0.96 + target * 0.04
        self.phase = (self.phase + self.current_speed) % 1.0
        
        if not self.active and self.current_speed < 0.0001:
            return 
            
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Dim when inactive
        if not self.active:
            painter.setOpacity(0.4)
            
        w, h = self.width(), self.height()
        cx, cy = w / 2, h / 2
        
        # Triangle Geometry (Centered)
        tri_size = 20
        p1 = QPointF(cx, cy - tri_size)
        p2 = QPointF(cx + tri_size * 0.866, cy + tri_size * 0.5)
        p3 = QPointF(cx - tri_size * 0.866, cy + tri_size * 0.5)
        triangle = QPolygonF([p1, p2, p3])

        # 1. Input Beam (Perfectly Horizontal & Centered)
        painter.setPen(QPen(QColor(255, 255, 255, 200), 2))
        painter.drawLine(QPointF(0, cy), QPointF(cx - 8, cy))

        # 2. Output Rainbow (Scattered/Fanned)
        for i in range(7):
            hue = (i / 7.0 - self.phase) % 1.0
            pulse = (math.sin(self.phase * 6.28 + i) + 1) / 2
            alpha = int(100 + 155 * pulse)
            col = QColor.fromHslF(hue, 0.7, 0.7, alpha/255.0)
            painter.setPen(QPen(col, 1.5))
            
            # Origin: Vertically centered on the right face of the triangle
            origin = QPointF(cx + 6, cy)
            
            # Fanned Angles (Restoring the scatter effect)
            angle_deg = -24 + (i * 8) 
            angle_rad = math.radians(angle_deg)
            
            dest_x = w
            # Calculate Y based on angle to create the fan
            dest_y = cy + math.tan(angle_rad) * (w - cx)
            
            painter.drawLine(origin, QPointF(dest_x, dest_y))

        # 3. Triangle Overlay
        grad = QLinearGradient(p1, p3)
        grad.setColorAt(0.0, QColor(255, 255, 255, 100))
        grad.setColorAt(1.0, QColor(255, 255, 255, 10))
        painter.setBrush(grad)
        painter.setPen(QPen(QColor(240, 240, 255), 1.5))
        painter.drawPolygon(triangle)

class GlassFrame(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet("""
            GlassFrame { 
                background-color: #f6f9fc; 
                border: none;
                border-left: 1px solid #d0dbe5; 
            }
        """)

class WaveformWidget(QWidget):
    seek_requested = pyqtSignal(float)
    import_clicked = pyqtSignal()
    scrub_started = pyqtSignal()
    scrub_ended = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.data = None
        self.play_head_pos = 0.0
        self.setMinimumHeight(120)
        self.setMouseTracking(True) 
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._static_pixmap = None
        self._scanline_buffer = None 
        
        self.is_scrubbing = False 
        
        self._resize_timer = QTimer(self)
        self._resize_timer.setSingleShot(True)
        self._resize_timer.timeout.connect(self.update_static_waveform)

        self.phase = 0.0
        self.text_anim_timer = QTimer(self)
        self.text_anim_timer.timeout.connect(self.animate_text)
        self.text_anim_timer.start(50)
    
    def animate_text(self):
        # Only animate if we are showing the empty state text
        if self.data is None:
            self.phase = (self.phase + 0.005) % 1.0
            self.update()
    
    def set_data(self, data):
        if data.ndim > 1: d_mono = data.mean(axis=1)
        else: d_mono = data  
        target_points = 2000 
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
        # OPTIMIZATION: Reset buffer on resize
        self._scanline_buffer = None
        self._resize_timer.start(50) 
        super().resizeEvent(event)

    def mousePressEvent(self, event):
        if self.data is None: 
            self.import_clicked.emit()
        elif event.button() == Qt.MouseButton.LeftButton:
            self.is_scrubbing = True
            self.scrub_started.emit()
            self.handle_input(event.pos().x())

    def mouseMoveEvent(self, event):
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

        # Only create pixmap if we have data to draw
        if self.data is None:
            self._static_pixmap = None
            self.update()
            return

        self._static_pixmap = QPixmap(w, h)
        self._static_pixmap.fill(Qt.GlobalColor.transparent)
        
        painter = QPainter(self._static_pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        path = QPainterPath()
        cy = h / 2
        path.moveTo(0, cy)
        
        x_step = w / len(self.data)
        amp_scale = h * 0.45
        for i, val in enumerate(self.data):
            path.lineTo(i * x_step, cy - (val * amp_scale))
        
        # FIX: Instead of closing the path back to center, create a mirrored bottom half
        # to form a complete waveform area without the center line
        for i in range(len(self.data) - 1, -1, -1):
            val = self.data[i]
            path.lineTo(i * x_step, cy + (val * amp_scale))
        
        path.closeSubpath()  # This will connect back to the start
        
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
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing) 
        rect = self.rect()
        w, h = rect.width(), rect.height()

        # CHANGED: Removed rounded rect drawing and clipping
        # Background - Simple fill
        painter.fillRect(rect, QColor(255, 255, 255, 166)) 

        if self.data is None:
            # (Empty state drawing...)
            grad = QLinearGradient(0, 0, rect.width(), 0)
            for i in range(4):
                t = i / 3.0
                hue = (self.phase + t * 0.3) % 1.0
                col = QColor.fromHslF(hue, 0.45, 0.65, 1.0)
                grad.setColorAt(t, col)
            font = QFont("Segoe UI", 10)
            painter.setFont(font)
            pen = QPen(QBrush(grad), 0)
            painter.setPen(pen)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "drop file or click to import")
            return

        pm = self._static_pixmap
        if pm:
            pm_w = pm.width()
            sx = pm_w / w if w > 0 else 1.0

            if self._scanline_buffer is None or self._scanline_buffer.size() != self.size():
                self._scanline_buffer = QPixmap(self.size())
            
            self._scanline_buffer.fill(Qt.GlobalColor.transparent)
            
            cw_painter = QPainter(self._scanline_buffer)
            cw_painter.drawPixmap(self.rect(), pm)
            cw_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
            
            t = time.time() * 0.08 
            wave_grad = QLinearGradient(0, 0, w, h)
            c1 = QColor.fromHslF((t) % 1.0, 0.6, 0.80, 1.0) 
            c2 = QColor.fromHslF((t + 0.2) % 1.0, 0.6, 0.80, 1.0)
            c3 = QColor.fromHslF((t + 0.4) % 1.0, 0.6, 0.80, 1.0)
            wave_grad.setColorAt(0.0, c1); wave_grad.setColorAt(0.5, c2); wave_grad.setColorAt(1.0, c3)
            cw_painter.fillRect(rect, wave_grad)
            cw_painter.end()
            
            painter.drawPixmap(0, 0, self._scanline_buffer)

            if self.play_head_pos >= 0:
                px = int(self.play_head_pos * w)
                
                # Ripple Effect
                ripple_w = 140 
                src_cx = px * sx
                src_rw = ripple_w * sx 
                
                source_rect_f = QRectF(src_cx - src_rw, 0, src_rw * 2, pm.height())
                source_rect = source_rect_f.toRect().intersected(pm.rect())
                
                if not source_rect.isEmpty():
                    wave_slice = pm.copy(source_rect)
                    dest_w = source_rect.width() / sx
                    if dest_w > 0:
                        wave_slice = wave_slice.scaled(int(dest_w), h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
                        slice_painter = QPainter(wave_slice)
                        slice_painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
                        dest_x = source_rect.x() / sx
                        grad_center = px - dest_x
                        r_grad = QLinearGradient(grad_center - ripple_w, 0, grad_center + ripple_w, 0)
                        hue_now = (t + 0.1) % 1.0
                        c_edge = QColor.fromHslF(hue_now, 0.8, 0.5, 0.0)
                        c_center = QColor.fromHslF(hue_now, 0.85, 0.45, 0.9)
                        r_grad.setColorAt(0.0, c_edge); r_grad.setColorAt(0.5, c_center); r_grad.setColorAt(1.0, c_edge)
                        slice_painter.fillRect(wave_slice.rect(), r_grad)
                        slice_painter.end()
                        painter.drawPixmap(int(dest_x), 0, wave_slice)

                # Playhead Line
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
    layout.setSpacing(2) # Reduced internal spacing (Label <-> Slider gap)
    return layout

class PrismSlider(QSlider):
    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(orientation, parent)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(26)
        # Removed setMouseTracking(True) as we don't need hover anymore
        
        self.phase = 0.0
        self.morph_progress = 0.0 # Renamed for clarity
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(40)

    def animate(self):
        self.phase = (self.phase + 0.1) % (2 * math.pi)

        target = 0.0 

        dist = target - self.morph_progress

        speed = 0.9 if target > 0.6 else 0.3
        
        self.morph_progress += dist * speed
        
        if abs(dist) > 0.001 or self.isSliderDown():
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            val = self.pixel_to_value(event.pos().x())
            self.setValue(val)
            event.accept()
            # Force immediate update to start animation feel
            self.setSliderDown(True) 
            
    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            val = self.pixel_to_value(event.pos().x())
            self.setValue(val)
            event.accept()

    def mouseReleaseEvent(self, event):
        self.setSliderDown(False)
        super().mouseReleaseEvent(event)

    def pixel_to_value(self, x):
        w = self.width()
        if w == 0: return 0
        norm = max(0.0, min(1.0, x / w))
        return int(self.minimum() + norm * (self.maximum() - self.minimum()))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        
        groove_h = 5
        groove_y = rect.height() / 2 - groove_h / 2
        groove_rect = QRectF(rect.x(), groove_y, rect.width(), groove_h)
        
        painter.setBrush(QColor(63, 108, 155, 30))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(groove_rect, 3, 3)
        
        val_norm = (self.value() - self.minimum()) / (self.maximum() - self.minimum()) if (self.maximum() > self.minimum()) else 0
        
        if val_norm > 0.01:
            fill_width = rect.width() * val_norm
            fill_rect = QRectF(rect.x(), groove_y, fill_width, groove_h)
            grad = QLinearGradient(rect.x(), 0, rect.x() + fill_width, 0)
            stops = 10
            
            # Use morph_progress for pulse intensity
            pulse_shift = math.sin(self.phase) * 0.15 * self.morph_progress
            
            for i in range(stops):
                t = i / (stops - 1)
                base_hue = t * (val_norm * 0.85)
                final_hue = (base_hue + pulse_shift) % 1.0
                color = QColor.fromHslF(final_hue, 0.75, 0.75, 1.0)
                grad.setColorAt(t, color)
                
            painter.setBrush(grad)
            painter.drawRoundedRect(fill_rect, 3, 3)

        handle_size = 18
        scaled_size = handle_size 
        
        cx = rect.x() + val_norm * (rect.width() - handle_size) + handle_size / 2
        cy = rect.height() / 2
        
        painter.translate(cx, cy)
        
        handle_hue = (val_norm * 0.85)
        c1 = QColor.fromHslF(handle_hue, 0.55, 0.80, 1.0)
        c2 = QColor.fromHslF((handle_hue + 0.1)%1.0, 0.55, 0.80, 1.0)
        grad_handle = QLinearGradient(-scaled_size/2, -scaled_size/2, scaled_size/2, scaled_size/2)
        grad_handle.setColorAt(0.0, c1); grad_handle.setColorAt(1.0, c2)
        
        painter.setBrush(grad_handle)
        painter.setPen(QPen(QColor(255, 255, 255), 1.5))

        # UPDATED: Draw Triangle (fades out as morph_progress increases)
        if self.morph_progress < 0.95:
            painter.setOpacity(1.0 - self.morph_progress)
            h_half = scaled_size / 2
            p1 = QPointF(0, -h_half)
            p2 = QPointF(-h_half, h_half)
            p3 = QPointF(h_half, h_half)
            painter.drawPolygon(QPolygonF([p1, p2, p3]))

        # UPDATED: Draw Circle (fades in as morph_progress increases)
        if self.morph_progress > 0.05:
            painter.setOpacity(self.morph_progress)
            r = scaled_size / 2
            painter.drawEllipse(QPointF(0, 0), r, r)

        painter.setOpacity(1.0)
        painter.resetTransform()

class ControlRow(QWidget):
    def __init__(self, label, key, parent_data):
        super().__init__()
        self.key, self.parent_data = key, parent_data
        layout = setup_row_layout(self)
        header = QHBoxLayout()
        self.lbl_name = QLabel(label.lower())
        self.lbl_val = QLabel("0")
        self.lbl_val.setStyleSheet("color: #7a8fa3; font-weight: bold;")
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
        self.lbl_val.setStyleSheet("color: #7a8fa3; font-weight: bold;")
        header.addWidget(self.lbl_name)
        header.addStretch()
        header.addWidget(self.lbl_val)
        self.slider = PrismSlider(Qt.Orientation.Horizontal)
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
        bg_color = QColor(255, 255, 255, 0)
        if self.underMouse(): bg_color = QColor(245, 248, 250)
        if self.isDown(): bg_color = QColor(235, 240, 245)
        painter.setBrush(bg_color)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 25, 25) 
        if self.isEnabled():
            grad = QLinearGradient(0, 0, self.width(), self.height())
            grad.setColorAt(0.0, QColor("#2c4e70")) 
            grad.setColorAt(1.0, QColor("#3f6c9b")) 
            painter.setBrush(grad)
        else:
            painter.setBrush(QColor("#e0e0e0"))
        cx, cy, txt = self.width() / 2, self.height() / 2, self.text()
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
        
        # Ripple State
        self.ripple_r = 0.0
        self.ripple_alpha = 0.0
        self.click_pos = QPointF(0,0)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16) 
        
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedHeight(32)

    def animate(self):
        # Use wall-clock time so phase is continuous and doesn't reset on UI updates
        self.phase = (time.time() * 0.5) % 1.0
        
        if self.ripple_alpha > 0:
            self.ripple_r += 4.5
            self.ripple_alpha -= 0.06
            if self.ripple_alpha < 0: self.ripple_alpha = 0.0
            self.update()
        elif self.is_hovering:
            self.update()
    
    def enterEvent(self, event):
        self.is_hovering = True
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.is_hovering = False
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        self.click_pos = QPointF(event.pos())
        self.ripple_r = 5.0
        self.ripple_alpha = 0.3
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing) 
        rect = self.rect()
        
        # 1. Background
        painter.setBrush(QColor("white"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 6, 6)
        
        grad_bg = QLinearGradient(0, 0, rect.width(), 0)
        for i in range(4):
            t = i / 3.0
            hue = (self.phase + (t * 0.5)) % 1.0
            # Subtle hover brighten
            opacity = 200 if self.is_hovering else 150
            col = QColor.fromHslF(hue, 0.6, 0.92, opacity/255.0)
            grad_bg.setColorAt(t, col)
        
        painter.setBrush(grad_bg)
        painter.drawRoundedRect(rect, 6, 6)
        
        # 2. Border
        border_col = QColor.fromHslF(self.phase, 0.5, 0.8, 1.0)
        painter.setPen(QPen(border_col, 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(rect, 6, 6)

        # 3. Ripple Overlay
        if self.ripple_alpha > 0.01:
            painter.setPen(Qt.PenStyle.NoPen)
            r_col = QColor(255, 255, 255)
            r_col.setAlphaF(self.ripple_alpha)
            painter.setBrush(r_col)
            painter.drawEllipse(self.click_pos, self.ripple_r, self.ripple_r)

        # 4. Text
        font = QFont("Segoe UI", 9)
        font.setBold(True)
        font.setCapitalization(QFont.Capitalization.AllLowercase)
        painter.setFont(font)
        text_col = QColor.fromHslF(self.phase, 0.25, 0.50, 1.0)
        painter.setPen(text_col)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())

class RainbowLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.phase = 0.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        # REDUCED CPU
        self.timer.start(50)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    def animate(self):
        self.phase = (self.phase + 0.005) % 1.0
        self.update()

    def paintEvent(self, event):
        if not self.text(): return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        grad = QLinearGradient(0, 0, rect.width(), 0)
        for i in range(4):
            t = i / 3.0
            hue = (self.phase + (t * 0.5)) % 1.0
            col = QColor.fromHslF(hue, 0.75, 0.55, 1.0) 
            grad.setColorAt(t, col)
        font = QFont("Segoe UI")
        font.setPixelSize(11)
        font.setBold(True)
        painter.setFont(font)
        fm = self.fontMetrics()
        txt_w = fm.horizontalAdvance(self.text())
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
        self.lbl_val.setStyleSheet("color: #7a8fa3; font-weight: bold;")
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
        self.lbl_val.setStyleSheet("color: #7a8fa3; font-weight: bold;")
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
        # REDUCED CPU
        self.timer.start(50)

    def trigger_excitement(self):
        self.current_speed = 0.2
        self.target_speed = self.base_speed

    def animate(self):
        self.current_speed = self.current_speed * 0.92 + self.target_speed * 0.08
        self.phase = (self.phase + self.current_speed) % 1.0
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()
        grad = QLinearGradient(0, 0, rect.width(), 0)
        shift = self.phase
        c1 = QColor("#7aa6d4") 
        c2 = QColor.fromHslF(shift, 0.6, 0.6, 1.0) 
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
        
        font = QFont("Segoe UI", 10)
        font.setWeight(QFont.Weight.DemiBold)
        self.setFont(font)
        
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(30)
        self.setMinimumWidth(200)
        
        self.phase = 0.0
        self.pan_phase = 0.0
        self._opacity = 1.0 
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)

    def get_opacity(self): return self._opacity
    def set_opacity(self, o): 
        self._opacity = max(0.0, min(1.0, o))
        self.update()
    
    opacity = pyqtProperty(float, get_opacity, set_opacity)

    def animate(self):
        self.phase = (self.phase + 0.002) % 1.0
        # VARIABLE TO CHANGE: Increased from 0.015 to 0.04 for faster scroll
        self.pan_phase += 0.04
        self.update()

    def paintEvent(self, event):
        if self._opacity <= 0.01: return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._opacity)
        
        rect = self.rect()
        w, h = rect.width(), rect.height()
        
        fm = QFontMetrics(self.font())
        txt = self.text()
        txt_w = fm.horizontalAdvance(txt)
        txt_h = fm.capHeight()
        
        # Panning Logic
        x_offset = 0
        is_panning = False
        
        if txt_w > w:
            is_panning = True
            overflow = txt_w - w + 40 
            cycle = math.sin(self.pan_phase)
            if cycle > 0.8: cycle = 0.8
            if cycle < -0.8: cycle = -0.8
            norm = (cycle + 0.8) / 1.6 
            x_offset = -1 * (norm * overflow)
            x_offset += 20 
        else:
            x_offset = (w - txt_w) / 2

        # MATCHING Y-OFFSET
        y_offset = (h + txt_h) / 2 - 2 

        grad = QLinearGradient(0, 0, w, 0)
        for i in range(4):
            t = i / 3.0
            hue = (self.phase + t * 0.3) % 1.0
            col = QColor.fromHslF(hue, 0.30, 0.60, 1.0)
            grad.setColorAt(t, col)
        
        painter.setPen(QPen(QBrush(grad), 0))
        painter.setFont(self.font())
        
        painter.save()
        painter.setClipRect(rect)
        painter.drawText(int(x_offset), int(y_offset), txt)
        painter.restore()

        if is_panning:
            fade_w = 20
            c_bg = QColor("#f6f9fc") 
            l_grad = QLinearGradient(0, 0, fade_w, 0)
            c_bg.setAlpha(255); l_grad.setColorAt(0.0, c_bg)
            c_bg.setAlpha(0);   l_grad.setColorAt(1.0, c_bg)
            painter.fillRect(0, 0, fade_w, h, l_grad)
            
            r_grad = QLinearGradient(w - fade_w, 0, w, 0)
            c_bg.setAlpha(0);   r_grad.setColorAt(0.0, c_bg)
            c_bg.setAlpha(255); r_grad.setColorAt(1.0, c_bg)
            painter.fillRect(w - fade_w, 0, fade_w, h, r_grad)

class ExportMessageLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._opacity = 0.0 
        self.phase = 0.0
        
        font = QFont("Segoe UI", 10)
        font.setWeight(QFont.Weight.Bold)
        self.setFont(font)
        
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setMinimumHeight(30)
        self.setMinimumWidth(200)
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)

    def get_opacity(self): return self._opacity
    def set_opacity(self, o): 
        self._opacity = max(0.0, min(1.0, o))
        self.update()

    opacity = pyqtProperty(float, get_opacity, set_opacity)

    def animate(self):
        self.phase = (self.phase + 0.01) % 1.0
        if self._opacity > 0.01:
            self.update()

    def paintEvent(self, event):
        if self._opacity <= 0.01: return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(self._opacity)
        
        rect = self.rect()
        w, h = rect.width(), rect.height()

        # 1. Calculate Text Position exactly like PastelFileLabel
        fm = QFontMetrics(self.font())
        txt = self.text()
        txt_w = fm.horizontalAdvance(txt)
        txt_h = fm.capHeight()

        # Always centered
        x_offset = (w - txt_w) / 2
        # MATCHING Y-OFFSET
        y_offset = (h + txt_h) / 2 - 2 
        
        # 2. Gradient
        grad = QLinearGradient(0, 0, w, 0)
        shift = self.phase
        c1 = QColor("#8da6c0") 
        c2 = QColor.fromHslF(shift, 0.25, 0.65, 1.0)
        c3 = QColor("#8da6c0")
        grad.setColorAt(0.0, c1); grad.setColorAt(0.5, c2); grad.setColorAt(1.0, c3)
        
        painter.setPen(QPen(QBrush(grad), 0))
        painter.setFont(self.font())
        
        # 3. Draw using calculated offsets
        painter.drawText(int(x_offset), int(y_offset), txt)

class PastelClearButton(QPushButton):
    cleared = pyqtSignal()

    def __init__(self, text="clear", parent=None):
        super().__init__(text, parent)
        self.setFixedSize(60, 24) # Slightly smaller to fit nicely next to Play
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMouseTracking(True)
        
        self.phase = 0.0
        self.hover_progress = 0.0
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(50)

    def animate(self):
        self.phase = (self.phase + 0.02) % 1.0
        
        # Smooth hover transition
        target_hover = 1.0 if self.underMouse() else 0.0
        self.hover_progress = self.hover_progress * 0.8 + target_hover * 0.2
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.cleared.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect()

        # Dynamic Gradient
        grad = QLinearGradient(0, 0, rect.width(), 0)
        
        # Base colors: Subtle Red/Pink
        sat_base = 0.3 + (0.3 * self.hover_progress)
        alpha_base = 100 + (100 * self.hover_progress)
        
        for i in range(3):
            t = i / 2.0
            hue = (0.95 + self.phase * 0.1 + t * 0.1) % 1.0
            col = QColor.fromHslF(hue, sat_base, 0.85, alpha_base/255.0)
            grad.setColorAt(t, col)

        painter.setBrush(grad)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect, 6, 6)

        # Text
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        text_col = QColor("#d9534f")
        text_col.setAlpha(int(150 + 105 * self.hover_progress))
        painter.setPen(text_col)
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, self.text())

class MorphPlayButton(QWidget):
    clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self._ready = False
        self._playing = False 
        self._hover = False
        
        self.ready_progress = 0.0 
        self.icon_progress = 0.0
        self.hover_progress = 0.0 # New hover animation state
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(16)

    def set_ready(self, r): self._ready = r
    def set_playing(self, p): self._playing = p

    def mousePressEvent(self, e): 
        if self._ready and e.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit()
            
    def enterEvent(self, e): self._hover = True
    def leaveEvent(self, e): self._hover = False

    def animate(self):
        t_ready = 1.0 if self._ready else 0.0
        self.ready_progress += (t_ready - self.ready_progress) * 0.15
        
        t_icon = 1.0 if self._playing else 0.0
        self.icon_progress += (t_icon - self.icon_progress) * 0.2
        
        # Fast subtle fade for hover
        t_hover = 1.0 if (self._ready and self._hover) else 0.0
        self.hover_progress += (t_hover - self.hover_progress) * 0.25
        
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        r = self.rect()
        c = QRectF(r).center()
        
        # Interpolate Background Color: #ecf2f7 -> #dfe7ef
        base_r, base_g, base_b = 236, 242, 247
        hover_r, hover_g, hover_b = 223, 231, 239
        
        k = self.hover_progress
        curr_r = int(base_r + (hover_r - base_r) * k)
        curr_g = int(base_g + (hover_g - base_g) * k)
        curr_b = int(base_b + (hover_b - base_b) * k)
        
        bg_col = QColor(curr_r, curr_g, curr_b)
        p.setBrush(bg_col)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, 6, 6)
        
        # Foreground Color
        fg = QColor("#3f6c9b")
        if not self._ready:
            fg = QColor("#a0b0c0")
        p.setBrush(fg)

        # 1. Draw Dot
        if self.ready_progress < 0.99:
            dot_scale = 1.0 - self.ready_progress
            p.setOpacity(dot_scale)
            p.drawEllipse(c, 3, 3)
            p.setOpacity(1.0)

        # 2. Draw Icon
        if self.ready_progress > 0.01:
            p.translate(c)
            s = self.ready_progress
            p.scale(s, s)
            t = self.icon_progress
            
            p1x = -4.0 * (1.0-t) + (-5.0) * t
            p2x = -4.0 * (1.0-t) + (-5.0) * t
            tip_x = 6.0 * (1.0-t) + (-2.0) * t
            tip_y_top = 0.0 * (1.0-t) + (-6.0) * t
            tip_y_bot = 0.0 * (1.0-t) + (6.0) * t
            
            path = QPainterPath()
            path.moveTo(p1x, -6)
            path.lineTo(p2x, 6)
            path.lineTo(tip_x, tip_y_bot)
            path.lineTo(tip_x, tip_y_top)
            path.closeSubpath()
            p.drawPath(path)
            
            if t > 0.01:
                fg.setAlpha(int(255 * t))
                p.setBrush(fg)
                p.drawRoundedRect(QRectF(2.0, -6.0, 3.0, 12.0), 1.0, 1.0)

class MorphClearButton(QWidget):
    cleared = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(32)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        
        self._ready = False
        self._hover = False
        self.ready_progress = 0.0
        self.hover_progress = 0.0 # New hover animation state
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        self.timer.start(20)

    def set_ready(self, r): self._ready = r

    def mousePressEvent(self, e): 
        if self._ready and e.button() == Qt.MouseButton.LeftButton:
            self.cleared.emit()
            
    def enterEvent(self, e): self._hover = True
    def leaveEvent(self, e): self._hover = False

    def animate(self):
        t_ready = 1.0 if self._ready else 0.0
        self.ready_progress += (t_ready - self.ready_progress) * 0.15
        
        # Fast subtle fade for hover
        t_hover = 1.0 if (self._ready and self._hover) else 0.0
        self.hover_progress += (t_hover - self.hover_progress) * 0.25
        
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        r = self.rect()
        c = QRectF(r).center()
        
        # Interpolate Background Color: #ecf2f7 -> #fce8e8
        base_r, base_g, base_b = 236, 242, 247
        hover_r, hover_g, hover_b = 252, 232, 232
        
        k = self.hover_progress
        curr_r = int(base_r + (hover_r - base_r) * k)
        curr_g = int(base_g + (hover_g - base_g) * k)
        curr_b = int(base_b + (hover_b - base_b) * k)
        
        bg_col = QColor(curr_r, curr_g, curr_b)
        p.setBrush(bg_col)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawRoundedRect(r, 6, 6)
        
        dot_col = QColor("#a0b0c0")
        txt_col = QColor("#d9534f")
        
        if self.ready_progress < 0.99:
            dot_scale = 1.0 - self.ready_progress
            p.setOpacity(dot_scale)
            p.setBrush(dot_col)
            p.drawEllipse(c, 3, 3)
            p.setOpacity(1.0)

        if self.ready_progress > 0.01:
            p.setOpacity(self.ready_progress)
            font = QFont("Segoe UI", 9)
            font.setBold(True)
            font.setCapitalization(QFont.Capitalization.AllLowercase)
            p.setFont(font)
            p.setPen(txt_col)
            p.drawText(r, Qt.AlignmentFlag.AlignCenter, "clear")

class DebugLogWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.items = [] 
        self.wave_phase = 0.0 
        self.sticky_msg = None # Persistent state
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate)
        
        self.log_font = QFont("Consolas", 9)
        if not self.log_font.exactMatch(): 
            self.log_font = QFont("Courier New", 9)
            
        self.matrix_chars = "prism"

    def resizeEvent(self, event):
        if self.items:
            self.layout_items(self.width())
            if not self.timer.isActive():
                self.timer.start(10)
        super().resizeEvent(event)

    def set_sticky_status(self, text):
        self.sticky_msg = text
        
        # Clear old status logic
        self.items = [i for i in self.items if i.get('identity') != 'sticky_status']
        
        if text:
            item = {
                'identity': 'sticky_status',
                'text': text,
                'scramble_val': text, 
                'scramble_time': 5, 
                'scramble_delay': 0,
                'x': 0.0, 'y': 0.0, 
                'target_x': 0.0, 'target_y': 0.0,
                'alpha': 0.0
            }
            self.items.append(item)
            
        self.layout_items(self.width())
        self.timer.start(20)

    def log_process(self, params, time_ms):
        # 1. Map existing items
        existing_map = {item.get('identity'): item for item in self.items}
        new_items = []
        
        def add_item(identity, text):
            if identity in existing_map:
                item = existing_map[identity]
                item['text'] = text
                item['scramble_val'] = text
                item['scramble_time'] = 0 
                new_items.append(item)
            else:
                item = {
                    'identity': identity,
                    'text': text,
                    'scramble_val': text, 
                    'scramble_time': 10, 
                    'scramble_delay': 0,
                    'x': 0.0, 'y': 0.0, 
                    'target_x': 0.0, 'target_y': 0.0,
                    'alpha': 0.0
                }
                new_items.append(item)

        # 2. Build List
        timestamp = time.strftime("%H:%M:%S")
        add_item('__header__', f"[{timestamp}] done::{time_ms}ms")
        
        for k, v in params.items():
            if k in ['bpm', 'sr_select', 'chops_enabled']: continue 
            if isinstance(v, (int, float)) and v > 0.01:
                key_short = k.replace("_", " ")
                if len(key_short) > 10: key_short = key_short[:9] + "."
                add_item(k, f"{key_short}:{v:.2f}")
        
        add_item('__state__', "state:active")
        
        # CRITICAL FIX: Re-append sticky message if it exists
        if self.sticky_msg:
            add_item('sticky_status', self.sticky_msg)
        
        # 3. Apply
        self.items = new_items
        self.layout_items(self.width())
        self.timer.start(20)

    def layout_items(self, width):
        fm = QFontMetrics(self.log_font)
        y_start = 15 
        y = y_start
        line_h = 20
        spacing = 20
        
        lines = []
        current_line_items = []
        current_line_width = 0
        
        for item in self.items:
            w = fm.horizontalAdvance(item['text'])
            if current_line_items and (current_line_width + w > width - 20):
                lines.append((current_line_items, current_line_width))
                current_line_items = []
                current_line_width = 0
            
            if current_line_items:
                current_line_width += spacing
            
            current_line_items.append((item, w))
            current_line_width += w
            
        if current_line_items:
            lines.append((current_line_items, current_line_width))

        for line_items, line_w in lines:
            start_x = (width - line_w) / 2
            current_x = start_x
            for item, w in line_items:
                item['target_x'] = float(current_x)
                item['target_y'] = float(y)
                if item['x'] == 0.0 and item['y'] == 0.0:
                    item['x'] = float(current_x)
                    item['y'] = float(y)
                current_x += w + spacing
            y += line_h

    def animate(self):
        if not self.items:
            self.timer.stop()
            return
            
        self.wave_phase = (self.wave_phase + 0.025) % (math.pi * 200)
        rng = np.random.default_rng()
        
        for item in self.items:
            dx = item['target_x'] - item['x']
            dy = item['target_y'] - item['y']
            if abs(dx) > 0.1 or abs(dy) > 0.1:
                item['x'] += dx * 0.15
                item['y'] += dy * 0.15
            else:
                item['x'] = item['target_x']
                item['y'] = item['target_y']

            if item['alpha'] < 1.0:
                item['alpha'] += 0.05
                if item['alpha'] > 1.0: item['alpha'] = 1.0
            
            if item['scramble_time'] > 0:
                item['scramble_delay'] += 1
                if item['scramble_delay'] > 2:
                    item['scramble_time'] -= 1
                    item['scramble_delay'] = 0
                    chars = [self.matrix_chars[rng.integers(len(self.matrix_chars))] for _ in item['text']]
                    item['scramble_val'] = "".join(chars)
            else:
                item['scramble_val'] = item['text']

        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        painter.setFont(self.log_font)
        
        base_col = QColor(140, 160, 175)
        
        for item in self.items:
            if item['alpha'] <= 0: continue
            
            c = QColor(base_col)
            
            if item['scramble_time'] > 0:
                hue = (time.time() * 0.5 + item['x'] * 0.005) % 1.0
                c = QColor.fromHslF(hue, 0.4, 0.6, item['alpha'])
            else:
                wave = math.sin(item['x'] * 0.015 + self.wave_phase)
                wave_factor = 0.75 + (wave * 0.25) 
                final_alpha = item['alpha'] * wave_factor
                c.setAlphaF(final_alpha)
                
            painter.setPen(c)
            painter.drawText(QPointF(item['x'], item['y']), item['scramble_val'])

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("prism")
        self.resize(900, 550) 
        self.setAcceptDrops(True)
        self.file_path = None
        self.original_audio = None
        self.processed_audio = None
        self.sr = 44100
        self.temp_file = None
        self.is_processing = False 
        
        self.chops_enabled = True # Default state
        
        self.last_wall_clock = 0.0
        self.last_media_pos = 0
        
        self.params = {
            'glitch': 0.5, 'crush': 0.0, 'reverb': 0.0, 
            'sr_select': 44100.0, 'rate': 1.0, 
            'filter_amt': 0.0, 'vol_pan_amt': 0.0, 'bpm': 120.0,
            'swing': 0.0, 'tone': 0.5
        }
        
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.mediaStatusChanged.connect(self.media_status_changed)

        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(12)
        self.anim_timer.timeout.connect(self.high_freq_update)

        central = QWidget()
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0) 
        main_layout.setSpacing(0)

        # --- LEFT: Viewport ---
        self.viewport = QVBoxLayout()
        self.wave_view = WaveformWidget()
        self.wave_view.setMinimumHeight(120) 
        self.wave_view.seek_requested.connect(self.seek_audio)
        self.wave_view.import_clicked.connect(self.open_file_dialog)
        self.wave_view.scrub_started.connect(self.anim_timer.stop)
        self.wave_view.scrub_ended.connect(self.resume_sync)
        self.viewport.addWidget(self.wave_view, 1)
        
        # --- RIGHT: Sidebar ---
        self.sidebar = GlassFrame()
        self.sidebar.setFixedWidth(330) 
        side_layout = QVBoxLayout(self.sidebar)
        side_layout.setContentsMargins(12, 12, 12, 12) 
        side_layout.setSpacing(2) 

        # Header
        self.logo = PrismLogo()
        self.logo.setMinimumHeight(40)
        self.logo.clicked.connect(self.toggle_chops) # Connect new signal
        side_layout.addWidget(self.logo)
        self.lbl_status = StatusRainbowLabel("status: idle")
        side_layout.addWidget(self.lbl_status)
        
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: rgba(63, 108, 155, 0.2);")
        side_layout.addWidget(line)
        side_layout.addSpacing(8)

        # Controls
        columns_layout = QHBoxLayout()
        columns_layout.setContentsMargins(0, 0, 0, 0)
        columns_layout.setSpacing(10)
        left_col = QVBoxLayout(); left_col.setSpacing(2)
        right_col = QVBoxLayout(); right_col.setSpacing(2)

        left_col.addWidget(BpmControlRow("bpm", "bpm", self.params))
        left_col.addWidget(ControlRow("grid density", "glitch", self.params))
        left_col.addWidget(ControlRow("swing", "swing", self.params))
        left_col.addWidget(ControlRow("vol/pan", "vol_pan_amt", self.params))
        left_col.addWidget(RateControlRow("playback rate", "rate", self.params))
        
        right_col.addWidget(ControlRow("tone", "tone", self.params))
        right_col.addWidget(ControlRow("crush", "crush", self.params))
        right_col.addWidget(SampleRateRow("sample rate", "sr_select", self.params))
        right_col.addWidget(ControlRow("filter", "filter_amt", self.params))
        right_col.addWidget(ControlRow("reverb", "reverb", self.params))

        columns_layout.addLayout(left_col)
        columns_layout.addLayout(right_col)
        side_layout.addLayout(columns_layout)
        side_layout.addSpacing(15)

        # Transport
        transport_layout = QHBoxLayout()
        transport_layout.setContentsMargins(0, 0, 0, 0)
        transport_layout.setSpacing(10) # Horizontal spacing is 10
        self.btn_play = MorphPlayButton()
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_play.set_ready(False)
        transport_layout.addWidget(self.btn_play)
        self.btn_clear = MorphClearButton()
        self.btn_clear.cleared.connect(self.clear_state)
        self.btn_clear.set_ready(False)
        transport_layout.addWidget(self.btn_clear)
        side_layout.addLayout(transport_layout)
        
        # CHANGED: Reduced from 10 to 6. 
        # Math: 6 (spacer) + 2 (top gap) + 2 (bottom gap) = 10px total visual space
        side_layout.addSpacing(6)

        # Action Buttons
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 0, 0, 0)
        action_layout.setSpacing(10) # Horizontal spacing is 10
        
        self.btn_process = RainbowButton("process")
        self.btn_process.setObjectName("ProcessBtn")
        self.btn_process.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_process.clicked.connect(self.start_processing)
        # FIXED: Button is ENABLED by default so it clicks/ripples, but logic will block it.
        self.btn_process.setEnabled(True) 
        
        self.btn_save = RainbowButton("export")
        self.btn_save.setObjectName("SaveBtn")
        self.btn_save.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_save.clicked.connect(self.quick_export)
        # FIXED: Button is ENABLED by default.
        self.btn_save.setEnabled(True)

        action_layout.addWidget(self.btn_process)
        action_layout.addWidget(self.btn_save)
        side_layout.addLayout(action_layout)
        
        # CHANGED: Reduced spacing to move footer up
        side_layout.addSpacing(5) 

        # --- FOOTER & ANIMATION SETUP ---
        footer_layout = QHBoxLayout()
        footer_layout.setContentsMargins(0, 0, 0, 0)
        footer_layout.setSpacing(0)
        
        footer_layout.addStretch() 
        
        # 1. File Label
        self.lbl_file = PastelFileLabel("no file loaded")
        # Ensure it has a minimum height so it doesn't disappear
        self.lbl_file.setMinimumHeight(28) 
        footer_layout.addWidget(self.lbl_file)
        
        # 2. Export Message (Hidden by default)
        self.lbl_saved_msg = ExportMessageLabel("")
        self.lbl_saved_msg.setMinimumHeight(28)
        self.lbl_saved_msg.setVisible(False)
        self.lbl_saved_msg.set_opacity(0.0)
        footer_layout.addWidget(self.lbl_saved_msg)
        
        # Initialize Animations
        self.anim_export_fade = QPropertyAnimation(self.lbl_saved_msg, b"opacity")
        self.anim_export_fade.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        self.anim_file_fade = QPropertyAnimation(self.lbl_file, b"opacity")
        self.anim_file_fade.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        footer_layout.addStretch()
        
        side_layout.addLayout(footer_layout)
        side_layout.addSpacing(5) 

        # Debug Log
        self.debug_log = DebugLogWidget()
        side_layout.addWidget(self.debug_log, 1)

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
        fname, _ = QFileDialog.getOpenFileName(self, 'open', os.path.expanduser("~"), "audio files (*.wav *.mp3 *.flac *.ogg)")
        if fname: self.load_file(fname)

    def set_ui_locked(self, locked):
        self.is_processing = locked
        # We purposely do NOT disable buttons visually to avoid "refresh" flash

    def clear_state(self):
        self.stop_playback()
        self.player.setSource(QUrl())
        self.file_path = None
        self.original_audio = None
        self.processed_audio = None
        self.is_processing = False
        
        self.wave_view.data = None
        self.wave_view.update_static_waveform()
        self.wave_view.update()
        
        # FIX: Ensure File Label is reset to visible
        self.lbl_file.setText("no file loaded")
        self.lbl_file.set_opacity(1.0)
        self.lbl_file.setVisible(True)
        
        self.lbl_saved_msg.setVisible(False)
        self.lbl_status.setText("status: idle")
        
        self.btn_clear.set_ready(False)
        self.btn_play.set_ready(False)
        self.btn_play.set_playing(False)
    
    def load_file(self, path):
        self.stop_playback()
        self.file_path = path
        
        # FIX: Ensure File Label is reset to visible
        self.lbl_file.setText(os.path.basename(path).lower())
        self.lbl_file.set_opacity(1.0)
        self.lbl_file.setVisible(True)
        self.lbl_saved_msg.setVisible(False)
        
        self.btn_clear.set_ready(True)
        self.btn_play.set_ready(True)
        try:
            data, sr = AudioEngine.load_file(path)
            self.sr = sr
            self.original_audio = data
            self.player.setSource(QUrl.fromLocalFile(path))
            display_data = data[:sr*30] if len(data) > sr*30 else data
            self.wave_view.set_data(display_data)
            self.lbl_status.setText("status: ready")
        except Exception as e:
            QMessageBox.critical(self, "error", f"could not load file: {e}")

    def resume_sync(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.last_wall_clock = time.perf_counter()
            self.last_media_pos = self.player.position()
            self.anim_timer.start()

    def high_freq_update(self):
        duration = self.player.duration()
        if duration > 0 and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            now = time.perf_counter()
            delta = now - self.last_wall_clock
            
            # Do NOT multiply by rate here. 
            # The audio file was physically resampled/shortened by the engine.
            # The player plays the shortened file at 1.0x speed.
            interpolated_ms = self.last_media_pos + (delta * 1000.0)
            
            # Check for drift
            actual_pos = self.player.position()
            
            if abs(interpolated_ms - actual_pos) > 150: 
                self.last_media_pos = actual_pos
                self.last_wall_clock = now
                interpolated_ms = actual_pos
            elif actual_pos < self.last_media_pos and actual_pos < 500:
                # Loop detected
                self.last_media_pos = actual_pos
                self.last_wall_clock = now
                interpolated_ms = actual_pos

            self.wave_view.set_play_head(interpolated_ms / duration)

    def toggle_playback(self):
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            self.anim_timer.stop() 
            self.btn_play.set_playing(False) # Use set_playing instead of setText
        else:
            self.last_wall_clock = time.perf_counter()
            self.last_media_pos = self.player.position()
            self.player.play()
            self.anim_timer.start()
            self.btn_play.set_playing(True)

    def stop_playback(self):
        self.player.stop()
        self.anim_timer.stop() 
        self.btn_play.set_playing(False)
        self.wave_view.set_play_head(0)

    def seek_audio(self, pos_norm):
        duration = self.player.duration()
        if duration > 0: 
            ms = int(pos_norm * duration)
            self.player.setPosition(ms)
            self.last_media_pos = ms
            self.last_wall_clock = time.perf_counter()

    def media_status_changed(self, status):
        if status == QMediaPlayer.MediaStatus.EndOfMedia:
            self.btn_play.set_playing(False)
            self.wave_view.set_play_head(0)
            self.anim_timer.stop()

    def toggle_chops(self):
        self.chops_enabled = not self.chops_enabled
        msg = "" if self.chops_enabled else "[chops disabled]"
        self.debug_log.set_sticky_status(msg)

    def start_processing(self):
        # LOGIC LOCK: Check if we have file and aren't already busy
        if self.is_processing or self.original_audio is None: 
            return
            
        self.stop_playback()
        self.lbl_status.setText("status: processing...")
        self.set_ui_locked(True)
        
        self._proc_start_time = time.perf_counter()
        
        # Inject chops_enabled state
        p = self.params.copy()
        p['chops_enabled'] = self.chops_enabled
        
        self.thread = ProcessThread(self.original_audio, self.sr, p)
        self.thread.finished_ok.connect(self.processing_done)
        self.thread.error.connect(self.processing_error)
        self.thread.start()

    def processing_done(self, data, sr):
        t_end = time.perf_counter()
        dur_ms = int((t_end - self._proc_start_time) * 1000)
        
        self.processed_audio = data
        self.wave_view.set_data(data)
        self.lbl_status.setText("status: done")
        self.debug_log.log_process(self.params, dur_ms)
        self.set_ui_locked(False)
        
        try:
            fd, temp_path = tempfile.mkstemp(suffix='.wav')
            os.close(fd)
            AudioEngine.save_file(temp_path, data, sr)
            self.temp_file = temp_path
            self.player.setSource(QUrl.fromLocalFile(temp_path))
            self.last_media_pos = 0
            self.last_wall_clock = time.perf_counter()
            self.player.play()
            self.btn_play.set_playing(True)
            self.anim_timer.start()
        except Exception as e: print(f"Temp file error: {e}")

    def processing_error(self, msg):
        QMessageBox.critical(self, "processing error", msg)
        self.lbl_status.setText("status: error")
        self.set_ui_locked(False)

    def quick_export(self):
        if self.is_processing or self.processed_audio is None: 
            return
            
        self.logo.trigger_excitement()
        
        home_dir = os.path.expanduser("~")
        save_dir = os.path.join(home_dir, "Music", "prism")
        if not os.path.exists(save_dir):
            try: os.makedirs(save_dir)
            except: pass

        timestamp = int(time.time())
        filename = f"prism_{timestamp}.wav"
        save_path = os.path.join(save_dir, filename)
        
        try:
            AudioEngine.save_file(save_path, self.processed_audio, self.sr)
            self.lbl_status.setText("status: exported")
            
            # 1. Stop any running animations
            self.anim_file_fade.stop()
            self.anim_export_fade.stop()
            
            # 2. Fade OUT File Label
            self.anim_file_fade.setDuration(200)
            self.anim_file_fade.setStartValue(self.lbl_file.opacity)
            self.anim_file_fade.setEndValue(0.0)
            
            try: self.anim_file_fade.finished.disconnect()
            except: pass
            self.anim_file_fade.finished.connect(self._on_file_faded_out_for_export)
            
            self.anim_file_fade.start()
            
        except Exception as e: 
            QMessageBox.critical(self, "error", f"could not save: {e}")

    def _on_file_faded_out_for_export(self):
        # 3. Swap and Fade IN Export Message
        self.lbl_file.setVisible(False)
        self.lbl_saved_msg.setText("saved to: Music/prism")
        self.lbl_saved_msg.set_opacity(0.0)
        self.lbl_saved_msg.setVisible(True)
        
        self.anim_export_fade.setDuration(300)
        self.anim_export_fade.setStartValue(0.0)
        self.anim_export_fade.setEndValue(1.0)
        
        try: self.anim_export_fade.finished.disconnect()
        except: pass
        self.anim_export_fade.finished.connect(self.schedule_fade_out)
        
        self.anim_export_fade.start()

    def schedule_fade_out(self):
        # 4. Wait 2 seconds (No self.msg_timer needed here)
        QTimer.singleShot(2000, self.start_return_sequence)

    def start_return_sequence(self):
        # 5. Fade OUT Export Message (Faster)
        self.anim_export_fade.stop()
        try: self.anim_export_fade.finished.disconnect()
        except: pass
        
        self.anim_export_fade.setDuration(200) # Reduced from 500
        self.anim_export_fade.setStartValue(self.lbl_saved_msg.opacity)
        self.anim_export_fade.setEndValue(0.0)
        
        self.anim_export_fade.finished.connect(self._on_export_faded_out)
        self.anim_export_fade.start()

    def _on_export_faded_out(self):
        # 6. Swap and Fade IN File Label (Faster)
        self.lbl_saved_msg.setVisible(False)
        self.lbl_file.set_opacity(0.0)
        self.lbl_file.setVisible(True)
        
        self.anim_file_fade.setDuration(150) # Reduced from 300
        self.anim_file_fade.setStartValue(0.0)
        self.anim_file_fade.setEndValue(1.0)
        
        try: self.anim_file_fade.finished.disconnect()
        except: pass
        
        self.anim_file_fade.start()

    def start_fade_out(self):
        self.fade_anim.stop()
        try: self.fade_anim.finished.disconnect()
        except: pass
        
        self.fade_anim.setDuration(600)
        self.fade_anim.setStartValue(1.0)
        self.fade_anim.setEndValue(0.0)
        
        # When fade out is done, swap back
        self.fade_anim.finished.connect(self.finalize_hide)
        self.fade_anim.start()

    def finalize_hide(self):
        # Restore State
        self.lbl_saved_msg.setVisible(False)
        self.lbl_file.setVisible(True)
    
    def start_collapse(self):
        # Step 3: Animate Width (The smooth ease back)
        self.collapse_anim = QPropertyAnimation(self.lbl_saved_msg, b"maximumWidth")
        self.collapse_anim.setDuration(250)
        # Animate from current width down to 0
        self.collapse_anim.setStartValue(self.lbl_saved_msg.width())
        self.collapse_anim.setEndValue(0)
        self.collapse_anim.setEasingCurve(QEasingCurve.Type.InOutQuad)
        
        self.collapse_anim.finished.connect(self.finalize_hide)
        self.collapse_anim.start()
    
    def closeEvent(self, event):
        if self.temp_file and os.path.exists(self.temp_file):
            try: os.remove(self.temp_file)
            except: pass
        event.accept()

if __name__ == '__main__':
    try:
        myappid = 'prism.audio.tool.v1'
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(myappid)
    except Exception:
        pass 

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLES)
    icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prism.ico")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    elif os.path.exists("prism.ico"):
        app.setWindowIcon(QIcon("prism.ico"))

    if hasattr(Qt.ApplicationAttribute, 'AA_EnableHighDpiScaling'):
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
    if hasattr(Qt.ApplicationAttribute, 'AA_UseHighDpiPixmaps'):
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
        
    window = MainWindow()
    window.show()
    sys.exit(app.exec())