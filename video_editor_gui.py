
# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import platform
import locale


try:
    from bidi.algorithm import get_display
    from arabic_reshaper import reshape
    ARABIC_SUPPORT = True
except ImportError:
    print("تحذير: مكتبات دعم العربية غير مثبتة. سيتم تثبيتها تلقائياً...")
    try:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "python-bidi", "arabic-reshaper"])
        from bidi.algorithm import get_display
        from arabic_reshaper import reshape
        ARABIC_SUPPORT = True
        print("تم تثبيت مكتبات دعم العربية بنجاح!")
    except Exception as e:
        print(f"فشل في تثبيت مكتبات دعم العربية: {e}")
        ARABIC_SUPPORT = False

import imageio_ffmpeg
from pydub import AudioSegment

try:
    base_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_path = os.path.abspath(".")

def find_executable(name):
    
    try:
        if name == 'ffmpeg':
            return imageio_ffmpeg.get_ffmpeg_exe()
        if name == 'ffprobe':
            ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            ffprobe_path = os.path.join(os.path.dirname(ffmpeg_path), 'ffprobe')
            if sys.platform == 'win32' and not ffprobe_path.endswith('.exe'):
                ffprobe_path += '.exe'
            if os.path.exists(ffprobe_path):
                return ffprobe_path
    except Exception:
        pass  

    
    executable = shutil.which(name)
    if executable:
        return os.path.normpath(executable)

    
    exe_name_with_ext = f"{name}.exe" if sys.platform == 'win32' else name
    
    local_paths_to_check = [
        os.path.join(base_path, exe_name_with_ext),
        os.path.join(base_path, 'bin', exe_name_with_ext),
        os.path.join(base_path, 'ffmpeg', 'bin', exe_name_with_ext),
        os.path.join(base_path, 'ffmpeg', 'ffmpeg-7.1.1-essentials_build', 'bin', exe_name_with_ext),
    ]

    for path in local_paths_to_check:
        if os.path.exists(path):
            return os.path.normpath(path)
    
    return None

ffmpeg_exe_path = find_executable('ffmpeg')
ffprobe_exe_path = find_executable('ffprobe')
ffplay_exe_path = find_executable('ffplay')

if ffmpeg_exe_path:
    AudioSegment.converter = ffmpeg_exe_path
if ffprobe_exe_path:
    AudioSegment.ffprobe = ffprobe_exe_path



import cv2
import numpy as np
import tempfile
import threading
import json
from subprocess import Popen
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time
from functools import partial
import pickle
from multiprocessing import shared_memory
try:
    import psutil
except ImportError:
    print("تحذير: مكتبة psutil غير متوفرة. بعض ميزات مراقبة الأداء قد لا تعمل.")
    psutil = None

import gc
import mmap
from collections import deque
import queue
import weakref
import re
import codecs

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

SUBPROCESS_CREATION_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

def fix_arabic_text(text):
    if not text or not isinstance(text, str):
        return text

    if not ARABIC_SUPPORT:
        return text

    try:
        
        reshaped_text = reshape(text)
        
        bidi_text = get_display(reshaped_text)
        return bidi_text
    except Exception as e:
        print(f"خطأ في معالجة النص العربي '{text}': {e}")
        return text

def get_arabic_font():
    try:
        system = platform.system()

        if system == "Linux":
            
            arabic_fonts = [
                "Noto Sans Arabic",
                "Noto Kufi Arabic",
                "Amiri",
                "Cairo",
                "Tajawal",
                "Almarai",
                "Scheherazade New",
                "Lateef",
                "DejaVu Sans",
                "Liberation Sans",
                "Ubuntu",
                "Arial"
            ]

            
            for font in arabic_fonts:
                if is_font_available(font):
                    return font

            
            return "Arial"

        elif system == "Windows":
            
            windows_fonts = ["Segoe UI", "Tahoma", "Arial Unicode MS", "Arial"]
            for font in windows_fonts:
                if is_font_available(font):
                    return font
            return "Arial"

        elif system == "Darwin":  
            mac_fonts = ["SF Arabic", "Arial Unicode MS", "Arial"]
            for font in mac_fonts:
                if is_font_available(font):
                    return font
            return "Arial"

        return "Arial"
    except Exception as e:
        print(f"خطأ في الحصول على الخط العربي: {e}")
        return "Arial"

def is_font_available(font_name):
    try:
        import tkinter.font as tkFont
        available_fonts = tkFont.families()
        return font_name in available_fonts
    except Exception as e:
        print(f"خطأ في فحص الخط {font_name}: {e}")
        return False

def configure_arabic_support():
    try:
        
        if hasattr(sys.stdout, 'reconfigure'):
            sys.stdout.reconfigure(encoding='utf-8')
        if hasattr(sys.stderr, 'reconfigure'):
            sys.stderr.reconfigure(encoding='utf-8')

        
        try:
            if platform.system() == "Linux":
                
                for arabic_locale in ['ar_SA.UTF-8', 'ar_EG.UTF-8', 'ar.UTF-8', 'C.UTF-8']:
                    try:
                        locale.setlocale(locale.LC_ALL, arabic_locale)
                        break
                    except locale.Error:
                        continue
        except:
            pass

        return True
    except Exception as e:
        print(f"تحذير: لا يمكن تكوين دعم اللغة العربية: {e}")
        return False

def configure_rtl_widget(widget, widget_type="label"):
    try:
        if platform.system() == "Linux":
            if widget_type == "entry":
                widget.configure(justify='right')
            elif widget_type == "label":
                widget.configure(anchor='e')
            elif widget_type == "text":
                widget.configure(justify='right')

        
        arabic_font = get_arabic_font()
        if hasattr(widget, 'configure'):
            try:
                current_font = widget.cget('font')
                if current_font:
                    
                    if isinstance(current_font, tuple) and len(current_font) >= 2:
                        size = current_font[1]
                        style = current_font[2] if len(current_font) > 2 else "normal"
                        widget.configure(font=(arabic_font, size, style))
                    else:
                        widget.configure(font=(arabic_font, 10))
                else:
                    widget.configure(font=(arabic_font, 10))
            except:
                pass

    except Exception as e:
        print(f"تحذير: لا يمكن تكوين عنصر الواجهة للعربية: {e}")

def create_arabic_label(parent, text, use_ttk=False, **kwargs):
    arabic_font = get_arabic_font()

    
    fixed_text = fix_arabic_text(text)

    if use_ttk:
        
        default_kwargs = {
            'font': (arabic_font, 10)
        }
        default_kwargs.update(kwargs)
        label = ttk.Label(parent, text=fixed_text, **default_kwargs)
    else:
        
        default_kwargs = {
            'font': (arabic_font, 10),
            'anchor': 'e' if platform.system() == "Linux" else 'w',
            'justify': 'right' if platform.system() == "Linux" else 'left'
        }
        default_kwargs.update(kwargs)
        label = tk.Label(parent, text=fixed_text, **default_kwargs)

    return label

def create_arabic_button(parent, text, **kwargs):
    arabic_font = get_arabic_font()

    
    fixed_text = fix_arabic_text(text)

    
    default_kwargs = {
        'font': (arabic_font, 10, 'bold')
    }

    
    default_kwargs.update(kwargs)

    button = tk.Button(parent, text=fixed_text, **default_kwargs)
    return button

def create_arabic_entry(parent, **kwargs):
    arabic_font = get_arabic_font()

    
    default_kwargs = {
        'font': (arabic_font, 10),
        'justify': 'right' if platform.system() == "Linux" else 'left'
    }

    
    default_kwargs.update(kwargs)

    entry = tk.Entry(parent, **default_kwargs)
    return entry

def update_widget_text(widget, text):
    fixed_text = fix_arabic_text(text)
    try:
        if hasattr(widget, 'config'):
            widget.config(text=fixed_text)
        elif hasattr(widget, 'configure'):
            widget.configure(text=fixed_text)
    except Exception as e:
        print(f"خطأ في تحديث نص العنصر: {e}")

def create_modern_frame(parent, title=None, **kwargs):
    
    outer_frame = tk.Frame(parent, bg="#1A1A1A", **kwargs)

    
    if title:
        inner_frame = ttk.LabelFrame(outer_frame, text=title, style="TLabelFrame")
    else:
        inner_frame = ttk.Frame(outer_frame, style="Secondary.TFrame")

    inner_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    return outer_frame, inner_frame

def create_status_label(parent, text="", status_type="normal"):
    colors = {
        "normal": ("#FFFFFF", "#2D2D2D"),
        "success": ("#FFFFFF", "#107C10"),
        "warning": ("#FFFFFF", "#FF8C00"),
        "error": ("#FFFFFF", "#D13438"),
        "info": ("#FFFFFF", "#0078D4")
    }

    fg, bg = colors.get(status_type, colors["normal"])
    arabic_font = get_arabic_font()
    fixed_text = fix_arabic_text(text)

    label = tk.Label(parent,
                    text=fixed_text,
                    bg=bg,
                    fg=fg,
                    font=(arabic_font, 10),
                    padx=12,
                    pady=6,
                    anchor='e' if platform.system() == "Linux" else 'w',
                    justify='right' if platform.system() == "Linux" else 'left')

    return label

def add_hover_effect(widget, enter_color=None, leave_color=None):
    if enter_color is None:
        enter_color = "#404040"
    if leave_color is None:
        leave_color = "#2D2D2D"

    def on_enter(event):
        try:
            widget.configure(bg=enter_color)
        except:
            pass

    def on_leave(event):
        try:
            widget.configure(bg=leave_color)
        except:
            pass

    widget.bind("<Enter>", on_enter)
    widget.bind("<Leave>", on_leave)

def create_separator(parent, orientation="horizontal"):
    if orientation == "horizontal":
        separator = tk.Frame(parent, height=1, bg="#555555")
        separator.pack(fill=tk.X, padx=10, pady=5)
    else:
        separator = tk.Frame(parent, width=1, bg="#555555")
        separator.pack(fill=tk.Y, padx=5, pady=10)

    return separator

class SubtitleProcessor:
    """معالج ملفات الترجمة SRT"""

    @staticmethod
    def parse_srt_file(srt_path):
        """تحليل ملف SRT وإرجاع قائمة بالترجمات"""
        try:
            # محاولة قراءة الملف بترميزات مختلفة
            encodings = ['utf-8', 'utf-8-sig', 'cp1256', 'iso-8859-1', 'windows-1252']
            content = None

            for encoding in encodings:
                try:
                    with codecs.open(srt_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except (UnicodeDecodeError, UnicodeError):
                    continue

            if content is None:
                raise Exception("لا يمكن قراءة ملف الترجمة بأي ترميز مدعوم")

            # تنظيف المحتوى
            content = content.strip()
            if not content:
                raise Exception("ملف الترجمة فارغ")

            # تقسيم إلى كتل الترجمة
            subtitle_blocks = re.split(r'\n\s*\n', content)
            subtitles = []

            for i, block in enumerate(subtitle_blocks):
                if not block.strip():
                    continue

                lines = block.strip().split('\n')
                if len(lines) < 3:
                    continue

                try:
                    # رقم الترجمة
                    subtitle_num = int(lines[0].strip())

                    # التوقيت
                    time_line = lines[1].strip()
                    time_match = re.match(r'(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})', time_line)

                    if not time_match:
                        print(f"تحذير: تنسيق التوقيت غير صحيح في السطر {i+1}: {time_line}")
                        continue

                    start_time = time_match.group(1)
                    end_time = time_match.group(2)

                    # النص
                    text_lines = lines[2:]
                    text = '\n'.join(text_lines).strip()

                    if text:
                        subtitles.append({
                            'number': subtitle_num,
                            'start': start_time,
                            'end': end_time,
                            'text': text
                        })

                except (ValueError, IndexError) as e:
                    print(f"تحذير: خطأ في تحليل كتلة الترجمة {i+1}: {e}")
                    continue

            if not subtitles:
                raise Exception("لم يتم العثور على ترجمات صالحة في الملف")

            print(f"تم تحليل {len(subtitles)} ترجمة من الملف")
            return subtitles

        except Exception as e:
            raise Exception(f"خطأ في تحليل ملف SRT: {e}")

    @staticmethod
    def time_to_seconds(time_str):
        """تحويل تنسيق الوقت SRT إلى ثوان"""
        try:
            # تنسيق: HH:MM:SS,mmm
            time_parts = time_str.replace(',', '.').split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds = float(time_parts[2])

            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0

    @staticmethod
    def validate_srt_file(srt_path):
        """التحقق من صحة ملف SRT"""
        try:
            if not os.path.exists(srt_path):
                return False, "الملف غير موجود"

            if not srt_path.lower().endswith('.srt'):
                return False, "الملف ليس من نوع SRT"

            subtitles = SubtitleProcessor.parse_srt_file(srt_path)

            if len(subtitles) == 0:
                return False, "الملف لا يحتوي على ترجمات صالحة"

            return True, f"ملف صالح يحتوي على {len(subtitles)} ترجمة"

        except Exception as e:
            return False, f"خطأ في التحقق من الملف: {e}"

    @staticmethod
    def create_subtitle_filter(subtitle_path, font_size=24, font_color="white",
                              outline_color="black", outline_width=2, position="bottom",
                              background_enabled=False, background_color="black",
                              background_opacity=80, background_padding=10):
        """إنشاء فلتر FFmpeg للترجمة مع دعم الخلفية"""
        try:
            # تحويل مسار الملف للتوافق مع FFmpeg
            subtitle_path_escaped = subtitle_path.replace('\\', '/').replace(':', '\\:')

            # إعدادات الألوان (BGR format for FFmpeg)
            color_map = {
                'white': '&Hffffff', 'black': '&H000000', 'red': '&H0000ff',
                'green': '&H00ff00', 'blue': '&Hff0000', 'yellow': '&H00ffff',
                'gray': '&H808080', 'darkgray': '&H404040', 'orange': '&H0080ff',
                'purple': '&Hff00ff'
            }

            primary_color = color_map.get(font_color.lower(), '&Hffffff')
            outline_color_hex = color_map.get(outline_color.lower(), '&H000000')

            # إعدادات الموضع
            alignment_map = {
                'bottom': 2,  # أسفل الوسط
                'top': 8,     # أعلى الوسط
                'center': 5   # الوسط
            }
            alignment = alignment_map.get(position.lower(), 2)

            # إعدادات الخلفية
            background_style = ""
            if background_enabled:
                bg_color_hex = color_map.get(background_color.lower(), '&H000000')
                # تحويل الشفافية من نسبة مئوية إلى قيمة hex (0-255)
                bg_alpha = int((100 - background_opacity) * 255 / 100)
                bg_alpha_hex = f"&H{bg_alpha:02x}"

                background_style = (
                    f",BackColour={bg_color_hex}"
                    f",BorderStyle=4"  # Box background
                    f",Shadow=0"
                    f",MarginV={background_padding}"
                    f",MarginL={background_padding}"
                    f",MarginR={background_padding}"
                )

            # إنشاء فلتر الترجمة
            subtitle_filter = (
                f"subtitles='{subtitle_path_escaped}'"
                f":force_style='FontSize={font_size},"
                f"PrimaryColour={primary_color},"
                f"OutlineColour={outline_color_hex},"
                f"Outline={outline_width},"
                f"Alignment={alignment}"
                f"{background_style}'"
            )

            return subtitle_filter

        except Exception as e:
            raise Exception(f"خطأ في إنشاء فلتر الترجمة: {e}")

def apply_arabic_fixes_to_app(app):
    if not ARABIC_SUPPORT:
        return

    def fix_widget_recursive(widget):
        try:
            
            if hasattr(widget, 'cget'):
                try:
                    current_text = widget.cget('text')
                    if current_text and isinstance(current_text, str):
                        fixed_text = fix_arabic_text(current_text)
                        widget.configure(text=fixed_text)
                except:
                    pass

            
            try:
                arabic_font = get_arabic_font()
                current_font = widget.cget('font')
                if current_font:
                    if isinstance(current_font, tuple) and len(current_font) >= 2:
                        size = current_font[1]
                        style = current_font[2] if len(current_font) > 2 else "normal"
                        widget.configure(font=(arabic_font, size, style))
                    else:
                        widget.configure(font=(arabic_font, 10))
            except:
                pass

            
            if platform.system() == "Linux":
                try:
                    widget_class = widget.winfo_class()
                    
                    if hasattr(widget, 'master') and not str(type(widget)).startswith("<class 'tkinter.ttk"):
                        if widget_class in ['Label', 'Button']:
                            widget.configure(anchor='e', justify='right')
                        elif widget_class == 'Entry':
                            widget.configure(justify='right')
                except:
                    pass

            
            for child in widget.winfo_children():
                fix_widget_recursive(child)

        except Exception as e:
            print(f"خطأ في إصلاح العنصر: {e}")

    
    fix_widget_recursive(app)

class ArabicText:

    @staticmethod
    def set_text(widget, text):
        fixed_text = fix_arabic_text(text)
        try:
            widget.configure(text=fixed_text)
        except:
            pass

    @staticmethod
    def set_title(window, title):
        try:
            fixed_title = fix_arabic_text(title)
            window.title(fixed_title)
        except Exception as e:
            print(f"خطأ في تعيين عنوان النافذة: {e}")
            try:
                
                window.title(title)
            except:
                
                window.title("Video Editor")

    @staticmethod
    def get_text(widget):
        try:
            text = widget.cget('text')
            return fix_arabic_text(text) if text else ""
        except:
            return ""

    @staticmethod
    def messagebox_info(title, message):
        from tkinter import messagebox
        fixed_title = fix_arabic_text(title)
        fixed_message = fix_arabic_text(message)
        return messagebox.showinfo(fixed_title, fixed_message)

    @staticmethod
    def messagebox_error(title, message):
        from tkinter import messagebox
        fixed_title = fix_arabic_text(title)
        fixed_message = fix_arabic_text(message)
        return messagebox.showerror(fixed_title, fixed_message)

    @staticmethod
    def messagebox_warning(title, message):
        from tkinter import messagebox
        fixed_title = fix_arabic_text(title)
        fixed_message = fix_arabic_text(message)
        return messagebox.showwarning(fixed_title, fixed_message)

    @staticmethod
    def messagebox_question(title, message):
        from tkinter import messagebox
        fixed_title = fix_arabic_text(title)
        fixed_message = fix_arabic_text(message)
        return messagebox.askyesno(fixed_title, fixed_message)

class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 25
        y += self.widget.winfo_rooty() + 25
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)
        self.tooltip_window.wm_geometry(f"+{x}+{y}")

        
        arabic_font = get_arabic_font()
        tooltip_font = (arabic_font, 9)
        fixed_text = fix_arabic_text(self.text)

        label = tk.Label(self.tooltip_window,
                        text=fixed_text,
                        justify=tk.RIGHT if platform.system() == "Linux" else tk.LEFT,
                        background="#2D2D2D",
                        relief=tk.SOLID,
                        borderwidth=1,
                        font=tooltip_font,
                        wraplength=250,
                        fg="#FFFFFF",
                        padx=8,
                        pady=6)
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

def process_frame_batch(frame_batch, settings, new_width, new_height, original_width, original_height, other_overlays=None):
    
    memory_manager = MemoryManager()
    frame_processor = FrameProcessor(memory_manager)

    
    return frame_processor.process_frame_batch_optimized(
        frame_batch, settings, new_width, new_height,
        original_width, original_height, other_overlays
    )

def process_audio_chunk_parallel(chunk_data):
    chunk, wave_fade, is_first, is_last = chunk_data
    
    if is_first:
        chunk = chunk.fade_out(wave_fade)
    elif is_last:
        chunk = chunk.fade_in(wave_fade)
    else:
        chunk = chunk.fade_in(wave_fade).fade_out(wave_fade)
    
    return chunk

class MemoryManager:

    def __init__(self):
        self.memory_threshold = 0.85  
        self.cleanup_interval = 100   
        self.frame_cache = deque(maxlen=1000)  
        self.temp_files = weakref.WeakSet()  

    def get_memory_usage(self):
        if psutil:
            return psutil.virtual_memory().percent / 100.0
        else:
            
            return 0.5  

    def get_available_memory_gb(self):
        if psutil:
            return psutil.virtual_memory().available / (1024**3)
        else:
            
            return 8.0  

    def should_cleanup(self):
        return self.get_memory_usage() > self.memory_threshold

    def cleanup_memory(self, force=False):
        if force or self.should_cleanup():
            
            self.frame_cache.clear()
            
            gc.collect()
            
            self._cleanup_temp_files()

    def _cleanup_temp_files(self):
        for temp_file in list(self.temp_files):
            try:
                if hasattr(temp_file, 'close'):
                    temp_file.close()
            except:
                pass

    def get_optimal_batch_size(self, frame_count, frame_size_mb=25):
        cpu_count = multiprocessing.cpu_count()
        available_memory_gb = self.get_available_memory_gb()

        
        available_memory_mb = available_memory_gb * 1024
        max_frames_in_memory = int(available_memory_mb * 0.4 / frame_size_mb)  

        
        optimal_batch_size = min(
            max_frames_in_memory,
            max(cpu_count * 8, 128),  
            frame_count
        )

        return max(1, optimal_batch_size)

def optimize_memory_usage():
    gc.collect()

class ResourceMonitor:

    def __init__(self):
        self.cpu_threshold = 90.0  
        self.memory_threshold = 85.0  
        self.disk_threshold = 90.0  
        self.monitoring_interval = 5.0  
        self.last_check = time.time()

    def get_system_stats(self):
        if psutil:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3)
            }
        else:
            
            return {
                'cpu_percent': 50.0,
                'memory_percent': 60.0,
                'memory_available_gb': 8.0,
                'disk_percent': 70.0,
                'disk_free_gb': 100.0
            }

    def should_reduce_load(self):
        current_time = time.time()
        if current_time - self.last_check < self.monitoring_interval:
            return False

        self.last_check = current_time
        stats = self.get_system_stats()

        return (stats['cpu_percent'] > self.cpu_threshold or
                stats['memory_percent'] > self.memory_threshold or
                stats['disk_percent'] > self.disk_threshold)

    def get_adaptive_settings(self, base_batch_size, base_workers):
        stats = self.get_system_stats()

        
        memory_factor = max(0.3, 1.0 - (stats['memory_percent'] / 100.0))
        adaptive_batch_size = max(1, int(base_batch_size * memory_factor))

        
        cpu_factor = max(0.5, 1.0 - (stats['cpu_percent'] / 100.0))
        adaptive_workers = max(1, int(base_workers * cpu_factor))

        return {
            'batch_size': adaptive_batch_size,
            'workers': adaptive_workers,
            'stats': stats
        }

class OptimizedVideoIO:

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.read_buffer_size = 1024 * 1024 * 50  
        self.write_buffer_size = 1024 * 1024 * 100  

    def create_optimized_video_reader(self, video_path):
        cap = cv2.VideoCapture(video_path)

        
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  
        cap.set(cv2.CAP_PROP_FPS, 30)  

        return cap

    def create_optimized_video_writer(self, output_path, fps, width, height, use_hardware=True):
        
        if use_hardware and self._is_hardware_encoding_available():
            
            fourcc = cv2.VideoWriter_fourcc(*'H264')
        else:
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        return writer

    def _is_hardware_encoding_available(self):
        try:
            import subprocess
            result = subprocess.run([ffmpeg_exe_path, '-encoders'],
                                  capture_output=True, text=True, timeout=10,
                                  creationflags=SUBPROCESS_CREATION_FLAGS)

            encoders = result.stdout.lower()

            
            hardware_encoders = {
                'nvidia': ['h264_nvenc', 'hevc_nvenc', 'av1_nvenc'],
                'intel': ['h264_qsv', 'hevc_qsv', 'av1_qsv'],
                'amd': ['h264_amf', 'hevc_amf'],
                'apple': ['h264_videotoolbox', 'hevc_videotoolbox']
            }

            available_encoders = {}
            for vendor, encoder_list in hardware_encoders.items():
                available_encoders[vendor] = []
                for encoder in encoder_list:
                    if encoder in encoders:
                        available_encoders[vendor].append(encoder)

            return available_encoders
        except Exception as e:
            print(f"خطأ في فحص مُرمزات الأجهزة: {e}")
            return {}

    def get_best_hardware_encoder(self):
        available = self._is_hardware_encoding_available()

        
        priority_order = ['nvidia', 'intel', 'amd', 'apple']

        for vendor in priority_order:
            if vendor in available and available[vendor]:
                
                for preferred_codec in ['h264_nvenc', 'h264_qsv', 'h264_amf', 'h264_videotoolbox']:
                    if preferred_codec in available[vendor]:
                        return preferred_codec, vendor

                
                return available[vendor][0], vendor

        return None, None

    def analyze_video_content(self, video_path):
        try:
            import subprocess
            import json

            
            probe_command = [
                ffprobe_exe_path, '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', '-show_frames', '-select_streams', 'v:0',
                '-read_intervals', '%+#10', video_path  
            ]

            result = subprocess.run(probe_command, capture_output=True, text=True,
                                  timeout=30, creationflags=SUBPROCESS_CREATION_FLAGS)

            if result.returncode != 0:
                return self._get_default_analysis()

            data = json.loads(result.stdout)

            
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break

            if not video_stream:
                return self._get_default_analysis()

            
            width = int(video_stream.get('width', 1920))
            height = int(video_stream.get('height', 1080))
            fps = eval(video_stream.get('r_frame_rate', '30/1'))
            duration = float(data.get('format', {}).get('duration', 0))
            bitrate = int(data.get('format', {}).get('bit_rate', 0))

            
            frames = data.get('frames', [])
            motion_analysis = self._analyze_motion_complexity(frames)

            
            content_type = self._determine_content_type(width, height, fps, motion_analysis, duration)

            
            suggested_settings = self._suggest_compression_settings(
                width, height, fps, duration, bitrate, content_type, motion_analysis
            )

            return {
                'width': width,
                'height': height,
                'fps': fps,
                'duration': duration,
                'bitrate': bitrate,
                'content_type': content_type,
                'motion_complexity': motion_analysis,
                'suggested_settings': suggested_settings
            }

        except Exception as e:
            print(f"خطأ في تحليل الفيديو: {e}")
            return self._get_default_analysis()

    def _analyze_motion_complexity(self, frames):
        if not frames:
            return 'medium'

        try:
            
            frame_sizes = []
            for frame in frames[:10]:  
                if 'pkt_size' in frame:
                    frame_sizes.append(int(frame['pkt_size']))

            if not frame_sizes:
                return 'medium'

            avg_size = sum(frame_sizes) / len(frame_sizes)
            size_variance = sum((x - avg_size) ** 2 for x in frame_sizes) / len(frame_sizes)

            
            if size_variance < avg_size * 0.1:
                return 'low'  
            elif size_variance > avg_size * 0.5:
                return 'high'  
            else:
                return 'medium'  

        except:
            return 'medium'

    def _determine_content_type(self, width, height, fps, motion_complexity, duration):
        
        if fps >= 50:
            return 'sports_gaming'  
        elif width >= 3840:  
            return '4k_content'
        elif motion_complexity == 'low' and duration > 1800:  
            return 'presentation_lecture'  
        elif motion_complexity == 'high':
            return 'action_gaming'  
        elif width <= 720:
            return 'web_content'  
        else:
            return 'general_video'  

    def _suggest_compression_settings(self, width, height, fps, duration, bitrate, content_type, motion_complexity):
        suggestions = {}

        
        if content_type == 'sports_gaming':
            suggestions['preset'] = 'أعلى جودة ممكنة (ملف كبير)'
            suggestions['reason'] = 'محتوى رياضي أو ألعاب يتطلب جودة عالية'
        elif content_type == '4k_content':
            suggestions['preset'] = 'جودة عالية جداً (4K)'
            suggestions['reason'] = 'محتوى 4K يحتاج إعدادات خاصة'
        elif content_type == 'presentation_lecture':
            suggestions['preset'] = '720p (HD) - سريع'
            suggestions['reason'] = 'محتوى تعليمي يمكن ضغطه بكفاءة'
        elif content_type == 'action_gaming':
            suggestions['preset'] = '1080p (Full HD) - جودة ممتازة'
            suggestions['reason'] = 'محتوى حركة يحتاج توازن بين الجودة والحجم'
        elif content_type == 'web_content':
            suggestions['preset'] = '480p (SD) - جودة جيدة'
            suggestions['reason'] = 'محتوى ويب مناسب للمشاركة'
        else:
            suggestions['preset'] = '1080p (Full HD) - متوازن'
            suggestions['reason'] = 'إعدادات متوازنة للمحتوى العام'

        
        suggestions['hardware_encoding'] = True
        suggestions['hardware_reason'] = 'استخدام تشفير الأجهزة لسرعة أكبر'

        if duration > 3600:  
            suggestions['chunking'] = True
            suggestions['chunking_reason'] = 'تقسيم الفيديو الطويل لمعالجة أفضل'

        return suggestions

    def _get_default_analysis(self):
        return {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'duration': 0,
            'bitrate': 0,
            'content_type': 'general_video',
            'motion_complexity': 'medium',
            'suggested_settings': {
                'preset': '1080p (Full HD) - متوازن',
                'reason': 'إعدادات افتراضية آمنة',
                'hardware_encoding': True,
                'hardware_reason': 'استخدام تشفير الأجهزة إذا كان متاحاً'
            }
        }

    def get_optimized_ffmpeg_params(self, input_path, output_path, settings):
        base_params = [
            ffmpeg_exe_path,
            '-i', input_path,
            '-threads', str(multiprocessing.cpu_count()),  
        ]

        
        if settings.get('compression_enabled', False):
            preset_name = settings.get('quality_preset', '1080p (Full HD) - متوازن')
            preset_config = QUALITY_PRESETS.get(preset_name, QUALITY_PRESETS['1080p (Full HD) - Balanced'])

            
            hardware_encoder, vendor = self.get_best_hardware_encoder()
            use_hardware = settings.get('use_hardware_encoding', True) and hardware_encoder

            if use_hardware:
                
                base_params.extend(['-c:v', hardware_encoder])

                
                if 'nvenc' in hardware_encoder:
                    
                    base_params.extend([
                        '-preset', 'p4',
                        '-tune', 'hq',
                        '-rc', 'vbr',
                        '-cq', preset_config.get('crf', '23'),
                        '-b:v', '0',
                        '-profile:v', 'main'
                    ])
                elif 'qsv' in hardware_encoder:
                    
                    base_params.extend([
                        '-preset', 'medium',
                        '-global_quality', preset_config.get('crf', '23'),
                        '-look_ahead', '1',
                        '-profile:v', 'main'
                    ])
                elif 'amf' in hardware_encoder:
                    
                    base_params.extend([
                        '-quality', 'balanced',
                        '-rc', 'cqp',
                        '-qp_i', preset_config.get('crf', '23'),
                        '-qp_p', preset_config.get('crf', '23'),
                        '-profile:v', 'main'
                    ])
                elif 'videotoolbox' in hardware_encoder:
                    
                    base_params.extend([
                        '-q:v', preset_config.get('crf', '23'),
                        '-profile:v', 'main'
                    ])
            else:
                
                base_params.extend([
                    '-c:v', 'libx264',
                    '-crf', preset_config.get('crf', '23'),
                    '-preset', preset_config.get('preset', 'medium'),
                    '-tune', preset_config.get('tune', 'film'),
                    '-profile:v', preset_config.get('profile', 'main'),
                    '-level', preset_config.get('level', '4.0')
                ])

                
                if preset_config.get('additional_params'):
                    base_params.extend(preset_config['additional_params'])

                
                base_params.extend([
                    '-g', '250',  
                    '-keyint_min', '25',  
                    '-sc_threshold', '40',  
                    '-bf', '3',  
                    '-b_strategy', '2',  
                    '-refs', '3',  
                    '-coder', '1',  
                    '-me_method', 'hex',  
                    '-subq', '7',  
                    '-trellis', '1',  
                    '-aq-mode', '1',  
                    '-aq-strength', '1.0'  
                ])


            if preset_config.get('resolution'):
                base_params.extend(['-vf', f"scale=-2:{preset_config['resolution']}:flags=lanczos"])

        else:
            
            base_params.extend([
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-tune', 'film',
                '-profile:v', 'main'
            ])

        
        base_params.extend([
            '-c:a', 'aac',
            '-b:a', '192k',
            '-ar', '48000',  
            '-ac', '2',      
            '-aac_coder', 'twoloop'  
        ])

        
        base_params.extend([
            '-movflags', '+faststart+use_metadata_tags',  
            '-pix_fmt', 'yuv420p',      
            '-colorspace', 'bt709',     
            '-color_primaries', 'bt709',
            '-color_trc', 'bt709',
            '-avoid_negative_ts', 'make_zero',  
            '-fflags', '+genpts',       
            '-y',                       
            output_path
        ])

        return base_params

class StreamingVideoProcessor:

    def __init__(self, memory_manager, io_manager):
        self.memory_manager = memory_manager
        self.io_manager = io_manager
        self.chunk_duration = 300  

    def process_large_video_streaming(self, input_path, output_path, settings,
                                    status_callback, cancel_event):
        try:
            
            video_info = self._get_video_info(input_path)
            total_duration = video_info['duration']

            if total_duration <= self.chunk_duration:
                
                return self._process_normal_video(input_path, output_path, settings,
                                                status_callback, cancel_event)

            
            return self._process_video_in_chunks(input_path, output_path, settings,
                                               video_info, status_callback, cancel_event)

        except Exception as e:
            status_callback(f"خطأ في معالجة الفيديو: {e}")
            return False

    def _get_video_info(self, video_path):
        probe_command = [
            ffprobe_exe_path, '-v', 'quiet', '-print_format', 'json',
            '-show_format', '-show_streams', video_path
        ]

        result = subprocess.run(probe_command, capture_output=True, text=True,
                              creationflags=SUBPROCESS_CREATION_FLAGS)

        if result.returncode != 0:
            raise Exception(f"فشل في قراءة معلومات الفيديو: {result.stderr}")

        info = json.loads(result.stdout)
        duration = float(info['format']['duration'])

        return {'duration': duration, 'info': info}

    def _process_normal_video(self, input_path, output_path, settings,
                            status_callback, cancel_event):
        
        chunk_settings = settings.copy()
        chunk_settings.update({
            'input_path': input_path,
            'output_path': output_path,
            'chunk_index': 0,
            'total_chunks': 1
        })

        result = process_video_chunk(chunk_settings, cancel_event, status_callback)
        return result is not None

    def _process_video_in_chunks(self, input_path, output_path, settings,
                               video_info, status_callback, cancel_event):
        total_duration = video_info['duration']
        num_chunks = int(np.ceil(total_duration / self.chunk_duration))

        status_callback(f"سيتم تقسيم الفيديو إلى {num_chunks} جزء للمعالجة")

        temp_chunks = []
        temp_dir = os.path.join(base_path, "temp_videos")

        try:
            
            for i in range(num_chunks):
                if cancel_event.is_set():
                    return False

                start_time = i * self.chunk_duration
                chunk_output = os.path.join(temp_dir, f"chunk_{i}.mp4")
                temp_chunks.append(chunk_output)

                
                split_command = [
                    ffmpeg_exe_path, '-ss', str(start_time), '-i', input_path,
                    '-t', str(self.chunk_duration), '-c', 'copy', '-y', chunk_output
                ]

                result = subprocess.run(split_command, capture_output=True, text=True,
                                      creationflags=SUBPROCESS_CREATION_FLAGS)

                if result.returncode != 0:
                    raise Exception(f"فشل في تقسيم الجزء {i}: {result.stderr}")

                status_callback(f"تم تقسيم الجزء {i+1}/{num_chunks}")

            
            processed_chunks = []
            for i, chunk_path in enumerate(temp_chunks):
                if cancel_event.is_set():
                    return False

                chunk_output = os.path.join(temp_dir, f"processed_chunk_{i}.mp4")
                processed_chunks.append(chunk_output)

                chunk_settings = settings.copy()
                chunk_settings.update({
                    'input_path': chunk_path,
                    'output_path': chunk_output,
                    'chunk_index': i,
                    'total_chunks': num_chunks
                })

                result = process_video_chunk(chunk_settings, cancel_event, status_callback)
                if not result:
                    raise Exception(f"فشل في معالجة الجزء {i}")

                status_callback(f"تمت معالجة الجزء {i+1}/{num_chunks}")

            
            self._merge_processed_chunks(processed_chunks, output_path, status_callback)

            return True

        finally:
            
            for chunk_file in temp_chunks + processed_chunks:
                if os.path.exists(chunk_file):
                    try:
                        os.remove(chunk_file)
                    except:
                        pass

    def _merge_processed_chunks(self, chunk_files, output_path, status_callback):
        status_callback("بدء دمج الأجزاء المعالجة...")

        
        concat_file = os.path.join(os.path.dirname(output_path), "concat_list.txt")

        try:
            with open(concat_file, 'w', encoding='utf-8') as f:
                for chunk_file in chunk_files:
                    if os.path.exists(chunk_file):
                        f.write(f"file '{os.path.abspath(chunk_file)}'\n")

            
            merge_command = [
                ffmpeg_exe_path, '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-c', 'copy', '-y', output_path
            ]

            result = subprocess.run(merge_command, capture_output=True, text=True,
                                  creationflags=SUBPROCESS_CREATION_FLAGS)

            if result.returncode != 0:
                raise Exception(f"فشل في دمج الأجزاء: {result.stderr}")

            status_callback("تم دمج جميع الأجزاء بنجاح")

        finally:
            if os.path.exists(concat_file):
                try:
                    os.remove(concat_file)
                except:
                    pass

class AdaptiveProcessingController:

    def __init__(self):
        self.memory_manager = MemoryManager()
        self.resource_monitor = ResourceMonitor()
        self.io_manager = OptimizedVideoIO(self.memory_manager)
        self.streaming_processor = StreamingVideoProcessor(self.memory_manager, self.io_manager)

        
        self.performance_history = deque(maxlen=10)
        self.adjustment_factor = 0.1  

        
        self.performance_data_file = os.path.join(base_path, "performance_history.json")
        self.load_performance_history()

    def get_processing_strategy(self, video_path, settings):
        try:
            
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            
            duration = frame_count / fps if fps > 0 else 0
            estimated_size_gb = (frame_count * width * height * 3) / (1024**3)

            
            system_stats = self.resource_monitor.get_system_stats()

            strategy = {
                'use_chunking': False,
                'chunk_size_minutes': 5,
                'use_streaming': False,
                'parallel_processing': True,
                'batch_size': self.memory_manager.get_optimal_batch_size(frame_count),
                'max_workers': multiprocessing.cpu_count(),
                'estimated_duration_minutes': duration / 60,
                'estimated_size_gb': estimated_size_gb
            }

            
            if duration > 1800:  
                strategy['use_chunking'] = True
                strategy['chunk_size_minutes'] = min(10, max(5, duration / 20))

            if estimated_size_gb > system_stats['memory_available_gb'] * 0.5:
                strategy['use_streaming'] = True
                strategy['batch_size'] = max(1, strategy['batch_size'] // 2)

            if system_stats['cpu_percent'] > 80:
                strategy['max_workers'] = max(1, strategy['max_workers'] // 2)

            if system_stats['memory_percent'] > 80:
                strategy['batch_size'] = max(1, strategy['batch_size'] // 2)
                strategy['parallel_processing'] = False

            return strategy

        except Exception as e:
            print(f"خطأ في تحديد استراتيجية المعالجة: {e}")
            
            return {
                'use_chunking': True,
                'chunk_size_minutes': 5,
                'use_streaming': False,
                'parallel_processing': False,
                'batch_size': 32,
                'max_workers': 2,
                'estimated_duration_minutes': 0,
                'estimated_size_gb': 0
            }

    def monitor_and_adjust_processing(self, current_settings, performance_data):
        self.performance_history.append(performance_data)

        if len(self.performance_history) < 3:
            return current_settings  

        
        recent_performance = list(self.performance_history)[-3:]
        avg_cpu = sum(p['cpu_percent'] for p in recent_performance) / len(recent_performance)
        avg_memory = sum(p['memory_percent'] for p in recent_performance) / len(recent_performance)
        avg_processing_time = sum(p.get('processing_time', 0) for p in recent_performance) / len(recent_performance)

        adjusted_settings = current_settings.copy()

        
        if avg_cpu > 95:
            
            adjusted_settings['max_workers'] = max(1, int(current_settings['max_workers'] * 0.8))
            adjusted_settings['batch_size'] = max(1, int(current_settings['batch_size'] * 0.8))
        elif avg_cpu < 60:
            
            adjusted_settings['max_workers'] = min(multiprocessing.cpu_count(),
                                                 int(current_settings['max_workers'] * 1.2))
            adjusted_settings['batch_size'] = int(current_settings['batch_size'] * 1.1)

        
        if avg_memory > 90:
            
            adjusted_settings['batch_size'] = max(1, int(current_settings['batch_size'] * 0.7))
            adjusted_settings['force_cleanup'] = True
        elif avg_memory < 50:
            
            adjusted_settings['batch_size'] = int(current_settings['batch_size'] * 1.1)

        
        if avg_processing_time > 0:
            target_time = 2.0  
            if avg_processing_time > target_time * 1.5:
                
                adjusted_settings['batch_size'] = max(1, int(current_settings['batch_size'] * 0.9))
            elif avg_processing_time < target_time * 0.5:
                
                adjusted_settings['batch_size'] = int(current_settings['batch_size'] * 1.1)

        return adjusted_settings

    def estimate_processing_time(self, video_path, settings):
        try:
            strategy = self.get_processing_strategy(video_path, settings)

            
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            
            resolution_factor = (width * height) / (1920 * 1080)  
            duration_minutes = (frame_count / fps) / 60 if fps > 0 else 0

            
            base_time_per_minute = 2.0  

            
            base_time_per_minute *= resolution_factor

            
            if strategy['parallel_processing']:
                cpu_cores = multiprocessing.cpu_count()
                parallel_efficiency = min(0.8, cpu_cores * 0.15)  
                base_time_per_minute *= (1 - parallel_efficiency)

            if strategy['use_streaming']:
                base_time_per_minute *= 1.3  

            if strategy['use_chunking']:
                base_time_per_minute *= 1.1  

            
            if settings.get('compression_enabled', False):
                preset = settings.get('quality_preset', '1080p (Full HD) - متوازن')
                if 'أعلى جودة' in preset:
                    base_time_per_minute *= 3.0
                elif '1080p' in preset:
                    base_time_per_minute *= 2.0
                elif '720p' in preset:
                    base_time_per_minute *= 1.5
                elif '480p' in preset:
                    base_time_per_minute *= 1.2

            
            if settings.get('overlays'):
                overlay_count = len(settings['overlays'])
                base_time_per_minute *= (1 + overlay_count * 0.2)

            
            system_stats = self.resource_monitor.get_system_stats()
            if system_stats['memory_percent'] > 80:
                base_time_per_minute *= 1.4  
            elif system_stats['cpu_percent'] > 80:
                base_time_per_minute *= 1.3

            estimated_minutes = duration_minutes * base_time_per_minute

            
            video_info = {
                'resolution_factor': resolution_factor,
                'duration_minutes': duration_minutes
            }
            historical_factor = self.get_historical_accuracy_factor(video_info, settings)
            estimated_minutes *= historical_factor

            
            estimated_minutes *= 1.15

            return {
                'estimated_minutes': estimated_minutes,
                'estimated_hours': estimated_minutes / 60,
                'strategy': strategy,
                'factors': {
                    'resolution_factor': resolution_factor,
                    'duration_minutes': duration_minutes,
                    'base_time_per_minute': base_time_per_minute,
                    'system_load': system_stats['cpu_percent'],
                    'historical_factor': historical_factor,
                    'similar_cases': len([r for r in self.performance_history if abs(r.get('video_info', {}).get('resolution_factor', 1) - resolution_factor) < 0.5])
                }
            }

        except Exception as e:
            print(f"خطأ في تقدير وقت المعالجة: {e}")
            return {
                'estimated_minutes': 0,
                'estimated_hours': 0,
                'strategy': {},
                'factors': {}
            }

    def load_performance_history(self):
        try:
            if os.path.exists(self.performance_data_file):
                with open(self.performance_data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.performance_history.extend(data.get('history', []))
        except Exception as e:
            print(f"خطأ في تحميل بيانات الأداء: {e}")

    def save_performance_data(self, actual_time, estimated_time, video_info, settings):
        try:
            performance_record = {
                'timestamp': time.time(),
                'actual_time_minutes': actual_time / 60,
                'estimated_time_minutes': estimated_time,
                'accuracy_ratio': (actual_time / 60) / estimated_time if estimated_time > 0 else 1,
                'video_info': video_info,
                'settings': {
                    'compression_enabled': settings.get('compression_enabled', False),
                    'quality_preset': settings.get('quality_preset', ''),
                    'parallel_processing': settings.get('processing_mode', '') == 'parallel',
                    'chunking_enabled': settings.get('enable_chunking', False)
                }
            }

            self.performance_history.append(performance_record)

            
            data = {'history': list(self.performance_history)}
            with open(self.performance_data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

        except Exception as e:
            print(f"خطأ في حفظ بيانات الأداء: {e}")

    def get_historical_accuracy_factor(self, video_info, settings):
        if not self.performance_history:
            return 1.0

        try:
            
            similar_cases = []
            for record in self.performance_history:
                similarity_score = 0

                
                if abs(record['video_info'].get('resolution_factor', 1) - video_info.get('resolution_factor', 1)) < 0.5:
                    similarity_score += 1

                
                if record['settings']['compression_enabled'] == settings.get('compression_enabled', False):
                    similarity_score += 1

                if record['settings']['parallel_processing'] == (settings.get('processing_mode', '') == 'parallel'):
                    similarity_score += 1

                if similarity_score >= 2:  
                    similar_cases.append(record['accuracy_ratio'])

            if similar_cases:
                
                return sum(similar_cases) / len(similar_cases)
            else:
                
                all_ratios = [r['accuracy_ratio'] for r in self.performance_history]
                return sum(all_ratios) / len(all_ratios)

        except Exception as e:
            print(f"خطأ في حساب معامل الدقة: {e}")
            return 1.0

class CheckpointManager:

    def __init__(self, base_path):
        self.base_path = base_path
        self.checkpoints_dir = os.path.join(base_path, "checkpoints")
        os.makedirs(self.checkpoints_dir, exist_ok=True)

    def save_checkpoint(self, job_id, progress_data):
        try:
            checkpoint_file = os.path.join(self.checkpoints_dir, f"{job_id}.json")
            checkpoint_data = {
                'timestamp': time.time(),
                'progress': progress_data,
                'version': '1.0'
            }

            with open(checkpoint_file, 'w', encoding='utf-8') as f:
                json.dump(checkpoint_data, f, indent=2, ensure_ascii=False)

            return True
        except Exception as e:
            print(f"خطأ في حفظ نقطة التحكم: {e}")
            return False

    def load_checkpoint(self, job_id):
        try:
            checkpoint_file = os.path.join(self.checkpoints_dir, f"{job_id}.json")
            if not os.path.exists(checkpoint_file):
                return None

            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)

            return checkpoint_data['progress']
        except Exception as e:
            print(f"خطأ في تحميل نقطة التحكم: {e}")
            return None

    def delete_checkpoint(self, job_id):
        try:
            checkpoint_file = os.path.join(self.checkpoints_dir, f"{job_id}.json")
            if os.path.exists(checkpoint_file):
                os.remove(checkpoint_file)
            return True
        except Exception as e:
            print(f"خطأ في حذف نقطة التحكم: {e}")
            return False

    def list_checkpoints(self):
        try:
            checkpoints = []
            for file in os.listdir(self.checkpoints_dir):
                if file.endswith('.json'):
                    job_id = file[:-5]  
                    checkpoint_data = self.load_checkpoint(job_id)
                    if checkpoint_data:
                        checkpoints.append({
                            'job_id': job_id,
                            'data': checkpoint_data
                        })
            return checkpoints
        except Exception as e:
            print(f"خطأ في قراءة نقاط التحكم: {e}")
            return []

class LongVideoProcessor:

    def __init__(self, adaptive_controller):
        self.adaptive_controller = adaptive_controller
        self.checkpoint_manager = CheckpointManager(base_path)
        self.progress_callback = None
        self.cancel_event = None

    def process_long_video(self, input_path, output_path, settings,
                          progress_callback, cancel_event, resume_job_id=None):
        self.progress_callback = progress_callback
        self.cancel_event = cancel_event

        
        job_id = resume_job_id or f"job_{int(time.time())}_{os.path.basename(input_path)}"

        try:
            
            checkpoint_data = None
            if resume_job_id:
                checkpoint_data = self.checkpoint_manager.load_checkpoint(resume_job_id)
                if checkpoint_data:
                    progress_callback(f"تم العثور على نقطة تحكم، استئناف المعالجة من {checkpoint_data.get('completed_chunks', 0)} جزء")

            
            strategy = self.adaptive_controller.get_processing_strategy(input_path, settings)
            progress_callback(f"استراتيجية المعالجة: {self._format_strategy(strategy)}")

            
            time_estimate = self.adaptive_controller.estimate_processing_time(input_path, settings)
            progress_callback(f"الوقت المقدر للمعالجة: {time_estimate['estimated_hours']:.1f} ساعة")

            
            if strategy['use_streaming']:
                return self._process_with_streaming(input_path, output_path, settings,
                                                  strategy, job_id, checkpoint_data)
            else:
                return self._process_with_chunking(input_path, output_path, settings,
                                                 strategy, job_id, checkpoint_data)

        except Exception as e:
            progress_callback(f"خطأ في معالجة الفيديو الطويل: {e}")
            return False
        finally:
            
            if not cancel_event.is_set():
                self.checkpoint_manager.delete_checkpoint(job_id)

    def _format_strategy(self, strategy):
        parts = []
        if strategy['use_chunking']:
            parts.append(f"تقسيم إلى أجزاء ({strategy['chunk_size_minutes']} دقائق)")
        if strategy['use_streaming']:
            parts.append("معالجة تدفقية")
        if strategy['parallel_processing']:
            parts.append(f"معالجة متوازية ({strategy['max_workers']} عامل)")
        parts.append(f"حجم الدفعة: {strategy['batch_size']}")
        return ", ".join(parts)

    def _process_with_streaming(self, input_path, output_path, settings,
                              strategy, job_id, checkpoint_data):
        return self.adaptive_controller.streaming_processor.process_large_video_streaming(
            input_path, output_path, settings, self.progress_callback, self.cancel_event
        )

    def _process_with_chunking(self, input_path, output_path, settings,
                             strategy, job_id, checkpoint_data):
        try:
            
            start_chunk = 0
            completed_chunks = []

            if checkpoint_data:
                start_chunk = checkpoint_data.get('completed_chunks', 0)
                completed_chunks = checkpoint_data.get('chunk_files', [])

            
            video_info = self.adaptive_controller.streaming_processor._get_video_info(input_path)
            total_duration = video_info['duration']
            chunk_duration_seconds = strategy['chunk_size_minutes'] * 60
            total_chunks = int(np.ceil(total_duration / chunk_duration_seconds))

            self.progress_callback(f"سيتم تقسيم الفيديو إلى {total_chunks} جزء")

            temp_dir = os.path.join(base_path, "temp_videos")
            chunk_files = []

            
            for chunk_index in range(start_chunk, total_chunks):
                if self.cancel_event.is_set():
                    
                    self._save_progress_checkpoint(job_id, chunk_index, chunk_files, total_chunks)
                    return False

                
                chunk_result = self._process_single_chunk(
                    input_path, chunk_index, chunk_duration_seconds,
                    settings, strategy, temp_dir
                )

                if chunk_result:
                    chunk_files.append(chunk_result)

                    
                    if chunk_index % 5 == 0:  
                        self._save_progress_checkpoint(job_id, chunk_index + 1, chunk_files, total_chunks)

                    progress = ((chunk_index + 1) / total_chunks) * 90  
                    self.progress_callback(f"تمت معالجة الجزء {chunk_index + 1}/{total_chunks}", progress)
                else:
                    self.progress_callback(f"فشل في معالجة الجزء {chunk_index + 1}")
                    return False

            
            self.progress_callback("بدء دمج الأجزاء النهائية...", 90)
            merge_success = self._merge_chunks(chunk_files, output_path)

            if merge_success:
                self.progress_callback("تمت المعالجة بنجاح!", 100)
                return True
            else:
                self.progress_callback("فشل في دمج الأجزاء")
                return False

        except Exception as e:
            self.progress_callback(f"خطأ في المعالجة بالتقسيم: {e}")
            return False
        finally:
            
            self._cleanup_temp_files(chunk_files)

    def _process_single_chunk(self, input_path, chunk_index, chunk_duration,
                            settings, strategy, temp_dir):
        try:
            start_time = chunk_index * chunk_duration
            chunk_output = os.path.join(temp_dir, f"processed_chunk_{chunk_index}.mp4")

            
            chunk_settings = settings.copy()
            chunk_settings.update({
                'input_path': input_path,
                'output_path': chunk_output,
                'chunk_index': chunk_index,
                'total_chunks': 1,  
                'start_time': start_time,
                'duration': chunk_duration
            })

            
            result = process_video_chunk(chunk_settings, self.cancel_event, self.progress_callback)

            return result if result and os.path.exists(chunk_output) else None

        except Exception as e:
            self.progress_callback(f"خطأ في معالجة الجزء {chunk_index}: {e}")
            return None

    def _save_progress_checkpoint(self, job_id, completed_chunks, chunk_files, total_chunks):
        progress_data = {
            'completed_chunks': completed_chunks,
            'total_chunks': total_chunks,
            'chunk_files': [f for f in chunk_files if os.path.exists(f)],
            'timestamp': time.time()
        }

        self.checkpoint_manager.save_checkpoint(job_id, progress_data)

    def _merge_chunks(self, chunk_files, output_path):
        try:
            
            valid_chunks = [f for f in chunk_files if os.path.exists(f)]

            if not valid_chunks:
                return False

            
            concat_file = os.path.join(os.path.dirname(output_path), "concat_list.txt")

            with open(concat_file, 'w', encoding='utf-8') as f:
                for chunk_file in valid_chunks:
                    f.write(f"file '{os.path.abspath(chunk_file)}'\n")

            
            merge_command = [
                ffmpeg_exe_path, '-f', 'concat', '-safe', '0', '-i', concat_file,
                '-c', 'copy', '-y', output_path
            ]

            result = subprocess.run(merge_command, capture_output=True, text=True,
                                  creationflags=SUBPROCESS_CREATION_FLAGS)

            
            if os.path.exists(concat_file):
                os.remove(concat_file)

            return result.returncode == 0

        except Exception as e:
            self.progress_callback(f"خطأ في دمج الأجزاء: {e}")
            return False

    def _cleanup_temp_files(self, chunk_files):
        for chunk_file in chunk_files:
            if os.path.exists(chunk_file):
                try:
                    os.remove(chunk_file)
                except:
                    pass

class PerformanceTester:

    def __init__(self):
        self.test_results = []

    def run_comprehensive_test(self, video_path, settings, status_callback):
        results = {
            'timestamp': time.time(),
            'video_path': video_path,
            'settings': settings.copy(),
            'system_info': self._get_system_info(),
            'tests': {}
        }

        try:
            status_callback("بدء الاختبار الشامل للأداء...")

            
            results['tests']['video_reading'] = self._test_video_reading(video_path, status_callback)

            
            results['tests']['frame_processing'] = self._test_frame_processing(video_path, settings, status_callback)

            
            results['tests']['memory_usage'] = self._test_memory_usage(video_path, settings, status_callback)

            
            results['tests']['parallel_performance'] = self._test_parallel_performance(video_path, settings, status_callback)

            
            self.test_results.append(results)
            self._save_test_results(results)

            status_callback("اكتمل الاختبار الشامل")
            return results

        except Exception as e:
            status_callback(f"خطأ في الاختبار الشامل: {e}")
            return None

    def _get_system_info(self):
        return {
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None,
            'memory_total_gb': psutil.virtual_memory().total / (1024**3),
            'disk_total_gb': psutil.disk_usage('/').total / (1024**3),
            'platform': sys.platform
        }

    def _test_video_reading(self, video_path, status_callback):
        status_callback("اختبار قراءة الفيديو...")

        start_time = time.time()
        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        
        sample_size = min(200, frame_count)
        frames_read = 0

        for i in range(sample_size):
            ret, frame = cap.read()
            if ret:
                frames_read += 1
            else:
                break

        cap.release()
        read_time = time.time() - start_time

        return {
            'frames_read': frames_read,
            'total_frames': frame_count,
            'read_time': read_time,
            'fps_reading': frames_read / read_time if read_time > 0 else 0,
            'video_fps': fps,
            'resolution': f"{width}x{height}"
        }

    def _test_frame_processing(self, video_path, settings, status_callback):
        status_callback("اختبار معالجة الإطارات...")

        try:
            cap = cv2.VideoCapture(video_path)
            frames = []

            
            for i in range(10):
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
            cap.release()

            if not frames:
                return {'error': 'لا يمكن قراءة الإطارات'}

            
            start_time = time.time()
            memory_manager = MemoryManager()
            frame_processor = FrameProcessor(memory_manager)

            original_height, original_width = frames[0].shape[:2]
            new_width = original_width - settings.get('crop_left', 0) - settings.get('crop_right', 0)
            new_height = original_height - settings.get('crop_top', 0) - settings.get('crop_bottom', 0)

            processed_frames = frame_processor.process_frame_batch_optimized(
                frames, settings, new_width, new_height, original_width, original_height
            )

            sequential_time = time.time() - start_time

            return {
                'frames_processed': len(processed_frames),
                'sequential_time': sequential_time,
                'fps_processing': len(frames) / sequential_time if sequential_time > 0 else 0,
                'memory_usage_mb': memory_manager.get_available_memory_gb() * 1024
            }

        except Exception as e:
            return {'error': str(e)}

    def _test_memory_usage(self, video_path, settings, status_callback):
        status_callback("اختبار استخدام الذاكرة...")

        memory_before = psutil.virtual_memory()

        try:
            
            cap = cv2.VideoCapture(video_path)
            frames = []

            for i in range(50):  
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
            cap.release()

            memory_after_load = psutil.virtual_memory()

            
            memory_manager = MemoryManager()
            frame_processor = FrameProcessor(memory_manager)

            if frames:
                original_height, original_width = frames[0].shape[:2]
                new_width = original_width - settings.get('crop_left', 0) - settings.get('crop_right', 0)
                new_height = original_height - settings.get('crop_top', 0) - settings.get('crop_bottom', 0)

                processed_frames = frame_processor.process_frame_batch_optimized(
                    frames, settings, new_width, new_height, original_width, original_height
                )

            memory_after_process = psutil.virtual_memory()

            
            del frames
            if 'processed_frames' in locals():
                del processed_frames
            memory_manager.cleanup_memory(force=True)

            memory_after_cleanup = psutil.virtual_memory()

            return {
                'memory_before_mb': memory_before.used / (1024**2),
                'memory_after_load_mb': memory_after_load.used / (1024**2),
                'memory_after_process_mb': memory_after_process.used / (1024**2),
                'memory_after_cleanup_mb': memory_after_cleanup.used / (1024**2),
                'peak_usage_mb': (memory_after_process.used - memory_before.used) / (1024**2),
                'cleanup_efficiency': (memory_after_process.used - memory_after_cleanup.used) / (1024**2)
            }

        except Exception as e:
            return {'error': str(e)}

    def _test_parallel_performance(self, video_path, settings, status_callback):
        status_callback("اختبار الأداء المتوازي...")

        try:
            
            cap = cv2.VideoCapture(video_path)
            frames = []

            for i in range(20):  
                ret, frame = cap.read()
                if ret:
                    frames.append(frame)
                else:
                    break
            cap.release()

            if not frames:
                return {'error': 'لا يمكن قراءة الإطارات'}

            original_height, original_width = frames[0].shape[:2]
            new_width = original_width - settings.get('crop_left', 0) - settings.get('crop_right', 0)
            new_height = original_height - settings.get('crop_top', 0) - settings.get('crop_bottom', 0)

            
            start_time = time.time()
            memory_manager = MemoryManager()
            frame_processor = FrameProcessor(memory_manager)

            sequential_result = frame_processor.process_frame_batch_optimized(
                frames.copy(), settings, new_width, new_height, original_width, original_height
            )
            sequential_time = time.time() - start_time

            
            start_time = time.time()
            parallel_result = _process_frames_parallel_optimized(
                frames.copy(), settings, new_width, new_height,
                original_width, original_height, None,
                min(4, multiprocessing.cpu_count()), frame_processor
            )
            parallel_time = time.time() - start_time

            speedup = sequential_time / parallel_time if parallel_time > 0 else 0

            return {
                'sequential_time': sequential_time,
                'parallel_time': parallel_time,
                'speedup_factor': speedup,
                'efficiency': speedup / min(4, multiprocessing.cpu_count()),
                'frames_tested': len(frames)
            }

        except Exception as e:
            return {'error': str(e)}

    def _save_test_results(self, results):
        try:
            results_dir = os.path.join(base_path, "performance_tests")
            os.makedirs(results_dir, exist_ok=True)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"performance_test_{timestamp}.json"
            filepath = os.path.join(results_dir, filename)

            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)

        except Exception as e:
            print(f"خطأ في حفظ نتائج الاختبار: {e}")

    def generate_performance_report(self):
        if not self.test_results:
            return "لا توجد نتائج اختبار متاحة"

        latest_result = self.test_results[-1]

        report = f"""
تقرير الأداء الشامل
==================

معلومات النظام:
- عدد أنوية المعالج: {latest_result['system_info']['cpu_count']}
- إجمالي الذاكرة: {latest_result['system_info']['memory_total_gb']:.1f} GB
- المنصة: {latest_result['system_info']['platform']}

نتائج الاختبارات:
"""

        if 'frame_processing' in latest_result['tests']:
            fp = latest_result['tests']['frame_processing']
            if 'error' not in fp:
                report += f"""
معالجة الإطارات:
- الإطارات المعالجة: {fp.get('frames_processed', 0)}
- سرعة المعالجة: {fp.get('fps_processing', 0):.1f} إطار/ثانية
- وقت المعالجة: {fp.get('sequential_time', 0):.2f} ثانية
"""

        if 'parallel_performance' in latest_result['tests']:
            pp = latest_result['tests']['parallel_performance']
            if 'error' not in pp:
                report += f"""
الأداء المتوازي:
- تسريع المعالجة: {pp.get('speedup_factor', 0):.2f}x
- الكفاءة: {pp.get('efficiency', 0):.2f}
- الوقت التسلسلي: {pp.get('sequential_time', 0):.2f} ثانية
- الوقت المتوازي: {pp.get('parallel_time', 0):.2f} ثانية
"""

        return report

QUALITY_PRESETS = {
    "Highest Quality (Large File)": {
        "crf": "15",
        "preset": "slow",
        "resolution": None,
        "tune": "film",
        "profile": "high",
        "level": "4.1",
        "additional_params": ["-x264opts", "ref=16:bframes=16:b-adapt=2:direct=auto:me=umh:subme=11:analyse=all:trellis=2:psy-rd=1.0,0.15"]
    },
    "1080p (Full HD) - Balanced": {
        "crf": "22",
        "preset": "medium",
        "resolution": 1080,
        "tune": "film",
        "profile": "main",
        "level": "4.0"
    },
    "1080p (Full HD) - Excellent Quality": {
        "crf": "20",
        "preset": "slow",
        "resolution": 1080,
        "tune": "film",
        "profile": "high",
        "level": "4.0"
    },
    "720p (HD) - Fast": {
        "crf": "23",
        "preset": "fast",
        "resolution": 720,
        "tune": "film",
        "profile": "main",
        "level": "3.1"
    },
    "720p (HD) - High Quality": {
        "crf": "21",
        "preset": "medium",
        "resolution": 720,
        "tune": "film",
        "profile": "high",
        "level": "3.1"
    },
    "480p (SD) - Good Quality": {
        "crf": "24",
        "preset": "medium",
        "resolution": 480,
        "tune": "film",
        "profile": "main",
        "level": "3.0"
    },
    "360p - Fast": {
        "crf": "26",
        "preset": "fast",
        "resolution": 360,
        "tune": "film",
        "profile": "baseline",
        "level": "3.0"
    },
    "240p - Fastest": {
        "crf": "28",
        "preset": "veryfast",
        "resolution": 240,
        "tune": "fastdecode",
        "profile": "baseline",
        "level": "2.1"
    }
}

def get_optimal_batch_size(frame_count):
    memory_manager = MemoryManager()
    return memory_manager.get_optimal_batch_size(frame_count)

class FrameProcessor:

    def __init__(self, memory_manager):
        self.memory_manager = memory_manager
        self.processed_count = 0

    def process_frame_batch_optimized(self, frames, settings, new_width, new_height,
                                    original_width, original_height, overlays=None):
        processed_frames = []

        try:
            for i, frame in enumerate(frames):
                
                if self.processed_count % self.memory_manager.cleanup_interval == 0:
                    self.memory_manager.cleanup_memory()

                
                processed_frame = self._process_single_frame(
                    frame, settings, new_width, new_height,
                    original_width, original_height, overlays
                )

                if processed_frame is not None:
                    processed_frames.append(processed_frame)

                self.processed_count += 1

                
                del frame

        except Exception as e:
            print(f"خطأ في معالجة دفعة الإطارات: {e}")

        return processed_frames

    def _process_single_frame(self, frame, settings, new_width, new_height,
                            original_width, original_height, overlays):
        try:
            
            cropped = frame[
                settings['crop_top']:original_height - settings['crop_bottom'],
                settings['crop_left']:original_width - settings['crop_right']
            ]

            
            mirrored = cv2.flip(cropped, 1) if settings.get('mirror_enabled', True) else cropped

            
            final_frame = cv2.convertScaleAbs(
                mirrored,
                alpha=settings['contrast'],
                beta=(settings['brightness'] - 1) * 100
            )

            
            if overlays:
                final_frame = self._apply_overlays(final_frame, overlays, new_width, new_height)

            
            final_frame = self._apply_x_effect(final_frame, settings, new_width, new_height)

            return final_frame

        except Exception as e:
            print(f"خطأ في معالجة الإطار: {e}")
            return None

    def _apply_overlays(self, frame, overlays, new_width, new_height):
        for overlay in overlays:
            try:
                o_type = overlay.get('type')
                x, y, w, h = overlay['x'], overlay['y'], overlay['w'], overlay['h']

                if w <= 0 or h <= 0:
                    continue

                if o_type == 'logo' and 'data' in overlay:
                    frame = self._apply_logo_overlay(frame, overlay, x, y, w, h, new_width, new_height)
                elif o_type == 'blur':
                    frame = self._apply_blur_overlay(frame, x, y, w, h)
                elif o_type == 'pixelate':
                    frame = self._apply_pixelate_overlay(frame, x, y, w, h)
                elif o_type in ['rect', 'circle']:
                    frame = self._apply_shape_overlay(frame, overlay, x, y, w, h)
                elif o_type == 'subtitle':
                    frame = self._apply_subtitle_overlay(frame, overlay, x, y, w, h, new_width, new_height)

            except Exception as e:
                print(f"خطأ في تطبيق العنصر {o_type}: {e}")

        return frame

    def _apply_logo_overlay(self, frame, overlay, x, y, w, h, new_width, new_height):
        logo_data = overlay['data']
        oh, ow = logo_data.shape[:2]

        y1, x1 = max(0, y), max(0, x)
        y2, x2 = min(new_height, y + oh), min(new_width, x + ow)

        if y1 < y2 and x1 < x2:
            logo_y1, logo_x1 = y1 - y, x1 - x
            logo_y2, logo_x2 = logo_y1 + (y2 - y1), logo_x1 + (x2 - x1)

            if logo_data.shape[2] == 4:  
                alpha_s = logo_data[logo_y1:logo_y2, logo_x1:logo_x2, 3] / 255.0
                alpha_l = 1.0 - alpha_s
                for c in range(3):
                    frame[y1:y2, x1:x2, c] = (
                        alpha_s * logo_data[logo_y1:logo_y2, logo_x1:logo_x2, c] +
                        alpha_l * frame[y1:y2, x1:x2, c]
                    )
            else:
                frame[y1:y2, x1:x2] = logo_data[logo_y1:logo_y2, logo_x1:logo_x2]

        return frame

    def _apply_blur_overlay(self, frame, x, y, w, h):
        roi = frame[y:y+h, x:x+w]
        ksize = (max(1, w // 4) | 1, max(1, h // 4) | 1)
        frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, ksize, 0)
        return frame

    def _apply_pixelate_overlay(self, frame, x, y, w, h):
        pixel_size = 20
        roi = frame[y:y+h, x:x+w]

        if roi.size == 0:
            return frame

        
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        cv2.circle(mask, (w//2, h//2), w//2, 255, -1)

        
        small_roi = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)),
                              interpolation=cv2.INTER_LINEAR)
        pixelated_roi = cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)

        
        frame[y:y+h, x:x+w] = np.where(mask[..., None].astype(bool), pixelated_roi, roi)
        return frame

    def _apply_shape_overlay(self, frame, overlay, x, y, w, h):
        color_hex = overlay.get('color', '#FFFF00').lstrip('#')
        color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0))
        thickness = overlay.get('thickness', 2)

        if overlay['type'] == 'rect':
            cv2.rectangle(frame, (x, y), (x+w, y+h), color_bgr, thickness)
        else:  
            cv2.ellipse(frame, (x + w//2, y + h//2), (w//2, h//2), 0, 0, 360, color_bgr, thickness)

        return frame

    def _apply_subtitle_overlay(self, frame, overlay, x, y, w, h, new_width, new_height):
        """Apply subtitle overlay with background support"""
        try:
            from PIL import Image, ImageDraw, ImageFont
            import cv2

            # Get subtitle properties
            text = overlay.get('text', 'Subtitle')
            font_size = overlay.get('font_size', 24)
            font_color = overlay.get('font_color', 'white')
            outline_color = overlay.get('outline_color', 'black')
            outline_width = overlay.get('outline_width', 2)
            bg_enabled = overlay.get('background_enabled', False)
            bg_color = overlay.get('background_color', 'black')
            bg_opacity = overlay.get('background_opacity', 80)
            bg_padding = overlay.get('background_padding', 10)

            # Convert frame to PIL Image
            frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(frame_pil)

            # Load font
            try:
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                font = ImageFont.load_default()

            # Get text size
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Calculate text position (center in the subtitle area)
            text_x = x + (w - text_width) // 2
            text_y = y + (h - text_height) // 2

            # Color mapping
            color_map = {
                'white': (255, 255, 255), 'black': (0, 0, 0),
                'yellow': (255, 255, 0), 'red': (255, 0, 0),
                'blue': (0, 0, 255), 'green': (0, 255, 0),
                'orange': (255, 165, 0), 'purple': (128, 0, 128),
                'gray': (128, 128, 128), 'darkgray': (64, 64, 64)
            }

            # Draw background if enabled
            if bg_enabled:
                bg_rgb = color_map.get(bg_color.lower(), (0, 0, 0))
                bg_alpha = int(bg_opacity * 255 / 100)

                # Calculate background rectangle
                bg_x1 = text_x - bg_padding
                bg_y1 = text_y - bg_padding
                bg_x2 = text_x + text_width + bg_padding
                bg_y2 = text_y + text_height + bg_padding

                # Create background overlay
                bg_overlay = Image.new('RGBA', frame_pil.size, (0, 0, 0, 0))
                bg_draw = ImageDraw.Draw(bg_overlay)
                bg_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2],
                                fill=(*bg_rgb, bg_alpha))

                # Composite background
                frame_pil = Image.alpha_composite(frame_pil.convert('RGBA'), bg_overlay).convert('RGB')
                draw = ImageDraw.Draw(frame_pil)

            # Get actual colors
            actual_font_color = color_map.get(font_color.lower(), (255, 255, 255))
            actual_outline_color = color_map.get(outline_color.lower(), (0, 0, 0))

            # Draw text with outline
            if outline_width > 0:
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((text_x + dx, text_y + dy), text, font=font, fill=actual_outline_color)

            # Draw main text
            draw.text((text_x, text_y), text, font=font, fill=actual_font_color)

            # Convert back to OpenCV format
            frame_result = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
            return frame_result

        except Exception as e:
            print(f"Error applying subtitle overlay: {e}")
            return frame

    def _apply_x_effect(self, frame, settings, new_width, new_height):
        x_mask = np.zeros((new_height, new_width), dtype=np.uint8)
        cv2.line(x_mask, (0, 0), (new_width, new_height), 255, int(settings['x_thickness']))
        cv2.line(x_mask, (new_width, 0), (0, new_height), 255, int(settings['x_thickness']))
        frame[x_mask > 0] = np.clip(
            frame[x_mask > 0].astype(np.int16) + int(settings['x_lighten']),
            0, 255
        ).astype(np.uint8)
        return frame

def process_frame_in_shared_memory(args):
    shm_name, frame_idx, shape, dtype, settings, new_width, new_height, original_width, original_height, overlays_to_apply = args

    try:
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        shm_np_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

        
        memory_manager = MemoryManager()
        frame_processor = FrameProcessor(memory_manager)

        
        frame_to_process = shm_np_array[frame_idx].copy()
        processed_frame = frame_processor._process_single_frame(
            frame_to_process, settings, new_width, new_height,
            original_width, original_height, overlays_to_apply
        )

        if processed_frame is not None:
            shm_np_array[frame_idx] = processed_frame

    except Exception as e:
        print(f"خطأ في معالجة الإطار {frame_idx}: {e}")
    finally:
        if 'existing_shm' in locals():
            existing_shm.close()

    return frame_idx

def _process_frames_parallel_optimized(frames, settings, new_width, new_height,
                                      original_width, original_height, overlays,
                                      max_workers, frame_processor):
    try:
        
        chunk_size = max(1, len(frames) // max_workers)
        frame_chunks = [frames[i:i + chunk_size] for i in range(0, len(frames), chunk_size)]

        processed_frames = []

        
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="frame_proc") as executor:
            
            futures = []
            for chunk in frame_chunks:
                future = executor.submit(
                    frame_processor.process_frame_batch_optimized,
                    chunk, settings, new_width, new_height,
                    original_width, original_height, overlays
                )
                futures.append(future)

            
            for future in futures:
                try:
                    chunk_result = future.result(timeout=30)  
                    processed_frames.extend(chunk_result)
                except Exception as e:
                    print(f"خطأ في معالجة مجموعة الإطارات: {e}")

        return processed_frames

    except Exception as e:
        print(f"خطأ في المعالجة المتوازية: {e}")
        
        return frame_processor.process_frame_batch_optimized(
            frames, settings, new_width, new_height,
            original_width, original_height, overlays
        )

def process_video_chunk(chunk_settings, cancel_event, status_callback=None, status_queue=None):
    def send_status(msg, progress=None):
        if status_queue:
            status_queue.put((msg, progress))
        elif status_callback:
            status_callback(msg, progress)
        else:
            print(msg)

    temp_audio_file_for_chunk, temp_video_file_for_chunk = None, None
    try:
        input_path, output_path, chunk_index, total_chunks = chunk_settings['input_path'], chunk_settings['output_path'], chunk_settings['chunk_index'], chunk_settings['total_chunks']
        send_status(f"بدء معالجة الجزء {chunk_index + 1}/{total_chunks}: {os.path.basename(input_path)}")
        
        audio = AudioSegment.from_file(os.path.normpath(input_path), format=os.path.splitext(input_path)[1][1:], ffmpeg=ffmpeg_exe_path)
        
        audio_chunks_data = [(audio[i:i + chunk_settings['wave_chunk_duration']], chunk_settings['wave_fade'], i == 0, i + chunk_settings['wave_chunk_duration'] >= len(audio)) for i in range(0, len(audio), chunk_settings['wave_chunk_duration'])]
        processed_audio_for_chunk = sum(process_audio_chunk_parallel(d) for d in audio_chunks_data)

        temp_audio_fd, temp_audio_file_for_chunk = tempfile.mkstemp(suffix=f'_chunk{chunk_index}.aac', dir=os.path.join(base_path, "temp_videos"))
        os.close(temp_audio_fd)
        processed_audio_for_chunk.export(temp_audio_file_for_chunk, format="adts")
        
        cap = cv2.VideoCapture(os.path.normpath(input_path))
        original_width, original_height, original_fps, frame_count = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        new_width, new_height = original_width - chunk_settings['crop_left'] - chunk_settings['crop_right'], original_height - chunk_settings['crop_top'] - chunk_settings['crop_bottom']
        
        overlays_to_apply = []
        if 'overlays' in chunk_settings and chunk_settings['overlays']:
            from PIL import Image
            preview_dims = chunk_settings.get('logo_preview_dimensions')
            if preview_dims and preview_dims['w'] > 0 and preview_dims['h'] > 0:
                scale_w, scale_h = new_width / preview_dims['w'], new_height / preview_dims['h']
                for info in chunk_settings['overlays']:
                    try:
                        prep_info = info.copy()
                        prep_info.update({'x': int(info['x'] * scale_w), 'y': int(info['y'] * scale_h), 'w': int(info['w'] * scale_w), 'h': int(info['h'] * scale_h)})
                        if info['type'] == 'logo' and os.path.exists(info['path']) and prep_info['w'] > 0 and prep_info['h'] > 0:
                            img = Image.open(info['path']).convert('RGBA')
                            resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
                            prep_info['data'] = np.array(img.resize((prep_info['w'], prep_info['h']), resample))
                        elif info['type'] == 'subtitle':
                            # Keep subtitle info for processing
                            prep_info['type'] = 'subtitle'
                        overlays_to_apply.append(prep_info)
                    except Exception as e: send_status(f"Warning: Could not prepare overlay: {e}")
        
        temp_video_fd, temp_video_file_for_chunk = tempfile.mkstemp(suffix=f'_chunk{chunk_index}.mp4', dir=os.path.join(base_path, "temp_videos"))
        os.close(temp_video_fd)
        out = cv2.VideoWriter(temp_video_file_for_chunk, cv2.VideoWriter_fourcc(*'mp4v'), original_fps, (new_width, new_height))
        
        
        memory_manager = MemoryManager()
        resource_monitor = ResourceMonitor()
        frame_processor = FrameProcessor(memory_manager)

        
        base_batch_size = memory_manager.get_optimal_batch_size(frame_count)
        cpu_cores = multiprocessing.cpu_count()

        processed_count = 0
        frames_buffer = deque()  

        
        read_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="frame_reader")

        def read_frames_async(cap, batch_size, start_frame):
            frames = []
            for i in range(batch_size):
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            return frames

        try:
            while processed_count < frame_count:
                if cancel_event.is_set():
                    send_status(f"الجزء {chunk_index + 1}: تم طلب الإلغاء، إيقاف معالجة الإطارات.")
                    break

                
                adaptive_settings = resource_monitor.get_adaptive_settings(base_batch_size, cpu_cores)
                current_batch_size = adaptive_settings['batch_size']
                current_workers = adaptive_settings['workers']

                
                if not frames_buffer:
                    frames = []
                    for _ in range(current_batch_size):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frames.append(frame)
                    if not frames:
                        break
                else:
                    
                    frames = []
                    for _ in range(min(current_batch_size, len(frames_buffer))):
                        if frames_buffer:
                            frames.append(frames_buffer.popleft())

                if not frames:
                    break

                
                if chunk_settings.get('frame_parallel', False) and len(frames) > 1:
                    
                    processed_batch = _process_frames_parallel_optimized(
                        frames, chunk_settings, new_width, new_height,
                        original_width, original_height, overlays_to_apply,
                        current_workers, frame_processor
                    )
                else:
                    
                    processed_batch = frame_processor.process_frame_batch_optimized(
                        frames, chunk_settings, new_width, new_height,
                        original_width, original_height, overlays_to_apply
                    )

                
                for p_frame in processed_batch:
                    if p_frame is not None:
                        out.write(p_frame)

                processed_count += len(frames)
                progress = (processed_count / frame_count) * 100

                
                stats = adaptive_settings['stats']
                send_status(
                    f"الجزء {chunk_index + 1}: تمت معالجة {processed_count}/{frame_count} إطار "
                    f"(CPU: {stats['cpu_percent']:.1f}%, RAM: {stats['memory_percent']:.1f}%)",
                    progress=progress
                )

                
                if processed_count % (current_batch_size * 5) == 0:
                    memory_manager.cleanup_memory()

                
                del frames, processed_batch

        finally:
            read_executor.shutdown(wait=False)
        cap.release()
        out.release()
        if cancel_event.is_set():
            return None 
        
        send_status(f"الجزء {chunk_index + 1}: دمج الصوت والفيديو...")
        
        
        command = [
            ffmpeg_exe_path, '-i', os.path.normpath(temp_video_file_for_chunk), '-i', os.path.normpath(temp_audio_file_for_chunk),
            '-c:a', 'aac', '-b:a', '192k',
            '-r', str(original_fps), '-vsync', 'cfr',
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-y'
        ]

        if chunk_settings.get('compression_enabled', False):
            preset_name = chunk_settings.get('quality_preset', '1080p (Full HD) - متوازن')
            preset_config = QUALITY_PRESETS.get(preset_name, QUALITY_PRESETS['1080p (Full HD) - Balanced'])

            # إعداد فلاتر الفيديو
            video_filters = [f"setpts={1/chunk_settings['speed_factor']}*PTS"]
            if preset_config.get('resolution'):
                video_filters.append(f"scale=-2:{preset_config['resolution']}")

            # إضافة فلتر الترجمة إذا كان مفعلاً
            if chunk_settings.get('subtitle_enabled', False) and chunk_settings.get('subtitle_path'):
                try:
                    subtitle_filter = SubtitleProcessor.create_subtitle_filter(
                        chunk_settings['subtitle_path'],
                        chunk_settings.get('subtitle_font_size', 24),
                        chunk_settings.get('subtitle_font_color', 'white'),
                        chunk_settings.get('subtitle_outline_color', 'black'),
                        chunk_settings.get('subtitle_outline_width', 2),
                        chunk_settings.get('subtitle_position', 'bottom'),
                        chunk_settings.get('subtitle_background_enabled', False),
                        chunk_settings.get('subtitle_background_color', 'black'),
                        chunk_settings.get('subtitle_background_opacity', 80),
                        chunk_settings.get('subtitle_background_padding', 10)
                    )
                    video_filters.append(subtitle_filter)
                    send_status(f"Chunk {chunk_index + 1}: Subtitle filter added")
                except Exception as e:
                    send_status(f"Chunk {chunk_index + 1}: Warning - Failed to add subtitles: {e}")

            # إضافة فلاتر الفيديو والصوت
            command.extend(['-vf', ",".join(video_filters)])
            command.extend(['-filter:a', f"atempo={chunk_settings['speed_factor']}"])

            # إعدادات الضغط للفيديو
            command.extend(['-c:v', 'libx264', '-crf', preset_config.get('crf', '23'), '-preset', preset_config.get('preset', 'medium')])

            # إضافة إعدادات إضافية للضغط إذا كانت متوفرة
            if preset_config.get('tune'):
                command.extend(['-tune', preset_config.get('tune')])
            if preset_config.get('profile'):
                command.extend(['-profile:v', preset_config.get('profile')])
            if preset_config.get('level'):
                command.extend(['-level', preset_config.get('level')])

            # إضافة المعاملات الإضافية إذا كانت متوفرة
            if preset_config.get('additional_params'):
                command.extend(preset_config['additional_params'])

            # إعدادات الصوت للضغط
            command.extend(['-c:a', 'aac'])

        else:
            # إعداد فلاتر الفيديو للوضع العادي
            video_filters = [f"setpts={1/chunk_settings['speed_factor']}*PTS"]

            # إضافة فلتر الترجمة إذا كان مفعلاً
            if chunk_settings.get('subtitle_enabled', False) and chunk_settings.get('subtitle_path'):
                try:
                    subtitle_filter = SubtitleProcessor.create_subtitle_filter(
                        chunk_settings['subtitle_path'],
                        chunk_settings.get('subtitle_font_size', 24),
                        chunk_settings.get('subtitle_font_color', 'white'),
                        chunk_settings.get('subtitle_outline_color', 'black'),
                        chunk_settings.get('subtitle_outline_width', 2),
                        chunk_settings.get('subtitle_position', 'bottom'),
                        chunk_settings.get('subtitle_background_enabled', False),
                        chunk_settings.get('subtitle_background_color', 'black'),
                        chunk_settings.get('subtitle_background_opacity', 80),
                        chunk_settings.get('subtitle_background_padding', 10)
                    )
                    video_filters.append(subtitle_filter)
                    send_status(f"Chunk {chunk_index + 1}: Subtitle filter added")
                except Exception as e:
                    send_status(f"Chunk {chunk_index + 1}: Warning - Failed to add subtitles: {e}")

            # استخدام فلتر مركب
            if len(video_filters) > 1:
                video_filter_chain = ",".join(video_filters)
                command.extend(['-filter_complex', f"[0:v]{video_filter_chain}[v];[1:a]atempo={chunk_settings['speed_factor']}[a]"])
                command.extend(['-map', '[v]', '-map', '[a]'])
            else:
                command.extend(['-filter_complex', f"[0:v]setpts={1/chunk_settings['speed_factor']}*PTS[v];[1:a]atempo={chunk_settings['speed_factor']}[a]"])
                command.extend(['-map', '[v]', '-map', '[a]'])

        command.append(os.path.normpath(output_path))

        # طباعة الأمر للتشخيص (اختياري)
        if chunk_settings.get('debug_mode', False):
            send_status(f"الجزء {chunk_index + 1}: تشغيل الأمر: {' '.join(command[:10])}...")

        try:
            process = subprocess.run(command, capture_output=True, text=True, creationflags=SUBPROCESS_CREATION_FLAGS, timeout=3600)

            if process.returncode != 0:
                error_msg = f"الجزء {chunk_index + 1}: خطأ FFmpeg (رمز الخطأ: {process.returncode})"
                if process.stderr:
                    error_msg += f"\nتفاصيل الخطأ: {process.stderr[:500]}"  # أول 500 حرف من رسالة الخطأ
                if process.stdout:
                    error_msg += f"\nمعلومات إضافية: {process.stdout[:200]}"
                send_status(error_msg)
                return None

            # التحقق من وجود الملف الناتج
            if not os.path.exists(output_path) or os.path.getsize(output_path) == 0:
                send_status(f"الجزء {chunk_index + 1}: تحذير - الملف الناتج غير موجود أو فارغ")
                return None

            send_status(f"الجزء {chunk_index + 1}: تم إنتاج الملف بنجاح ({os.path.getsize(output_path)} بايت)")
            return output_path

        except subprocess.TimeoutExpired:
            send_status(f"الجزء {chunk_index + 1}: انتهت مهلة المعالجة (أكثر من ساعة)")
            return None
        except Exception as e:
            send_status(f"الجزء {chunk_index + 1}: خطأ في تشغيل FFmpeg: {e}")
            return None

    except Exception as e:
        send_status(f"الجزء {chunk_index + 1}: خطأ غير متوقع: {e}")
        return None
    finally:
        for f in [temp_audio_file_for_chunk, temp_video_file_for_chunk]:
            if f and os.path.exists(f): os.remove(f)

def process_one_frame_pickled(frame_bytes, chunk_settings, new_width, new_height, original_width, original_height, overlays_to_apply):
    frame = pickle.loads(frame_bytes)
    return process_frame_batch([frame], chunk_settings, new_width, new_height, original_width, original_height, overlays_to_apply)[0]

def process_one_frame(frame, settings, new_width, new_height, original_width, original_height, other_overlays=None):
    
    processed_frames = process_frame_batch([frame], settings, new_width, new_height, original_width, original_height, other_overlays)
    return processed_frames[0] if processed_frames else None

def run_ffmpeg_split(command):
    try:
        
        result = subprocess.run(command, check=True, capture_output=True, text=True, creationflags=SUBPROCESS_CREATION_FLAGS)
        return (True, "") 
    except subprocess.CalledProcessError as e:
        return (False, e.stderr) 

def process_video_core(settings, status_callback, cancel_event):
    original_input_path = settings['input_path']
    original_output_path = settings['output_path']
    temp_input_video_path_main_copy = None
    created_temp_files = []
    start_time = time.time()
    shm = None
    final_output_path = None

    
    adaptive_controller = AdaptiveProcessingController()
    long_video_processor = LongVideoProcessor(adaptive_controller)

    try:
        
        status_callback("تحليل الفيديو وتحديد استراتيجية المعالجة المثلى...")
        strategy = adaptive_controller.get_processing_strategy(original_input_path, settings)

        
        status_callback(f"الاستراتيجية المختارة: {long_video_processor._format_strategy(strategy)}")

        
        time_estimate = adaptive_controller.estimate_processing_time(original_input_path, settings)
        if time_estimate['estimated_hours'] > 0:
            status_callback(f"الوقت المقدر: {time_estimate['estimated_hours']:.1f} ساعة")

        
        resume_job_id = settings.get('resume_job_id')
        if resume_job_id:
            status_callback(f"محاولة استئناف المهمة: {resume_job_id}")

        
        if strategy['estimated_duration_minutes'] > 30 or strategy['estimated_size_gb'] > 2:
            
            status_callback("استخدام معالج الفيديوهات الطويلة...")
            success = long_video_processor.process_long_video(
                original_input_path, original_output_path, settings,
                status_callback, cancel_event, resume_job_id
            )

            if success:
                elapsed_time = time.time() - start_time
                return (True, original_output_path, elapsed_time)
            else:
                return (False, None, 0)

        
        status_callback("إنشاء نسخة مؤقتة آمنة من ملف الإدخال...")
        try:
            _, ext = os.path.splitext(original_input_path)
            temp_dir = os.path.join(base_path, "temp_videos")
            os.makedirs(temp_dir, exist_ok=True)
            temp_file_main = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=temp_dir)
            temp_input_video_path_main_copy = temp_file_main.name
            temp_file_main.close()
            shutil.copy(os.path.normpath(original_input_path), temp_input_video_path_main_copy)
            created_temp_files.append(temp_input_video_path_main_copy)
            file_size = os.path.getsize(temp_input_video_path_main_copy)
            status_callback(f"تم إنشاء نسخة مؤقتة في: {temp_input_video_path_main_copy} (الحجم: {file_size} بايت)")
            if file_size == 0:
                status_callback("خطأ: ملف الفيديو المؤقت فارغ (0 بايت).")
                return (False, None, 0)
        except Exception as e:
            status_callback(f"خطأ في إنشاء الملف المؤقت: {e}")
            return (False, None, 0)

        if cancel_event.is_set(): return (False, None, 0)

        if not ffmpeg_exe_path or not ffprobe_exe_path:
            error_msg = (
                f"لم يتم العثور على FFmpeg أو FFprobe.\n"
                "الرجاء تشغيل سكربت الإعداد (setup) أو التأكد من وجودها في مسار النظام (PATH)."
            )
            status_callback(error_msg)
            return (False, None, 0)

        if settings.get('enable_chunking', False) and float(settings.get('chunk_size_seconds', 0)) > 0:
            chunk_duration_seconds = float(settings['chunk_size_seconds']) * 60
            probe_command = [ffprobe_exe_path, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', temp_input_video_path_main_copy]
            try:
                result = subprocess.run(probe_command, capture_output=True, text=True, check=True, creationflags=SUBPROCESS_CREATION_FLAGS)
                video_duration = float(result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError) as e:
                status_callback(f"خطأ في الحصول على مدة الفيديو: {e}")
                return (False, None, 0)

            num_chunks = int(np.ceil(video_duration / chunk_duration_seconds))
            if num_chunks <= 0: num_chunks = 1
            status_callback(f"[تقسيم] سيتم تقسيم الفيديو إلى {num_chunks} جزء (أجزاء).", progress=1)
            chunk_input_files = [None] * num_chunks
            commands_to_run = []
            output_dir = os.path.dirname(original_output_path)
            base_output_name, output_ext = os.path.splitext(os.path.basename(original_output_path))
            temp_dir = os.path.join(base_path, "temp_videos")
            for i in range(num_chunks):
                start_time_chunk = i * chunk_duration_seconds
                chunk_output_name = os.path.join(temp_dir, f"temp_chunk_{i}{output_ext}")
                chunk_input_files[i] = chunk_output_name
                created_temp_files.append(chunk_output_name)
                split_command = [
                    ffmpeg_exe_path, '-y', '-ss', str(start_time_chunk), '-i', temp_input_video_path_main_copy,
                    '-t', str(chunk_duration_seconds), '-c:v', 'libx264', '-preset', 'veryfast', '-c:a', 'aac',
                    '-strict', '-2', chunk_output_name
                ]
                commands_to_run.append(split_command)

            status_callback(f"[تقسيم] بدء تقسيم الفيديو إلى {num_chunks} جزء بشكل متوازٍ...")
            with ThreadPoolExecutor(max_workers=min(num_chunks, multiprocessing.cpu_count())) as executor:
                results = list(executor.map(run_ffmpeg_split, commands_to_run))
            
            if cancel_event.is_set():
                status_callback("تم إلغاء العملية بعد تقسيم الفيديو.")
                return (False, None, 0)
            
            all_splits_successful = True
            for i, (success, error_msg) in enumerate(results):
                if not success:
                    all_splits_successful = False
                    status_callback(f"خطأ في تقسيم الجزء {i+1}: {error_msg}")
            
            if not all_splits_successful:
                status_callback("فشل تقسيم الفيديو إلى أجزاء. راجع سجل الحالة.")
                return (False, None, 0)
            
            status_callback("[تقسيم] اكتمل تقسيم جميع الأجزاء بنجاح.")
            processed_chunk_paths = [None] * num_chunks
            cpu_cores = multiprocessing.cpu_count()
            parallel_level = settings.get('parallel_level', "تفرع على مستوى الأجزاء (Chunks)")

            if settings.get('processing_mode', 'parallel') == 'parallel' and parallel_level == "تفرع على مستوى الأجزاء (Chunks)":
                from multiprocessing import Manager
                import queue as pyqueue
                max_concurrent_chunks = max(1, cpu_cores - 1)
                status_callback(f"بدء معالجة الأجزاء بشكل متوازٍ (حتى {max_concurrent_chunks} أجزاء في نفس الوقت)...")
                manager = Manager()
                status_queue = manager.Queue()
                stop_flag = threading.Event()
                def status_watcher():
                    while not stop_flag.is_set():
                        try:
                            msg, progress = status_queue.get(timeout=0.2)
                            status_callback(msg, progress)
                        except pyqueue.Empty:
                            continue
                watcher_thread = threading.Thread(target=status_watcher, daemon=True)
                watcher_thread.start()

                with ProcessPoolExecutor(max_workers=max_concurrent_chunks) as executor:
                    futures = {}
                    for i, chunk_file_path in enumerate(chunk_input_files):
                        chunk_specific_output_path = os.path.join(output_dir, f"{base_output_name}_part_{i+1}{output_ext}")
                        chunk_settings = settings.copy()
                        chunk_settings.update({'input_path': chunk_file_path, 'output_path': chunk_specific_output_path, 'chunk_index': i, 'total_chunks': num_chunks})
                        futures[executor.submit(process_video_chunk, chunk_settings, cancel_event, None, status_queue)] = i
                    
                    for future in as_completed(futures):
                        if cancel_event.is_set():
                            status_callback("تم طلب الإلغاء، إيقاف إرسال مهام جديدة...")
                            for f in futures: f.cancel()
                            break
                        i = futures[future]
                        try:
                            result_path = future.result()
                            if result_path:
                                processed_chunk_paths[i] = result_path
                                status_callback(f"[تفـرعي] انتهى الجزء {i+1}: {os.path.basename(result_path)}", progress=(10 + ((i + 1) / num_chunks) * 80))
                            else:
                                status_callback(f"فشلت معالجة أحد الأجزاء (متوازي).")
                        except Exception as exc:
                            if not isinstance(exc, futures.CancelledError):
                                status_callback(f"حدث خطأ أثناء معالجة جزء (متوازي): {exc}")
                stop_flag.set()
                watcher_thread.join()
                manager.shutdown()
            else:
                for i, chunk_file_path in enumerate(chunk_input_files):
                    if cancel_event.is_set():
                        status_callback("تم إلغاء العملية أثناء معالجة الأجزاء.")
                        break
                    chunk_specific_output_path = os.path.join(output_dir, f"{base_output_name}_part_{i+1}{output_ext}")
                    chunk_settings = settings.copy()
                    chunk_settings.update({'input_path': chunk_file_path, 'output_path': chunk_specific_output_path, 'chunk_index': i, 'total_chunks': num_chunks, 'frame_parallel': settings.get('processing_mode', 'parallel') == 'parallel' and parallel_level == "تفرع على مستوى الإطارات داخل الجزء (Frames in Chunk)"})
                    status_callback(f"[تسلسلي] بدء معالجة الجزء {i+1}/{num_chunks}: {os.path.basename(chunk_file_path)}")
                    try:
                        result_path = process_video_chunk(chunk_settings, cancel_event, status_callback)
                        if result_path:
                            processed_chunk_paths[i] = result_path
                            status_callback(f"[تسلسلي] انتهى الجزء {i+1}: {os.path.basename(result_path)}", progress=(10 + ((i + 1) / num_chunks) * 80))
                        else:
                            status_callback(f"فشلت معالجة الجزء {i+1} (تسلسلي).")
                    except Exception as exc:
                        status_callback(f"حدث خطأ أثناء معالجة الجزء {i+1} (تسلسلي): {exc}")

            if cancel_event.is_set():
                status_callback("تم إلغاء العملية قبل دمج الأجزاء.")
                return (False, None, 0)
            
            if any(p for p in processed_chunk_paths if p and os.path.exists(p)):
                status_callback("[دمج] بدء دمج جميع الأجزاء النهائية...")
                concat_list_path = os.path.join(temp_dir, "concat_list.txt")
                with open(concat_list_path, 'w', encoding='utf-8') as f:
                    for part_path in processed_chunk_paths:
                        if part_path and os.path.exists(part_path): f.write(f"file '{os.path.abspath(part_path)}'\n")
                
                merged_output_path = original_output_path
                merge_command = [ffmpeg_exe_path, '-f', 'concat', '-safe', '0', '-i', concat_list_path, '-c', 'copy', '-y', merged_output_path]
                process = subprocess.run(merge_command, capture_output=True, text=True, creationflags=SUBPROCESS_CREATION_FLAGS)
                
                if process.returncode != 0:
                    status_callback(f"خطأ في دمج الأجزاء: {process.stderr}")
                    return (False, None, 0)
                
                status_callback(f"[دمج] تم دمج جميع الأجزاء بنجاح في: {merged_output_path}", progress=100)
                final_output_path = merged_output_path
            else:
                if not cancel_event.is_set():
                    status_callback("لم يتم إنتاج أي أجزاء صالحة للدمج.")
                return (False, None, 0)
        else:
            chunk_settings = settings.copy()
            chunk_settings.update({'input_path': temp_input_video_path_main_copy, 'output_path': original_output_path, 'chunk_index': 0, 'total_chunks': 1})
            result_path = process_video_chunk(chunk_settings, cancel_event, status_callback)
            
            if result_path:
                status_callback(f"تمت معالجة الفيديو بالكامل بنجاح: {result_path}", progress=100)
                final_output_path = result_path
            else:
                if not cancel_event.is_set():
                    status_callback(f"فشلت معالجة الفيديو بالكامل.")
                return (False, None, 0)

        elapsed_time = time.time() - start_time

        
        try:
            if hasattr(adaptive_controller, 'save_performance_data'):
                
                time_estimate = adaptive_controller.estimate_processing_time(original_input_path, settings)
                estimated_minutes = time_estimate.get('estimated_minutes', 0)

                
                video_info = time_estimate.get('factors', {})

                
                adaptive_controller.save_performance_data(
                    elapsed_time, estimated_minutes, video_info, settings
                )
        except Exception as e:
            print(f"خطأ في حفظ بيانات الأداء: {e}")

        return (True, final_output_path, elapsed_time)

    except Exception as e:
        status_callback(f"خطأ غير متوقع في المعالجة الأساسية: {e}")
        return (False, None, 0)
    finally:
        for f in created_temp_files:
            if f and os.path.exists(f):
                try: os.remove(f)
                except Exception: pass
        if shm is not None:
            try:
                shm.close()
                shm.unlink()
            except Exception: pass


class App(tk.Tk):

    
    BG_COLOR = "#1E1E1E"           
    FRAME_COLOR = "#2D2D2D"        
    SECONDARY_FRAME_COLOR = "#3A3A3A"  
    TEXT_COLOR = "#FFFFFF"         
    SECONDARY_TEXT_COLOR = "#B0B0B0"   
    ENTRY_BG_COLOR = "#404040"     
    BUTTON_COLOR = "#0078D4"       
    BUTTON_ACTIVE_COLOR = "#106EBE"  
    SUCCESS_COLOR = "#107C10"      
    WARNING_COLOR = "#FF8C00"      
    ERROR_COLOR = "#D13438"        
    VIEW_BUTTON_BG = "#404040"
    VIEW_BUTTON_ACTIVE = "#505050"
    TOOLTIP_BG = "#2D2D2D"
    TOOLTIP_FG = "#FFFFFF"
    BORDER_COLOR = "#555555"       
    ACCENT_COLOR = "#0078D4"       

    def __init__(self):
        super().__init__()

        
        try:
            configure_arabic_support()

            
            self.arabic_font = get_arabic_font()
        except Exception as e:
            print(f"خطأ في تكوين دعم العربية: {e}")
            self.arabic_font = "Arial"


        title_text = "Advanced Video Editor"
        self.title(title_text)
        self.geometry("950x750")
        self.minsize(800, 600)
        self.configure(bg=self.BG_COLOR)

        
        try:
            
            if platform.system() == "Windows":
                self.iconbitmap(default="")  

            
            if platform.system() == "Windows":
                try:
                    import ctypes
                    ctypes.windll.dwmapi.DwmSetWindowAttribute(
                        ctypes.windll.user32.GetParent(self.winfo_id()),
                        35, ctypes.byref(ctypes.c_int(0x1E1E1E)), ctypes.sizeof(ctypes.c_int)
                    )
                except:
                    pass
        except:
            pass

        
        if platform.system() == "Linux":
            try:
                self.option_add("*Text.direction", "rtl")
                self.option_add("*Entry.justify", "right")
                self.option_add("*Label.justify", "right")
            except:
                pass

        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)

        
        self.is_maximized = False
        self.last_geometry = "950x750"

        import threading
        self.cancel_event = threading.Event()

        self.settings = {}
        
        self.default_values = {
            "crop_top": 10, "crop_bottom": 10, "crop_left": 10, "crop_right": 10,
            "brightness": 1.1, "contrast": 1.2, "speed_factor": 1.01,
            "logo_scale": 0.1, "wave_chunk_duration": 1200, "wave_fade": 200,
            "x_thickness": 50, "x_lighten": 50,
            "mirror_enabled": True,
            "processing_mode": "parallel",
            "compression_enabled": False,
            "quality_preset": "1080p (Full HD) - Balanced",
            "subtitle_enabled": False,
            "subtitle_path": "",
            "subtitle_font_size": 24,
            "subtitle_font_color": "white",
            "subtitle_outline_color": "black",
            "subtitle_outline_width": 2,
            "subtitle_position": "bottom"
        }
        self.settings_file = os.path.join(base_path, "settings.json")
        self.processed_chunk_files = [] 
        

        self.mirror_enabled_var = tk.BooleanVar(value=self.default_values['mirror_enabled'])
        self.processing_mode_var = tk.StringVar(value=self.default_values["processing_mode"])
        self.compression_enabled_var = tk.BooleanVar(value=self.default_values["compression_enabled"])
        self.quality_preset_var = tk.StringVar(value=self.default_values["quality_preset"])

        # متغيرات الترجمة
        self.subtitle_enabled_var = tk.BooleanVar(value=self.default_values["subtitle_enabled"])
        self.subtitle_path_var = tk.StringVar(value=self.default_values["subtitle_path"])
        self.subtitle_font_size_var = tk.StringVar(value=str(self.default_values["subtitle_font_size"]))
        self.subtitle_font_color_var = tk.StringVar(value=self.default_values["subtitle_font_color"])
        self.subtitle_outline_color_var = tk.StringVar(value=self.default_values["subtitle_outline_color"])
        self.subtitle_outline_width_var = tk.StringVar(value=str(self.default_values["subtitle_outline_width"]))
        self.subtitle_position_var = tk.StringVar(value=self.default_values["subtitle_position"])
        
        self.setup_styles()
        self.create_widgets()
        self.load_settings()
        self.show_view('proc')

        # Update control states
        self._update_compression_controls()
        self._update_subtitle_controls()

        self.after(100, lambda: apply_arabic_fixes_to_app(self))

        # بدء مراقبة النظام
        self.after(1000, self.start_system_monitoring)

    def on_closing(self):
        try:
            
            self.save_settings()

            
            if hasattr(self, 'cancel_event') and self.cancel_event:
                self.cancel_event.set()

            
            temp_dir = os.path.join(base_path, "temp_videos")
            if os.path.exists(temp_dir):
                try:
                    import shutil
                    shutil.rmtree(temp_dir, ignore_errors=True)
                except:
                    pass

            
            self.destroy()

        except Exception as e:
            print(f"خطأ أثناء إغلاق التطبيق: {e}")
            self.destroy()

    def setup_styles(self):
        style = ttk.Style(self)

        
        try:
            style.theme_use('clam')
        except Exception as e:
            print(f"خطأ في تعيين theme: {e}")
            try:
                style.theme_use('default')
            except:
                pass

        
        try:
            style.configure(".",
                           background=self.BG_COLOR,
                           foreground=self.TEXT_COLOR)

            style.configure("TFrame",
                           background=self.FRAME_COLOR)

            style.configure("TLabel",
                           background=self.FRAME_COLOR,
                           foreground=self.TEXT_COLOR)

            style.configure("TButton",
                           background=self.BUTTON_COLOR,
                           foreground="white")

        except Exception as e:
            print(f"خطأ في تكوين الأنماط: {e}")
            
            pass

        
        try:
            style.configure("TLabelFrame",
                           background=self.FRAME_COLOR)
            style.configure("TEntry",
                           fieldbackground=self.ENTRY_BG_COLOR,
                           foreground=self.TEXT_COLOR)
        except Exception as e:
            print(f"خطأ في الإعدادات الإضافية: {e}")

        





    def create_widgets(self):
        paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        
        left_panel_container = ttk.Frame(paned_window, style="TFrame")
        paned_window.add(left_panel_container, weight=2)

        left_canvas = tk.Canvas(left_panel_container, bg=self.FRAME_COLOR, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_scrollbar.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)
        left_panel = ttk.Frame(left_canvas, style="TFrame")
        left_panel_id = left_canvas.create_window((0, 0), window=left_panel, anchor="nw")
        
        def _on_left_panel_configure(event): left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        def _on_left_canvas_configure(event): left_canvas.itemconfig(left_panel_id, width=event.width)
        def _on_mousewheel(event): left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_panel.bind("<Configure>", _on_left_panel_configure)
        left_canvas.bind("<Configure>", _on_left_canvas_configure)
        left_canvas.bind("<MouseWheel>", _on_mousewheel)

        # Create right panel for status and progress
        right_panel_container = ttk.Frame(paned_window, style="TFrame")
        paned_window.add(right_panel_container, weight=1)

        # Status area
        status_frame = ttk.LabelFrame(right_panel_container, text="Processing Status", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Progress bar
        progress_frame = ttk.Frame(status_frame, style="TFrame")
        progress_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Label(progress_frame, text="Progress:").pack(anchor='w')
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var,
                                          maximum=100, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)

        # Status text area
        text_frame = ttk.Frame(status_frame, style="TFrame")
        text_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(text_frame, text="Processing Log:").pack(anchor='w')

        text_container = ttk.Frame(text_frame, style="TFrame")
        text_container.pack(fill=tk.BOTH, expand=True, pady=5)

        self.status_text = tk.Text(text_container, height=15, wrap=tk.WORD,
                                  bg=self.ENTRY_BG_COLOR, fg=self.TEXT_COLOR,
                                  font=('Arial', 10), state=tk.DISABLED)
        status_scrollbar = ttk.Scrollbar(text_container, orient="vertical",
                                       command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=status_scrollbar.set)

        status_scrollbar.pack(side="right", fill="y")
        self.status_text.pack(side="left", fill="both", expand=True)

        # System information
        system_frame = ttk.LabelFrame(right_panel_container, text="System Information", padding=10)
        system_frame.pack(fill=tk.X, padx=5, pady=(0, 5))

        self.cpu_info_var = tk.StringVar(value="Loading system information...")
        self.memory_info_var = tk.StringVar(value="")
        self.disk_info_var = tk.StringVar(value="")

        ttk.Label(system_frame, textvariable=self.cpu_info_var,
                 font=('Arial', 9), foreground='#888888').pack(anchor='w')
        ttk.Label(system_frame, textvariable=self.memory_info_var,
                 font=('Arial', 9), foreground='#888888').pack(anchor='w')
        ttk.Label(system_frame, textvariable=self.disk_info_var,
                 font=('Arial', 9), foreground='#888888').pack(anchor='w')


        file_frame = ttk.LabelFrame(left_panel, text="1. File Selection", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=(5, 10))
        self.input_path_var = tk.StringVar(value="No file selected")
        self.output_path_var = tk.StringVar(value="No output location selected")
        ttk.Button(file_frame, text="Select Video for Processing", command=self.select_input).grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(file_frame, textvariable=self.input_path_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(file_frame, text="Select Output Location", command=self.select_output).grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(file_frame, textvariable=self.output_path_var).grid(row=1, column=1, sticky="ew", padx=5)
        file_frame.columnconfigure(1, weight=1)


        tools_frame = ttk.Frame(left_panel, style="TFrame")
        tools_frame.pack(fill=tk.X, padx=5, pady=5)
        self.waveform_button = ttk.Button(tools_frame, text="Audio Waveform Editor", command=self.open_waveform_editor)
        self.waveform_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.preview_logo_button = ttk.Button(tools_frame, text="Element Editor (Overlays & Subtitles)", command=self.open_overlay_editor_window)
        self.preview_logo_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        
        view_buttons_frame = ttk.Frame(left_panel, style="TFrame")
        view_buttons_frame.pack(fill=tk.X, padx=5, pady=(10, 0))

        # First row of buttons
        view_buttons_row1 = ttk.Frame(view_buttons_frame, style="TFrame")
        view_buttons_row1.pack(fill=tk.X, pady=(0, 2))
        self.proc_opts_button = ttk.Button(view_buttons_row1, text="Processing Options", command=lambda: self.show_view('proc'), style="ViewToggle.TButton")
        self.proc_opts_button.pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        self.comp_opts_button = ttk.Button(view_buttons_row1, text="Video Compression", command=lambda: self.show_view('comp'), style="ViewToggle.TButton")
        self.comp_opts_button.pack(side=tk.LEFT, padx=(2, 0), fill=tk.X, expand=True)

        # Second row of buttons
        view_buttons_row2 = ttk.Frame(view_buttons_frame, style="TFrame")
        view_buttons_row2.pack(fill=tk.X)
        self.subtitle_opts_button = ttk.Button(view_buttons_row2, text="Subtitles", command=lambda: self.show_view('subtitle'), style="ViewToggle.TButton")
        self.subtitle_opts_button.pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        self.perf_opts_button = ttk.Button(view_buttons_row2, text="Performance Monitor", command=lambda: self.show_view('perf'), style="ViewToggle.TButton")
        self.perf_opts_button.pack(side=tk.LEFT, padx=(2, 0), fill=tk.X, expand=True)

        # Settings control buttons
        settings_buttons_frame = ttk.Frame(left_panel, style="TFrame")
        settings_buttons_frame.pack(fill=tk.X, padx=5, pady=(5, 0))

        self.save_settings_button = ttk.Button(settings_buttons_frame, text="Save Settings", command=self.save_settings_manual)
        self.save_settings_button.pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)

        self.reset_settings_button = ttk.Button(settings_buttons_frame, text="Reset to Default", command=self.reset_to_default)
        self.reset_settings_button.pack(side=tk.LEFT, padx=(2, 0), fill=tk.X, expand=True)


        self.options_views_container = ttk.Frame(left_panel)
        self.options_views_container.pack(fill=tk.BOTH, expand=True)

        


        self.processing_options_view = ttk.Frame(self.options_views_container, style="TFrame")
        proc_main_frame = ttk.LabelFrame(self.processing_options_view, text="Basic Processing Options", padding=10)
        proc_main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.entries = {}
        options = [
            ("Top Crop:", "crop_top"), ("Bottom Crop:", "crop_bottom"),
            ("Left Crop:", "crop_left"), ("Right Crop:", "crop_right"),
            ("Brightness:", "brightness"), ("Contrast:", "contrast"),
            ("Speed Factor:", "speed_factor"), ("Logo Scale:", "logo_scale"),
            ("Audio Wave Duration:", "wave_chunk_duration"), ("Wave Fade Duration:", "wave_fade"),
            ("X Line Thickness:", "x_thickness"), ("X Light Intensity:", "x_lighten")
        ]

        for i, (text, name) in enumerate(options):
            ttk.Label(proc_main_frame, text=text).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value=str(self.default_values.get(name, '')))
            entry = ttk.Entry(proc_main_frame, textvariable=var)
            entry.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            self.entries[name] = var


        mirror_check = ttk.Checkbutton(proc_main_frame, text="Mirror Video", variable=self.mirror_enabled_var)
        mirror_check.grid(row=len(options), column=0, columnspan=2, sticky='w', padx=5, pady=5)


        proc_mode_label = ttk.Label(proc_main_frame, text="Processing Mode:")
        proc_mode_label.grid(row=len(options) + 1, column=0, sticky='w', padx=5, pady=5)
        proc_mode_frame = ttk.Frame(proc_main_frame, style="TFrame")
        proc_mode_frame.grid(row=len(options) + 1, column=1, sticky='ew')


        ttk.Radiobutton(proc_mode_frame, text="Parallel (Recommended)", variable=self.processing_mode_var, value="parallel").pack(side=tk.LEFT, expand=True)
        ttk.Radiobutton(proc_mode_frame, text="Sequential", variable=self.processing_mode_var, value="sequential").pack(side=tk.LEFT, expand=True)


        smart_info_frame = ttk.LabelFrame(proc_main_frame, text="Automatic Smart Processing", padding=5)
        smart_info_frame.grid(row=len(options) + 2, column=0, columnspan=2, sticky='ew', padx=5, pady=10)

        info_text = """
🤖 The system automatically:
- Analyzes video size and duration
- Selects optimal processing strategy
- Splits long videos (>30 minutes) automatically
- Optimizes resource usage based on system
- Saves checkpoints for resuming when needed
"""

        info_label = ttk.Label(smart_info_frame, text=info_text, justify='left',
                              font=('Arial', 9), foreground='#666666')
        info_label.grid(row=0, column=0, sticky='w', padx=5, pady=5)

        # Create compression options view
        self.compression_options_view = ttk.Frame(self.options_views_container, style="TFrame")
        comp_main_frame = ttk.LabelFrame(self.compression_options_view, text="Video Compression Options", padding=10)
        comp_main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Enable compression
        ttk.Checkbutton(comp_main_frame, text="Enable Video Compression",
                       variable=self.compression_enabled_var,
                       command=self._update_compression_controls).pack(anchor='w', pady=5)

        # Quality options
        quality_frame = ttk.Frame(comp_main_frame, style="TFrame")
        quality_frame.pack(fill=tk.X, pady=5)

        ttk.Label(quality_frame, text="Video Quality:").pack(anchor='w')
        self.quality_preset_combo = ttk.Combobox(quality_frame, textvariable=self.quality_preset_var,
                                               values=list(QUALITY_PRESETS.keys()),
                                               state="readonly")
        self.quality_preset_combo.pack(fill=tk.X, pady=2)

        # إنشاء عرض خيارات الترجمة
        self.subtitle_options_view = ttk.Frame(self.options_views_container, style="TFrame")
        subtitle_main_frame = ttk.LabelFrame(self.subtitle_options_view, text="Subtitle Options", padding=10)
        subtitle_main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # تفعيل الترجمة
        ttk.Checkbutton(subtitle_main_frame, text="Enable Subtitles",
                       variable=self.subtitle_enabled_var,
                       command=self._update_subtitle_controls).pack(anchor='w', pady=5)

        # اختيار ملف الترجمة
        subtitle_file_frame = ttk.Frame(subtitle_main_frame, style="TFrame")
        subtitle_file_frame.pack(fill=tk.X, pady=5)

        ttk.Label(subtitle_file_frame, text="Subtitle File (.srt):").pack(anchor='w')

        subtitle_path_frame = ttk.Frame(subtitle_file_frame, style="TFrame")
        subtitle_path_frame.pack(fill=tk.X, pady=2)

        self.subtitle_path_entry = ttk.Entry(subtitle_path_frame, textvariable=self.subtitle_path_var, state="readonly")
        self.subtitle_path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.subtitle_browse_button = ttk.Button(subtitle_path_frame, text="Browse", command=self.select_subtitle_file)
        self.subtitle_browse_button.pack(side=tk.RIGHT)

        # إعدادات الخط
        font_frame = ttk.LabelFrame(subtitle_main_frame, text="Font Settings", padding=5)
        font_frame.pack(fill=tk.X, pady=5)

        # حجم الخط
        font_size_frame = ttk.Frame(font_frame, style="TFrame")
        font_size_frame.pack(fill=tk.X, pady=2)
        ttk.Label(font_size_frame, text="Font Size:").pack(side=tk.LEFT)
        self.subtitle_font_size_spinbox = ttk.Spinbox(font_size_frame, from_=12, to=72, textvariable=self.subtitle_font_size_var, width=10)
        self.subtitle_font_size_spinbox.pack(side=tk.RIGHT)

        # لون الخط
        font_color_frame = ttk.Frame(font_frame, style="TFrame")
        font_color_frame.pack(fill=tk.X, pady=2)
        ttk.Label(font_color_frame, text="Font Color:").pack(side=tk.LEFT)
        self.subtitle_font_color_combo = ttk.Combobox(font_color_frame, textvariable=self.subtitle_font_color_var,
                                                     values=["white", "black", "yellow", "red", "blue", "green"],
                                                     state="readonly", width=10)
        self.subtitle_font_color_combo.pack(side=tk.RIGHT)

        # لون الحدود
        outline_color_frame = ttk.Frame(font_frame, style="TFrame")
        outline_color_frame.pack(fill=tk.X, pady=2)
        ttk.Label(outline_color_frame, text="Outline Color:").pack(side=tk.LEFT)
        self.subtitle_outline_color_combo = ttk.Combobox(outline_color_frame, textvariable=self.subtitle_outline_color_var,
                                                        values=["black", "white", "gray", "darkgray"],
                                                        state="readonly", width=10)
        self.subtitle_outline_color_combo.pack(side=tk.RIGHT)

        # عرض الحدود
        outline_width_frame = ttk.Frame(font_frame, style="TFrame")
        outline_width_frame.pack(fill=tk.X, pady=2)
        ttk.Label(outline_width_frame, text="Outline Width:").pack(side=tk.LEFT)
        self.subtitle_outline_width_spinbox = ttk.Spinbox(outline_width_frame, from_=0, to=10, textvariable=self.subtitle_outline_width_var, width=10)
        self.subtitle_outline_width_spinbox.pack(side=tk.RIGHT)

        # موضع الترجمة
        position_frame = ttk.Frame(font_frame, style="TFrame")
        position_frame.pack(fill=tk.X, pady=2)
        ttk.Label(position_frame, text="Position:").pack(side=tk.LEFT)
        self.subtitle_position_combo = ttk.Combobox(position_frame, textvariable=self.subtitle_position_var,
                                                   values=["bottom", "top", "center"],
                                                   state="readonly", width=10)
        self.subtitle_position_combo.pack(side=tk.RIGHT)

        # معلومات الملف
        self.subtitle_info_label = ttk.Label(subtitle_main_frame, text="No subtitle file selected", foreground="gray")
        self.subtitle_info_label.pack(anchor='w', pady=5)

        # إنشاء عرض مراقبة الأداء
        self.performance_options_view = ttk.Frame(self.options_views_container, style="TFrame")
        perf_main_frame = ttk.LabelFrame(self.performance_options_view, text="مراقبة الأداء", padding=10)
        perf_main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # أزرار الاختبار
        test_buttons_frame = ttk.Frame(perf_main_frame, style="TFrame")
        test_buttons_frame.pack(fill=tk.X, pady=5)

        ttk.Button(test_buttons_frame, text="تقدير وقت المعالجة",
                  command=self.estimate_processing_time).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(test_buttons_frame, text="اختبار الأداء",
                  command=self.run_performance_test).pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

        # سجل الأداء
        ttk.Label(perf_main_frame, text="سجل اختبارات الأداء:").pack(anchor='w', pady=(10, 5))

        perf_log_frame = ttk.Frame(perf_main_frame, style="TFrame")
        perf_log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.performance_log = tk.Text(perf_log_frame, height=8, wrap=tk.WORD,
                                     bg=self.ENTRY_BG_COLOR, fg=self.TEXT_COLOR,
                                     font=('Arial', 9), state=tk.DISABLED)
        perf_scrollbar = ttk.Scrollbar(perf_log_frame, orient="vertical",
                                     command=self.performance_log.yview)
        self.performance_log.configure(yscrollcommand=perf_scrollbar.set)

        perf_scrollbar.pack(side="right", fill="y")
        self.performance_log.pack(side="left", fill="both", expand=True)

        # Add processing buttons
        processing_frame = ttk.LabelFrame(left_panel, text="3. Start Processing", padding=10)
        processing_frame.pack(fill=tk.X, padx=5, pady=(10, 5))

        buttons_frame = ttk.Frame(processing_frame, style="TFrame")
        buttons_frame.pack(fill=tk.X, pady=5)

        self.start_button = tk.Button(buttons_frame, text="Start Video Processing",
                                     command=self.start_processing,
                                     bg=self.BUTTON_COLOR, fg="white",
                                     font=('Arial', 12, 'bold'),
                                     pady=10)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        self.stop_button = tk.Button(buttons_frame, text="Stop Processing",
                                    command=self.stop_processing,
                                    bg="#D13438", fg="white",
                                    font=('Arial', 12, 'bold'),
                                    state=tk.DISABLED,
                                    pady=10)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0))

    def select_input(self):
        from tkinter import filedialog
        path = filedialog.askopenfilename(title="Select Video", filetypes=[("Video Files", "*.mp4 *.mov *.avi"), ("All Files", "*.*")])
        if path:
            self.input_path_var.set(os.path.basename(path))
            self.settings['input_path'] = path

    def select_output(self):
        from tkinter import filedialog
        path = filedialog.asksaveasfilename(title="Save As", defaultextension=".mp4", filetypes=[("MP4 File", "*.mp4")])
        if path:
            self.output_path_var.set(os.path.basename(path))
            self.settings['output_path'] = path

    def select_subtitle_file(self):
        """Select subtitle file"""
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="Select Subtitle File",
            filetypes=[("Subtitle Files", "*.srt"), ("All Files", "*.*")]
        )
        if path:
            # Validate the file
            is_valid, message = SubtitleProcessor.validate_srt_file(path)

            if is_valid:
                self.subtitle_path_var.set(path)
                # Convert message to English
                english_message = self._translate_validation_message(message)
                self.subtitle_info_label.config(text=f"✅ {english_message}", foreground="green")

                # Automatically enable subtitles when a valid file is selected
                self.subtitle_enabled_var.set(True)
                self._update_subtitle_controls()
            else:
                # Convert error message to English
                english_error = self._translate_validation_message(message)
                self.subtitle_info_label.config(text=f"❌ {english_error}", foreground="red")
                messagebox.showerror("Subtitle File Error", english_error)

    def _translate_validation_message(self, arabic_message):
        """Convert validation messages from Arabic to English"""
        translations = {
            "الملف غير موجود": "File not found",
            "الملف ليس من نوع SRT": "File is not an SRT file",
            "الملف لا يحتوي على ترجمات صالحة": "File contains no valid subtitles",
            "ملف صالح يحتوي على": "Valid file containing",
            "ترجمة": "subtitle(s)"
        }

        english_message = arabic_message
        for arabic, english in translations.items():
            english_message = english_message.replace(arabic, english)

        return english_message

    def load_settings(self):
        """Loads all settings from the JSON file."""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)

                # Load basic settings
                for name, value in loaded_settings.items():
                    if hasattr(self, 'entries') and name in self.entries:
                        self.entries[name].set(value)

                # Load specific settings if they exist
                if hasattr(self, 'compression_enabled_var'):
                    self.compression_enabled_var.set(loaded_settings.get('compression_enabled', False))
                if hasattr(self, 'quality_preset_var'):
                    self.quality_preset_var.set(loaded_settings.get('quality_preset', '1080p (Full HD) - Balanced'))

                # Load subtitle settings
                if hasattr(self, 'subtitle_enabled_var'):
                    self.subtitle_enabled_var.set(loaded_settings.get('subtitle_enabled', False))
                if hasattr(self, 'subtitle_path_var'):
                    self.subtitle_path_var.set(loaded_settings.get('subtitle_path', ''))
                if hasattr(self, 'subtitle_font_size_var'):
                    self.subtitle_font_size_var.set(str(loaded_settings.get('subtitle_font_size', 24)))
                if hasattr(self, 'subtitle_font_color_var'):
                    self.subtitle_font_color_var.set(loaded_settings.get('subtitle_font_color', 'white'))
                if hasattr(self, 'subtitle_outline_color_var'):
                    self.subtitle_outline_color_var.set(loaded_settings.get('subtitle_outline_color', 'black'))
                if hasattr(self, 'subtitle_outline_width_var'):
                    self.subtitle_outline_width_var.set(str(loaded_settings.get('subtitle_outline_width', 2)))
                if hasattr(self, 'subtitle_position_var'):
                    self.subtitle_position_var.set(loaded_settings.get('subtitle_position', 'bottom'))

                # تحميل إعدادات خلفية الترجمة
                if hasattr(self, 'subtitle_background_enabled_var'):
                    self.subtitle_background_enabled_var.set(loaded_settings.get('subtitle_background_enabled', False))
                if hasattr(self, 'subtitle_background_color_var'):
                    self.subtitle_background_color_var.set(loaded_settings.get('subtitle_background_color', 'black'))
                if hasattr(self, 'subtitle_background_opacity_var'):
                    self.subtitle_background_opacity_var.set(loaded_settings.get('subtitle_background_opacity', 80))
                if hasattr(self, 'subtitle_background_padding_var'):
                    self.subtitle_background_padding_var.set(loaded_settings.get('subtitle_background_padding', 10))

                # Update control states after loading
                self._update_compression_controls()
                self._update_subtitle_controls()

                print("Settings loaded successfully.")
        except Exception as e:
            print(f"No saved settings found or error occurred: {e}")

    def save_settings(self):
        """حفظ جميع الإعدادات في ملف JSON"""
        try:
            # جمع الإعدادات من واجهة المستخدم
            settings_to_save = {}

            # حفظ الإعدادات الأساسية من الحقول
            if hasattr(self, 'entries'):
                for name, var in self.entries.items():
                    try:
                        settings_to_save[name] = var.get()
                    except:
                        pass

            # حفظ الإعدادات الخاصة
            if hasattr(self, 'compression_enabled_var'):
                settings_to_save['compression_enabled'] = self.compression_enabled_var.get()
            if hasattr(self, 'quality_preset_var'):
                settings_to_save['quality_preset'] = self.quality_preset_var.get()
            if hasattr(self, 'mirror_enabled_var'):
                settings_to_save['mirror_enabled'] = self.mirror_enabled_var.get()
            if hasattr(self, 'processing_mode_var'):
                settings_to_save['processing_mode'] = self.processing_mode_var.get()

            # حفظ إعدادات الترجمة
            if hasattr(self, 'subtitle_enabled_var'):
                settings_to_save['subtitle_enabled'] = self.subtitle_enabled_var.get()
            if hasattr(self, 'subtitle_path_var'):
                settings_to_save['subtitle_path'] = self.subtitle_path_var.get()
            if hasattr(self, 'subtitle_font_size_var'):
                settings_to_save['subtitle_font_size'] = int(self.subtitle_font_size_var.get() or 24)
            if hasattr(self, 'subtitle_font_color_var'):
                settings_to_save['subtitle_font_color'] = self.subtitle_font_color_var.get()
            if hasattr(self, 'subtitle_outline_color_var'):
                settings_to_save['subtitle_outline_color'] = self.subtitle_outline_color_var.get()
            if hasattr(self, 'subtitle_outline_width_var'):
                settings_to_save['subtitle_outline_width'] = int(self.subtitle_outline_width_var.get() or 2)
            if hasattr(self, 'subtitle_position_var'):
                settings_to_save['subtitle_position'] = self.subtitle_position_var.get()

            # حفظ إعدادات خلفية الترجمة
            if hasattr(self, 'subtitle_background_enabled_var'):
                settings_to_save['subtitle_background_enabled'] = self.subtitle_background_enabled_var.get()
            if hasattr(self, 'subtitle_background_color_var'):
                settings_to_save['subtitle_background_color'] = self.subtitle_background_color_var.get()
            if hasattr(self, 'subtitle_background_opacity_var'):
                settings_to_save['subtitle_background_opacity'] = int(self.subtitle_background_opacity_var.get() or 80)
            if hasattr(self, 'subtitle_background_padding_var'):
                settings_to_save['subtitle_background_padding'] = int(self.subtitle_background_padding_var.get() or 10)

            # حفظ مسارات الملفات
            if hasattr(self, 'settings'):
                if 'input_path' in self.settings:
                    settings_to_save['input_path'] = self.settings['input_path']
                if 'output_path' in self.settings:
                    settings_to_save['output_path'] = self.settings['output_path']
                if 'overlays' in self.settings:
                    settings_to_save['overlays'] = self.settings['overlays']

            # كتابة الإعدادات إلى الملف
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings_to_save, f, ensure_ascii=False, indent=2)

            print("Settings saved successfully.")

        except Exception as e:
            print(f"Error saving settings: {e}")

    def save_settings_manual(self):
        """Manual save settings with user feedback"""
        try:
            self.save_settings()
            messagebox.showinfo("Settings Saved", "Settings have been saved successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")

    def reset_to_default(self):
        """Reset all settings to default values"""
        try:
            # Ask for confirmation
            if messagebox.askyesno("Reset Settings", "Are you sure you want to reset all settings to default values?"):
                # Reset all entry fields
                if hasattr(self, 'entries'):
                    for name, var in self.entries.items():
                        if name in self.default_values:
                            var.set(str(self.default_values[name]))

                # Reset boolean variables
                if hasattr(self, 'mirror_enabled_var'):
                    self.mirror_enabled_var.set(self.default_values['mirror_enabled'])
                if hasattr(self, 'compression_enabled_var'):
                    self.compression_enabled_var.set(self.default_values['compression_enabled'])
                if hasattr(self, 'subtitle_enabled_var'):
                    self.subtitle_enabled_var.set(self.default_values['subtitle_enabled'])

                # Reset string variables
                if hasattr(self, 'processing_mode_var'):
                    self.processing_mode_var.set(self.default_values['processing_mode'])
                if hasattr(self, 'quality_preset_var'):
                    self.quality_preset_var.set(self.default_values['quality_preset'])
                if hasattr(self, 'subtitle_path_var'):
                    self.subtitle_path_var.set(self.default_values['subtitle_path'])
                if hasattr(self, 'subtitle_font_size_var'):
                    self.subtitle_font_size_var.set(str(self.default_values['subtitle_font_size']))
                if hasattr(self, 'subtitle_font_color_var'):
                    self.subtitle_font_color_var.set(self.default_values['subtitle_font_color'])
                if hasattr(self, 'subtitle_outline_color_var'):
                    self.subtitle_outline_color_var.set(self.default_values['subtitle_outline_color'])
                if hasattr(self, 'subtitle_outline_width_var'):
                    self.subtitle_outline_width_var.set(str(self.default_values['subtitle_outline_width']))
                if hasattr(self, 'subtitle_position_var'):
                    self.subtitle_position_var.set(self.default_values['subtitle_position'])

                # Update control states
                self._update_compression_controls()
                self._update_subtitle_controls()

                messagebox.showinfo("Reset Complete", "All settings have been reset to default values!")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to reset settings: {e}")

    def start_processing(self):
        if not self.settings.get('input_path') or not self.settings.get('output_path'):
            messagebox.showerror("Error", "Please select input file and output location first.")
            return

        try:
            # تحديث الإعدادات من واجهة المستخدم
            for name in ["brightness", "contrast", "speed_factor", "logo_scale"]:
                if hasattr(self, 'entries') and name in self.entries:
                    self.settings[name] = float(self.entries[name].get())

            for name in ["crop_top", "crop_bottom", "crop_left", "crop_right",
                         "wave_chunk_duration", "wave_fade", "x_thickness", "x_lighten"]:
                if hasattr(self, 'entries') and name in self.entries:
                    self.settings[name] = int(float(self.entries[name].get()))

            # إعدادات إضافية
            if hasattr(self, 'mirror_enabled_var'):
                self.settings['mirror_enabled'] = self.mirror_enabled_var.get()
            if hasattr(self, 'processing_mode_var'):
                self.settings['processing_mode'] = self.processing_mode_var.get()
            if hasattr(self, 'compression_enabled_var'):
                self.settings['compression_enabled'] = self.compression_enabled_var.get()
            if hasattr(self, 'quality_preset_var'):
                self.settings['quality_preset'] = self.quality_preset_var.get()

            # إعدادات الترجمة
            if hasattr(self, 'subtitle_enabled_var'):
                self.settings['subtitle_enabled'] = self.subtitle_enabled_var.get()
            if hasattr(self, 'subtitle_path_var'):
                self.settings['subtitle_path'] = self.subtitle_path_var.get()
            if hasattr(self, 'subtitle_font_size_var'):
                self.settings['subtitle_font_size'] = int(self.subtitle_font_size_var.get() or 24)
            if hasattr(self, 'subtitle_font_color_var'):
                self.settings['subtitle_font_color'] = self.subtitle_font_color_var.get()
            if hasattr(self, 'subtitle_outline_color_var'):
                self.settings['subtitle_outline_color'] = self.subtitle_outline_color_var.get()
            if hasattr(self, 'subtitle_outline_width_var'):
                self.settings['subtitle_outline_width'] = int(self.subtitle_outline_width_var.get() or 2)
            if hasattr(self, 'subtitle_position_var'):
                self.settings['subtitle_position'] = self.subtitle_position_var.get()

            # إعدادات خلفية الترجمة
            if hasattr(self, 'subtitle_background_enabled_var'):
                self.settings['subtitle_background_enabled'] = self.subtitle_background_enabled_var.get()
            if hasattr(self, 'subtitle_background_color_var'):
                self.settings['subtitle_background_color'] = self.subtitle_background_color_var.get()
            if hasattr(self, 'subtitle_background_opacity_var'):
                self.settings['subtitle_background_opacity'] = int(self.subtitle_background_opacity_var.get() or 80)
            if hasattr(self, 'subtitle_background_padding_var'):
                self.settings['subtitle_background_padding'] = int(self.subtitle_background_padding_var.get() or 10)

        except (ValueError, KeyError) as e:
            messagebox.showerror("Input Error", f"Please enter valid numbers in the options.\nError: {e}")
            return

        # Disable processing buttons
        if hasattr(self, 'start_button'):
            self.start_button.config(state=tk.DISABLED)
        if hasattr(self, 'stop_button'):
            self.stop_button.config(state=tk.NORMAL)

        # Reset cancel event
        self.cancel_event.clear()

        # Start processing in separate thread
        import threading
        processing_thread = threading.Thread(target=self._processing_thread, daemon=True)
        processing_thread.start()

    def _processing_thread(self):
        try:
            self.update_status("Starting video processing...")

            # Use advanced processor
            adaptive_controller = AdaptiveProcessingController()
            long_processor = LongVideoProcessor(adaptive_controller)

            success = long_processor.process_long_video(
                self.settings['input_path'],
                self.settings['output_path'],
                self.settings,
                self.update_status,
                self.cancel_event
            )

            if success:
                self.update_status("Video processing completed successfully!")
                messagebox.showinfo("Success", "Video processing completed successfully!")
            else:
                self.update_status("Video processing failed.")
                if not self.cancel_event.is_set():
                    messagebox.showerror("Error", "Video processing failed. Check messages for details.")

        except Exception as e:
            self.update_status(f"Processing error: {e}")
            messagebox.showerror("Error", f"An error occurred during processing:\n{e}")
        finally:
            # Re-enable buttons
            if hasattr(self, 'start_button'):
                self.start_button.config(state=tk.NORMAL)
            if hasattr(self, 'stop_button'):
                self.stop_button.config(state=tk.DISABLED)

    def update_status(self, message, progress=None):
        """Updates the status text and progress bar."""
        if hasattr(self, 'status_text'):
            self.status_text.config(state=tk.NORMAL)
            self.status_text.insert(tk.END, message + "\n")
            self.status_text.see(tk.END)
            self.status_text.config(state=tk.DISABLED)

            if hasattr(self, 'progress_bar') and hasattr(self, 'progress_var'):
                if progress == "indeterminate":
                    self.progress_bar.config(mode='indeterminate')
                    self.progress_bar.start(10)
                elif progress is not None:
                    self.progress_bar.stop()
                    self.progress_bar.config(mode='determinate')
                    self.progress_var.set(progress)
                    if progress >= 100:
                        self.after(1000, lambda: self.progress_var.set(0))
                else:
                    self.progress_bar.stop()
                    self.progress_bar.config(mode='determinate')

            self.update_idletasks()
        else:
            print(message)

    def start_system_monitoring(self):
        self.update_system_info()

        self.after(5000, self.start_system_monitoring)

    def update_system_info(self):
        try:
            
            cpu_percent = psutil.cpu_percent(interval=0.1)
            cpu_count = psutil.cpu_count()
            cpu_freq = psutil.cpu_freq()

            cpu_info = f"CPU: {cpu_percent:.1f}% ({cpu_count} أنوية"
            if cpu_freq:
                cpu_info += f", {cpu_freq.current:.0f} MHz"
            cpu_info += ")"

            
            memory = psutil.virtual_memory()
            memory_info = f"الذاكرة: {memory.percent:.1f}% ({memory.used / (1024**3):.1f} / {memory.total / (1024**3):.1f} GB)"

            
            disk = psutil.disk_usage('/')
            disk_info = f"القرص: {disk.percent:.1f}% ({disk.free / (1024**3):.1f} GB متاح)"

            
            self.cpu_info_var.set(cpu_info)
            self.memory_info_var.set(memory_info)
            self.disk_info_var.set(disk_info)

        except Exception as e:
            self.cpu_info_var.set(f"خطأ في قراءة معلومات النظام: {e}")

    def show_view(self, view_name):
        """تبديل العروض بين خيارات المعالجة والضغط ومراقبة الأداء"""
        # إخفاء جميع العروض
        for child in self.options_views_container.winfo_children():
            child.pack_forget()

        # إعادة تعيين ألوان الأزرار
        for button in [self.proc_opts_button, self.comp_opts_button, self.subtitle_opts_button, self.perf_opts_button]:
            button.configure(style="ViewToggle.TButton")

        # عرض العرض المطلوب وتمييز الزر
        if view_name == 'proc':
            self.processing_options_view.pack(fill=tk.BOTH, expand=True)
            self.proc_opts_button.configure(style="ViewToggleActive.TButton")
        elif view_name == 'comp':
            if hasattr(self, 'compression_options_view'):
                self.compression_options_view.pack(fill=tk.BOTH, expand=True)
            self.comp_opts_button.configure(style="ViewToggleActive.TButton")
        elif view_name == 'subtitle':
            if hasattr(self, 'subtitle_options_view'):
                self.subtitle_options_view.pack(fill=tk.BOTH, expand=True)
            self.subtitle_opts_button.configure(style="ViewToggleActive.TButton")
        elif view_name == 'perf':
            if hasattr(self, 'performance_options_view'):
                self.performance_options_view.pack(fill=tk.BOTH, expand=True)
            self.perf_opts_button.configure(style="ViewToggleActive.TButton")

    def estimate_processing_time(self):
        if not self.settings.get('input_path'):
            messagebox.showwarning("تحذير", "يرجى اختيار ملف فيديو أولاً")
            return

        try:
            adaptive_controller = AdaptiveProcessingController()
            time_estimate = adaptive_controller.estimate_processing_time(
                self.settings['input_path'], self.settings
            )

            strategy = time_estimate['strategy']

            factors = time_estimate.get('factors', {})

            historical_factor = factors.get('historical_factor', 1.0)
            similar_cases = factors.get('similar_cases', 0)

            
            accuracy_level = "متوسط"
            if similar_cases >= 3:
                accuracy_level = "عالي"
            elif similar_cases >= 1:
                accuracy_level = "جيد"

            estimated_hours = time_estimate['estimated_hours']
            estimated_minutes = time_estimate['estimated_minutes']
            duration_minutes = factors.get('duration_minutes', 0)
            resolution_factor = factors.get('resolution_factor', 1)
            base_time_per_minute = factors.get('base_time_per_minute', 0)
            system_load = factors.get('system_load', 0)

            estimate_text = "تقدير وقت المعالجة (محسن مع التعلم):\n"
            estimate_text += "- الوقت المقدر: " + str(round(estimated_hours, 1)) + " ساعة (" + str(round(estimated_minutes, 0)) + " دقيقة)\n"
            estimate_text += "- مدة الفيديو: " + str(round(duration_minutes, 1)) + " دقيقة\n"
            estimate_text += "- نسبة التعقيد: " + str(round(resolution_factor, 2)) + "x\n"
            estimate_text += "- معامل الوقت الأساسي: " + str(round(base_time_per_minute, 1)) + " دقيقة/دقيقة فيديو\n"
            estimate_text += "- حمولة النظام: " + str(round(system_load, 1)) + "%\n\n"

            estimate_text += "التحسين التاريخي:\n"
            estimate_text += "- معامل التصحيح: " + str(round(historical_factor, 2)) + "x\n"
            estimate_text += "- حالات مشابهة: " + str(similar_cases) + "\n"
            estimate_text += "- دقة التقدير: " + accuracy_level + "\n\n"

            estimate_text += "الاستراتيجية:\n"
            chunking_text = "نعم" if strategy.get('use_chunking') else "لا"
            processing_text = "متوازية" if strategy.get('parallel_processing') else "تسلسلية"
            estimate_text += "- التقسيم: " + chunking_text + "\n"
            estimate_text += "- المعالجة: " + processing_text + "\n"
            estimate_text += "- حجم الدفعة: " + str(strategy.get('batch_size', 0)) + "\n"
            estimate_text += "- عدد العمال: " + str(strategy.get('max_workers', 0)) + "\n\n"

            estimate_text += "ملاحظة: التقدير محسن بناءً على الأداء السابق ويتضمن هامش أمان 15%"

            messagebox.showinfo("تقدير وقت المعالجة", estimate_text)

        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ في تقدير الوقت: {str(e)}")

    def show_estimation_stats(self):
        try:
            adaptive_controller = AdaptiveProcessingController()

            if not adaptive_controller.performance_history:
                messagebox.showinfo("إحصائيات التقدير", "لا توجد بيانات أداء سابقة متاحة")
                return

            
            history = list(adaptive_controller.performance_history)
            accuracy_ratios = [r['accuracy_ratio'] for r in history]

            avg_accuracy = sum(accuracy_ratios) / len(accuracy_ratios)
            min_accuracy = min(accuracy_ratios)
            max_accuracy = max(accuracy_ratios)

            
            accurate_predictions = sum(1 for ratio in accuracy_ratios if 0.5 <= ratio <= 1.5)
            accuracy_percentage = (accurate_predictions / len(accuracy_ratios)) * 100

            
            total_jobs = len(history)
            recent_jobs = [r for r in history if time.time() - r['timestamp'] < 30 * 24 * 3600]  

            optimistic_count = sum(1 for r in accuracy_ratios if r < 1.0)
            pessimistic_count = sum(1 for r in accuracy_ratios if r > 1.0)

            stats_text = "إحصائيات دقة تقدير الوقت:\n\n"
            stats_text += "الإحصائيات العامة:\n"
            stats_text += "- إجمالي المهام: " + str(total_jobs) + "\n"
            stats_text += "- المهام الحديثة (30 يوم): " + str(len(recent_jobs)) + "\n"
            stats_text += "- متوسط نسبة الدقة: " + str(round(avg_accuracy, 2)) + "x\n"
            stats_text += "- أفضل تقدير: " + str(round(min_accuracy, 2)) + "x (أسرع من المتوقع)\n"
            stats_text += "- أسوأ تقدير: " + str(round(max_accuracy, 2)) + "x (أبطأ من المتوقع)\n\n"

            stats_text += "دقة التقديرات:\n"
            stats_text += "- التقديرات الدقيقة (±50%): " + str(round(accuracy_percentage, 1)) + "%\n"
            stats_text += "- التقديرات المتفائلة (<1.0x): " + str(optimistic_count) + "/" + str(total_jobs) + "\n"
            stats_text += "- التقديرات المتشائمة (>1.0x): " + str(pessimistic_count) + "/" + str(total_jobs) + "\n\n"

            stats_text += "نصائح لتحسين الدقة:"

            messagebox.showinfo("إحصائيات التقدير", stats_text)

        except Exception as e:
            messagebox.showerror("خطأ", f"حدث خطأ في عرض الإحصائيات: {str(e)}")

    def show_stats_window(self, title, content):
        stats_window = tk.Toplevel(self)
        stats_window.title(title)
        stats_window.geometry("500x400")
        stats_window.configure(bg=self.BG_COLOR)

        
        content_frame = ttk.Frame(stats_window, style="TFrame")
        content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        
        stats_text = tk.Text(content_frame, wrap=tk.WORD, state=tk.DISABLED,
                           background=self.ENTRY_BG_COLOR, foreground=self.TEXT_COLOR,
                           relief="solid", borderwidth=1, padx=10, pady=10)

        
        scrollbar = ttk.Scrollbar(content_frame, orient="vertical", command=stats_text.yview)
        stats_text['yscrollcommand'] = scrollbar.set

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        stats_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        
        stats_text.config(state=tk.NORMAL)
        stats_text.insert(tk.END, content)
        stats_text.config(state=tk.DISABLED)

        
        ttk.Button(stats_window, text="إغلاق", command=stats_window.destroy).pack(pady=10)

    def run_performance_test(self):
        if not self.settings.get('input_path'):
            messagebox.showwarning("تحذير", "يرجى اختيار ملف فيديو أولاً")
            return

        
        threading.Thread(target=self._performance_test_thread, daemon=True).start()

    def _performance_test_thread(self):
        try:
            self.after(0, lambda: self.log_performance("بدء اختبار الأداء..."))

            
            start_time = time.time()
            cap = cv2.VideoCapture(self.settings['input_path'])
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            
            sample_frames = min(100, frame_count)
            frames_read = 0

            for i in range(sample_frames):
                ret, frame = cap.read()
                if ret:
                    frames_read += 1
                else:
                    break

            cap.release()
            read_time = time.time() - start_time

            
            if frames_read > 0:
                fps_read = frames_read / read_time if read_time > 0 else 0

                self.after(0, lambda: self.log_performance(
                    f"اختبار القراءة: {frames_read} إطار في {read_time:.2f} ثانية ({fps_read:.1f} إطار/ثانية)"
                ))

                
                memory_before = psutil.virtual_memory().percent

                
                test_frames = []
                cap = cv2.VideoCapture(self.settings['input_path'])
                for i in range(min(10, frames_read)):
                    ret, frame = cap.read()
                    if ret:
                        test_frames.append(frame)
                cap.release()

                memory_after = psutil.virtual_memory().percent
                memory_usage = memory_after - memory_before

                self.after(0, lambda frames_count=len(test_frames), usage=memory_usage: self.log_performance(
                    f"اختبار الذاكرة: استخدام إضافي {usage:.1f}% لـ {frames_count} إطار"
                ))

                
                del test_frames

                self.after(0, lambda: self.log_performance("اكتمل اختبار الأداء"))
            else:
                self.after(0, lambda: self.log_performance("فشل في قراءة الإطارات"))

        except Exception as e:
            self.after(0, lambda: self.log_performance(f"خطأ في اختبار الأداء: {e}"))

    def run_comprehensive_test(self):
        if not self.settings.get('input_path'):
            messagebox.showwarning("تحذير", "يرجى اختيار ملف فيديو أولاً")
            return

        
        threading.Thread(target=self._comprehensive_test_thread, daemon=True).start()

    def _comprehensive_test_thread(self):
        try:
            self.after(0, lambda: self.log_performance("بدء الاختبار الشامل للأداء..."))

            
            performance_tester = PerformanceTester()

            
            results = performance_tester.run_comprehensive_test(
                self.settings['input_path'],
                self.settings,
                lambda msg: self.after(0, lambda m=msg: self.log_performance(m))
            )

            if results:
                
                report = performance_tester.generate_performance_report()

                
                self.after(0, lambda: self._show_performance_report(report))
                self.after(0, lambda: self.log_performance("اكتمل الاختبار الشامل"))
            else:
                self.after(0, lambda: self.log_performance("فشل الاختبار الشامل"))

        except Exception as e:
            self.after(0, lambda: self.log_performance(f"خطأ في الاختبار الشامل: {e}"))

    def _show_performance_report(self, report):
        report_window = tk.Toplevel(self)
        report_window.title("تقرير الأداء الشامل")
        report_window.geometry("600x500")
        report_window.configure(bg=self.BG_COLOR)

        
        report_frame = ttk.Frame(report_window, style="TFrame")
        report_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        
        report_text = tk.Text(report_frame, wrap=tk.WORD, state=tk.DISABLED,
                             background=self.ENTRY_BG_COLOR, foreground=self.TEXT_COLOR,
                             relief="solid", borderwidth=1, padx=10, pady=10)

        
        scrollbar = ttk.Scrollbar(report_frame, orient="vertical", command=report_text.yview)
        report_text['yscrollcommand'] = scrollbar.set

        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        report_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        
        report_text.config(state=tk.NORMAL)
        report_text.insert(tk.END, report)
        report_text.config(state=tk.DISABLED)

        
        buttons_frame = ttk.Frame(report_window, style="TFrame")
        buttons_frame.pack(fill=tk.X, padx=10, pady=(0, 10))

        ttk.Button(buttons_frame, text="حفظ التقرير",
                  command=lambda: self._save_report(report)).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(buttons_frame, text="إغلاق",
                  command=report_window.destroy).pack(side=tk.RIGHT)

    def _save_report(self, report):
        try:
            filename = filedialog.asksaveasfilename(
                title="حفظ تقرير الأداء",
                defaultextension=".txt",
                filetypes=[("ملفات نصية", "*.txt"), ("جميع الملفات", "*.*")]
            )

            if filename:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(report)
                messagebox.showinfo("نجح الحفظ", f"تم حفظ التقرير في:\n{filename}")

        except Exception as e:
            messagebox.showerror("خطأ في الحفظ", f"فشل في حفظ التقرير:\n{e}")

    def log_performance(self, message):
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"

        self.performance_log.config(state=tk.NORMAL)
        self.performance_log.insert(tk.END, log_message)
        self.performance_log.see(tk.END)
        self.performance_log.config(state=tk.DISABLED)

    def open_waveform_editor(self):
        if not MATPLOTLIB_INSTALLED:
            is_in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            if not is_in_venv:
                error_msg = "البرنامج لا يعمل في البيئة الصحيحة.\n\nالرجاء إغلاقه وتشغيله عبر ملف 'run_app.bat' لضمان تفعيل البيئة الافتراضية."
                messagebox.showerror("بيئة غير صحيحة", error_msg)
            else:
                error_msg = "ميزة تعديل الموجة الصوتية تتطلب مكتبة 'matplotlib'.\n\nيرجى تثبيتها أولاً في بيئتك الحالية باستخدام الأمر:\nvenv\\Scripts\\pip.exe install matplotlib"
                messagebox.showerror("مكتبة مفقودة", error_msg)
            return

        if not self.settings.get('input_path'):
            messagebox.showerror("ملف غير محدد", "الرجاء اختيار ملف فيديو للمعالجة أولاً.")
            return
        
        
        try:
            chunk_duration = int(float(self.entries["wave_chunk_duration"].get()))
            fade_duration = int(float(self.entries["wave_fade"].get()))
        except ValueError:
            messagebox.showerror("قيمة غير صالحة", "الرجاء التأكد من أن قيم مدة الموجة والتلاشي هي أرقام صالحة.")
            return

        editor = WaveformEditorWindow(self, 
                                      self.settings['input_path'],
                                      chunk_duration,
                                      fade_duration)
        editor.grab_set() 



    def open_overlay_editor_window(self):
        input_path = self.settings.get('input_path')
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("خطأ", "يرجى اختيار ملف فيديو صالح أولاً.")
            return
        from PIL import Image, ImageTk
        try:
            cap = cv2.VideoCapture(input_path)
            ret, frame = cap.read()
            if not ret: raise Exception("لا يمكن قراءة الإطار الأول")
            h, w = frame.shape[:2]
        finally:
            cap.release()
        editor = OverlayEditorWindow(self, (w, h), frame, self.settings.get('overlays', []), self.settings.get('logo_preview_dimensions'))
        self.wait_window(editor)
        if editor.saved:
            self.settings['overlays'] = editor.get_overlays()
            self.settings['logo_preview_dimensions'] = editor.get_preview_dimensions()
            self.update_status(f"Saved {len(self.settings['overlays'])} element(s) from the visual editor.")

    def stop_processing(self):
        self.update_status("Requesting to stop processing...")
        self.cancel_event.set()
        self.stop_button.config(state=tk.DISABLED)

    def _update_compression_controls(self, event=None):
        """Update compression control states"""
        is_enabled = self.compression_enabled_var.get()

        # Enable/disable compression controls
        try:
            # Update quality preset combo state
            if hasattr(self, 'quality_preset_combo'):
                state = "readonly" if is_enabled else "disabled"
                self.quality_preset_combo.config(state=state)

            # Update other compression-related elements
            if hasattr(self, 'compression_options_view'):
                try:
                    for child in self.compression_options_view.winfo_children():
                        if hasattr(child, 'winfo_children'):
                            for subchild in child.winfo_children():
                                if isinstance(subchild, (ttk.Combobox, ttk.Scale, ttk.Entry)):
                                    try:
                                        if isinstance(subchild, ttk.Combobox):
                                            subchild.config(state="readonly" if is_enabled else "disabled")
                                        else:
                                            subchild.config(state="normal" if is_enabled else "disabled")
                                    except:
                                        pass  # Ignore errors in updating elements
                except:
                    pass  # Ignore access errors

        except Exception as e:
            print(f"Error updating compression controls: {e}")
            # Don't let this error stop the application

    def _update_subtitle_controls(self, event=None):
        """تحديث حالة عناصر التحكم في الترجمة"""
        is_enabled = self.subtitle_enabled_var.get()

        # تفعيل/تعطيل عناصر التحكم في الترجمة
        try:
            controls = [
                'subtitle_path_entry', 'subtitle_browse_button',
                'subtitle_font_size_spinbox', 'subtitle_font_color_combo',
                'subtitle_outline_color_combo', 'subtitle_outline_width_spinbox',
                'subtitle_position_combo'
            ]

            state = "normal" if is_enabled else "disabled"
            readonly_state = "readonly" if is_enabled else "disabled"

            for control_name in controls:
                if hasattr(self, control_name):
                    control = getattr(self, control_name)
                    try:
                        if isinstance(control, (ttk.Combobox, ttk.Entry)) and control_name.endswith('_combo'):
                            control.config(state=readonly_state)
                        elif isinstance(control, ttk.Entry) and 'path' in control_name:
                            control.config(state="readonly" if is_enabled else "disabled")
                        else:
                            control.config(state=state)
                    except:
                        pass

        except Exception as e:
            print(f"خطأ في تحديث عناصر التحكم في الترجمة: {e}")

    def toggle_simple_compression_widgets(self):
        """تفعيل/تعطيل عناصر التحكم البسيطة في الضغط"""
        try:
            state = tk.NORMAL if self.compression_enabled_var.get() else tk.DISABLED
            if hasattr(self, 'quality_preset_combo'):
                self.quality_preset_combo.config(state=state)
        except Exception as e:
            print(f"خطأ في تحديث عناصر الضغط البسيطة: {e}")


class OverlayEditorWindow(tk.Toplevel):
    def __init__(self, parent, video_dimensions, first_frame, existing_overlays, preview_dims):
        super().__init__(parent)
        self.parent = parent
        self.parent_app = parent  # Reference to main app for subtitle settings
        self.video_w, self.video_h = video_dimensions
        self.first_frame = first_frame
        self.existing_overlays = existing_overlays
        self.saved = False

        self.title("Visual Element & Subtitle Editor")
        self.configure(bg=App.BG_COLOR)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        max_w, max_h = 1024, 576
        scale = min(max_w / self.video_w, max_h / self.video_h)
        self.disp_w, self.disp_h = int(self.video_w * scale), int(self.video_h * scale)
        self.scale_x = self.video_w / self.disp_w
        self.scale_y = self.video_h / self.disp_h

        # Make window larger to ensure all controls are visible
        window_width = max(1400, self.disp_w + 500)
        window_height = max(900, self.disp_h + 300)
        self.geometry(f"{window_width}x{window_height}")
        self.minsize(1200, 700)  # Set minimum size to ensure buttons are always visible

        # Center the window on screen
        self.update_idletasks()
        x = (self.winfo_screenwidth() // 2) - (window_width // 2)
        y = (self.winfo_screenheight() // 2) - (window_height // 2)
        self.geometry(f"{window_width}x{window_height}+{x}+{y}")

        self._overlays = {}
        self._subtitles = {}  # Store subtitle elements
        self._selected_id = None
        self._drag_data = {}
        self.brush_size = tk.IntVar(value=30)

        # Subtitle editing variables
        self.subtitle_text_var = tk.StringVar(value="Sample Subtitle Text")
        self.subtitle_font_size_var = tk.IntVar(value=24)
        self.subtitle_font_color_var = tk.StringVar(value="white")
        self.subtitle_outline_color_var = tk.StringVar(value="black")
        self.subtitle_outline_width_var = tk.IntVar(value=2)
        self.subtitle_background_enabled_var = tk.BooleanVar(value=False)
        self.subtitle_background_color_var = tk.StringVar(value="black")
        self.subtitle_background_opacity_var = tk.IntVar(value=80)
        self.subtitle_background_padding_var = tk.IntVar(value=10)

        self.create_widgets()
        self.load_overlays()
        # Initialize selection status and color previews
        self.update_selection_status()
        self.update_color_previews()
        # Force initial preview update
        self.after(100, self.force_update_preview)
    
    def get_overlays(self):
        return list(self._overlays.values())

    def get_preview_dimensions(self):
        return {'w': self.disp_w, 'h': self.disp_h}

    def create_widgets(self):
        from PIL import Image, ImageTk, ImageDraw, ImageFont
        frame_rgb = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
        self.tk_img = ImageTk.PhotoImage(pil_img.resize((self.disp_w, self.disp_h), resample))

        # Create main horizontal layout
        main_frame = ttk.Frame(self)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left side - Video preview and controls
        left_frame = ttk.Frame(main_frame)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Add simple welcome message
        welcome_frame = tk.Frame(left_frame, bg="#0078D4", relief=tk.RAISED, bd=2)
        welcome_frame.pack(fill=tk.X, padx=5, pady=5)

        welcome_label = tk.Label(welcome_frame, text="🎬 Video Preview",
                                bg="#0078D4", fg="white", font=("Arial", 14, "bold"))
        welcome_label.pack(pady=5)

        instructions_label = tk.Label(welcome_frame,
                                    text="Use the controls on the right panel to edit elements",
                                    bg="#0078D4", fg="white", font=("Arial", 10))
        instructions_label.pack(pady=(0, 5))

        # Right side - Subtitle editor with scrollbar
        right_container = ttk.Frame(main_frame)
        right_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Create scrollable frame for subtitle editor
        right_canvas = tk.Canvas(right_container, width=300, bg=App.BG_COLOR, highlightthickness=0)
        right_scrollbar = ttk.Scrollbar(right_container, orient="vertical", command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scrollbar.set)

        right_scrollbar.pack(side="right", fill="y")
        right_canvas.pack(side="left", fill="both", expand=True)

        right_frame = ttk.LabelFrame(right_canvas, text="Element Editor & Subtitle Controls", padding=10)
        right_canvas_window = right_canvas.create_window((0, 0), window=right_frame, anchor="nw")

        def _on_right_configure(event):
            right_canvas.configure(scrollregion=right_canvas.bbox("all"))
        def _on_right_canvas_configure(event):
            right_canvas.itemconfig(right_canvas_window, width=event.width)
        def _on_right_mousewheel(event):
            right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

        right_frame.bind("<Configure>", _on_right_configure)
        right_canvas.bind("<Configure>", _on_right_canvas_configure)
        right_canvas.bind("<MouseWheel>", _on_right_mousewheel)

        # Create main controls and subtitle editor
        self.create_main_controls(right_frame)
        self.create_subtitle_editor(right_frame)




        # Create canvas with scrollbars
        canvas_frame = ttk.Frame(left_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True, pady=10)

        # Create scrollbars
        h_scrollbar = ttk.Scrollbar(canvas_frame, orient="horizontal")
        v_scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical")

        # Create canvas with scrolling capability
        self.canvas = tk.Canvas(canvas_frame, bg='black', highlightthickness=0,
                               xscrollcommand=h_scrollbar.set,
                               yscrollcommand=v_scrollbar.set)

        # Configure scrollbars
        h_scrollbar.config(command=self.canvas.xview)
        v_scrollbar.config(command=self.canvas.yview)

        # Pack scrollbars and canvas
        h_scrollbar.pack(side="bottom", fill="x")
        v_scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Set canvas size and scroll region
        self.canvas.configure(width=min(self.disp_w, 800), height=min(self.disp_h, 600))
        self.canvas.configure(scrollregion=(0, 0, self.disp_w, self.disp_h))

        # Create image on canvas
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img, tags="bg_image")



        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_motion)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)

        # Add mouse wheel scrolling support
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Shift-MouseWheel>", self.on_shift_mousewheel)
        self.canvas.bind("<Control-MouseWheel>", self.on_ctrl_mousewheel)

        # Add keyboard scrolling support
        self.canvas.bind("<Key>", self.on_key_press)
        self.canvas.focus_set()  # Allow canvas to receive keyboard events

    def create_main_controls(self, parent):
        """Create main control buttons and tools"""

        # Export Options Section
        export_options_frame = ttk.LabelFrame(parent, text="📤 Export Options", padding=10)
        export_options_frame.pack(fill=tk.X, pady=(0, 10))

        # Option to disable subtitle overlays during export
        self.disable_subtitle_overlays_var = tk.BooleanVar(value=False)
        disable_subtitles_check = ttk.Checkbutton(export_options_frame,
                                                 text="Disable subtitle overlays in final video (to avoid conflict with SRT subtitles)",
                                                 variable=self.disable_subtitle_overlays_var)
        disable_subtitles_check.pack(anchor="w", pady=2)

        # Option to disable all overlays
        self.disable_all_overlays_var = tk.BooleanVar(value=False)
        disable_all_check = ttk.Checkbutton(export_options_frame,
                                           text="Disable all overlays in final video (preview only)",
                                           variable=self.disable_all_overlays_var)
        disable_all_check.pack(anchor="w", pady=2)

        # Main Action Buttons Section
        main_buttons_frame = ttk.LabelFrame(parent, text="🎬 Main Controls", padding=10)
        main_buttons_frame.pack(fill=tk.X, pady=(0, 15))

        # Add Logo button
        add_logo_btn = tk.Button(main_buttons_frame, text="🖼️ Add Logo", command=self.add_logo,
                                bg="#0078D4", fg="white", font=("Arial", 11, "bold"),
                                relief=tk.RAISED, bd=2, padx=20, pady=10)
        add_logo_btn.pack(fill=tk.X, pady=2)
        ToolTip(add_logo_btn, "إضافة شعار أو صورة إلى الفيديو")

        # Delete Selected button
        delete_btn = tk.Button(main_buttons_frame, text="🗑️ Delete Selected", command=self.delete_selected,
                              bg="#D13438", fg="white", font=("Arial", 11, "bold"),
                              relief=tk.RAISED, bd=2, padx=20, pady=10)
        delete_btn.pack(fill=tk.X, pady=2)
        ToolTip(delete_btn, "حذف العنصر المحدد")

        # Apply Changes button
        apply_btn = tk.Button(main_buttons_frame, text="✅ Apply Changes", command=self.apply_changes,
                             bg="#107C10", fg="white", font=("Arial", 11, "bold"),
                             relief=tk.RAISED, bd=2, padx=20, pady=10)
        apply_btn.pack(fill=tk.X, pady=2)
        ToolTip(apply_btn, "تطبيق التغييرات دون إغلاق النافذة")

        # Save & Close button
        save_close_btn = tk.Button(main_buttons_frame, text="💾 Save & Close", command=self.save_and_close,
                                  bg="#FF8C00", fg="white", font=("Arial", 11, "bold"),
                                  relief=tk.RAISED, bd=2, padx=20, pady=10)
        save_close_btn.pack(fill=tk.X, pady=2)
        ToolTip(save_close_btn, "حفظ التغييرات وإغلاق النافذة")

        # Editing Tools Section
        tools_frame = ttk.LabelFrame(parent, text="🛠️ Editing Tools", padding=10)
        tools_frame.pack(fill=tk.X, pady=(0, 15))

        self.current_mode = tk.StringVar(value="move")
        modes = [
            ("🖱️ Move/Select", "move"),
            ("🔲 Pixelate", "pixelate"),
            ("⬜ Rectangle", "rect"),
            ("⭕ Circle", "circle"),
            ("📝 Subtitle", "subtitle")
        ]

        for text, mode in modes:
            btn = tk.Radiobutton(tools_frame, text=text, variable=self.current_mode, value=mode,
                               bg="#2D2D2D", fg="white", selectcolor="#0078D4",
                               font=("Arial", 10, "bold"), relief=tk.RAISED, bd=1,
                               padx=10, pady=8, indicatoron=0)
            btn.pack(fill=tk.X, pady=1)

        # Brush Settings Section
        brush_frame = ttk.LabelFrame(parent, text="🖌️ Brush Settings", padding=10)
        brush_frame.pack(fill=tk.X, pady=(0, 15))

        tk.Label(brush_frame, text="Brush Size:", font=("Arial", 10, "bold")).pack(anchor="w")

        brush_scale = tk.Scale(brush_frame, from_=10, to=150, orient=tk.HORIZONTAL,
                              variable=self.brush_size, bg="#f0f0f0",
                              highlightthickness=0, length=250)
        brush_scale.pack(fill=tk.X, pady=5)

        brush_entry = tk.Entry(brush_frame, textvariable=self.brush_size, width=10,
                              font=("Arial", 10), justify="center")
        brush_entry.pack(pady=2)

        # Zoom Controls Section
        zoom_frame = ttk.LabelFrame(parent, text="🔍 Zoom Controls", padding=10)
        zoom_frame.pack(fill=tk.X, pady=(0, 15))

        # Zoom level display
        if not hasattr(self, 'zoom_level'):
            self.zoom_level = 1.0
        self.zoom_label = tk.Label(zoom_frame, text=f"Zoom: {int(self.zoom_level * 100)}%",
                                  font=("Arial", 12, "bold"))
        self.zoom_label.pack(pady=5)

        # Zoom buttons
        tk.Button(zoom_frame, text="🔍+ Zoom In", command=self.zoom_in,
                 bg="#555555", fg="white", font=("Arial", 10, "bold"),
                 relief=tk.RAISED, bd=1, padx=15, pady=5).pack(fill=tk.X, pady=1)

        tk.Button(zoom_frame, text="🔍- Zoom Out", command=self.zoom_out,
                 bg="#555555", fg="white", font=("Arial", 10, "bold"),
                 relief=tk.RAISED, bd=1, padx=15, pady=5).pack(fill=tk.X, pady=1)

        tk.Button(zoom_frame, text="🎯 Reset Zoom", command=self.reset_zoom,
                 bg="#555555", fg="white", font=("Arial", 10, "bold"),
                 relief=tk.RAISED, bd=1, padx=15, pady=5).pack(fill=tk.X, pady=1)

    def create_subtitle_editor(self, parent):
        """Create subtitle editing controls"""

        # Information about subtitle settings
        info_frame = ttk.LabelFrame(parent, text="ℹ️ Subtitle Settings Info", padding=5)
        info_frame.pack(fill=tk.X, pady=(0, 10))

        info_text = tk.Text(info_frame, height=3, wrap=tk.WORD, bg="#f0f0f0", relief=tk.FLAT)
        info_text.pack(fill=tk.X, padx=5, pady=5)
        info_text.insert('1.0',
            "📝 These settings will be applied to your SRT subtitle file.\n"
            "🎨 Adjust font, colors, and background to preview how your SRT subtitles will look.\n"
            "💾 Click 'Save & Close' to apply these settings to your external subtitle file.")
        info_text.config(state=tk.DISABLED)

        # Text input (for preview only)
        text_frame = ttk.LabelFrame(parent, text="Preview Text (for testing only)", padding=5)
        text_frame.pack(fill=tk.X, pady=(0, 10))

        self.subtitle_text_entry = tk.Text(text_frame, height=3, width=30, wrap=tk.WORD)
        self.subtitle_text_entry.pack(fill=tk.BOTH, expand=True)
        self.subtitle_text_entry.insert('1.0', "Sample subtitle text for preview")
        self.subtitle_text_entry.bind('<KeyRelease>', self.on_subtitle_text_change)
        self.subtitle_text_entry.bind('<Button-1>', self.on_subtitle_text_change)
        self.subtitle_text_entry.bind('<FocusOut>', self.on_subtitle_text_change)

        # Font settings
        font_frame = ttk.LabelFrame(parent, text="Font Settings", padding=5)
        font_frame.pack(fill=tk.X, pady=(0, 10))

        # Font size with scale
        size_frame = ttk.Frame(font_frame)
        size_frame.pack(fill=tk.X, pady=2)
        ttk.Label(size_frame, text="Size (px):").pack(side=tk.LEFT)

        # Use Scale widget for better control
        self.size_scale = tk.Scale(size_frame, from_=8, to=72, orient=tk.HORIZONTAL,
                                  variable=self.subtitle_font_size_var,
                                  command=self.on_font_size_change,
                                  length=150, bg="#f0f0f0")
        self.size_scale.pack(side=tk.RIGHT, padx=5)

        # Add entry for precise input
        self.size_entry = tk.Entry(size_frame, textvariable=self.subtitle_font_size_var,
                                  width=5, justify="center")
        self.size_entry.pack(side=tk.RIGHT, padx=2)
        self.size_entry.bind('<KeyRelease>', lambda e: self.after_idle(self.force_update_preview))
        self.size_entry.bind('<FocusOut>', lambda e: self.after_idle(self.force_update_preview))

        # Font color
        color_frame = ttk.Frame(font_frame)
        color_frame.pack(fill=tk.X, pady=2)
        ttk.Label(color_frame, text="Font Color:").pack(side=tk.LEFT)
        self.color_combo = ttk.Combobox(color_frame, textvariable=self.subtitle_font_color_var,
                                       values=["white", "black", "yellow", "red", "blue", "green", "orange", "purple"],
                                       state="readonly", width=12)
        self.color_combo.pack(side=tk.RIGHT)
        self.color_combo.bind('<<ComboboxSelected>>', self.on_font_color_change)

        # Add color preview
        self.font_color_preview = tk.Label(color_frame, text="●", font=("Arial", 16),
                                          fg=self.subtitle_font_color_var.get())
        self.font_color_preview.pack(side=tk.RIGHT, padx=5)

        # Outline color
        outline_frame = ttk.Frame(font_frame)
        outline_frame.pack(fill=tk.X, pady=2)
        ttk.Label(outline_frame, text="Outline Color:").pack(side=tk.LEFT)
        self.outline_combo = ttk.Combobox(outline_frame, textvariable=self.subtitle_outline_color_var,
                                         values=["black", "white", "gray", "darkgray", "red", "blue", "green", "yellow"],
                                         state="readonly", width=12)
        self.outline_combo.pack(side=tk.RIGHT)
        self.outline_combo.bind('<<ComboboxSelected>>', self.on_outline_color_change)

        # Add color preview
        self.outline_color_preview = tk.Label(outline_frame, text="●", font=("Arial", 16),
                                             fg=self.subtitle_outline_color_var.get())
        self.outline_color_preview.pack(side=tk.RIGHT, padx=5)

        # Outline width with scale
        outline_width_frame = ttk.Frame(font_frame)
        outline_width_frame.pack(fill=tk.X, pady=2)
        ttk.Label(outline_width_frame, text="Outline Width (px):").pack(side=tk.LEFT)

        # Use Scale widget for better control
        self.width_scale = tk.Scale(outline_width_frame, from_=0, to=10, orient=tk.HORIZONTAL,
                                   variable=self.subtitle_outline_width_var,
                                   command=self.on_outline_width_change,
                                   length=150, bg="#f0f0f0")
        self.width_scale.pack(side=tk.RIGHT, padx=5)

        # Add entry for precise input
        self.width_entry = tk.Entry(outline_width_frame, textvariable=self.subtitle_outline_width_var,
                                   width=5, justify="center")
        self.width_entry.pack(side=tk.RIGHT, padx=2)
        self.width_entry.bind('<KeyRelease>', lambda e: self.after_idle(self.force_update_preview))
        self.width_entry.bind('<FocusOut>', lambda e: self.after_idle(self.force_update_preview))

        # Background settings
        bg_frame = ttk.LabelFrame(parent, text="Background Settings", padding=5)
        bg_frame.pack(fill=tk.X, pady=(0, 10))

        # Enable background
        bg_enable_frame = ttk.Frame(bg_frame)
        bg_enable_frame.pack(fill=tk.X, pady=2)
        self.bg_enable_check = ttk.Checkbutton(bg_enable_frame, text="Enable Background Box",
                                              variable=self.subtitle_background_enabled_var,
                                              command=self.on_background_enable_change)
        self.bg_enable_check.pack(side=tk.LEFT)

        # Background color
        bg_color_frame = ttk.Frame(bg_frame)
        bg_color_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bg_color_frame, text="Background Color:").pack(side=tk.LEFT)
        self.bg_color_combo = ttk.Combobox(bg_color_frame, textvariable=self.subtitle_background_color_var,
                                          values=["black", "white", "gray", "darkgray", "blue", "red", "green", "yellow"],
                                          state="readonly", width=12)
        self.bg_color_combo.pack(side=tk.RIGHT)
        self.bg_color_combo.bind('<<ComboboxSelected>>', lambda e: self.after_idle(self.force_update_preview))

        # Add color preview
        self.bg_color_preview = tk.Label(bg_color_frame, text="■", font=("Arial", 16),
                                        fg=self.subtitle_background_color_var.get())
        self.bg_color_preview.pack(side=tk.RIGHT, padx=5)

        # Background opacity
        bg_opacity_frame = ttk.Frame(bg_frame)
        bg_opacity_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bg_opacity_frame, text="Background Opacity (%):").pack(side=tk.LEFT)

        # Use Scale widget for better control
        self.opacity_scale = tk.Scale(bg_opacity_frame, from_=10, to=100, orient=tk.HORIZONTAL,
                                     variable=self.subtitle_background_opacity_var,
                                     command=self.on_background_opacity_change,
                                     length=150, bg="#f0f0f0")
        self.opacity_scale.pack(side=tk.RIGHT, padx=5)

        # Add entry for precise input
        self.opacity_entry = tk.Entry(bg_opacity_frame, textvariable=self.subtitle_background_opacity_var,
                                     width=5, justify="center")
        self.opacity_entry.pack(side=tk.RIGHT, padx=2)
        self.opacity_entry.bind('<KeyRelease>', lambda e: self.after_idle(self.force_update_preview))
        self.opacity_entry.bind('<FocusOut>', lambda e: self.after_idle(self.force_update_preview))

        # Background padding
        bg_padding_frame = ttk.Frame(bg_frame)
        bg_padding_frame.pack(fill=tk.X, pady=2)
        ttk.Label(bg_padding_frame, text="Background Padding (px):").pack(side=tk.LEFT)

        # Use Scale widget for better control
        self.padding_scale = tk.Scale(bg_padding_frame, from_=0, to=50, orient=tk.HORIZONTAL,
                                     variable=self.subtitle_background_padding_var,
                                     command=self.on_background_padding_change,
                                     length=150, bg="#f0f0f0")
        self.padding_scale.pack(side=tk.RIGHT, padx=5)

        # Add entry for precise input
        self.padding_entry = tk.Entry(bg_padding_frame, textvariable=self.subtitle_background_padding_var,
                                     width=5, justify="center")
        self.padding_entry.pack(side=tk.RIGHT, padx=2)
        self.padding_entry.bind('<KeyRelease>', lambda e: self.after_idle(self.force_update_preview))
        self.padding_entry.bind('<FocusOut>', lambda e: self.after_idle(self.force_update_preview))

        # Position controls
        position_frame = ttk.LabelFrame(parent, text="Position", padding=5)
        position_frame.pack(fill=tk.X, pady=(0, 10))

        # Selection status
        self.selection_status_label = tk.Label(position_frame, text="No subtitle selected",
                                              bg="#FF6B6B", fg="white", font=("Arial", 9, "bold"),
                                              relief=tk.RAISED, bd=1, padx=5, pady=2)
        self.selection_status_label.pack(fill=tk.X, pady=2)

        ttk.Button(position_frame, text="Add Subtitle", command=self.add_subtitle).pack(fill=tk.X, pady=2)
        ttk.Button(position_frame, text="Update Selected", command=self.update_selected_subtitle).pack(fill=tk.X, pady=2)
        ttk.Button(position_frame, text="Refresh Preview", command=self.update_subtitle_preview).pack(fill=tk.X, pady=2)

        # Preview
        preview_frame = ttk.LabelFrame(parent, text="Preview", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        self.subtitle_preview_canvas = tk.Canvas(preview_frame, width=250, height=120, bg='black', relief=tk.SUNKEN, bd=2)
        self.subtitle_preview_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Initial preview update
        self.after(100, self.update_subtitle_preview)

    def on_subtitle_text_change(self, event=None):
        """Update subtitle text when user types"""
        try:
            text = self.subtitle_text_entry.get('1.0', tk.END).strip()
            self.subtitle_text_var.set(text)
            # Force immediate update for better responsiveness
            self.force_update_preview()
            print(f"Text changed to: {text[:20]}...")
        except Exception as e:
            print(f"Error in text change: {e}")

    def force_update_preview(self):
        """Force immediate preview update"""
        try:
            self.update_subtitle_preview()
            # Also update selected subtitle if one is selected
            self.auto_update_selected_subtitle()
            # Update color previews
            self.update_color_previews()
        except Exception as e:
            print(f"Error in force update: {e}")

    def update_color_previews(self):
        """Update color preview indicators"""
        try:
            # Color mapping for preview
            color_map = {
                'white': 'white', 'black': 'black', 'yellow': 'yellow',
                'red': 'red', 'blue': 'blue', 'green': 'green',
                'orange': 'orange', 'purple': 'purple', 'gray': 'gray',
                'darkgray': 'darkgray'
            }

            # Update font color preview
            if hasattr(self, 'font_color_preview'):
                font_color = self.subtitle_font_color_var.get()
                preview_color = color_map.get(font_color, font_color)
                self.font_color_preview.config(fg=preview_color)

            # Update outline color preview
            if hasattr(self, 'outline_color_preview'):
                outline_color = self.subtitle_outline_color_var.get()
                preview_color = color_map.get(outline_color, outline_color)
                self.outline_color_preview.config(fg=preview_color)

            # Update background color preview
            if hasattr(self, 'bg_color_preview'):
                bg_color = self.subtitle_background_color_var.get()
                preview_color = color_map.get(bg_color, bg_color)
                self.bg_color_preview.config(fg=preview_color)

        except Exception as e:
            print(f"Error updating color previews: {e}")

    def on_font_size_change(self, value):
        """Handle font size scale change"""
        try:
            # Force immediate update
            self.force_update_preview()
            print(f"Font size changed to: {value}")
        except Exception as e:
            print(f"Error in font size change: {e}")

    def on_outline_width_change(self, value):
        """Handle outline width scale change"""
        try:
            # Force immediate update
            self.force_update_preview()
            print(f"Outline width changed to: {value}")
        except Exception as e:
            print(f"Error in outline width change: {e}")

    def on_font_color_change(self, event=None):
        """Handle font color change"""
        try:
            # Force immediate update
            self.force_update_preview()
            print(f"Font color changed to: {self.subtitle_font_color_var.get()}")
        except Exception as e:
            print(f"Error in font color change: {e}")

    def on_outline_color_change(self, event=None):
        """Handle outline color change"""
        try:
            # Force immediate update
            self.force_update_preview()
            print(f"Outline color changed to: {self.subtitle_outline_color_var.get()}")
        except Exception as e:
            print(f"Error in outline color change: {e}")

    def on_background_enable_change(self):
        """Handle background enable/disable change"""
        try:
            # Force immediate update
            self.force_update_preview()
            print(f"Background enabled: {self.subtitle_background_enabled_var.get()}")
        except Exception as e:
            print(f"Error in background enable change: {e}")

    def on_background_opacity_change(self, value):
        """Handle background opacity change"""
        try:
            # Force immediate update
            self.force_update_preview()
            print(f"Background opacity changed to: {value}")
        except Exception as e:
            print(f"Error in background opacity change: {e}")

    def on_background_padding_change(self, value):
        """Handle background padding change"""
        try:
            # Force immediate update
            self.force_update_preview()
            print(f"Background padding changed to: {value}")
        except Exception as e:
            print(f"Error in background padding change: {e}")

    def auto_update_selected_subtitle(self):
        """Automatically update selected subtitle with current settings"""
        if self._selected_id and self._selected_id in self._subtitles:
            try:
                text = self.subtitle_text_entry.get('1.0', tk.END).strip()
                if text:  # Only update if there's text
                    subtitle_data = self._overlays[self._selected_id]
                    subtitle_data.update({
                        'text': text,
                        'font_size': self.subtitle_font_size_var.get(),
                        'font_color': self.subtitle_font_color_var.get(),
                        'outline_color': self.subtitle_outline_color_var.get(),
                        'outline_width': self.subtitle_outline_width_var.get(),
                        'background_enabled': self.subtitle_background_enabled_var.get(),
                        'background_color': self.subtitle_background_color_var.get(),
                        'background_opacity': self.subtitle_background_opacity_var.get(),
                        'background_padding': self.subtitle_background_padding_var.get()
                    })
                    # Redraw the canvas to show changes
                    self.redraw_all()
            except Exception as e:
                print(f"Error in auto update: {e}")

    def update_subtitle_preview(self, event=None):
        """Update the subtitle preview canvas"""
        try:
            from PIL import Image, ImageDraw, ImageFont, ImageTk

            # Create preview image (larger for better visibility)
            preview_img = Image.new('RGB', (250, 120), 'black')
            draw = ImageDraw.Draw(preview_img)

            # Get current text from text widget with safe fallback
            try:
                text = self.subtitle_text_entry.get('1.0', tk.END).strip() or "Sample Text"
            except:
                text = "Sample Text"

            # Get values with safe fallbacks
            try:
                font_size = self.subtitle_font_size_var.get() or 24
            except:
                font_size = 24

            try:
                font_color = self.subtitle_font_color_var.get() or "white"
            except:
                font_color = "white"

            try:
                outline_color = self.subtitle_outline_color_var.get() or "black"
            except:
                outline_color = "black"

            try:
                outline_width = self.subtitle_outline_width_var.get() or 2
            except:
                outline_width = 2

            try:
                bg_enabled = self.subtitle_background_enabled_var.get()
            except:
                bg_enabled = False

            try:
                bg_color = self.subtitle_background_color_var.get() or "black"
            except:
                bg_color = "black"

            try:
                bg_opacity = self.subtitle_background_opacity_var.get() or 80
            except:
                bg_opacity = 80

            try:
                bg_padding = self.subtitle_background_padding_var.get() or 10
            except:
                bg_padding = 10

            # Update the text variable
            self.subtitle_text_var.set(text)

            # Try to use a system font with better scaling
            try:
                # Scale font size for preview but make it more visible
                preview_font_size = max(12, min(font_size * 0.8, 32))  # Better scaling
                font = ImageFont.truetype("arial.ttf", preview_font_size)
            except:
                try:
                    # Try other common fonts
                    preview_font_size = max(12, min(font_size * 0.8, 32))
                    font = ImageFont.truetype("calibri.ttf", preview_font_size)
                except:
                    try:
                        # Try default with size
                        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", preview_font_size)
                    except:
                        font = ImageFont.load_default()

            # Get text size
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # Center text in the larger preview
            x = (250 - text_width) // 2
            y = (120 - text_height) // 2

            # Draw background box if enabled
            if bg_enabled:
                # Calculate background rectangle
                bg_x1 = x - bg_padding // 2
                bg_y1 = y - bg_padding // 2
                bg_x2 = x + text_width + bg_padding // 2
                bg_y2 = y + text_height + bg_padding // 2

                # Create semi-transparent background
                bg_overlay = Image.new('RGBA', (250, 120), (0, 0, 0, 0))
                bg_draw = ImageDraw.Draw(bg_overlay)

                # Convert color name to RGB
                color_map = {
                    'black': (0, 0, 0),
                    'white': (255, 255, 255),
                    'gray': (128, 128, 128),
                    'darkgray': (64, 64, 64),
                    'blue': (0, 0, 255),
                    'red': (255, 0, 0),
                    'green': (0, 255, 0),
                    'yellow': (255, 255, 0),
                    'orange': (255, 165, 0),
                    'purple': (128, 0, 128)
                }
                bg_rgb = color_map.get(bg_color, (0, 0, 0))
                bg_alpha = int(255 * bg_opacity / 100)

                bg_draw.rectangle([bg_x1, bg_y1, bg_x2, bg_y2],
                                fill=(*bg_rgb, bg_alpha))

                # Composite background with main image
                preview_img = Image.alpha_composite(preview_img.convert('RGBA'), bg_overlay).convert('RGB')
                draw = ImageDraw.Draw(preview_img)

            # Convert color names to actual colors for drawing
            color_map = {
                'white': 'white', 'black': 'black', 'yellow': 'yellow',
                'red': 'red', 'blue': 'blue', 'green': 'green',
                'orange': 'orange', 'purple': 'purple', 'gray': 'gray',
                'darkgray': 'darkgray'
            }

            actual_font_color = color_map.get(font_color.lower(), font_color)
            actual_outline_color = color_map.get(outline_color.lower(), outline_color)

            # Draw text with outline (make outline more visible)
            if outline_width > 0:
                # Draw outline with multiple passes for better visibility
                for dx in range(-outline_width, outline_width + 1):
                    for dy in range(-outline_width, outline_width + 1):
                        if dx != 0 or dy != 0:
                            draw.text((x + dx, y + dy), text, font=font, fill=actual_outline_color)

            # Draw main text
            draw.text((x, y), text, font=font, fill=actual_font_color)

            # Convert to PhotoImage and display
            self.subtitle_preview_img = ImageTk.PhotoImage(preview_img)
            self.subtitle_preview_canvas.delete("all")
            self.subtitle_preview_canvas.create_image(125, 60, image=self.subtitle_preview_img)

            # Force canvas update
            self.subtitle_preview_canvas.update_idletasks()

        except Exception as e:
            print(f"Error updating subtitle preview: {e}")
            # Create a simple error preview
            try:
                self.subtitle_preview_canvas.delete("all")
                self.subtitle_preview_canvas.create_text(100, 50, text="Preview Error", fill="red", font=("Arial", 10))
            except:
                pass

    def add_subtitle(self):
        """Add a new subtitle element to the canvas"""
        text = self.subtitle_text_entry.get('1.0', tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter subtitle text first.")
            return

        # Create subtitle data
        subtitle_id = f"subtitle_{len(self._subtitles)}"
        subtitle_data = {
            'type': 'subtitle',
            'text': text,
            'font_size': self.subtitle_font_size_var.get(),
            'font_color': self.subtitle_font_color_var.get(),
            'outline_color': self.subtitle_outline_color_var.get(),
            'outline_width': self.subtitle_outline_width_var.get(),
            'background_enabled': self.subtitle_background_enabled_var.get(),
            'background_color': self.subtitle_background_color_var.get(),
            'background_opacity': self.subtitle_background_opacity_var.get(),
            'background_padding': self.subtitle_background_padding_var.get(),
            'x': self.disp_w // 4,
            'y': self.disp_h - 100,  # Default to bottom
            'w': self.disp_w // 2,
            'h': 50
        }

        self._overlays[subtitle_id] = subtitle_data
        self._subtitles[subtitle_id] = subtitle_data
        self.redraw_all()

    def update_selected_subtitle(self):
        """Update the selected subtitle with current settings"""
        if self._selected_id and self._selected_id in self._subtitles:
            text = self.subtitle_text_entry.get('1.0', tk.END).strip()
            if not text:
                messagebox.showwarning("تحذير", "يرجى إدخال نص الترجمة أولاً.")
                return

            subtitle_data = self._overlays[self._selected_id]
            subtitle_data.update({
                'text': text,
                'font_size': self.subtitle_font_size_var.get(),
                'font_color': self.subtitle_font_color_var.get(),
                'outline_color': self.subtitle_outline_color_var.get(),
                'outline_width': self.subtitle_outline_width_var.get(),
                'background_enabled': self.subtitle_background_enabled_var.get(),
                'background_color': self.subtitle_background_color_var.get(),
                'background_opacity': self.subtitle_background_opacity_var.get(),
                'background_padding': self.subtitle_background_padding_var.get()
            })

            self.redraw_all()
            messagebox.showinfo("تم التحديث", "تم تحديث إعدادات الترجمة المحددة بنجاح!")
        else:
            messagebox.showinfo("معلومات", "يرجى تحديد عنصر ترجمة أولاً.\nانقر على أي ترجمة في الفيديو لتحديدها.")

    def update_selection_status(self):
        """Update the selection status display"""
        if hasattr(self, 'selection_status_label'):
            if self._selected_id and self._selected_id in self._subtitles:
                self.selection_status_label.config(text=f"Selected: Subtitle {self._selected_id[-8:]}",
                                                  bg="#4CAF50", fg="white")
            else:
                self.selection_status_label.config(text="No subtitle selected",
                                                  bg="#FF6B6B", fg="white")

    def load_subtitle_settings(self, subtitle_id):
        """Load settings from selected subtitle into the editor"""
        if subtitle_id in self._subtitles:
            data = self._subtitles[subtitle_id]

            # Load text
            self.subtitle_text_entry.delete('1.0', tk.END)
            self.subtitle_text_entry.insert('1.0', data.get('text', ''))

            # Load font settings
            self.subtitle_font_size_var.set(data.get('font_size', 24))
            self.subtitle_font_color_var.set(data.get('font_color', 'white'))
            self.subtitle_outline_color_var.set(data.get('outline_color', 'black'))
            self.subtitle_outline_width_var.set(data.get('outline_width', 2))

            # Load background settings
            self.subtitle_background_enabled_var.set(data.get('background_enabled', False))
            self.subtitle_background_color_var.set(data.get('background_color', 'black'))
            self.subtitle_background_opacity_var.set(data.get('background_opacity', 80))
            self.subtitle_background_padding_var.set(data.get('background_padding', 10))

            # Update preview and selection status
            self.force_update_preview()  # Use force update for immediate response
            self.update_selection_status()

    def on_mousewheel(self, event):
        """Handle vertical mouse wheel scrolling"""
        try:
            # Scroll vertically
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except:
            pass

    def on_shift_mousewheel(self, event):
        """Handle horizontal mouse wheel scrolling (with Shift key)"""
        try:
            # Scroll horizontally when Shift is held
            self.canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        except:
            pass

    def on_ctrl_mousewheel(self, event):
        """Handle zoom with Ctrl + mouse wheel"""
        try:
            # Get current zoom level (if not set, default to 1.0)
            if not hasattr(self, 'zoom_level'):
                self.zoom_level = 1.0

            # Calculate zoom change
            zoom_change = 0.1 if event.delta > 0 else -0.1
            new_zoom = max(0.1, min(3.0, self.zoom_level + zoom_change))

            if new_zoom != self.zoom_level:
                self.zoom_level = new_zoom
                self.update_zoom()
                # Update zoom label if it exists
                if hasattr(self, 'zoom_label'):
                    self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")
        except:
            pass

    def update_zoom(self):
        """Update canvas zoom level"""
        try:
            from PIL import Image, ImageTk

            # Calculate new display size
            new_w = int(self.video_w * self.zoom_level)
            new_h = int(self.video_h * self.zoom_level)

            # Resize the background image
            frame_rgb = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
            zoomed_img = pil_img.resize((new_w, new_h), resample)
            self.tk_img = ImageTk.PhotoImage(zoomed_img)

            # Update canvas
            self.canvas.delete("bg_image")
            self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img, tags="bg_image")

            # Update scale factors
            self.scale_x = self.video_w / new_w
            self.scale_y = self.video_h / new_h

            # Update scroll region
            self.canvas.configure(scrollregion=(0, 0, new_w, new_h))

            # Redraw all overlays
            self.redraw_all()

        except Exception as e:
            print(f"Error updating zoom: {e}")

    def zoom_in(self):
        """Zoom in by 10%"""
        if not hasattr(self, 'zoom_level'):
            self.zoom_level = 1.0
        self.zoom_level = min(3.0, self.zoom_level + 0.1)
        self.update_zoom()
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")

    def zoom_out(self):
        """Zoom out by 10%"""
        if not hasattr(self, 'zoom_level'):
            self.zoom_level = 1.0
        self.zoom_level = max(0.1, self.zoom_level - 0.1)
        self.update_zoom()
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")

    def reset_zoom(self):
        """Reset zoom to 100%"""
        self.zoom_level = 1.0
        self.update_zoom()
        self.zoom_label.config(text=f"Zoom: {int(self.zoom_level * 100)}%")

    def on_key_press(self, event):
        """Handle keyboard scrolling"""
        try:
            if event.keysym == "Up":
                self.canvas.yview_scroll(-1, "units")
            elif event.keysym == "Down":
                self.canvas.yview_scroll(1, "units")
            elif event.keysym == "Left":
                self.canvas.xview_scroll(-1, "units")
            elif event.keysym == "Right":
                self.canvas.xview_scroll(1, "units")
            elif event.keysym == "Prior":  # Page Up
                self.canvas.yview_scroll(-1, "pages")
            elif event.keysym == "Next":   # Page Down
                self.canvas.yview_scroll(1, "pages")
            elif event.keysym == "Home":
                self.canvas.xview_moveto(0)
                self.canvas.yview_moveto(0)
            elif event.keysym == "End":
                self.canvas.xview_moveto(1)
                self.canvas.yview_moveto(1)
        except:
            pass

    def redraw_all(self):
        self.canvas.delete("overlay", "handle", "temp")
        for oid, data in self._overlays.items():
            tags = ("overlay", data['type'], oid)
            x, y, w, h = data['x'], data['y'], data['w'], data['h']
            if data['type'] == 'logo' and 'pil_img' in data:
                from PIL import Image, ImageTk
                resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
                if w > 0 and h > 0:
                    logo_resized = data['pil_img'].resize((w, h), resample)
                    data['tk_logo'] = ImageTk.PhotoImage(logo_resized)
                    self.canvas.create_image(x, y, anchor='nw', image=data['tk_logo'], tags=tags)
            elif data['type'] == 'subtitle':
                # Draw subtitle with visual representation
                bg_enabled = data.get('background_enabled', False)
                if bg_enabled:
                    # Draw background box representation
                    bg_color = data.get('background_color', 'black')
                    bg_opacity = data.get('background_opacity', 80)
                    bg_padding = data.get('background_padding', 10)

                    color_map = {
                        'black': '#000000',
                        'white': '#FFFFFF',
                        'gray': '#808080',
                        'darkgray': '#404040',
                        'blue': '#0000FF',
                        'red': '#FF0000',
                        'green': '#00FF00',
                        'yellow': '#FFFF00',
                        'orange': '#FFA500',
                        'purple': '#800080'
                    }
                    bg_hex = color_map.get(bg_color, '#000000')

                    # Draw background with padding representation
                    bg_x = x - bg_padding//2
                    bg_y = y - bg_padding//2
                    bg_w = w + bg_padding
                    bg_h = h + bg_padding

                    # Create semi-transparent effect by using stipple
                    stipple_pattern = "gray75" if bg_opacity > 50 else "gray50"
                    self.canvas.create_rectangle(bg_x, bg_y, bg_x + bg_w, bg_y + bg_h,
                                               fill=bg_hex, stipple=stipple_pattern,
                                               outline="orange", width=2, tags=tags)
                    text_color = "white" if bg_color in ['black', 'darkgray', 'blue', 'red'] else "black"
                else:
                    # Draw transparent subtitle box
                    self.canvas.create_rectangle(x, y, x + w, y + h, fill="yellow", stipple="gray25", outline="orange", width=2, tags=tags)
                    text_color = "black"

                # Add text label
                text_preview = data.get('text', 'Subtitle')[:20] + ('...' if len(data.get('text', '')) > 20 else '')
                self.canvas.create_text(x + w//2, y + h//2, text=text_preview, fill=text_color, font=("Arial", 8, "bold"), tags=tags)
            elif data['type'] == 'blur':
                self.canvas.create_rectangle(x, y, x + w, y + h, fill="red", stipple="gray50", outline="white", width=1, tags=tags)
            elif data['type'] == 'pixelate':
                 self.canvas.create_oval(x, y, x + w, y + h, outline="#FFD700", width=1, dash=(3,5), tags=tags)
            elif data['type'] in ['rect', 'circle']:
                color = data.get('color', '#FFFF00')
                thickness = data.get('thickness', 2)
                if data['type'] == 'rect':
                    self.canvas.create_rectangle(x, y, x + w, y + h, outline=color, width=thickness, tags=tags)
                else:
                    self.canvas.create_oval(x, y, x + w, y + h, outline=color, width=thickness, tags=tags)

            if oid == self._selected_id:
                self.canvas.create_rectangle(x, y, x + w, y + h, outline="#007ACC", width=2, dash=(6, 2), tags=("handle", oid))

        # Update scroll region to include all elements
        self.update_scroll_region()

    def update_scroll_region(self):
        """Update the scroll region to include all elements"""
        try:
            # Calculate the bounding box of all elements
            min_x, min_y = 0, 0
            max_x, max_y = self.disp_w, self.disp_h

            for data in self._overlays.values():
                x, y, w, h = data['x'], data['y'], data['w'], data['h']
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x + w)
                max_y = max(max_y, y + h)

            # Add some padding
            padding = 50
            min_x -= padding
            min_y -= padding
            max_x += padding
            max_y += padding

            # Update scroll region
            self.canvas.configure(scrollregion=(min_x, min_y, max_x, max_y))
        except:
            # Fallback to default scroll region
            self.canvas.configure(scrollregion=(0, 0, self.disp_w, self.disp_h))

    def load_overlays(self):
        from PIL import Image
        for i, info in enumerate(self.existing_overlays):
            oid = f"{info.get('type', 'item')}_{time.time()}_{i}"
            if info.get('type') == 'logo' and 'path' in info and os.path.exists(info['path']):
                try:
                    info['pil_img'] = Image.open(info['path']).convert("RGBA")
                except Exception as e:
                    print(f"Skipping logo, could not load image {info['path']}: {e}")
                    continue
            self._overlays[oid] = info
        self.redraw_all()

    def add_logo(self):
        from PIL import Image
        path = filedialog.askopenfilename(title="Select Logo", filetypes=[("PNG Images", "*.png")], parent=self)
        if not path: return
        try:
            img = Image.open(path).convert("RGBA")
            w, h = img.size
            new_w = min(100, self.disp_w // 4)
            new_h = int(new_w * (h / w)) if w > 0 and h > 0 else new_w
            oid = f"logo_{time.time()}"
            self._overlays[oid] = {'type': 'logo', 'path': path, 'x': 50, 'y': 50, 'w': new_w, 'h': new_h, 'pil_img': img}
            self._selected_id = oid
            self.update_selection_status()
            self.redraw_all()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}", parent=self)

    def delete_selected(self):
        if self._selected_id in self._overlays:
            # Remove from subtitles dict if it's a subtitle
            if self._selected_id in self._subtitles:
                del self._subtitles[self._selected_id]
            del self._overlays[self._selected_id]
            self._selected_id = None
            self.update_selection_status()
            self.redraw_all()

    def on_close(self):
        self.destroy()

    def apply_changes(self):
        """Apply changes without closing the window"""
        try:
            # Clean up image references for overlays
            for data in self._overlays.values():
                data.pop('pil_img', None)
                data.pop('tk_logo', None)

            # Apply subtitle settings to main app if there are subtitle elements
            subtitle_elements = [data for data in self._overlays.values() if data.get('type') == 'subtitle']
            if subtitle_elements and hasattr(self, 'parent_app'):
                # Use the first subtitle element's settings for SRT subtitles
                subtitle_settings = subtitle_elements[0]
                self.apply_subtitle_settings_to_main_app(subtitle_settings)

                # Remove subtitle elements from overlays to avoid duplication
                self._overlays = {k: v for k, v in self._overlays.items() if v.get('type') != 'subtitle'}
                print(f"Applied subtitle settings and removed {len(subtitle_elements)} subtitle elements")

            self.saved = True

            # Show confirmation message
            from tkinter import messagebox
            if subtitle_elements:
                messagebox.showinfo("تم تطبيق التغييرات",
                    "✅ تم تطبيق تغييرات العناصر بنجاح!\n"
                    "🎨 إعدادات الخط والخلفية طُبقت على ترجمة SRT.\n"
                    "📝 يمكنك الآن الاستمرار في التحرير أو إغلاق النافذة.")
            else:
                messagebox.showinfo("تم تطبيق التغييرات", "تم تطبيق تغييرات العناصر بنجاح!\nيمكنك الآن الاستمرار في التحرير أو إغلاق النافذة.")

        except Exception as e:
            print(f"Error in apply_changes: {e}")
            from tkinter import messagebox
            messagebox.showerror("خطأ", f"حدث خطأ أثناء تطبيق التغييرات: {e}")

    def save_and_close(self):
        """Save changes and close the window"""
        try:
            # Clean up image references for overlays
            for data in self._overlays.values():
                data.pop('pil_img', None)
                data.pop('tk_logo', None)

            # Apply subtitle settings to main app if there are subtitle elements
            subtitle_elements = [data for data in self._overlays.values() if data.get('type') == 'subtitle']
            if subtitle_elements and hasattr(self, 'parent_app'):
                # Use the first subtitle element's settings for SRT subtitles
                subtitle_settings = subtitle_elements[0]
                self.apply_subtitle_settings_to_main_app(subtitle_settings)

                # Remove subtitle elements from overlays to avoid duplication
                # Keep only non-subtitle overlays
                self._overlays = {k: v for k, v in self._overlays.items() if v.get('type') != 'subtitle'}
                print(f"Removed {len(subtitle_elements)} subtitle elements to avoid duplication with SRT")

            self.saved = True

            # Show confirmation message
            from tkinter import messagebox
            if subtitle_elements:
                messagebox.showinfo("تم الحفظ",
                    "✅ تم حفظ جميع التغييرات بنجاح!\n"
                    "🎨 إعدادات الخط والخلفية ستطبق على ترجمة SRT.\n"
                    "📝 النصوص التجريبية تم حذفها لتجنب التداخل.")
            else:
                messagebox.showinfo("تم الحفظ", "تم حفظ جميع التغييرات بنجاح!")

            # Close the window
            self.destroy()

        except Exception as e:
            print(f"Error in save_and_close: {e}")
            from tkinter import messagebox
            messagebox.showerror("خطأ", f"حدث خطأ أثناء الحفظ: {e}")
            # Still close the window even if there's an error
            self.destroy()

    def apply_subtitle_settings_to_main_app(self, subtitle_settings):
        """Apply subtitle settings from element editor to main app SRT settings"""
        try:
            if hasattr(self, 'parent_app'):
                app = self.parent_app

                # Apply font settings
                if hasattr(app, 'subtitle_font_size_var'):
                    app.subtitle_font_size_var.set(subtitle_settings.get('font_size', 24))
                if hasattr(app, 'subtitle_font_color_var'):
                    app.subtitle_font_color_var.set(subtitle_settings.get('font_color', 'white'))
                if hasattr(app, 'subtitle_outline_color_var'):
                    app.subtitle_outline_color_var.set(subtitle_settings.get('outline_color', 'black'))
                if hasattr(app, 'subtitle_outline_width_var'):
                    app.subtitle_outline_width_var.set(subtitle_settings.get('outline_width', 2))

                # Apply background settings
                if hasattr(app, 'subtitle_background_enabled_var'):
                    app.subtitle_background_enabled_var.set(subtitle_settings.get('background_enabled', False))
                if hasattr(app, 'subtitle_background_color_var'):
                    app.subtitle_background_color_var.set(subtitle_settings.get('background_color', 'black'))
                if hasattr(app, 'subtitle_background_opacity_var'):
                    app.subtitle_background_opacity_var.set(subtitle_settings.get('background_opacity', 80))
                if hasattr(app, 'subtitle_background_padding_var'):
                    app.subtitle_background_padding_var.set(subtitle_settings.get('background_padding', 10))

                print("Applied subtitle settings to main app")

        except Exception as e:
            print(f"Error applying subtitle settings to main app: {e}")

    def get_item_at(self, x, y):
        items = self.canvas.find_overlapping(x - 2, y - 2, x + 2, y + 2)
        for priority in ['resize_handle', 'overlay']:
            for item in reversed(items):
                tags = self.canvas.gettags(item)
                if priority in tags:
                    oid = next((t for t in tags if t not in ['overlay', 'handle', 'resize_handle', 'logo', 'rect', 'circle', 'blur', 'pixelate', 'subtitle']), None)
                    return oid, tags
        return None, []

    def on_press(self, event):
        self._drag_data.clear()
        oid, tags = self.get_item_at(event.x, event.y)
        mode = self.current_mode.get()
        
        if mode == 'move':
            if self._selected_id != oid:
                self._selected_id = oid
                # Load subtitle settings if a subtitle is selected
                if oid and oid in self._subtitles:
                    self.load_subtitle_settings(oid)
                else:
                    self.update_selection_status()
                self.redraw_all()
            if oid:
                self._drag_data = {'id': oid, 'x': event.x, 'y': event.y}
                if 'resize_handle' in tags:
                    self._drag_data['mode'] = 'resize'
                    self._drag_data['orig'] = self._overlays[oid].copy()
                else:
                    self._drag_data['mode'] = 'move'
        elif mode == 'pixelate':
            self._selected_id = None
            self._drag_data = {'mode': 'draw_pixelate', 'last_point': (event.x, event.y)}
            
            self._add_pixelate_at(event.x, event.y)
        elif mode in ['rect', 'circle']:
            self._selected_id = None
            self._drag_data = {'x': event.x, 'y': event.y, 'mode': f'draw_{mode}'}
            self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="white", width=2, dash=(), tags="temp")
        elif mode == 'subtitle':
            # Add subtitle at clicked position
            text = self.subtitle_text_entry.get('1.0', tk.END).strip()
            if not text:
                messagebox.showwarning("Warning", "Please enter subtitle text first.")
                return

            subtitle_id = f"subtitle_{time.time()}"
            subtitle_data = {
                'type': 'subtitle',
                'text': text,
                'font_size': self.subtitle_font_size_var.get(),
                'font_color': self.subtitle_font_color_var.get(),
                'outline_color': self.subtitle_outline_color_var.get(),
                'outline_width': self.subtitle_outline_width_var.get(),
                'background_enabled': self.subtitle_background_enabled_var.get(),
                'background_color': self.subtitle_background_color_var.get(),
                'background_opacity': self.subtitle_background_opacity_var.get(),
                'background_padding': self.subtitle_background_padding_var.get(),
                'x': event.x - 100,  # Center around click
                'y': event.y - 25,
                'w': 200,
                'h': 50
            }

            self._overlays[subtitle_id] = subtitle_data
            self._subtitles[subtitle_id] = subtitle_data
            self._selected_id = subtitle_id
            self.load_subtitle_settings(subtitle_id)  # Load settings for the new subtitle
            self.redraw_all()

    def _add_pixelate_at(self, x, y):
        radius = self.brush_size.get() // 2
        
        x0 = max(0, min(self.disp_w - radius * 2, x - radius))
        y0 = max(0, min(self.disp_h - radius * 2, y - radius))
        w = h = radius * 2

        oid = f"pixelate_{time.time()}_{x0}_{y0}"
        self._overlays[oid] = {'type': 'pixelate', 'x': x0, 'y': y0, 'w': w, 'h': h}
        self.redraw_all()

    def on_motion(self, event):
        mode = self._drag_data.get('mode')
        if not mode: return
        
        if mode == 'draw_pixelate':
            self._add_pixelate_at(event.x, event.y)
            self._drag_data['last_point'] = (event.x, event.y)
        elif mode.startswith('draw_'):
            self.canvas.coords("temp", self._drag_data['x'], self._drag_data['y'], event.x, event.y)
        elif 'id' in self._drag_data:
            oid = self._drag_data['id']
            data = self._overlays[oid]
            if mode == 'move':
                dx, dy = event.x - self._drag_data['x'], event.y - self._drag_data['y']
                data['x'] += dx
                data['y'] += dy
                self._drag_data.update({'x': event.x, 'y': event.y})
            elif mode == 'resize':
                orig = self._drag_data['orig']
                data['w'] = max(20, orig['w'] + (event.x - (orig['x'] + orig['w'])))
                data['h'] = max(20, orig['h'] + (event.y - (orig['y'] + orig['h'])))
            self.redraw_all()

    def on_release(self, event):
        mode = self._drag_data.get('mode')
        if mode == 'draw_pixelate':
            pass  
        elif mode and mode.startswith('draw_'):
            self.canvas.delete("temp")
            x1, y1 = self._drag_data['x'], self._drag_data['y']
            oid = f"{mode.split('_')[1]}_{time.time()}"
            info = {'type': mode.split('_')[1], 'x': min(x1, event.x), 'y': min(y1, event.y), 'w': abs(x1 - event.x), 'h': abs(y1 - event.y)}
            if info['type'] in ['rect', 'circle']: info.update({'color': '#FFFF00', 'thickness': 2})
            if info['w'] > 5 and info['h'] > 5: self._overlays[oid] = info
            self.redraw_all()
        self._drag_data.clear()

class WaveformEditorWindow(tk.Toplevel):
    def __init__(self, parent, video_path, initial_chunk_duration, initial_fade_duration):
        super().__init__(parent)
        self.parent = parent
        self.video_path = video_path

        self.title("محرر الموجة الصوتية")
        self.geometry("1000x750")
        self.configure(bg=App.BG_COLOR)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        self.audio_segment = None
        self.samples = None
        self.time_axis = None
        self.initial_chunk_duration = initial_chunk_duration
        self.initial_fade_duration = initial_fade_duration
        self.preview_process = None

        self.create_widgets()
        
        self.parent.update_status("جاري تحميل بيانات الصوت للرسم البياني...")
        threading.Thread(target=self.load_audio, daemon=True).start()

    def load_audio(self):
        self.after(0, lambda: self.status_label_var.set("جاري استخراج الصوت إلى ملف مؤقت..."))
        temp_wav_file = None
        try:
            temp_dir = os.path.join(base_path, "temp_videos")
            os.makedirs(temp_dir, exist_ok=True)
            fd, temp_wav_file = tempfile.mkstemp(suffix='.wav', dir=temp_dir)
            os.close(fd)

            extract_command = [ffmpeg_exe_path, '-i', os.path.normpath(self.video_path), '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', '-y', os.path.normpath(temp_wav_file)]
            
            subprocess.run(extract_command, check=True, capture_output=True, text=True, creationflags=SUBPROCESS_CREATION_FLAGS)
            
            self.after(0, lambda: self.status_label_var.set("جاري تحميل بيانات الموجة..."))

            audio = AudioSegment.from_file(temp_wav_file)
            self.audio_segment = audio
            self.samples = np.array(audio.get_array_of_samples())
            
            duration_in_ms = len(audio)
            self.time_axis = np.linspace(0, duration_in_ms, num=len(self.samples))
            
            self.after(0, lambda: self.parent.update_status("تم تحميل الصوت بنجاح. يتم الآن رسم الموجة..."))
            self.after(0, self.initial_plot)
        except Exception as e:
            error_message = f"خطأ في تحميل الصوت: {e}"
            if isinstance(e, subprocess.CalledProcessError):
                error_message += f"\nFFmpeg stderr: {e.stderr}"
            self.after(0, lambda: self.parent.update_status(error_message))
            self.after(0, lambda: messagebox.showerror("خطأ", error_message, parent=self))
            self.after(0, self.destroy)
        finally:
            if temp_wav_file and os.path.exists(temp_wav_file):
                try: os.remove(temp_wav_file)
                except OSError as ex: self.after(0, lambda: self.parent.update_status(f"فشل حذف الملف المؤقت: {ex}"))

    def create_widgets(self):
        main_frame = ttk.Frame(self, style="TFrame")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        plot_frame = ttk.Frame(main_frame, style="TFrame", relief="solid", borderwidth=1)
        plot_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fig, self.ax = plt.subplots(facecolor=App.FRAME_COLOR)
        self.ax.set_facecolor(App.ENTRY_BG_COLOR)
        self.ax.tick_params(colors=App.TEXT_COLOR)
        [spine.set_color(App.TEXT_COLOR) for spine in self.ax.spines.values()]
        self.ax.set_xlabel("الزمن (مللي ثانية)", color=App.TEXT_COLOR)
        self.ax.set_ylabel("السعة", color=App.TEXT_COLOR)
        self.fig.tight_layout()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.status_label_var = tk.StringVar(value="جاري تحميل الصوت...")
        ttk.Label(plot_frame, textvariable=self.status_label_var, anchor="center").pack(fill=tk.X)
        
        controls_frame = ttk.LabelFrame(main_frame, text="التحكم في التأثيرات", padding=10)
        controls_frame.pack(fill=tk.X, pady=10)
        controls_frame.columnconfigure(1, weight=1)

        self.chunk_duration_var = tk.IntVar(value=self.initial_chunk_duration)
        ttk.Label(controls_frame, text="مدة موجة الصوت:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Scale(controls_frame, from_=100, to=5000, orient=tk.HORIZONTAL, variable=self.chunk_duration_var, command=self._update_controls).grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        ttk.Entry(controls_frame, textvariable=self.chunk_duration_var, width=8, justify='right').grid(row=0, column=2, padx=5, pady=5)
        ttk.Label(controls_frame, text="ms").grid(row=0, column=3, sticky="w")

        self.fade_duration_var = tk.IntVar(value=self.initial_fade_duration)
        ttk.Label(controls_frame, text="مدة تلاشي الموجة:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.fade_slider = ttk.Scale(controls_frame, from_=0, to=self.initial_chunk_duration/2, orient=tk.HORIZONTAL, variable=self.fade_duration_var, command=self._update_controls)
        self.fade_slider.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        ttk.Entry(controls_frame, textvariable=self.fade_duration_var, width=8, justify='right').grid(row=1, column=2, padx=5, pady=5)
        ttk.Label(controls_frame, text="ms").grid(row=1, column=3, sticky="w")

        buttons_frame = ttk.Frame(main_frame, style="TFrame")
        buttons_frame.pack(fill=tk.X, pady=5)
        
        self.preview_button = ttk.Button(buttons_frame, text="تجربة الصوت", command=self.toggle_preview)
        self.preview_button.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        ttk.Button(buttons_frame, text="تطبيق وإغلاق", command=self.apply_and_close).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)
        ttk.Button(buttons_frame, text="إلغاء", command=self.on_close).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

    def initial_plot(self):
        if self.samples is None:
            self.status_label_var.set("فشل تحميل الصوت.")
            return

        self.status_label_var.set("جاري رسم شكل الموجة...")
        self.update() 
        
        max_points = 50000 
        step = max(1, len(self.samples) // max_points)
        plot_samples = self.samples[::step]
        plot_time = self.time_axis[::step]

        self.ax.plot(plot_time, plot_samples, color=App.BUTTON_COLOR, linewidth=0.5)
        self.ax.set_xlim(0, self.time_axis[-1])
        self.status_label_var.set("اكتمل الرسم. يمكنك الآن تعديل القيم.")
        self.update_visuals()

    def _update_controls(self, _=None):
        try:
            chunk_val = self.chunk_duration_var.get()
            fade_val = self.fade_duration_var.get()
        except tk.TclError:
            return

        new_fade_max = chunk_val / 2
        self.fade_slider.config(to=new_fade_max)
        if fade_val > new_fade_max: self.fade_duration_var.set(int(new_fade_max))

        self.update_visuals()

    def update_visuals(self):
        if self.audio_segment is None: return
        self.canvas.draw_idle()
    
    def apply_and_close(self):
        if self.preview_process: self.toggle_preview()
        chunk_duration = self.chunk_duration_var.get()
        fade_duration = self.fade_duration_var.get()
        self.parent.entries["wave_chunk_duration"].set(str(chunk_duration))
        self.parent.entries["wave_fade"].set(str(fade_duration))
        self.parent.update_status(f"تم تحديث قيم الموجة إلى: مدة {chunk_duration}ms، تلاشي {fade_duration}ms.")
        self.destroy()

    def on_close(self):
        if self.preview_process: self.toggle_preview()
        self.destroy()

    def toggle_preview(self):
        if self.preview_process and self.preview_process.poll() is None:
            self.preview_process.terminate()
            self.preview_process = None
            self.preview_button.config(text="تجربة الصوت")
        else:
            if self.audio_segment:
                self.preview_button.config(state=tk.DISABLED, text="جاري التحضير...")
                threading.Thread(target=self.run_preview_thread, daemon=True).start()

    def run_preview_thread(self):
        temp_preview_file = None
        try:
            chunk_duration = self.chunk_duration_var.get()
            fade_duration = self.fade_duration_var.get()
            
            chunks_data = [(self.audio_segment[i:i+chunk_duration], fade_duration, i==0, i+chunk_duration >= len(self.audio_segment)) for i in range(0, len(self.audio_segment), chunk_duration)]
            processed_audio = sum(process_audio_chunk_parallel(d) for d in chunks_data)

            temp_dir = os.path.join(base_path, "temp_videos")
            fd, temp_preview_file = tempfile.mkstemp(suffix='.wav', dir=temp_dir)
            os.close(fd)
            processed_audio.export(temp_preview_file, format="wav")

            self.preview_process = Popen([ffplay_exe_path, "-nodisp", "-autoexit", os.path.normpath(temp_preview_file)], creationflags=SUBPROCESS_CREATION_FLAGS)
            self.after(0, lambda: self.preview_button.config(text="إيقاف التجربة"))
            self.preview_process.wait()

        except Exception as e:
            self.after(0, lambda: messagebox.showerror("خطأ في التجربة", f"فشل تشغيل الصوت: {e}", parent=self))
        finally:
            self.after(0, lambda: self.preview_button.config(state=tk.NORMAL, text="تجربة الصوت"))
            if temp_preview_file and os.path.exists(temp_preview_file):
                try: os.remove(temp_preview_file)
                except OSError: pass



if __name__ == "__main__":
    if not os.path.exists(os.path.join(base_path, "temp_videos")):
        os.makedirs(os.path.join(base_path, "temp_videos"))
    app = App()
    app.mainloop()
