# -*- coding: utf-8 -*-
import os
import sys
import shutil
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# --- FFmpeg & System Configuration (Early Setup) ---
import imageio_ffmpeg
from pydub import AudioSegment

try:
    base_path = os.path.dirname(os.path.abspath(__file__))
except NameError:
    base_path = os.path.abspath(".")

def find_executable(name):
    """Finds an executable using imageio-ffmpeg, system PATH, or local folders."""
    # 1. Try imageio-ffmpeg first
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
        pass  # Fallback if imageio-ffmpeg fails

    # 2. Try shutil.which (searches system PATH)
    executable = shutil.which(name)
    if executable:
        return os.path.normpath(executable)

    # 3. Check common local project folders if not found in PATH
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

# --- End of Early Setup ---

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

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    MATPLOTLIB_INSTALLED = True
except ImportError:
    MATPLOTLIB_INSTALLED = False

SUBPROCESS_CREATION_FLAGS = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0

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
        label = tk.Label(self.tooltip_window, text=self.text, justify=tk.LEFT, background="#FFFFE0", relief=tk.SOLID, borderwidth=1, font=("SegoeUI", "9"), wraplength=250, fg="#000000", padx=4, pady=4)
        label.pack(ipadx=1)

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
        self.tooltip_window = None

def process_frame_batch(frame_batch, settings, new_width, new_height, original_width, original_height, other_overlays=None):
    processed_frames = []
    
    for frame in frame_batch:
        cropped = frame[settings['crop_top']:original_height - settings['crop_bottom'], settings['crop_left']:original_width - settings['crop_right']]
        
        mirrored = cv2.flip(cropped, 1) if settings.get('mirror_enabled', True) else cropped
            
        final_frame = cv2.convertScaleAbs(mirrored, alpha=settings['contrast'], beta=(settings['brightness'] - 1) * 100)
        
        if other_overlays:
            for overlay in other_overlays:
                try:
                    o_type = overlay.get('type')
                    x, y, w, h = overlay['x'], overlay['y'], overlay['w'], overlay['h']
                    
                    if w <= 0 or h <= 0: continue

                    if o_type == 'logo':
                        logo_data, oy, ox = overlay['data'], y, x
                        oh, ow = logo_data.shape[:2]
                        
                        y1, x1 = max(0, oy), max(0, ox)
                        y2, x2 = min(new_height, oy + oh), min(new_width, ox + ow)
                        
                        if y1 < y2 and x1 < x2:
                            logo_y1, logo_x1 = y1 - oy, x1 - ox
                            logo_y2, logo_x2 = logo_y1 + (y2 - y1), logo_x1 + (x2 - x1)
                            
                            if logo_data.shape[2] == 4:
                                alpha_s = logo_data[logo_y1:logo_y2, logo_x1:logo_x2, 3] / 255.0
                                alpha_l = 1.0 - alpha_s
                                for c in range(3):
                                    final_frame[y1:y2, x1:x2, c] = (alpha_s * logo_data[logo_y1:logo_y2, logo_x1:logo_x2, c] + alpha_l * final_frame[y1:y2, x1:x2, c])
                            else:
                                final_frame[y1:y2, x1:x2] = logo_data[logo_y1:logo_y2, logo_x1:logo_x2]
                    
                    elif o_type == 'blur':
                        roi = final_frame[y:y+h, x:x+w]
                        # Kernel size must be odd
                        ksize = (max(1, w // 4) | 1, max(1, h // 4) | 1)
                        final_frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, ksize, 0)
                    
                    elif o_type == 'pixelate':
                        # حجم البكسل (كلما زاد الرقم زاد التشويش)
                        pixel_size = 20
                        # استخراج منطقة الاهتمام (ROI)
                        x, y, w, h = overlay['x'], overlay['y'], overlay['w'], overlay['h']
                        roi = final_frame[y:y+h, x:x+w]
                        if roi.size == 0: continue
                        
                        # إنشاء قناع دائري
                        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                        cv2.circle(mask, (w//2, h//2), w//2, 255, -1)

                        # تصغير الصورة لإنشاء تأثير البكسلة
                        small_roi = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
                        pixelated_roi = cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)

                        # دمج المنطقة المبكسلة مع الأصلية باستخدام القناع
                        final_frame[y:y+h, x:x+w] = np.where(mask[..., None].astype(bool), pixelated_roi, roi)

                    elif o_type in ['rect', 'circle']:
                        color_hex = overlay.get('color', '#FFFF00').lstrip('#')
                        color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0)) # BGR for OpenCV
                        thickness = overlay.get('thickness', 2)
                        if o_type == 'rect':
                            cv2.rectangle(final_frame, (x, y), (x+w, y+h), color_bgr, thickness)
                        else: # circle
                            cv2.ellipse(final_frame, (x + w//2, y + h//2), (w//2, h//2), 0, 0, 360, color_bgr, thickness)

                except Exception as e:
                    print(f"Error applying overlay: {e}")

        x_mask = np.zeros((new_height, new_width), dtype=np.uint8)
        cv2.line(x_mask, (0, 0), (new_width, new_height), 255, int(settings['x_thickness']))
        cv2.line(x_mask, (new_width, 0), (0, new_height), 255, int(settings['x_thickness']))
        final_frame[x_mask > 0] = np.clip(final_frame[x_mask > 0].astype(np.int16) + int(settings['x_lighten']), 0, 255).astype(np.uint8)
        
        processed_frames.append(final_frame)
    
    return processed_frames

def process_audio_chunk_parallel(chunk_data):
    chunk, wave_fade, is_first, is_last = chunk_data
    
    if is_first:
        chunk = chunk.fade_out(wave_fade)
    elif is_last:
        chunk = chunk.fade_in(wave_fade)
    else:
        chunk = chunk.fade_in(wave_fade).fade_out(wave_fade)
    
    return chunk

def optimize_memory_usage():
    import gc
    gc.collect()  
    
def get_optimal_batch_size(frame_count, available_memory_gb=64):
    cpu_count = multiprocessing.cpu_count()
    estimated_frame_memory_mb = 5  # تقدير حجم إطار واحد في الذاكرة
    available_memory_mb = available_memory_gb * 1024
    # استخدم 80% من الذاكرة المتاحة
    max_frames_in_memory = int(available_memory_mb * 0.8 / estimated_frame_memory_mb)
    # لا تتجاوز عدد الإطارات الكلي
    optimal_batch_size = min(max_frames_in_memory, max(cpu_count * 4, 32), frame_count)
    return max(1, optimal_batch_size)

def process_video_chunk(chunk_settings, status_callback=None, status_queue=None):
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
                        overlays_to_apply.append(prep_info)
                    except Exception as e: send_status(f"Warning: Could not prepare overlay: {e}")
        
        temp_video_fd, temp_video_file_for_chunk = tempfile.mkstemp(suffix=f'_chunk{chunk_index}.mp4', dir=os.path.join(base_path, "temp_videos"))
        os.close(temp_video_fd)
        out = cv2.VideoWriter(temp_video_file_for_chunk, cv2.VideoWriter_fourcc(*'mp4v'), original_fps, (new_width, new_height))
        
        batch_size = get_optimal_batch_size(frame_count)
        processed_count = 0
        cpu_cores = multiprocessing.cpu_count()
        while processed_count < frame_count:
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            if not frames: break
            # --- تعديل: استخدم ProcessPoolExecutor بدلاً من ThreadPoolExecutor ---
            if chunk_settings.get('frame_parallel', False):
                from concurrent.futures import ProcessPoolExecutor
                import pickle
                def process_one_frame_pickled(frame_bytes):
                    frame = pickle.loads(frame_bytes)
                    return process_frame_batch([frame], chunk_settings, new_width, new_height, original_width, original_height, overlays_to_apply)[0]
                with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
                    # يجب تمرير البيانات بشكل قابل للنقل بين العمليات (pickle)
                    frames_bytes = [pickle.dumps(f) for f in frames]
                    processed_batch = list(executor.map(process_one_frame_pickled, frames_bytes))
            else:
                processed_batch = process_frame_batch(frames, chunk_settings, new_width, new_height, original_width, original_height, overlays_to_apply)
            for p_frame in processed_batch: out.write(p_frame)
            processed_count += len(frames)
            send_status(f"الجزء {chunk_index + 1}: تمت معالجة {processed_count}/{frame_count} إطار", progress=(processed_count / frame_count) * 100)
        
        cap.release()
        out.release()
        
        send_status(f"الجزء {chunk_index + 1}: دمج الصوت والفيديو...")
        command = [
            ffmpeg_exe_path, '-i', os.path.normpath(temp_video_file_for_chunk), '-i', os.path.normpath(temp_audio_file_for_chunk),
            '-c:v', 'libx264', '-c:a', 'aac', '-b:a', '192k', '-r', str(original_fps), '-vsync', 'cfr',
            '-filter:v', f"setpts={1/chunk_settings['speed_factor']}*PTS", '-filter:a', f"atempo={chunk_settings['speed_factor']}",
            '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-y', os.path.normpath(output_path)
        ]
        if chunk_settings.get('compression_enabled', False):
            crf = {"عالية": "18", "متوسطة": "23", "منخفضة": "28"}.get(chunk_settings.get('compression_quality', 'متوسطة'), '23')
            preset = {"سريعة": "veryfast", "متوسطة": "medium", "بطيئة": "slow"}.get(chunk_settings.get('compression_speed', 'متوسطة'), 'medium')
            command.extend(['-crf', crf, '-preset', preset])
            # --- جديد: فلتر الدقة ---
            res = chunk_settings.get('compression_resolution', 'أصلي')
            scale_map = {
                "140p": 140, "240p": 240, "360p": 360, "480p": 480, "720p": 720, "1080p": 1080
            }
            if res in scale_map:
                command.extend(['-vf', f'scale=-2:{scale_map[res]}'])
        
        process = subprocess.run(command, capture_output=True, text=True, creationflags=SUBPROCESS_CREATION_FLAGS)
        if process.returncode != 0:
            send_status(f"الجزء {chunk_index + 1}: خطأ FFmpeg: {process.stderr}")
            return None
        return output_path
    except Exception as e:
        send_status(f"الجزء {chunk_index + 1}: خطأ غير متوقع: {e}")
        return None
    finally:
        for f in [temp_audio_file_for_chunk, temp_video_file_for_chunk]:
            if f and os.path.exists(f): os.remove(f)

def process_video_core(settings, status_callback):
    original_input_path = settings['input_path']
    original_output_path = settings['output_path']
    temp_input_video_path_main_copy = None
    created_temp_files = []
    import time
    start_time = time.time()

    try:
        
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
                messagebox.showerror("خطأ", "ملف الفيديو المؤقت فارغ (0 بايت).")
                return
        except Exception as e:
            status_callback(f"خطأ في إنشاء الملف المؤقت: {e}")
            messagebox.showerror("خطأ", f"لا يمكن إنشاء نسخة مؤقتة من ملف الإدخال.\nالخطأ: {e}")
            return

        if not ffmpeg_exe_path or not ffprobe_exe_path:
            error_msg = (
                f"لم يتم العثور على FFmpeg أو FFprobe.\n"
                "الرجاء تشغيل سكربت الإعداد (setup) أو التأكد من وجودها في مسار النظام (PATH)."
            )
            status_callback(error_msg)
            messagebox.showerror("خطأ في FFmpeg", error_msg)
            return

        
        if settings.get('enable_chunking', False) and float(settings.get('chunk_size_seconds', 0)) > 0:
            status_callback("[تقسيم] تقسيم الفيديو الأصلي إلى أجزاء...")
            chunk_duration_seconds = float(settings['chunk_size_seconds']) * 60

            
            probe_command = [ffprobe_exe_path, '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', temp_input_video_path_main_copy]
            try:
                result = subprocess.run(probe_command, capture_output=True, text=True, check=True, creationflags=SUBPROCESS_CREATION_FLAGS)
                video_duration = float(result.stdout.strip())
            except (subprocess.CalledProcessError, ValueError) as e:
                status_callback(f"خطأ في الحصول على مدة الفيديو: {e}")
                messagebox.showerror("خطأ", f"لا يمكن تحديد مدة الفيديو للتقسيم.\n{e}")
                return

            num_chunks = int(np.ceil(video_duration / chunk_duration_seconds))
            if num_chunks <= 0: num_chunks = 1
            status_callback(f"[تقسيم] سيتم تقسيم الفيديو إلى {num_chunks} جزء (أجزاء).", progress=1)

            chunk_input_files = []
            output_dir = os.path.dirname(original_output_path)
            base_output_name, output_ext = os.path.splitext(os.path.basename(original_output_path))

            for i in range(num_chunks):
                start_time_chunk = i * chunk_duration_seconds
                chunk_output_name = os.path.join(temp_dir, f"temp_chunk_{i}{output_ext}")

                split_command = [
                    ffmpeg_exe_path,
                    '-ss', str(start_time_chunk),
                    '-i', temp_input_video_path_main_copy,
                    '-t', str(chunk_duration_seconds),
                    '-c:v', 'libx264',
                    '-c:a', 'aac',
                    '-strict', '-2',
                    '-y', chunk_output_name
                ]
                status_callback(f"[تقسيم] إنشاء الجزء {i+1}/{num_chunks}: {os.path.basename(chunk_output_name)}")
                try:
                    subprocess.run(split_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=SUBPROCESS_CREATION_FLAGS)
                    chunk_input_files.append(chunk_output_name)
                    created_temp_files.append(chunk_output_name)
                except subprocess.CalledProcessError as e:
                    status_callback(f"خطأ في تقسيم الجزء {i+1}: {e.stderr.decode()}")
                    messagebox.showerror("خطأ في التقسيم", f"فشل تقسيم الفيديو إلى أجزاء.\n{e.stderr.decode()}")
                    return

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
                import threading
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
                        chunk_settings['input_path'] = chunk_file_path
                        chunk_settings['output_path'] = chunk_specific_output_path
                        chunk_settings['chunk_index'] = i
                        chunk_settings['total_chunks'] = num_chunks
                        futures[executor.submit(process_video_chunk, chunk_settings, None, status_queue)] = i
                    for future in as_completed(futures):
                        i = futures[future]
                        try:
                            result_path = future.result()
                            if result_path:
                                processed_chunk_paths[i] = result_path
                                current_progress = 10 + ((i + 1) / num_chunks) * 80
                                status_callback(f"[تفرعي] انتهى الجزء {i+1}: {os.path.basename(result_path)}", progress=current_progress)
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
                    chunk_specific_output_path = os.path.join(output_dir, f"{base_output_name}_part_{i+1}{output_ext}")
                    chunk_settings = settings.copy()
                    chunk_settings['input_path'] = chunk_file_path
                    chunk_settings['output_path'] = chunk_specific_output_path
                    chunk_settings['chunk_index'] = i
                    chunk_settings['total_chunks'] = num_chunks
                    if settings.get('processing_mode', 'parallel') == 'parallel' and parallel_level == "تفرع على مستوى الإطارات داخل الجزء (Frames in Chunk)":
                        chunk_settings['frame_parallel'] = True
                    else:
                        chunk_settings['frame_parallel'] = False
                    status_callback(f"[تسلسلي] بدء معالجة الجزء {i+1}/{num_chunks}: {os.path.basename(chunk_file_path)}")
                    try:
                        result_path = process_video_chunk(chunk_settings, status_callback)
                        if result_path:
                            processed_chunk_paths[i] = result_path
                            current_progress = 10 + ((i + 1) / num_chunks) * 80
                            status_callback(f"[تسلسلي] انتهى الجزء {i+1}: {os.path.basename(result_path)}", progress=current_progress)
                        else:
                            status_callback(f"فشلت معالجة الجزء {i+1} (تسلسلي).")
                    except Exception as exc:
                        status_callback(f"حدث خطأ أثناء معالجة الجزء {i+1} (تسلسلي): {exc}")

            
            if any(processed_chunk_paths):
                status_callback("[دمج] بدء دمج جميع الأجزاء النهائية...")
                concat_list_path = os.path.join(temp_dir, "concat_list.txt")
                with open(concat_list_path, 'w', encoding='utf-8') as f:
                    for part_path in processed_chunk_paths:
                        if part_path:
                            f.write(f"file '{os.path.abspath(part_path)}'\n")
                merged_output_path = original_output_path
                merge_command = [
                    ffmpeg_exe_path,
                    '-f', 'concat',
                    '-safe', '0',
                    '-i', concat_list_path,
                    '-c', 'copy',
                    '-y', merged_output_path
                ]
                try:
                    subprocess.run(merge_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=SUBPROCESS_CREATION_FLAGS)
                    status_callback(f"[دمج] تم دمج جميع الأجزاء بنجاح في: {merged_output_path}", progress=100)
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    minutes = int(total_time // 60)
                    seconds = int(total_time % 60)
                    time_str = f"{minutes}:{seconds:02d}" if minutes > 0 else f"{seconds} ثانية"
                    import tkinter.messagebox as messagebox
                    messagebox.showinfo("نجاح", f"🎉 اكتملت معالجة الفيديو بنجاح!\n📁 المسار: {merged_output_path}\n⏱️ الوقت المستغرق: {time_str}")
                except subprocess.CalledProcessError as e:
                    status_callback(f"[دمج] فشل دمج الأجزاء: {e.stderr.decode()}")
                    messagebox.showerror("خطأ في الدمج", f"فشل دمج الأجزاء.\n{e.stderr.decode()}")
            else:
                status_callback("لم يتم إنتاج أي أجزاء معالجة.")

        else: 
            cpu_cores = multiprocessing.cpu_count()
            status_callback("بدء المعالجة القياسية (بدون تقسيم)...")
            
            status_callback("معالجة الصوت: تحميل الملف...")
            try:
                audio = AudioSegment.from_file(os.path.normpath(temp_input_video_path_main_copy), format=os.path.splitext(temp_input_video_path_main_copy)[1][1:], ffmpeg=ffmpeg_exe_path)
            except Exception as e:
                status_callback(f"حدث خطأ أثناء تحميل الصوت: {e}")
                messagebox.showerror("خطأ في الصوت", f"فشل تحميل ملف الصوت من الفيديو.\nالخطأ: {e}")
                return

            status_callback("معالجة الصوت: تطبيق تأثير 'الموجة'...")
            audio_chunks_data = []
            for i in range(0, len(audio), settings['wave_chunk_duration']):
                chunk = audio[i:i + settings['wave_chunk_duration']]
                is_first = (i == 0)
                is_last = (i + settings['wave_chunk_duration'] >= len(audio))
                audio_chunks_data.append((chunk, settings['wave_fade'], is_first, is_last))

            processed_audio = AudioSegment.empty()
            if settings.get('processing_mode', 'parallel'):
                
                with ThreadPoolExecutor(max_workers=min(cpu_cores, len(audio_chunks_data) if audio_chunks_data else 1)) as executor:
                    future_to_index = {executor.submit(process_audio_chunk_parallel, chunk_data): i
                                     for i, chunk_data in enumerate(audio_chunks_data)}
                    results = [None] * len(audio_chunks_data)
                    completed_audio_chunks = 0
                    for future in as_completed(future_to_index):
                        index = future_to_index[future]
                        results[index] = future.result()
                        completed_audio_chunks +=1
                        progress = (completed_audio_chunks / len(audio_chunks_data)) * 30
                        status_callback(f"معالجة الصوت (متوازي): {completed_audio_chunks}/{len(audio_chunks_data)} مقطع", progress)
                for processed_chunk_audio in results:
                    if processed_chunk_audio: processed_audio += processed_chunk_audio
            else: 
                status_callback("معالجة الصوت (تسلسلي)...")
                completed_audio_chunks = 0
                for chunk_data in audio_chunks_data:
                    processed_chunk_audio = process_audio_chunk_parallel(chunk_data)
                    if processed_chunk_audio: processed_audio += processed_chunk_audio
                    completed_audio_chunks +=1
                    progress = (completed_audio_chunks / len(audio_chunks_data)) * 30
                    status_callback(f"معالجة الصوت (تسلسلي): {completed_audio_chunks}/{len(audio_chunks_data)} مقطع", progress)

            status_callback("معالجة الصوت: تعديل السرعة...")
            new_audio_speed = processed_audio.speedup(playback_speed=settings['speed_factor'])

            temp_audio_fd, temp_audio_file = tempfile.mkstemp(suffix='.aac', dir=temp_dir)
            os.close(temp_audio_fd)
            created_temp_files.append(temp_audio_file)
            status_callback("معالجة الصوت: تصدير الملف المؤقت...")
            new_audio_speed.export(temp_audio_file, format="adts")


            
            status_callback("معالجة الفيديو: فتح بث الفيديو...")
            cap = cv2.VideoCapture(os.path.normpath(temp_input_video_path_main_copy))
            original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            new_width = original_width - int(settings['crop_left']) - int(settings['crop_right'])
            new_height = original_height - int(settings['crop_top']) - int(settings['crop_bottom'])

            # Load and prepare ALL overlays once for the whole video
            overlays_to_apply = []
            if 'overlays' in settings and settings['overlays']:
                from PIL import Image
                preview_dims = settings.get('logo_preview_dimensions')

                for overlay_info in settings['overlays']:
                    try:
                        # Calculate scaled dimensions for all overlay types
                        final_x, final_y, final_w, final_h = overlay_info['x'], overlay_info['y'], overlay_info['w'], overlay_info['h']
                        if preview_dims:
                            scale_w = new_width / preview_dims['w']
                            scale_h = new_height / preview_dims['h']
                            final_x = int(overlay_info['x'] * scale_w)
                            final_y = int(overlay_info['y'] * scale_h)
                            final_w = int(overlay_info['w'] * scale_w)
                            final_h = int(overlay_info['h'] * scale_h)

                        # Prepare data based on type
                        prepared_overlay = overlay_info.copy()
                        prepared_overlay.update({'x': final_x, 'y': final_y, 'w': final_w, 'h': final_h})

                        if overlay_info['type'] == 'logo' and os.path.exists(overlay_info['path']):
                            if final_w > 0 and final_h > 0:
                                logo_img_pil = Image.open(overlay_info['path']).convert('RGBA')
                                resample_mode = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
                                resized_logo = logo_img_pil.resize((final_w, final_h), resample_mode)
                                prepared_overlay['data'] = np.array(resized_logo)
                                overlays_to_apply.append(prepared_overlay)
                        else:
                            overlays_to_apply.append(prepared_overlay)

                    except Exception as e:
                        status_callback(f"Warning: Could not load/prepare overlay: {e}")

            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            temp_video_fd, temp_video_file_path = tempfile.mkstemp(suffix='.mp4', dir=temp_dir)
            os.close(temp_video_fd)
            created_temp_files.append(temp_video_file_path)
            out = cv2.VideoWriter(temp_video_file_path, fourcc, original_fps, (new_width, new_height))

            batch_size = get_optimal_batch_size(frame_count)
            status_callback(f"معالجة الفيديو: استخدام المعالجة المتوازية بـ {cpu_cores} معالج وحجم مجموعة {batch_size} إطار...")
            optimize_memory_usage()
            processed_frames_count = 0

            while processed_frames_count < frame_count:
                frame_batch = []
                for _ in range(batch_size):
                    if processed_frames_count >= frame_count: break
                    ret, frame = cap.read()
                    if not ret: break
                    frame_batch.append(frame)
                    processed_frames_count += 1
                if not frame_batch: break
                
                if settings.get('processing_mode', 'parallel'):
                    
                    with ThreadPoolExecutor(max_workers=min(cpu_cores, len(frame_batch) if frame_batch else 1)) as executor:
                        process_func = partial(process_frame_batch, settings=settings,
                                             new_width=new_width, new_height=new_height,
                                             original_width=original_width, original_height=original_height,
                                             other_overlays=overlays_to_apply)

                        
                        processed_batch_frames_list = list(executor.map(process_func, [[f] for f in frame_batch])) 
                        for processed_single_frame_list in processed_batch_frames_list:
                            if processed_single_frame_list: 
                                 out.write(processed_single_frame_list[0])
                else: 
                    process_func = partial(process_frame_batch, settings=settings,
                                         new_width=new_width, new_height=new_height,
                                         original_width=original_width, original_height=original_height,
                                         other_overlays=overlays_to_apply)
                    for frame_to_process in frame_batch:
                        processed_single_frame_list = process_func([frame_to_process]) 
                        if processed_single_frame_list: 
                            out.write(processed_single_frame_list[0])

                progress = 30 + (processed_frames_count / frame_count) * 60
                status_callback(f"معالجة الفيديو ({settings.get('processing_mode', 'parallel')}): {processed_frames_count}/{frame_count} إطار", progress)
                if processed_frames_count % 100 == 0: optimize_memory_usage()

            cap.release()
            out.release()

            
            status_callback("دمج الصوت والفيديو باستخدام FFmpeg المحسن...", progress=90)
            command = [
                ffmpeg_exe_path, '-threads', str(cpu_cores),
                '-i', os.path.normpath(temp_video_file_path),
                '-i', os.path.normpath(temp_audio_file),
                '-c:v', 'libx264', '-c:a', 'aac', '-b:a', '192k',
                '-r', str(original_fps), '-vsync', 'cfr',
                '-filter:v', f"setpts={1/settings['speed_factor']}*PTS",
                '-filter:a', f"atempo={settings['speed_factor']}",
            ]
            if settings.get('compression_enabled', False):
                quality_map = {"عالية": "18", "متوسطة": "23", "منخفضة": "28"}
                crf = quality_map.get(settings.get('compression_quality', 'متوسطة'), '23')
                speed_map = {"سريعة": "veryfast", "متوسطة": "medium", "بطيئة": "slow"}
                preset = speed_map.get(settings.get('compression_speed', 'متوسطة'), 'medium')
                command.extend(['-crf', crf, '-preset', preset, '-tune', 'film'])
                # --- جديد: فلتر الدقة ---
                res = settings.get('compression_resolution', 'أصلي')
                scale_map = {
                    "140p": 140, "240p": 240, "360p": 360, "480p": 480, "720p": 720, "1080p": 1080
                }
                if res in scale_map:
                    command.extend(['-vf', f'scale=-2:{scale_map[res]}'])
            else:
                command.extend(['-b:v', '2000k', '-maxrate', '2500k', '-bufsize', '5000k'])
            command.extend([
                '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.0',
                '-movflags', '+faststart', '-y', os.path.normpath(original_output_path)
            ])

            status_callback("تشغيل FFmpeg للدمج النهائي...")
            try:
                process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=SUBPROCESS_CREATION_FLAGS)
                stdout, stderr = process.communicate()
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, command, stderr)
            except subprocess.CalledProcessError as e:
                status_callback(f"خطأ في FFmpeg: {e.stderr}")
                messagebox.showerror("خطأ FFmpeg", f"فشل دمج الفيديو والصوت.\n{e.stderr}")
                return

            end_time = time.time()
            total_time = end_time - start_time
            
            minutes = int(total_time // 60)
            seconds = int(total_time % 60)
            time_str = f"{minutes}:{seconds:02d}" if minutes > 0 else f"{seconds} ثانية"
            status_callback(f"اكتملت المعالجة بنجاح! تم حفظ الناتج في: {original_output_path}", 100)
            import tkinter.messagebox as messagebox
            messagebox.showinfo("نجاح", f"🎉 اكتملت معالجة الفيديو بنجاح!\n📁 المسار: {original_output_path}\n⏱️ الوقت المستغرق: {time_str}")

        
        if all(processed_chunk_paths):
            try:
                if os.path.exists(resume_state_file):
                    os.remove(resume_state_file)
            except Exception:
                pass

    except Exception as e:
        error_message = f"حدث خطأ رئيسي: {e}\n{type(e)}"
        if isinstance(e, subprocess.CalledProcessError):
            error_message += f"\nFFmpeg stderr: {e.stderr if hasattr(e, 'stderr') and e.stderr else 'N/A'}"
        status_callback(error_message)
        
    finally:
        status_callback("تنظيف الملفات المؤقتة...")
        deleted_files = set()
        for temp_f_path in created_temp_files:
            abs_temp_f_path = os.path.abspath(temp_f_path)
            if abs_temp_f_path in deleted_files:
                continue
            if os.path.exists(abs_temp_f_path):
                try:
                    os.remove(abs_temp_f_path)
                    status_callback(f"تم حذف الملف المؤقت: {abs_temp_f_path}")
                except Exception as e_clean:
                    status_callback(f"فشل حذف الملف المؤقت {abs_temp_f_path}: {e_clean}")
            else:
                status_callback(f"الملف المؤقت لم يتم العثور عليه للتنظيف: {abs_temp_f_path}")
            deleted_files.add(abs_temp_f_path)


class App(tk.Tk):
    
    BG_COLOR = "#2E2E2E"
    FRAME_COLOR = "#3C3C3C"
    TEXT_COLOR = "#F0F0F0"
    ENTRY_BG_COLOR = "#4A4A4A"
    BUTTON_COLOR = "#007ACC"
    BUTTON_ACTIVE_COLOR = "#005F9E"
    VIEW_BUTTON_BG = "#4A4A4A"
    TOOLTIP_BG = "#FFFFE0"
    TOOLTIP_FG = "#000000"

    def __init__(self):
        super().__init__()
        self.title("محرر الفيديو")
        self.geometry("900x800") 
        self.configure(bg=self.BG_COLOR)

        self.settings = {}
        self.default_values = {
            "crop_top": 10, "crop_bottom": 10, "crop_left": 10, "crop_right": 10,
            "brightness": 1.1, "contrast": 1.2, "speed_factor": 1.01,
            "logo_scale": 0.1, "wave_chunk_duration": 1500, "wave_fade": 400,
            "x_thickness": 50, "x_lighten": 60,
            "mirror_enabled": True,
            "compression_enabled": False,
            "compression_quality": "متوسطة",
            "compression_speed": "متوسط",
            "compression_resolution": "أصلي",  # <--- جديد
            "enable_chunking": False,
            "chunk_size_seconds": 60, 
            "merge_chunks_after_processing": True, 
            "crossfade_duration": 1, 
            "processing_mode": "parallel", 
            "parallel_level": "chunks" 
        }
        self.settings_file = os.path.join(base_path, "settings.json")
        self.processed_chunk_files = [] 
        
        self.processing_mode_var = tk.StringVar(value=self.default_values["processing_mode"])
        self.parallel_level_var = tk.StringVar(value=self.default_values["parallel_level"])

        style = ttk.Style(self)
        self.option_add("*Font", "SegoeUI 10")
        
        style.theme_use('clam')
        style.configure(".", background=self.BG_COLOR, foreground=self.TEXT_COLOR, bordercolor=self.FRAME_COLOR)
        style.configure("TFrame", background=self.FRAME_COLOR)
        style.configure("TLabel", background=self.FRAME_COLOR, foreground=self.TEXT_COLOR, padding=5)
        style.configure("TButton", background=self.BUTTON_COLOR, foreground="white", padding=8, relief="flat", font=("SegoeUI", 10, "bold"), borderwidth=0)
        style.map("TButton", background=[('active', self.BUTTON_ACTIVE_COLOR)], relief=[('pressed', 'sunken')])
        style.configure("TEntry", fieldbackground=self.ENTRY_BG_COLOR, foreground=self.TEXT_COLOR, insertcolor=self.TEXT_COLOR, borderwidth=1, relief="solid")
        style.configure("TLabelFrame", background=self.FRAME_COLOR)
        style.configure("TLabelFrame.Label", background=self.FRAME_COLOR, foreground=self.TEXT_COLOR, font=("SegoeUI", 11, "bold"))
        style.configure("Vertical.TScrollbar", background=self.BG_COLOR, troughcolor=self.FRAME_COLOR, arrowcolor=self.TEXT_COLOR)
        style.configure("TCheckbutton", background=self.FRAME_COLOR, foreground=self.TEXT_COLOR)
        style.map("TCheckbutton", background=[('active', self.FRAME_COLOR)], indicatorcolor=[('selected', self.BUTTON_COLOR), ('!selected', self.ENTRY_BG_COLOR)])
        style.configure("ViewToggle.TButton", background=self.VIEW_BUTTON_BG, foreground="white", font=("SegoeUI", 10, "normal"))
        style.map("ViewToggle.TButton", background=[('active', self.BUTTON_COLOR)])
        style.configure("Active.TButton", background=self.BUTTON_COLOR, foreground="white") 
        
        self.compression_resolution_var = tk.StringVar(value=self.default_values["compression_resolution"])

        self.create_widgets()
        self.load_settings() 
        
        if not MATPLOTLIB_INSTALLED:
            is_in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
            if not is_in_venv:
                error_msg_1 = "تحذير: أنت لا تقوم بتشغيل البرنامج من البيئة الافتراضية (venv)."
                error_msg_2 = "يرجى إغلاق البرنامج وتشغيله باستخدام الملف 'run_app.bat'."
                self.update_status(error_msg_1)
                self.update_status(error_msg_2)
                messagebox.showwarning("بيئة غير صحيحة", f"{error_msg_1}\n{error_msg_2}")
            else:
                error_msg_1 = "تحذير: مكتبة matplotlib غير مثبتة في بيئتك الافتراضية."
                error_msg_2 = "يرجى تثبيتها باستخدام: venv\\Scripts\\pip.exe install matplotlib"
                self.update_status(error_msg_1)
                self.update_status(error_msg_2)

        self.toggle_compression_widgets() 
        self.show_view('proc') 

    def create_widgets(self):
        
        paned_window = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        paned_window.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # --- LEFT PANEL WITH SCROLLBAR ---
        left_panel_container = ttk.Frame(paned_window, style="TFrame")
        paned_window.add(left_panel_container, weight=2)

        left_canvas = tk.Canvas(left_panel_container, bg=self.FRAME_COLOR, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scrollbar.set)
        left_scrollbar.pack(side="right", fill="y")
        left_canvas.pack(side="left", fill="both", expand=True)

        # Frame inside canvas
        left_panel = ttk.Frame(left_canvas, style="TFrame")
        left_panel_id = left_canvas.create_window((0, 0), window=left_panel, anchor="nw")

        def _on_left_panel_configure(event):
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))
        left_panel.bind("<Configure>", _on_left_panel_configure)

        def _on_left_canvas_configure(event):
            left_canvas.itemconfig(left_panel_id, width=event.width)
        left_canvas.bind("<Configure>", _on_left_canvas_configure)

        # Enable mousewheel scrolling
        def _on_mousewheel(event):
            left_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        left_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        
        file_frame = ttk.LabelFrame(left_panel, text="1. اختيار الملفات", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=(5, 10))
        self.input_path_var = tk.StringVar(value="لم يتم اختيار ملف")
        self.output_path_var = tk.StringVar(value="لم يتم اختيار مكان الحفظ")
        
        ttk.Button(file_frame, text="اختر فيديو للمعالجة", command=self.select_input).grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        input_label = ttk.Label(file_frame, textvariable=self.input_path_var, anchor="w", style="TLabel")
        input_label.grid(row=0, column=1, sticky="ew", padx=5)
        
        ttk.Button(file_frame, text="اختر مكان حفظ الناتج", command=self.select_output).grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        output_label = ttk.Label(file_frame, textvariable=self.output_path_var, anchor="w", style="TLabel")
        output_label.grid(row=1, column=1, sticky="ew", padx=5)
        
        file_frame.columnconfigure(1, weight=1)

        
        processing_mode_frame = ttk.LabelFrame(left_panel, text="وضع المعالجة", padding=10)
        processing_mode_frame.pack(fill=tk.X, padx=5, pady=(5, 10))

        parallel_rb = ttk.Radiobutton(processing_mode_frame, text="المعالجة التفرعية (أسرع)", variable=self.processing_mode_var, value="parallel")
        parallel_rb.pack(side=tk.LEFT, padx=5, pady=2, expand=True)
        ToolTip(parallel_rb, "استخدام نوى معالجات متعددة لتسريع العملية. موصى به لمعظم الحالات.")

        sequential_rb = ttk.Radiobutton(processing_mode_frame, text="المعالجة التسلسلية (أبطأ)", variable=self.processing_mode_var, value="sequential")
        sequential_rb.pack(side=tk.LEFT, padx=5, pady=2, expand=True)
        ToolTip(sequential_rb, "استخدام نواة معالج واحدة. قد يكون مفيدًا في حال وجود مشاكل مع المعالجة التفرعية أو لمقارنة الأداء.")

        
        parallel_level_frame = ttk.Frame(left_panel)
        parallel_level_frame.pack(fill=tk.X, padx=5, pady=(0, 10))
        ttk.Label(parallel_level_frame, text="مستوى التفرع:").pack(side=tk.LEFT, padx=5)
        self.parallel_level_combo = ttk.Combobox(parallel_level_frame, textvariable=self.parallel_level_var, state="readonly", width=28)
        self.parallel_level_combo['values'] = ("تفرع على مستوى الأجزاء (Chunks)", "تفرع على مستوى الإطارات داخل الجزء (Frames in Chunk)")
        self.parallel_level_combo.pack(side=tk.LEFT, padx=5)
        ToolTip(self.parallel_level_combo, "اختر نوع التفرع: تفرع الأجزاء (كل جزء في عملية مستقلة) أو تفرع الإطارات داخل كل جزء.")

        
        waveform_button_frame = ttk.Frame(left_panel, style="TFrame")
        waveform_button_frame.pack(fill=tk.X, padx=5, pady=5)
        self.waveform_button = ttk.Button(waveform_button_frame, text="عرض وتعديل الموجة الصوتية", command=self.open_waveform_editor)
        self.waveform_button.pack(fill=tk.X, expand=True)
        self.preview_logo_button = ttk.Button(left_panel, text="معاينة وإضافة عناصر", command=self.open_overlay_editor_window)
        self.preview_logo_button.pack(fill=tk.X, padx=5, pady=(0, 10))
        
        view_buttons_frame = ttk.Frame(left_panel, style="TFrame")
        view_buttons_frame.pack(fill=tk.X, padx=5, pady=(10, 0))

        self.proc_opts_button = ttk.Button(view_buttons_frame, text="خيارات المعالجة", command=lambda: self.show_view('proc'), style="ViewToggle.TButton")
        self.proc_opts_button.pack(side=tk.RIGHT, padx=(2, 0), fill=tk.X, expand=True)

        self.chunking_opts_button = ttk.Button(view_buttons_frame, text="تقسيم الفيديو", command=lambda: self.show_view('chunk'), style="ViewToggle.TButton")
        self.chunking_opts_button.pack(side=tk.RIGHT, padx=(2,0), fill=tk.X, expand=True)

        self.comp_opts_button = ttk.Button(view_buttons_frame, text="ضغط الفيديو", command=lambda: self.show_view('comp'), style="ViewToggle.TButton")
        self.comp_opts_button.pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)


        
        self.options_views_container = ttk.Frame(left_panel)
        self.options_views_container.pack(fill=tk.BOTH, expand=True)


        
        self.processing_options_view = ttk.Frame(self.options_views_container, style="TFrame")
        
        options_canvas = tk.Canvas(self.processing_options_view, bg=self.FRAME_COLOR, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self.processing_options_view, orient="vertical", command=options_canvas.yview)
        options_frame = ttk.Frame(options_canvas, style="TFrame") 

        options_canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        options_canvas.pack(side="left", fill="both", expand=True)
        canvas_window = options_canvas.create_window((0, 0), window=options_frame, anchor="nw")

        def on_frame_configure(event):
            options_canvas.configure(scrollregion=options_canvas.bbox("all"))

        def on_canvas_configure(event):
            options_canvas.itemconfig(canvas_window, width=event.width)

        options_frame.bind("<Configure>", on_frame_configure)
        options_canvas.bind("<Configure>", on_canvas_configure)

        self.entries = {} 
        processing_options_list = {
            "crop_top": ["قص علوي", "إزالة عدد محدد من البكسلات من الحافة العلوية للفيديو..."],
            "crop_bottom": ["قص سفلي", "إزالة عدد محدد من البكسلات من الحافة السفلية للفيديو..."],
            "crop_left": ["قص أيسر", "إزالة عدد محدد من البكسلات من الحافة اليسرى للفيديو..."],
            "crop_right": ["قص أيمن", "إزالة عدد محدد من البكسلات من الحافة اليمنى للفيديو..."],
            "brightness": ["السطوع", "يتحكم في مستوى السطوع العام للصورة..."],
            "contrast": ["التباين", "يتحكم في الفرق بين المناطق الأكثر سطوعًا والأكثر قتامة..."],
            "speed_factor": ["عامل السرعة", "يتحكم في سرعة تشغيل الفيديو والصوت..."],
            "logo_scale": ["حجم الشعار", "يحدد حجم الشعار كنسبة مئوية من ارتفاع الفيديو..."],
            "wave_chunk_duration": ["مدة موجة الصوت (مللي ثانية)", "المدة الزمنية لكل مقطع صوتي لتأثير الموجة..."],
            "wave_fade": ["مدة تلاشي الموجة (مللي ثانية)", "مدة التلاشي لكل مقطع من مقاطع موجة الصوت..."],
            "x_thickness": ["سماكة خط X", "يتحكم في سماكة الخطوط المتقاطعة (X)..."],
            "x_lighten": ["قوة تفتيح X", "يتحكم في مقدار الإضاءة تحت خطوط X..."]
        }
        
        i = 0
        for name, (label_text, tooltip_text) in processing_options_list.items():
            ttk.Label(options_frame, text=label_text).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
            self.entries[name] = tk.StringVar(value=str(self.default_values.get(name, "")))
            ttk.Entry(options_frame, textvariable=self.entries[name], width=15).grid(row=i, column=1, sticky=tk.EW, padx=5, pady=5)
            info_label = ttk.Label(options_frame, text="ⓘ", cursor="hand2")
            info_label.grid(row=i, column=2, sticky=tk.W, padx=(5, 0))
            ToolTip(info_label, tooltip_text)
            i += 1
        
        self.mirror_enabled_var = tk.BooleanVar(value=self.default_values["mirror_enabled"])
        mirror_checkbox = ttk.Checkbutton(options_frame, text="عكس الصورة أفقياً", variable=self.mirror_enabled_var)
        mirror_checkbox.grid(row=i, column=0, columnspan=2, sticky=tk.W, padx=5, pady=5)
        mirror_info_label = ttk.Label(options_frame, text="ⓘ", cursor="hand2")
        mirror_info_label.grid(row=i, column=2, sticky=tk.W, padx=(5,0))
        ToolTip(mirror_info_label, "عكس الصورة أفقياً (مثل المرآة).")
        options_frame.columnconfigure(1, weight=1)


        
        self.compression_options_view = ttk.Frame(self.options_views_container, style="TFrame")
        compression_main_frame = ttk.LabelFrame(self.compression_options_view, text="إعدادات ضغط الفيديو", padding=10)
        compression_main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.compression_enabled_var = tk.BooleanVar(value=self.default_values["compression_enabled"])
        self.compression_quality_var = tk.StringVar(value=self.default_values["compression_quality"])
        self.compression_speed_var = tk.StringVar(value=self.default_values["compression_speed"])
        # --- جديد: متغير الدقة ---
        self.compression_resolution_var = tk.StringVar(value=self.default_values["compression_resolution"])

        checkbutton = ttk.Checkbutton(compression_main_frame, text="تفعيل ضغط الفيديو",
                                      variable=self.compression_enabled_var, command=self.toggle_compression_widgets)
        checkbutton.grid(row=0, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        ToolTip(checkbutton, "تفعيل ضغط الفيديو لتقليل حجم الملف.")

        quality_label = ttk.Label(compression_main_frame, text="مستوى الجودة:")
        quality_label.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.quality_combo = ttk.Combobox(compression_main_frame, textvariable=self.compression_quality_var,
                                          values=["عالية", "متوسطة", "منخفضة"], state='readonly')
        self.quality_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        ToolTip(quality_label, "جودة الفيديو: عالية (ملف أكبر)، متوسطة (متوازن)، منخفضة (ملف أصغر).")

        speed_label = ttk.Label(compression_main_frame, text="سرعة الضغط:")
        speed_label.grid(row=2, column=0, sticky='w', padx=5, pady=5)
        self.speed_combo = ttk.Combobox(compression_main_frame, textvariable=self.compression_speed_var,
                                        values=["سريعة", "متوسطة", "بطيئة"], state='readonly')
        self.speed_combo.grid(row=2, column=1, sticky='ew', padx=5, pady=5)
        ToolTip(speed_label, "سرعة الضغط: سريعة (أقل كفاءة)، متوسطة (متوازن)، بطيئة (أكثر كفاءة).")
        # --- جديد: قائمة الدقة ---
        resolution_label = ttk.Label(compression_main_frame, text="الدقة المستهدفة:")
        resolution_label.grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.resolution_combo = ttk.Combobox(
            compression_main_frame, textvariable=self.compression_resolution_var,
            values=["أصلي", "140p", "240p", "360p", "480p", "720p", "1080p"], state='readonly')
        self.resolution_combo.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        ToolTip(resolution_label, "اختر الدقة النهائية للفيديو المضغوط. \n'أصلي' تعني الحفاظ على دقة الفيديو الأصلية.")
        compression_main_frame.columnconfigure(1, weight=1)


        
        self.chunking_options_view = ttk.Frame(self.options_views_container, style="TFrame")
        chunking_main_frame = ttk.LabelFrame(self.chunking_options_view, text="إعدادات تقسيم الفيديو", padding=10)
        chunking_main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.enable_chunking_var = tk.BooleanVar(value=self.default_values["enable_chunking"])
        self.chunk_size_seconds_var = tk.StringVar(value=str(self.default_values["chunk_size_seconds"]))
        self.merge_chunks_var = tk.BooleanVar(value=self.default_values["merge_chunks_after_processing"])
        self.crossfade_duration_var = tk.StringVar(value=str(self.default_values["crossfade_duration"]))

        
        enable_chunking_check = ttk.Checkbutton(chunking_main_frame, text="تفعيل تقسيم الفيديو إلى أجزاء",
                                                variable=self.enable_chunking_var, command=self.toggle_chunking_widgets_state)
        enable_chunking_check.grid(row=0, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        ToolTip(enable_chunking_check, "قسم الفيديو إلى أجزاء أصغر لمعالجتها بشكل منفصل. مفيد للملفات الكبيرة جداً.")

        
        chunk_size_label = ttk.Label(chunking_main_frame, text="حجم الجزء (بالدقائق):")
        chunk_size_label.grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.chunk_size_entry = ttk.Entry(chunking_main_frame, textvariable=self.chunk_size_seconds_var, width=10)
        self.chunk_size_entry.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        ToolTip(chunk_size_label, "المدة الزمنية لكل جزء من الفيديو بالدقائق (مثال: 1 لدقيقة واحدة). سيتم تحويلها تلقائياً إلى ثواني.")

        
        self.merge_chunks_check = ttk.Checkbutton(chunking_main_frame, text="دمج الأجزاء تلقائياً بعد المعالجة",
                                                 variable=self.merge_chunks_var, command=self.toggle_crossfade_widget_state)
        self.merge_chunks_check.grid(row=2, column=0, columnspan=3, sticky='w', padx=5, pady=5)
        ToolTip(self.merge_chunks_check, "إذا تم تفعيله، سيتم دمج جميع الأجزاء المعالجة في ملف واحد نهائي.")

        
        crossfade_label = ttk.Label(chunking_main_frame, text="مدة التلاشي المتقاطع (بالثواني):")
        crossfade_label.grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.crossfade_entry = ttk.Entry(chunking_main_frame, textvariable=self.crossfade_duration_var, width=10)
        self.crossfade_entry.grid(row=3, column=1, sticky='ew', padx=5, pady=5)
        ToolTip(crossfade_label, "مدة تأثير التلاشي المتقاطع (crossfade) بين الأجزاء عند الدمج (مثال: 1 لثانية واحدة).")
        
        chunking_main_frame.columnconfigure(1, weight=1)

        
        self.manual_merge_button = ttk.Button(chunking_main_frame, text="دمج الأجزاء المحددة يدوياً", command=self.manually_merge_chunks, state=tk.DISABLED)
        self.manual_merge_button.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
        ToolTip(self.manual_merge_button, "اختر أجزاء فيديو معالجة مسبقاً لدمجها في ملف واحد.")


        
        settings_buttons_frame = ttk.Frame(left_panel, style="TFrame")
        settings_buttons_frame.pack(fill=tk.X, padx=5, pady=(10, 5))
        
        ttk.Button(settings_buttons_frame, text="حفظ الإعدادات", command=self.save_settings).pack(side=tk.RIGHT, padx=5, expand=True, fill=tk.X)
        ttk.Button(settings_buttons_frame, text="استعادة الافتراضيات", command=self.restore_default_settings).pack(side=tk.RIGHT, padx=5, expand=True, fill=tk.X)

        
        bottom_frame = ttk.Frame(left_panel, style="TFrame")
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=(10, 5))

        self.process_button = ttk.Button(bottom_frame, text="بدء المعالجة", command=self.start_processing)
        self.process_button.pack(fill=tk.X, ipady=5, pady=(0, 5))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(bottom_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)

        
        right_panel = ttk.Frame(paned_window, padding=10, style="TFrame")
        paned_window.add(right_panel, weight=3) 

        status_frame = ttk.LabelFrame(right_panel, text="3. الحالة", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True)

        
        status_top_frame = ttk.Frame(status_frame, style="TFrame")
        status_top_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))

        self.process_button = ttk.Button(status_top_frame, text="بدء المعالجة", command=self.start_processing)
        self.process_button.pack(fill=tk.X, ipady=5, pady=(0, 5))

        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_top_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X)

        status_text_frame = ttk.Frame(status_frame, style="TFrame")
        status_text_frame.pack(fill=tk.BOTH, expand=True)

        self.status_text = tk.Text(status_text_frame, state=tk.DISABLED,
                                   background=self.ENTRY_BG_COLOR, foreground=self.TEXT_COLOR,
                                   relief="solid", borderwidth=1, wrap=tk.WORD,
                                   padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(status_text_frame, orient="vertical", command=self.status_text.yview, style="Vertical.TScrollbar")
        self.status_text['yscrollcommand'] = scrollbar.set
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def load_first_video_frame(self):
        
        input_path = self.settings.get('input_path')
        if not input_path or not os.path.exists(input_path):
            messagebox.showerror("خطأ", "يرجى اختيار ملف فيديو صالح أولاً.")
            return
        import cv2
        from PIL import Image, ImageTk
        cap = cv2.VideoCapture(input_path)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("خطأ", "تعذر قراءة أول إطار من الفيديو.")
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
        pil_img = pil_img.resize((320, 180), resample)
        self.video_frame_image = ImageTk.PhotoImage(pil_img)
        self.video_canvas.create_image(0, 0, anchor='nw', image=self.video_frame_image)
        
        if self.logo_canvas_image and self.logo_canvas_id:
            self.video_canvas.lift(self.logo_canvas_id)

    def select_logo_image(self):
        from PIL import Image, ImageTk
        path = filedialog.askopenfilename(title="اختر صورة لوغو", filetypes=[("صور", "*.png;*.jpg;*.jpeg")])
        if not path:
            return
        pil_img = Image.open(path).convert("RGBA")
        self.logo_pil_image = pil_img
        
        logo_resized = pil_img.resize((64, 64), Image.ANTIALIAS)
        self.logo_canvas_image = ImageTk.PhotoImage(logo_resized)
        
        if self.logo_canvas_id:
            self.video_canvas.delete(self.logo_canvas_id)
        self.logo_canvas_id = self.video_canvas.create_image(10, 10, anchor='nw', image=self.logo_canvas_image)
        
        self.video_canvas.lift(self.logo_canvas_id)

    def select_input(self):
        path = filedialog.askopenfilename(title="اختر فيديو", filetypes=[("ملفات الفيديو", "*.mp4 *.mov *.avi"), ("كل الملفات", "*.*")])
        if path:
            self.input_path_var.set(os.path.basename(path))
            self.settings['input_path'] = path

    def select_output(self):
        path = filedialog.asksaveasfilename(title="حفظ باسم", defaultextension=".mp4", filetypes=[("ملف MP4", "*.mp4")])
        if path:
            self.output_path_var.set(os.path.basename(path))
            self.settings['output_path'] = path

    def update_status(self, message, progress=None):
        self.status_text.config(state=tk.NORMAL)
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state=tk.DISABLED)

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

    def start_processing(self):
        
        if not self.settings.get('input_path') or not self.settings.get('output_path'):
            messagebox.showerror("خطأ", "الرجاء اختيار ملف الإدخال ومكان الحفظ أولاً.")
            return
            
        
        try:
            
            for name in ["brightness", "contrast", "speed_factor", "logo_scale"]:
                self.settings[name] = float(self.entries[name].get())
            
            for name in ["crop_top", "crop_bottom", "crop_left", "crop_right", 
                         "wave_chunk_duration", "wave_fade", "x_thickness", "x_lighten"]:
                
                self.settings[name] = int(float(self.entries[name].get()))
        except (ValueError, KeyError) as e:
            messagebox.showerror("خطأ في الإدخال", f"الرجاء إدخال أرقام صالحة في الخيارات.\nالخيار الذي به مشكلة على الأغلب: {name}")
            return

        
        self.settings['compression_enabled'] = self.compression_enabled_var.get()
        self.settings['compression_quality'] = self.compression_quality_var.get()
        self.settings['compression_speed'] = self.compression_speed_var.get()
        # --- جديد: حفظ الدقة ---
        self.settings['compression_resolution'] = self.compression_resolution_var.get()
        
        
        self.settings['mirror_enabled'] = self.mirror_enabled_var.get()

        
        self.settings['processing_mode'] = self.processing_mode_var.get()
        
        
        self.settings['enable_chunking'] = self.enable_chunking_var.get()
        self.settings['chunk_size_seconds'] = float(self.chunk_size_seconds_var.get())
        self.settings['merge_chunks_after_processing'] = self.merge_chunks_var.get()
        self.settings['crossfade_duration'] = float(self.crossfade_duration_var.get())
        
        
        self.settings['parallel_level'] = self.parallel_level_var.get()
        
        self.update_status(f"تم اختيار وضع المعالجة: {'تفرعي' if self.settings['processing_mode'] == 'parallel' else 'تسلسلي'}")

        self.process_button.config(state=tk.DISABLED)
        self.progress_var.set(0) 
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        
        
        thread = threading.Thread(target=self.run_processing_thread)
        thread.daemon = True
        thread.start()

    def run_processing_thread(self):
        try:
            process_video_core(self.settings, self.update_status)
        finally:
            self.process_button.config(state=tk.NORMAL)
            
            self.progress_bar.stop()
            self.progress_bar.config(mode='determinate')
            self.progress_var.set(0)

    def save_settings(self):
        settings_to_save = {name: var.get() for name, var in self.entries.items()}
        settings_to_save['compression_enabled'] = self.compression_enabled_var.get()
        settings_to_save['compression_quality'] = self.compression_quality_var.get()
        settings_to_save['compression_speed'] = self.compression_speed_var.get()
        # --- جديد: حفظ الدقة ---
        settings_to_save['compression_resolution'] = self.compression_resolution_var.get()
        settings_to_save['mirror_enabled'] = self.mirror_enabled_var.get()
        settings_to_save['processing_mode'] = self.processing_mode_var.get() 
        
        settings_to_save['enable_chunking'] = self.enable_chunking_var.get()
        settings_to_save['chunk_size_seconds'] = self.chunk_size_seconds_var.get()
        settings_to_save['merge_chunks_after_processing'] = self.merge_chunks_var.get()
        settings_to_save['crossfade_duration'] = self.crossfade_duration_var.get()
        settings_to_save['parallel_level'] = self.parallel_level_var.get()
        try:
            with open(self.settings_file, 'w', encoding='utf-8') as f:
                json.dump(settings_to_save, f, indent=4)
            self.update_status("تم حفظ جميع الإعدادات بنجاح.")
        except Exception as e:
            messagebox.showerror("خطأ في الحفظ", f"لا يمكن حفظ الإعدادات:\n{e}")

    def load_settings(self):
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r', encoding='utf-8') as f:
                    loaded_settings = json.load(f)
                for name, value in loaded_settings.items():
                    if name in self.entries:
                        self.entries[name].set(value)

                self.compression_enabled_var.set(loaded_settings.get('compression_enabled', self.default_values['compression_enabled']))
                self.compression_quality_var.set(loaded_settings.get('compression_quality', self.default_values['compression_quality']))
                self.compression_speed_var.set(loaded_settings.get('compression_speed', self.default_values['compression_speed']))
                # --- جديد: تحميل الدقة ---
                self.compression_resolution_var.set(loaded_settings.get('compression_resolution', self.default_values['compression_resolution']))
                self.mirror_enabled_var.set(loaded_settings.get('mirror_enabled', self.default_values['mirror_enabled']))
                self.processing_mode_var.set(loaded_settings.get('processing_mode', self.default_values['processing_mode'])) 
                
                self.enable_chunking_var.set(loaded_settings.get('enable_chunking', self.default_values['enable_chunking']))
                self.chunk_size_seconds_var.set(loaded_settings.get('chunk_size_seconds', str(self.default_values['chunk_size_seconds'])))
                self.merge_chunks_var.set(loaded_settings.get('merge_chunks_after_processing', self.default_values['merge_chunks_after_processing']))
                self.crossfade_duration_var.set(loaded_settings.get('crossfade_duration', str(self.default_values['crossfade_duration'])))
                self.parallel_level_var.set(loaded_settings.get('parallel_level', "تفرع على مستوى الأجزاء (Chunks)"))
                self.toggle_compression_widgets()
                self.toggle_chunking_widgets_state()
                self.toggle_crossfade_widget_state()
                self.update_status("تم تحميل الإعدادات المحفوظة.")
        except Exception as e:
            self.update_status(f"لم يتم العثور على إعدادات محفوظة أو حدث خطأ: {e}")
            self.restore_default_settings()

    def restore_default_settings(self):
        for name, value in self.default_values.items():
            if name in self.entries:
                self.entries[name].set(str(value))
        
        self.compression_enabled_var.set(self.default_values['compression_enabled'])
        self.compression_quality_var.set(self.default_values['compression_quality'])
        self.compression_speed_var.set(self.default_values['compression_speed'])
        self.mirror_enabled_var.set(self.default_values['mirror_enabled'])
        self.processing_mode_var.set(self.default_values['processing_mode']) 
        
        self.toggle_compression_widgets()
        self.update_status("تم استعادة جميع الإعدادات الافتراضية.")

    def toggle_compression_widgets(self):
        state = tk.NORMAL if self.compression_enabled_var.get() else tk.DISABLED
        self.quality_combo.config(state=state)
        self.speed_combo.config(state=state)

    def show_view(self, view_name):
        
        self.processing_options_view.pack_forget()
        self.compression_options_view.pack_forget()
        self.chunking_options_view.pack_forget()

        
        self.proc_opts_button.config(style="ViewToggle.TButton")
        self.comp_opts_button.config(style="ViewToggle.TButton")
        self.chunking_opts_button.config(style="ViewToggle.TButton")

        if view_name == 'proc':
            self.processing_options_view.pack(fill=tk.BOTH, expand=True)
            self.proc_opts_button.config(style="Active.TButton")
        elif view_name == 'comp':
            self.compression_options_view.pack(fill=tk.BOTH, expand=True)
            self.comp_opts_button.config(style="Active.TButton")
        elif view_name == 'chunk':
            self.chunking_options_view.pack(fill=tk.BOTH, expand=True)
            self.chunking_opts_button.config(style="Active.TButton")

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

    def toggle_chunking_widgets_state(self):
        state = tk.NORMAL if self.enable_chunking_var.get() else tk.DISABLED
        self.chunk_size_entry.config(state=state)
        self.merge_chunks_check.config(state=state)
        
        self.crossfade_entry.config(state=state if self.merge_chunks_var.get() and self.enable_chunking_var.get() else tk.DISABLED)
        self.manual_merge_button.config(state=state)

    def toggle_crossfade_widget_state(self):
        if self.enable_chunking_var.get() and self.merge_chunks_var.get():
            self.crossfade_entry.config(state=tk.NORMAL)
        else:
            self.crossfade_entry.config(state=tk.DISABLED)

    def manually_merge_chunks(self):
        from tkinter import messagebox
        messagebox.showinfo("دمج يدوي", "ميزة دمج الأجزاء يدوياً غير متوفرة حالياً.\nسيتم تطويرها لاحقاً.")

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
            self.update_status(f"تم حفظ {len(self.settings['overlays'])} عنصر/عناصر من محرر المعاينة.")


class OverlayEditorWindow(tk.Toplevel):
    def __init__(self, parent, video_dimensions, first_frame, existing_overlays, preview_dims):
        super().__init__(parent)
        self.parent = parent
        self.video_w, self.video_h = video_dimensions
        self.first_frame = first_frame
        self.existing_overlays = existing_overlays
        self.saved = False

        self.title("معاينة وإضافة عناصر")
        self.configure(bg=App.BG_COLOR)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        max_w, max_h = 1024, 576
        scale = min(max_w / self.video_w, max_h / self.video_h)
        self.disp_w, self.disp_h = int(self.video_w * scale), int(self.video_h * scale)
        self.scale_x = self.video_w / self.disp_w
        self.scale_y = self.video_h / self.disp_h
        
        self.geometry(f"{self.disp_w + 40}x{self.disp_h + 120}")

        self._overlays = {}
        self._selected_id = None
        self._drag_data = {}
        self.brush_size = tk.IntVar(value=30)

        self.create_widgets()
        self.load_overlays()
    
    def get_overlays(self):
        return list(self._overlays.values())

    def get_preview_dimensions(self):
        return {'w': self.disp_w, 'h': self.disp_h}

    def create_widgets(self):
        from PIL import Image, ImageTk
        frame_rgb = cv2.cvtColor(self.first_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        resample = getattr(getattr(Image, 'Resampling', Image), 'LANCZOS', Image.BICUBIC)
        self.tk_img = ImageTk.PhotoImage(pil_img.resize((self.disp_w, self.disp_h), resample))

        # --- SCROLLABLE MAIN CONTAINER ---
        main_canvas = tk.Canvas(self, bg=App.BG_COLOR, highlightthickness=0)
        vscroll = ttk.Scrollbar(self, orient="vertical", command=main_canvas.yview)
        main_canvas.configure(yscrollcommand=vscroll.set)
        vscroll.pack(side="right", fill="y")
        main_canvas.pack(side="left", fill="both", expand=True)
        content_frame = ttk.Frame(main_canvas, style="TFrame")
        content_id = main_canvas.create_window((0, 0), window=content_frame, anchor="nw")
        def _on_configure(event):
            main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        content_frame.bind("<Configure>", _on_configure)
        def _on_canvas_configure(event):
            main_canvas.itemconfig(content_id, width=event.width)
        main_canvas.bind("<Configure>", _on_canvas_configure)
        def _on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # --- Toolbar ---
        toolbar = ttk.Frame(content_frame)
        toolbar.pack(fill=tk.X, padx=10, pady=(5, 0))
        self.current_mode = tk.StringVar(value="move")
        modes = [("تحريك/تحديد", "move"), ("تجزئة بكسلات", "pixelate"), ("مربع", "rect"), ("دائرة", "circle")]
        style = ttk.Style(self)
        style.configure("Tool.TRadiobutton", background=App.VIEW_BUTTON_BG, foreground="white", padding=6, font=("SegoeUI", 9, "bold"))
        style.map("Tool.TRadiobutton", background=[('active', App.BUTTON_ACTIVE_COLOR), ('selected', App.BUTTON_COLOR)])
        for text, mode in modes:
            ttk.Radiobutton(toolbar, text=text, variable=self.current_mode, value=mode, style="Tool.TRadiobutton").pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        # --- Brush Size Slider ---
        brush_frame = ttk.Frame(content_frame)
        brush_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(brush_frame, text="حجم الفرشاة:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Scale(brush_frame, from_=10, to=150, orient=tk.HORIZONTAL, variable=self.brush_size).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Entry(brush_frame, textvariable=self.brush_size, width=5).pack(side=tk.LEFT, padx=5)
        # --- Canvas ---
        self.canvas = tk.Canvas(content_frame, width=self.disp_w, height=self.disp_h, bg='black', highlightthickness=0)
        self.canvas.pack(pady=10, padx=10)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img, tags="bg_image")
        # --- Buttons ---
        btn_frame = ttk.Frame(content_frame)
        btn_frame.pack(pady=10, fill=tk.X, padx=10)
        ttk.Button(btn_frame, text="إضافة شعار...", command=self.add_logo).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="حذف المحدد", command=self.delete_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="حفظ وإغلاق", command=self.save_and_close).pack(side=tk.RIGHT, padx=5)
        # --- Bindings ---
        self.canvas.bind('<ButtonPress-1>', self.on_press)
        self.canvas.bind('<B1-Motion>', self.on_motion)
        self.canvas.bind('<ButtonRelease-1>', self.on_release)

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
                self.canvas.create_rectangle(x + w - 5, y + h - 5, x + w + 5, y + h + 5, fill="#007ACC", outline="white", tags=("handle", "resize_handle", oid))

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
        path = filedialog.askopenfilename(title="اختر شعار", filetypes=[("PNG Images", "*.png")], parent=self)
        if not path: return
        try:
            img = Image.open(path).convert("RGBA")
            w, h = img.size
            new_w = min(100, self.disp_w // 4)
            new_h = int(new_w * (h / w)) if w > 0 and h > 0 else new_w
            oid = f"logo_{time.time()}"
            self._overlays[oid] = {'type': 'logo', 'path': path, 'x': 50, 'y': 50, 'w': new_w, 'h': new_h, 'pil_img': img}
            self._selected_id = oid
            self.redraw_all()
        except Exception as e:
            messagebox.showerror("خطأ", f"فشل تحميل الصورة: {e}", parent=self)

    def delete_selected(self):
        if self._selected_id in self._overlays:
            del self._overlays[self._selected_id]
            self._selected_id = None
            self.redraw_all()

    def on_close(self):
        self.destroy()

    def save_and_close(self):
        # Remove runtime-only objects before saving
        for data in self._overlays.values():
            data.pop('pil_img', None)
            data.pop('tk_logo', None)
        self.saved = True
        self.destroy()

    def get_item_at(self, x, y):
        items = self.canvas.find_overlapping(x - 2, y - 2, x + 2, y + 2)
        for priority in ['resize_handle', 'overlay']:
            for item in reversed(items):
                tags = self.canvas.gettags(item)
                if priority in tags:
                    oid = next((t for t in tags if t not in ['overlay', 'handle', 'resize_handle', 'logo', 'rect', 'circle', 'blur', 'pixelate']), None)
                    return oid, tags
        return None, []

    def on_press(self, event):
        self._drag_data.clear()
        oid, tags = self.get_item_at(event.x, event.y)
        mode = self.current_mode.get()
        
        if mode == 'move':
            if self._selected_id != oid:
                self._selected_id = oid
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
            # أضف أول نقطة تشويش
            self._add_pixelate_at(event.x, event.y)
        elif mode in ['rect', 'circle']:
            self._selected_id = None
            self._drag_data = {'x': event.x, 'y': event.y, 'mode': f'draw_{mode}'}
            self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="white", width=2, dash=(), tags="temp")

    def _add_pixelate_at(self, x, y):
        radius = self.brush_size.get() // 2
        # تأكد أن المستطيل ضمن حدود الصورة
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
            pass  # لا شيء، التشويش تم أثناء السحب
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
        
        max_points = 50000 # More points for detail
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

