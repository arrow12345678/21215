# -*- coding: utf-8 -*-

import os
import sys
import shutil
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox


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
                        
                        ksize = (max(1, w // 4) | 1, max(1, h // 4) | 1)
                        final_frame[y:y+h, x:x+w] = cv2.GaussianBlur(roi, ksize, 0)
                    
                    elif o_type == 'pixelate':
                        
                        pixel_size = 20
                        
                        x, y, w, h = overlay['x'], overlay['y'], overlay['w'], overlay['h']
                        roi = final_frame[y:y+h, x:x+w]
                        if roi.size == 0: continue
                        
                        
                        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
                        cv2.circle(mask, (w//2, h//2), w//2, 255, -1)

                        
                        small_roi = cv2.resize(roi, (max(1, w // pixel_size), max(1, h // pixel_size)), interpolation=cv2.INTER_LINEAR)
                        pixelated_roi = cv2.resize(small_roi, (w, h), interpolation=cv2.INTER_NEAREST)

                        
                        final_frame[y:y+h, x:x+w] = np.where(mask[..., None].astype(bool), pixelated_roi, roi)

                    elif o_type in ['rect', 'circle']:
                        color_hex = overlay.get('color', '#FFFF00').lstrip('#')
                        color_bgr = tuple(int(color_hex[i:i+2], 16) for i in (4, 2, 0)) 
                        thickness = overlay.get('thickness', 2)
                        if o_type == 'rect':
                            cv2.rectangle(final_frame, (x, y), (x+w, y+h), color_bgr, thickness)
                        else: 
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
    
    
    estimated_frame_memory_mb = 25 
    available_memory_mb = available_memory_gb * 1024
    
    
    max_frames_in_memory = int(available_memory_mb * 0.5 / estimated_frame_memory_mb)
    
    
    
    optimal_batch_size = min(max_frames_in_memory, max(cpu_count * 16, 256), frame_count)
    
    print(f"DEBUG: Optimal batch size calculated: {optimal_batch_size}") 
    return max(1, optimal_batch_size)

def process_frame_in_shared_memory(args):
    
    shm_name, frame_idx, shape, dtype, settings, new_width, new_height, original_width, original_height, overlays_to_apply = args

    try:
        
        existing_shm = shared_memory.SharedMemory(name=shm_name)
        
        
        shm_np_array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)

        
        
        frame_to_process = shm_np_array[frame_idx].copy()
        
        
        processed_frame_list = process_frame_batch(
            [frame_to_process], settings, new_width, new_height, original_width, original_height, overlays_to_apply
        )
        
        
        if processed_frame_list:
            shm_np_array[frame_idx] = processed_frame_list[0]

    except Exception as e:
        print(f"Error in worker process for frame {frame_idx}: {e}")
    finally:
        
        if 'existing_shm' in locals():
            existing_shm.close()
            
    return frame_idx 

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
                        overlays_to_apply.append(prep_info)
                    except Exception as e: send_status(f"Warning: Could not prepare overlay: {e}")
        
        temp_video_fd, temp_video_file_for_chunk = tempfile.mkstemp(suffix=f'_chunk{chunk_index}.mp4', dir=os.path.join(base_path, "temp_videos"))
        os.close(temp_video_fd)
        out = cv2.VideoWriter(temp_video_file_for_chunk, cv2.VideoWriter_fourcc(*'mp4v'), original_fps, (new_width, new_height))
        
        batch_size = get_optimal_batch_size(frame_count)
        processed_count = 0
        cpu_cores = multiprocessing.cpu_count()
        
        while processed_count < frame_count:
            
            if cancel_event.is_set():
                send_status(f"الجزء {chunk_index + 1}: تم طلب الإلغاء، إيقاف معالجة الإطارات.")
                break 
            frames = []
            for _ in range(batch_size):
                ret, frame = cap.read()
                if not ret: break
                frames.append(frame)
            if not frames: break
            if chunk_settings.get('frame_parallel', False):
                frames_bytes = [pickle.dumps(f) for f in frames]
                func = partial(process_one_frame_pickled,
                               chunk_settings=chunk_settings,
                               new_width=new_width,
                               new_height=new_height,
                               original_width=original_width,
                               original_height=original_height,
                               overlays_to_apply=overlays_to_apply)
                with ProcessPoolExecutor(max_workers=cpu_cores) as executor:
                    processed_batch = list(executor.map(func, frames_bytes))
            else:
                processed_batch = process_frame_batch(frames, chunk_settings, new_width, new_height, original_width, original_height, overlays_to_apply)
            for p_frame in processed_batch: out.write(p_frame)
            processed_count += len(frames)
            send_status(f"الجزء {chunk_index + 1}: تمت معالجة {processed_count}/{frame_count} إطار", progress=(processed_count / frame_count) * 100)
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
            preset_name = chunk_settings.get('quality_preset', '1080p (Full HD)')
            preset_config = QUALITY_PRESETS.get(preset_name, QUALITY_PRESETS['1080p (Full HD)'])
            
            filters = [f"setpts={1/chunk_settings['speed_factor']}*PTS"]
            if preset_config['resolution']:
                filters.append(f"scale=-2:{preset_config['resolution']}")
            
            command.extend(['-vf', ",".join(filters)])
            command.extend(['-filter:a', f"atempo={chunk_settings['speed_factor']}"])
            command.extend(['-c:v', 'libx264', '-crf', preset_config['crf'], '-preset', preset_config['preset']])
        else:
            command.extend(['-filter_complex', f"[0:v]setpts={1/chunk_settings['speed_factor']}*PTS[v];[1:a]atempo={chunk_settings['speed_factor']}[a]"])
            command.extend(['-map', '[v]', '-map', '[a]'])

        command.append(os.path.normpath(output_path))
        

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
            "enable_chunking": False, "chunk_size_seconds": 60, 
            "merge_chunks_after_processing": True, "crossfade_duration": 1, 
            "parallel_level": "\u062a\u0641\u0631\u0639 \u0639\u0644\u0649 \u0645\u0633\u062a\u0648\u0649 \u0627\u0644\u0625\u0637\u0627\u0631\u0627\u062a \u062f\u0627\u062e\u0644 \u0627\u0644\u062c\u0632\u0621 (Frames in Chunk)",
            "compression_enabled": False,
            "quality_preset": "1080p (Full HD)" 
        }
        self.settings_file = os.path.join(base_path, "settings.json")
        self.processed_chunk_files = [] 
        
        
        self.mirror_enabled_var = tk.BooleanVar(value=self.default_values['mirror_enabled'])
        self.processing_mode_var = tk.StringVar(value=self.default_values["processing_mode"])
        self.parallel_level_var = tk.StringVar(value=self.default_values["parallel_level"])
        self.compression_enabled_var = tk.BooleanVar(value=self.default_values["compression_enabled"])
        self.enable_chunking_var = tk.BooleanVar(value=self.default_values["enable_chunking"])
        self.chunk_size_seconds_var = tk.StringVar(value=str(self.default_values["chunk_size_seconds"]))
        self.merge_chunks_var = tk.BooleanVar(value=self.default_values["merge_chunks_after_processing"])
        self.crossfade_duration_var = tk.StringVar(value=str(self.default_values["crossfade_duration"]))
        self.quality_preset_var = tk.StringVar(value=self.default_values["quality_preset"]) 
        
        self.setup_styles()
        self.create_widgets()
        self.load_settings() 
        self.show_view('proc')

    def setup_styles(self):
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
        

        
        file_frame = ttk.LabelFrame(left_panel, text="1. اختيار الملفات", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=(5, 10))
        self.input_path_var = tk.StringVar(value="لم يتم اختيار ملف")
        self.output_path_var = tk.StringVar(value="لم يتم اختيار مكان الحفظ")
        ttk.Button(file_frame, text="اختر فيديو للمعالجة", command=self.select_input).grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(file_frame, textvariable=self.input_path_var, anchor="w").grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(file_frame, text="اختر مكان حفظ الناتج", command=self.select_output).grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ttk.Label(file_frame, textvariable=self.output_path_var, anchor="w").grid(row=1, column=1, sticky="ew", padx=5)
        file_frame.columnconfigure(1, weight=1)

        
        tools_frame = ttk.Frame(left_panel, style="TFrame")
        tools_frame.pack(fill=tk.X, padx=5, pady=5)
        self.waveform_button = ttk.Button(tools_frame, text="محرر الموجة الصوتية", command=self.open_waveform_editor)
        self.waveform_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        self.preview_logo_button = ttk.Button(tools_frame, text="محرر العناصر (Overlays)", command=self.open_overlay_editor_window)
        self.preview_logo_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        
        view_buttons_frame = ttk.Frame(left_panel, style="TFrame")
        view_buttons_frame.pack(fill=tk.X, padx=5, pady=(10, 0))
        self.proc_opts_button = ttk.Button(view_buttons_frame, text="خيارات المعالجة", command=lambda: self.show_view('proc'), style="ViewToggle.TButton")
        self.proc_opts_button.pack(side=tk.LEFT, padx=(0, 2), fill=tk.X, expand=True)
        self.chunking_opts_button = ttk.Button(view_buttons_frame, text="تقسيم الفيديو", command=lambda: self.show_view('chunk'), style="ViewToggle.TButton")
        self.chunking_opts_button.pack(side=tk.LEFT, padx=(2, 2), fill=tk.X, expand=True)
        self.comp_opts_button = ttk.Button(view_buttons_frame, text="ضغط الفيديو", command=lambda: self.show_view('comp'), style="ViewToggle.TButton")
        self.comp_opts_button.pack(side=tk.LEFT, padx=(2, 0), fill=tk.X, expand=True)

        
        self.options_views_container = ttk.Frame(left_panel)
        self.options_views_container.pack(fill=tk.BOTH, expand=True)

        

        
        self.processing_options_view = ttk.Frame(self.options_views_container, style="TFrame")
        proc_main_frame = ttk.LabelFrame(self.processing_options_view, text="خيارات المعالجة الأساسية", padding=10)
        proc_main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.entries = {}
        options = [
            ("اقتصاص علوي:", "crop_top"), ("اقتصاص سفلي:", "crop_bottom"), 
            ("اقتصاص يسار:", "crop_left"), ("اقتصاص يمين:", "crop_right"),
            ("سطوع (Brightness):", "brightness"), ("تباين (Contrast):", "contrast"), 
            ("عامل السرعة:", "speed_factor"), ("مقياس اللوجو:", "logo_scale"),
            ("مدة موجة الصوت:", "wave_chunk_duration"), ("مدة تلاشي الموجة:", "wave_fade"),
            ("سمك خط X:", "x_thickness"), ("قوة إضاءة X:", "x_lighten")
        ]
        
        for i, (text, name) in enumerate(options):
            ttk.Label(proc_main_frame, text=text).grid(row=i, column=0, sticky="w", padx=5, pady=2)
            var = tk.StringVar(value=str(self.default_values.get(name, '')))
            entry = ttk.Entry(proc_main_frame, textvariable=var)
            entry.grid(row=i, column=1, sticky="ew", padx=5, pady=2)
            self.entries[name] = var
        
        
        mirror_check = ttk.Checkbutton(proc_main_frame, text="عكس الفيديو (Mirror)", variable=self.mirror_enabled_var)
        mirror_check.grid(row=len(options), column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        
        proc_mode_label = ttk.Label(proc_main_frame, text="نوع المعالجة:")
        proc_mode_label.grid(row=len(options) + 1, column=0, sticky='w', padx=5, pady=5)
        proc_mode_frame = ttk.Frame(proc_main_frame, style="TFrame")
        proc_mode_frame.grid(row=len(options) + 1, column=1, sticky='ew')
        
        
        ttk.Radiobutton(proc_mode_frame, text="تفرعي", variable=self.processing_mode_var, value="parallel", command=self.toggle_chunking_widgets_state).pack(side=tk.LEFT, expand=True)
        ttk.Radiobutton(proc_mode_frame, text="تسلسلي", variable=self.processing_mode_var, value="sequential", command=self.toggle_chunking_widgets_state).pack(side=tk.LEFT, expand=True)
        

        proc_main_frame.columnconfigure(1, weight=1)

        
        self.chunking_options_view = ttk.Frame(self.options_views_container, style="TFrame")
        chunk_main_frame = ttk.LabelFrame(self.chunking_options_view, text="إعدادات تقسيم الفيديو", padding=10)
        chunk_main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        chunk_check = ttk.Checkbutton(chunk_main_frame, text="تفعيل تقسيم الفيديو إلى أجزاء", variable=self.enable_chunking_var, command=self.toggle_chunking_widgets_state)
        chunk_check.grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(chunk_main_frame, text="حجم الجزء (بالدقائق):").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        self.chunk_size_entry = ttk.Entry(chunk_main_frame, textvariable=self.chunk_size_seconds_var) 
        self.chunk_size_entry.grid(row=1, column=1, sticky='ew', padx=5)
        
        self.merge_chunks_check = ttk.Checkbutton(chunk_main_frame, text="دمج الأجزاء بعد المعالجة", variable=self.merge_chunks_var, command=self.toggle_crossfade_widget_state)
        self.merge_chunks_check.grid(row=2, column=0, columnspan=2, sticky='w', padx=5, pady=5)
        
        ttk.Label(chunk_main_frame, text="مدة التلاشي المتداخل (بالثواني):").grid(row=3, column=0, sticky='w', padx=5, pady=5)
        self.crossfade_entry = ttk.Entry(chunk_main_frame, textvariable=self.crossfade_duration_var)
        self.crossfade_entry.grid(row=3, column=1, sticky='ew', padx=5)

        
        ttk.Label(chunk_main_frame, text="مستوى التفرع:").grid(row=4, column=0, sticky='w', padx=5, pady=5)
        self.parallel_level_combo = ttk.Combobox(chunk_main_frame, textvariable=self.parallel_level_var, values=["تفرع على مستوى الأجزاء (Chunks)", "تفرع على مستوى الإطارات داخل الجزء (Frames in Chunk)"], state="readonly")
        self.parallel_level_combo.grid(row=4, column=1, sticky='ew', padx=5, pady=5)
        
        
        self.manual_merge_button = ttk.Button(chunk_main_frame, text="دمج أجزاء يدوياً...", command=self.manually_merge_chunks)
        self.manual_merge_button.grid(row=5, column=0, columnspan=2, sticky='ew', padx=5, pady=10)
        chunk_main_frame.columnconfigure(1, weight=1)

        
        self.compression_options_view = ttk.Frame(self.options_views_container, style="TFrame")
        comp_main_frame = ttk.LabelFrame(self.compression_options_view, text="إعدادات ضغط الفيديو", padding=10)
        comp_main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.compression_enabled_var.set(self.default_values['compression_enabled']) 
        comp_check = ttk.Checkbutton(comp_main_frame, text="تفعيل ضغط الفيديو", variable=self.compression_enabled_var, command=self.toggle_simple_compression_widgets)
        comp_check.grid(row=0, column=0, columnspan=2, sticky='w', padx=5, pady=10)
        
        ttk.Label(comp_main_frame, text="اختر جودة الفيديو:").grid(row=1, column=0, sticky='w', padx=5, pady=5)
        
        self.quality_preset_combo = ttk.Combobox(comp_main_frame, textvariable=self.quality_preset_var,
                                                 values=list(QUALITY_PRESETS.keys()), state='readonly')
        self.quality_preset_combo.grid(row=1, column=1, sticky='ew', padx=5, pady=5)
        
        comp_main_frame.columnconfigure(1, weight=1)
        

        
        settings_buttons_frame = ttk.Frame(left_panel, style="TFrame")
        settings_buttons_frame.pack(fill=tk.X, padx=5, pady=(10, 5), side=tk.BOTTOM)
        ttk.Button(settings_buttons_frame, text="حفظ الإعدادات", command=self.save_settings).pack(side=tk.LEFT, padx=(0, 2), expand=True, fill=tk.X)
        ttk.Button(settings_buttons_frame, text="استعادة الافتراضيات", command=self.restore_default_settings).pack(side=tk.LEFT, padx=(2, 0), expand=True, fill=tk.X)

        
        right_panel = ttk.Frame(paned_window, padding=10, style="TFrame")
        paned_window.add(right_panel, weight=3) 

        status_frame = ttk.LabelFrame(right_panel, text="الحالة والتحكم", padding=10)
        status_frame.pack(fill=tk.BOTH, expand=True)

        status_top_frame = ttk.Frame(status_frame, style="TFrame")
        status_top_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        self.process_button = ttk.Button(status_top_frame, text="بدء المعالجة", command=self.start_processing)
        self.process_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))

        self.stop_button = ttk.Button(status_top_frame, text="إيقاف المعالجة", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(side=tk.TOP, fill=tk.X, pady=(5, 10))

        status_text_frame = ttk.Frame(status_frame, style="TFrame")
        status_text_frame.pack(fill=tk.BOTH, expand=True)
        self.status_text = tk.Text(status_text_frame, state=tk.DISABLED, background=self.ENTRY_BG_COLOR, foreground=self.TEXT_COLOR, relief="solid", borderwidth=1, wrap=tk.WORD, padx=5, pady=5)
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

        
        self.settings['mirror_enabled'] = self.mirror_enabled_var.get()
        self.settings['processing_mode'] = self.processing_mode_var.get()
        
        
        self.settings['enable_chunking'] = self.enable_chunking_var.get()
        self.settings['chunk_size_seconds'] = float(self.chunk_size_seconds_var.get())
        self.settings['merge_chunks_after_processing'] = self.merge_chunks_var.get()
        self.settings['crossfade_duration'] = float(self.crossfade_duration_var.get())
        self.settings['parallel_level'] = self.parallel_level_var.get()

        
        self.settings['compression_enabled'] = self.compression_enabled_var.get()
        self.settings['quality_preset'] = self.quality_preset_var.get()
        
        
        self.cancel_event.clear()
        self.process_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        
        self.progress_var.set(0) 
        self.status_text.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        self.update_status(f"تم اختيار وضع المعالجة: {'تفرعي' if self.settings['processing_mode'] == 'parallel' else 'تسلسلي'}")
        
        
        if hasattr(self, 'optimal_temp_dir'):
            self.settings['temp_dir_path'] = self.optimal_temp_dir

        thread = threading.Thread(target=self.run_processing_thread)
        thread.daemon = True
        thread.start()

    def run_processing_thread(self):
        result = (False, None, 0)
        try:
            
            result = process_video_core(self.settings, self.update_status, self.cancel_event)
        except Exception as e:
            
            self.after(0, self.update_status, f"An unhandled error occurred in the processing thread: {e}")
        finally:
            
            self.after(0, self.finalize_processing, result)

    def finalize_processing(self, result):
        success, output_path, elapsed_time = result

        
        self.process_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress_bar.stop()
        self.progress_bar.config(mode='determinate')
        self.progress_var.set(100 if success else 0)

        if success and output_path and os.path.exists(output_path):
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)
            time_str = f"{minutes} دقيقة و {seconds} ثانية" if minutes > 0 else f"{seconds} ثانية"

            file_name = os.path.basename(output_path)
            save_path = os.path.dirname(output_path)
            
            message = (
                f"اكتملت المعالجة بنجاح!\n\n"
                f"اسم الملف: {file_name}\n"
                f"تم الحفظ في: {save_path}\n"
                f"الوقت المستغرق: {time_str}"
            )
            messagebox.showinfo("اكتملت المعالجة", message)
        elif not self.cancel_event.is_set():
            
            messagebox.showerror("فشل في المعالجة", "فشلت عملية المعالجة. يرجى مراجعة سجل الحالة لمعرفة التفاصيل.")
        
        
        self.after(2000, lambda: self.progress_var.set(0))

    def save_settings(self):
        settings_to_save = {name: var.get() for name, var in self.entries.items()}
        
        
        settings_to_save['mirror_enabled'] = self.mirror_enabled_var.get()
        settings_to_save['processing_mode'] = self.processing_mode_var.get() 
        
        
        settings_to_save['enable_chunking'] = self.enable_chunking_var.get()
        settings_to_save['chunk_size_seconds'] = self.chunk_size_seconds_var.get()
        settings_to_save['merge_chunks_after_processing'] = self.merge_chunks_var.get()
        settings_to_save['crossfade_duration'] = self.crossfade_duration_var.get()
        settings_to_save['parallel_level'] = self.parallel_level_var.get()

        
        settings_to_save['compression_enabled'] = self.compression_enabled_var.get()
        settings_to_save['quality_preset'] = self.quality_preset_var.get()
        
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

            
            self.mirror_enabled_var.set(loaded_settings.get('mirror_enabled', self.default_values['mirror_enabled']))
            self.processing_mode_var.set(loaded_settings.get('processing_mode', self.default_values['processing_mode'])) 
            self.enable_chunking_var.set(loaded_settings.get('enable_chunking', self.default_values['enable_chunking']))
            self.chunk_size_seconds_var.set(loaded_settings.get('chunk_size_seconds', str(self.default_values['chunk_size_seconds'])))
            self.merge_chunks_var.set(loaded_settings.get('merge_chunks_after_processing', self.default_values['merge_chunks_after_processing']))
            self.crossfade_duration_var.set(loaded_settings.get('crossfade_duration', str(self.default_values['crossfade_duration'])))
            self.parallel_level_var.set(loaded_settings.get('parallel_level', "تفرع على مستوى الأجزاء (Chunks)"))

            
            self.compression_enabled_var.set(loaded_settings.get('compression_enabled', self.default_values['compression_enabled']))
            self.quality_preset_var.set(loaded_settings.get('quality_preset', self.default_values['quality_preset']))
            
            
            self.toggle_chunking_widgets_state()
            self.toggle_crossfade_widget_state()
            self.toggle_simple_compression_widgets()
            self.update_status("تم تحميل الإعدادات المحفوظة.")
        except Exception as e:
            self.update_status(f"لم يتم العثور على إعدادات محفوظة أو حدث خطأ: {e}")
            self.restore_default_settings()

    def restore_default_settings(self):
        for name, value in self.default_values.items():
            if name in self.entries:
                self.entries[name].set(str(value))
        
        self.mirror_enabled_var.set(self.default_values['mirror_enabled'])
        self.processing_mode_var.set(self.default_values['processing_mode']) 
        
        
        self.compression_enabled_var.set(self.default_values['compression_enabled'])
        self.quality_preset_var.set(self.default_values['quality_preset'])
        
        
        self.toggle_simple_compression_widgets()
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
        
        is_chunking_enabled = self.enable_chunking_var.get()
        chunking_state = tk.NORMAL if is_chunking_enabled else tk.DISABLED
        
        
        self.chunk_size_entry.config(state=chunking_state)
        self.merge_chunks_check.config(state=chunking_state)
        self.manual_merge_button.config(state=chunking_state)

        
        
        is_parallel_mode = self.processing_mode_var.get() == 'parallel'
        
        
        parallel_level_state = tk.NORMAL if is_chunking_enabled and is_parallel_mode else tk.DISABLED
        self.parallel_level_combo.config(state=parallel_level_state)
        

        
        self.toggle_crossfade_widget_state()

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

    def stop_processing(self):
        self.update_status("جاري طلب إيقاف المعالجة...")
        self.cancel_event.set()
        self.stop_button.config(state=tk.DISABLED)

    def _update_compression_controls(self, event=None):
        is_enabled = self.compression_enabled_var.get()
        state = tk.NORMAL if is_enabled else tk.DISABLED

        
        for child in self.compression_options_view.winfo_children()[0].winfo_children():
            if isinstance(child, (ttk.Combobox, ttk.Scale, ttk.Entry, ttk.Checkbutton, ttk.Label)):
                
                if child != self.compression_options_view.winfo_children()[0].winfo_children()[0]:
                     child.config(state=state)

        if not is_enabled:
            return

        
        mode = self.encoding_mode_combo.get()
        if mode == "الجودة الثابتة (CRF)":
            self.quality_label.config(text="قيمة CRF (0-51):")
            self.quality_scale.config(from_=0, to=51, variable=self.quality_preset_var)
            self.quality_entry.config(textvariable=self.quality_preset_var)
            self.two_pass_check.config(state=tk.DISABLED) 
        else: 
            self.quality_label.config(text="معدل البت (kbps):")
            self.quality_scale.config(from_=500, to=10000, variable=self.quality_preset_var)
            self.quality_entry.config(textvariable=self.quality_preset_var)
            self.two_pass_check.config(state=tk.NORMAL)

    def toggle_simple_compression_widgets(self):
        state = tk.NORMAL if self.compression_enabled_var.get() else tk.DISABLED
        self.quality_preset_combo.config(state=state)


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
        main_canvas.bind("<MouseWheel>", _on_mousewheel)

        
        toolbar = ttk.Frame(content_frame)
        toolbar.pack(fill=tk.X, padx=10, pady=(5, 0))
        self.current_mode = tk.StringVar(value="move")
        modes = [("تحريك/تحديد", "move"), ("تجزئة بكسلات", "pixelate"), ("مربع", "rect"), ("دائرة", "circle")]
        style = ttk.Style(self)
        style.configure("Tool.TRadiobutton", background=App.VIEW_BUTTON_BG, foreground="white", padding=6, font=("SegoeUI", 9, "bold"))
        style.map("Tool.TRadiobutton", background=[('active', App.BUTTON_ACTIVE_COLOR), ('selected', App.BUTTON_COLOR)])
        for text, mode in modes:
            ttk.Radiobutton(toolbar, text=text, variable=self.current_mode, value=mode, style="Tool.TRadiobutton").pack(side=tk.LEFT, padx=2, fill=tk.X, expand=True)
        
        brush_frame = ttk.Frame(content_frame)
        brush_frame.pack(fill=tk.X, padx=10, pady=5)
        ttk.Label(brush_frame, text="حجم الفرشاة:").pack(side=tk.LEFT, padx=(0, 5))
        ttk.Scale(brush_frame, from_=10, to=150, orient=tk.HORIZONTAL, variable=self.brush_size).pack(side=tk.LEFT, fill=tk.X, expand=True)
        ttk.Entry(brush_frame, textvariable=self.brush_size, width=5).pack(side=tk.LEFT, padx=5)
        
        self.canvas = tk.Canvas(content_frame, width=self.disp_w, height=self.disp_h, bg='black', highlightthickness=0)
        self.canvas.pack(pady=10, padx=10)
        self.canvas.create_image(0, 0, anchor='nw', image=self.tk_img, tags="bg_image")
        
        btn_frame = ttk.Frame(content_frame)
        btn_frame.pack(pady=10, fill=tk.X, padx=10)
        ttk.Button(btn_frame, text="إضافة شعار...", command=self.add_logo).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="حذف المحدد", command=self.delete_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="حفظ وإغلاق", command=self.save_and_close).pack(side=tk.RIGHT, padx=5)
        
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
            
            self._add_pixelate_at(event.x, event.y)
        elif mode in ['rect', 'circle']:
            self._selected_id = None
            self._drag_data = {'x': event.x, 'y': event.y, 'mode': f'draw_{mode}'}
            self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="white", width=2, dash=(), tags="temp")

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



QUALITY_PRESETS = {
    "أعلى جودة ممكنة (ملف كبير)": {
        "crf": "17",        
        "preset": "slow",   
        "resolution": None  
    },
    "1080p (Full HD)": {
        "crf": "22",
        "preset": "medium",
        "resolution": 1080
    },
    "720p (HD)": {
        "crf": "23",
        "preset": "medium",
        "resolution": 720
    },
    "480p (SD)": {
        "crf": "24",
        "preset": "medium",
        "resolution": 480
    },
    "360p": {
        "crf": "26",
        "preset": "fast",
        "resolution": 360
    },
    "240p": {
        "crf": "28",
        "preset": "veryfast",
        "resolution": 240
    },
    "144p": {
        "crf": "30",
        "preset": "ultrafast",
        "resolution": 144
    }
}

if __name__ == "__main__":
    if not os.path.exists(os.path.join(base_path, "temp_videos")):
        os.makedirs(os.path.join(base_path, "temp_videos"))
    app = App()
    app.mainloop()
