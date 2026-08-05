from __future__ import annotations

import csv
import json
import math
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/mnt/data/Experiment_006_Real_Video_Story')
VIDEO_DIR = ROOT / 'video'
DATASET_DIR = ROOT / 'dataset'
ANALYSIS_DIR = ROOT / 'analysis'
SCRIPTS_DIR = ROOT / 'scripts'
CODE_DIR = ROOT / 'AEIOU_Local_Compressor_v2_1_JSON'
INPUT_DIR = CODE_DIR / 'input'
OUTPUT_DIR = CODE_DIR / 'output'
THUMBS_DIR = ANALYSIS_DIR / 'selected_keyframes'

W, H = 854, 480
FPS = 20
SCENE_SECONDS = 15
SCENE_FRAMES = FPS * SCENE_SECONDS
TOTAL_SCENES = 12
TOTAL_FRAMES = SCENE_FRAMES * TOTAL_SCENES
DURATION = TOTAL_FRAMES / FPS
FONT_REG = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
FONT_BOLD = '/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'

SCENES = [
    {'id': 1, 'title': '任务说明', 'desc': '小蓝需要把橙色箱子送到小红手中。'},
    {'id': 2, 'title': '小蓝进入', 'desc': '小蓝从画面左侧进入房间。'},
    {'id': 3, 'title': '接近箱子', 'desc': '小蓝走向放在中间的橙色箱子。'},
    {'id': 4, 'title': '拿起箱子', 'desc': '小蓝弯腰并把箱子拿起来。'},
    {'id': 5, 'title': '搬向闸门', 'desc': '小蓝抱着箱子走向关闭前的闸门。'},
    {'id': 6, 'title': '闸门关闭', 'desc': '闸门突然落下，小蓝停止前进。'},
    {'id': 7, 'title': '等待并求助', 'desc': '小蓝等待，同时挥手发出求助信号。'},
    {'id': 8, 'title': '闸门打开', 'desc': '闸门升起，通道重新开放。'},
    {'id': 9, 'title': '穿过通道', 'desc': '小蓝抱着箱子穿过闸门和桥面。'},
    {'id': 10, 'title': '交付箱子', 'desc': '小蓝把箱子递给小红。'},
    {'id': 11, 'title': '放入目标区', 'desc': '小红把箱子搬到右侧绿色目标区。'},
    {'id': 12, 'title': '任务完成', 'desc': '箱子到达目标区，两人完成任务。'},
]

# Important action windows in source frames. These are kept outside compressor input.
EVENTS = [
    {'event_id': 'E01', 'name': '小蓝进入画面', 'start_s': 15.0, 'end_s': 30.0},
    {'event_id': 'E02', 'name': '小蓝接近箱子', 'start_s': 30.0, 'end_s': 45.0},
    {'event_id': 'E03', 'name': '小蓝拿起箱子', 'start_s': 45.0, 'end_s': 60.0},
    {'event_id': 'E04', 'name': '小蓝搬向闸门', 'start_s': 60.0, 'end_s': 75.0},
    {'event_id': 'E05', 'name': '闸门落下阻挡', 'start_s': 75.0, 'end_s': 90.0},
    {'event_id': 'E06', 'name': '小蓝挥手求助', 'start_s': 90.0, 'end_s': 105.0},
    {'event_id': 'E07', 'name': '闸门升起开放', 'start_s': 105.0, 'end_s': 120.0},
    {'event_id': 'E08', 'name': '小蓝穿过通道', 'start_s': 120.0, 'end_s': 135.0},
    {'event_id': 'E09', 'name': '小蓝向小红交付箱子', 'start_s': 135.0, 'end_s': 150.0},
    {'event_id': 'E10', 'name': '小红把箱子放入目标区', 'start_s': 150.0, 'end_s': 165.0},
    {'event_id': 'E11', 'name': '任务完成庆祝', 'start_s': 165.0, 'end_s': 180.0},
]


def reset_dirs():
    if ROOT.exists():
        shutil.rmtree(ROOT)
    for p in [VIDEO_DIR, DATASET_DIR, ANALYSIS_DIR, SCRIPTS_DIR, THUMBS_DIR]:
        p.mkdir(parents=True, exist_ok=True)
    # Copy previously adapted JSON version, preserving user's five-dimensional core.
    src = Path('/mnt/data/Experiment_005_Video_Frame_JSON/AEIOU_Local_Compressor_v2_1_JSON')
    shutil.copytree(src, CODE_DIR)
    shutil.rmtree(INPUT_DIR)
    shutil.rmtree(OUTPUT_DIR)
    INPUT_DIR.mkdir(parents=True)
    OUTPUT_DIR.mkdir(parents=True)


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)


def text_layer(scene_index: int) -> np.ndarray:
    img = Image.new('RGB', (W, H), (241, 244, 248))
    d = ImageDraw.Draw(img)
    # Top banner
    d.rectangle([0, 0, W, 70], fill=(24, 35, 52))
    title = f"场景 {scene_index+1:02d}/12：{SCENES[scene_index]['title']}"
    d.text((24, 12), title, font=font(26, True), fill=(255, 255, 255))
    d.text((24, 43), SCENES[scene_index]['desc'], font=font(16), fill=(205, 219, 235))
    # Bottom explanation panel
    d.rectangle([0, H-54, W, H], fill=(255, 255, 255))
    d.line([0, H-54, W, H-54], fill=(190, 198, 210), width=2)
    d.text((20, H-43), '蓝色人物：小蓝    红色人物：小红    橙色方块：箱子    绿色区域：目标区',
           font=font(16), fill=(35, 43, 55))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def draw_person(frame, x, y, color, carrying=False, wave=0.0, label=''):
    # head
    cv2.circle(frame, (int(x), int(y-54)), 16, color, -1, cv2.LINE_AA)
    # body
    cv2.line(frame, (int(x), int(y-38)), (int(x), int(y+12)), color, 7, cv2.LINE_AA)
    # legs
    cv2.line(frame, (int(x), int(y+12)), (int(x-16), int(y+42)), color, 6, cv2.LINE_AA)
    cv2.line(frame, (int(x), int(y+12)), (int(x+16), int(y+42)), color, 6, cv2.LINE_AA)
    # arms
    if carrying:
        cv2.line(frame, (int(x), int(y-22)), (int(x+20), int(y-4)), color, 6, cv2.LINE_AA)
        cv2.line(frame, (int(x), int(y-22)), (int(x-20), int(y-4)), color, 6, cv2.LINE_AA)
    else:
        arm_y = int(y-18 - 30*wave)
        cv2.line(frame, (int(x), int(y-22)), (int(x+28), arm_y), color, 6, cv2.LINE_AA)
        cv2.line(frame, (int(x), int(y-22)), (int(x-25), int(y-2)), color, 6, cv2.LINE_AA)
    if label:
        cv2.putText(frame, label, (int(x-22), int(y+67)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def draw_box(frame, x, y, held=False):
    size = 34 if held else 40
    x1, y1 = int(x-size/2), int(y-size/2)
    x2, y2 = int(x+size/2), int(y+size/2)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (30, 145, 245), -1, cv2.LINE_AA)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (15, 91, 170), 3, cv2.LINE_AA)
    cv2.line(frame, (int(x), y1), (int(x), y2), (20, 105, 195), 2)


def ease(t):
    t = max(0.0, min(1.0, t))
    return t*t*(3-2*t)


def render_story_frame(frame_id: int, layers: list[np.ndarray]) -> np.ndarray:
    scene = min(TOTAL_SCENES-1, frame_id // SCENE_FRAMES)
    local = (frame_id % SCENE_FRAMES) / max(1, SCENE_FRAMES-1)
    p = ease(local)
    frame = layers[scene].copy()

    # Environment
    cv2.rectangle(frame, (20, 84), (W-20, H-65), (224, 230, 237), -1)
    cv2.rectangle(frame, (20, 315), (W-20, H-65), (197, 207, 216), -1)
    cv2.line(frame, (20, 315), (W-20, 315), (130, 142, 156), 3)
    # bridge / path
    cv2.rectangle(frame, (440, 275), (650, 330), (166, 176, 188), -1)
    for xx in range(450, 650, 26):
        cv2.line(frame, (xx, 278), (xx, 327), (125, 136, 150), 2)
    # Target area
    cv2.rectangle(frame, (760, 245), (830, 320), (82, 190, 114), -1)
    cv2.rectangle(frame, (760, 245), (830, 320), (36, 126, 70), 4)
    cv2.putText(frame, 'TARGET', (765, 338), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (36, 126, 70), 2, cv2.LINE_AA)

    # Default positions/states
    blue_x = 105
    blue_y = 270
    red_x = 690
    red_y = 270
    box_x, box_y = 270, 290
    blue_carry = False
    red_carry = False
    gate_y_top = 100
    gate_y_bottom = 315
    wave = 0.0
    delivered = False

    if scene == 0:
        blue_x = 110
        red_x = 690
        # arrows showing task
        cv2.arrowedLine(frame, (300, 240), (650, 240), (80, 90, 105), 4, tipLength=.04)
        cv2.putText(frame, 'DELIVERY MISSION', (320, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (50, 60, 75), 2, cv2.LINE_AA)
    elif scene == 1:
        blue_x = -30 + 190*p
    elif scene == 2:
        blue_x = 160 + 95*p
    elif scene == 3:
        blue_x = 255
        lift = ease(min(1, local*1.4))
        box_y = 290 - 68*lift
        box_x = 270 - 5*lift
        blue_carry = lift > 0.55
    elif scene == 4:
        blue_x = 255 + 150*p
        blue_carry = True
        box_x, box_y = blue_x, blue_y-6
    elif scene == 5:
        blue_x = 405
        blue_carry = True
        box_x, box_y = blue_x, blue_y-6
        gate_y_top = 100
        gate_y_bottom = int(100 + 215*ease(min(1, local*1.7)))
        # warning flash
        if int(local*8) % 2 == 0:
            cv2.circle(frame, (430, 110), 12, (25, 45, 240), -1)
    elif scene == 6:
        blue_x = 405
        blue_carry = True
        box_x, box_y = blue_x, blue_y-6
        wave = 0.5 + 0.5*math.sin(local*math.pi*6)
        cv2.putText(frame, 'HELP', (360, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (40, 40, 210), 3, cv2.LINE_AA)
    elif scene == 7:
        blue_x = 405
        blue_carry = True
        box_x, box_y = blue_x, blue_y-6
        gate_y_bottom = int(315 - 215*p)
    elif scene == 8:
        blue_x = 405 + 230*p
        blue_carry = True
        box_x, box_y = blue_x, blue_y-6
        gate_y_bottom = 100
    elif scene == 9:
        blue_x = 635
        red_x = 690
        hand = ease(local)
        box_x = blue_x + (red_x-blue_x)*hand
        box_y = blue_y-6
        blue_carry = hand < 0.55
        red_carry = hand >= 0.45
        delivered = True
        gate_y_bottom = 100
    elif scene == 10:
        blue_x = 635
        red_x = 690 + 100*p
        red_carry = local < 0.72
        if red_carry:
            box_x, box_y = red_x, red_y-6
        else:
            q = ease((local-0.72)/0.28)
            box_x, box_y = 790, (red_y-6)*(1-q)+285*q
        gate_y_bottom = 100
    elif scene == 11:
        blue_x = 635
        red_x = 790
        box_x, box_y = 790, 285
        gate_y_bottom = 100
        # celebration stars/confetti
        rng = np.random.default_rng(frame_id)
        for _ in range(35):
            xx = int(rng.integers(40, W-40))
            yy = int(rng.integers(95, 235))
            col = tuple(int(v) for v in rng.integers(50, 245, size=3))
            cv2.circle(frame, (xx, yy), 3, col, -1)
        cv2.putText(frame, 'MISSION COMPLETE', (285, 155), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (35, 130, 65), 4, cv2.LINE_AA)

    # Gate
    if scene < 5:
        gate_y_bottom = 100  # open before obstruction
    cv2.rectangle(frame, (423, gate_y_top), (438, max(gate_y_top+1, gate_y_bottom)), (78, 86, 98), -1)
    cv2.rectangle(frame, (416, 95), (445, 110), (55, 63, 75), -1)

    # Box/person drawing order
    if scene not in (9, 10, 11) or not delivered:
        if not blue_carry and not red_carry:
            draw_box(frame, box_x, box_y)
    if scene != 0 or True:
        draw_person(frame, blue_x, blue_y, (210, 95, 35), carrying=blue_carry, wave=wave, label='BLUE')
        draw_person(frame, red_x, red_y, (55, 60, 215), carrying=red_carry, label='RED')
    if blue_carry or red_carry or scene in (9,10,11):
        draw_box(frame, box_x, box_y, held=(blue_carry or red_carry))

    # Time and progress
    current_s = frame_id / FPS
    cv2.putText(frame, f'{current_s:06.1f}s / {DURATION:.0f}s', (W-190, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 220, 235), 1, cv2.LINE_AA)
    x0, x1, y = 20, W-20, H-12
    cv2.rectangle(frame, (x0, y-7), (x1, y), (210, 216, 224), -1)
    prog = int(x0 + (x1-x0)*(frame_id/(TOTAL_FRAMES-1)))
    cv2.rectangle(frame, (x0, y-7), (prog, y), (78, 139, 224), -1)
    return frame


def generate_video():
    layers = [text_layer(i) for i in range(TOTAL_SCENES)]
    temp = VIDEO_DIR / 'story_video_temp.mp4'
    final = VIDEO_DIR / '完整故事视频_小蓝送箱子.mp4'
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W, H))
    if not writer.isOpened():
        raise RuntimeError('无法创建视频写入器')
    started = time.time()
    for frame_id in range(TOTAL_FRAMES):
        writer.write(render_story_frame(frame_id, layers))
    writer.release()
    # Transcode to broadly compatible H.264.
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(temp), '-c:v', 'libx264', '-preset', 'medium', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(final)]
    subprocess.run(cmd, check=True)
    temp.unlink(missing_ok=True)
    return final, time.time()-started


def entropy(gray):
    hist = cv2.calcHist([gray], [0], None, [64], [0,256]).ravel()
    p = hist / max(1.0, hist.sum())
    p = p[p > 0]
    return float(-(p*np.log2(p)).sum()/6.0)


def extract_features(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    records = []
    prev_gray = None
    prev_hsv = None
    prev_edge_density = 0.0
    frame_id = 0
    started = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        small = cv2.resize(frame, (214,120), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        edges = cv2.Canny(gray, 70, 150)
        edge_density = float(np.mean(edges > 0))
        mean_luma = float(gray.mean()/255.0)
        luma_std = float(min(1.0, gray.std()/100.0))
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = float(1.0/(1.0 + blur_var/300.0))
        sat_mean = float(hsv[:,:,1].mean()/255.0)
        hue_mean = float(hsv[:,:,0].mean()/179.0)
        value_mean = float(hsv[:,:,2].mean()/255.0)
        ent = entropy(gray)
        if prev_gray is None:
            diff = np.zeros_like(gray)
            frame_difference = motion_energy = color_shift = 0.0
            change_cx = change_cy = 0.5
        else:
            diff = cv2.absdiff(gray, prev_gray)
            frame_difference = float(diff.mean()/255.0)
            motion_mask = diff > 15
            motion_energy = float(motion_mask.mean())
            color_shift = float(cv2.absdiff(hsv, prev_hsv).mean()/255.0)
            ys, xs = np.nonzero(motion_mask)
            if len(xs):
                change_cx = float(xs.mean()/(gray.shape[1]-1))
                change_cy = float(ys.mean()/(gray.shape[0]-1))
            else:
                change_cx = change_cy = 0.5
        scene_cut_score = float(min(1.0, frame_difference*5.0 + color_shift*2.0 + abs(edge_density-prev_edge_density)*3.0))
        foreground_ratio = float(np.mean((hsv[:,:,1] > 70) & (hsv[:,:,2] > 70)))
        records.append({
            'frame_id': frame_id,
            'timestamp_ms': int(round(frame_id*1000.0/fps)),
            'mean_luma': round(mean_luma, 6),
            'luma_std': round(luma_std, 6),
            'edge_density': round(edge_density, 6),
            'frame_difference': round(frame_difference, 6),
            'motion_energy': round(motion_energy, 6),
            'color_shift': round(color_shift, 6),
            'scene_cut_score': round(scene_cut_score, 6),
            'foreground_ratio': round(foreground_ratio, 6),
            'change_center_x': round(change_cx, 6),
            'change_center_y': round(change_cy, 6),
            'mean_saturation': round(sat_mean, 6),
            'mean_hue': round(hue_mean, 6),
            'mean_value': round(value_mean, 6),
            'texture_entropy': round(ent, 6),
            'blur_score': round(blur_score, 6),
        })
        prev_gray = gray
        prev_hsv = hsv
        prev_edge_density = edge_density
        frame_id += 1
    cap.release()
    payload = {
        'metadata': {
            'source_video': video_path.name,
            'frame_count': frame_id,
            'fps': fps,
            'duration_seconds': frame_id/fps,
            'width': W,
            'height': H,
            'note': '每一条记录均由程序重新读取 MP4 像素后计算；未把场景名称或事件标签写入压缩输入。',
        },
        'frames': records,
    }
    out = DATASET_DIR / '实际视频逐帧特征_3600帧.json'
    out.write_text(json.dumps(payload, ensure_ascii=False, separators=(',',':')), encoding='utf-8')
    shutil.copy2(out, INPUT_DIR / out.name)
    return out, records, time.time()-started


def configure_and_run():
    cfg = {
      'min_output_chars': 6500,
      'max_output_chars': 9500,
      'preferred_output_chars': 7800,
      'ngram_min': 2,
      'ngram_max': 5,
      'long_unit_fallback_chars': 800,
      'include_subfolders': True,
      'output_title': 'AEIOU 实际视频帧动态轨迹压缩',
      'json_ignored_keys': ['frame_id','frame_index','timestamp','timestamp_ms','time','id','index']
    }
    (CODE_DIR/'config.json').write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding='utf-8')
    proc = subprocess.run([sys.executable, 'run.py'], cwd=CODE_DIR, text=True, capture_output=True, check=True)
    (ANALYSIS_DIR/'运行控制台.txt').write_text(proc.stdout + ('\nSTDERR:\n'+proc.stderr if proc.stderr else ''), encoding='utf-8')
    summaries = list(OUTPUT_DIR.glob('*_运行摘要.json'))
    if not summaries:
        raise RuntimeError('未找到运行摘要')
    summary = json.loads(summaries[0].read_text(encoding='utf-8'))
    selected_file = list(OUTPUT_DIR.glob('*_AEIOU选择记录.jsonl'))[0]
    selected = [json.loads(line) for line in selected_file.read_text(encoding='utf-8').splitlines() if line.strip()]
    return summary, selected


def event_for_time(t):
    for ev in EVENTS:
        if ev['start_s'] <= t < ev['end_s']:
            return ev
    return None


def extract_frame(cap, frame_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError(f'无法读取帧 {frame_id}')
    return frame


def add_chinese_overlay(frame, lines, top=88):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    d = ImageDraw.Draw(img, 'RGBA')
    box_h = 38 + 30*len(lines)
    d.rounded_rectangle([15, top, W-15, top+box_h], radius=12, fill=(10,20,34,205), outline=(255,255,255,150), width=2)
    for i, line in enumerate(lines):
        d.text((30, top+18+i*30), line, font=font(18, bold=(i==0)), fill=(255,255,255,255))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def analyze_and_visualize(video_path, records, summary, selected):
    selected_ids = [int(r['frame_id']) for r in selected]
    selected_times = [x/FPS for x in selected_ids]

    # Event coverage: selected inside each 15-second action window.
    event_rows = []
    for ev in EVENTS:
        inside = [fid for fid in selected_ids if ev['start_s']*FPS <= fid < ev['end_s']*FPS]
        center = (ev['start_s']+ev['end_s'])/2
        nearest = min(selected_times, key=lambda t: abs(t-center)) if selected_times else None
        event_rows.append({
            **ev,
            'selected_inside_count': len(inside),
            'selected_inside_frames': inside,
            'nearest_selected_s': round(nearest,2) if nearest is not None else None,
            'distance_to_center_s': round(abs(nearest-center),2) if nearest is not None else None,
            'captured': bool(inside),
        })

    # Scene boundary coverage.
    boundaries = [i*SCENE_SECONDS for i in range(1,TOTAL_SCENES)]
    boundary_rows = []
    for t in boundaries:
        nearest = min(selected_times, key=lambda x: abs(x-t)) if selected_times else None
        boundary_rows.append({
            'boundary_s': t,
            'nearest_selected_s': round(nearest,2),
            'distance_s': round(abs(nearest-t),2),
            'within_1s': abs(nearest-t) <= 1.0,
            'within_3s': abs(nearest-t) <= 3.0,
        })

    cap = cv2.VideoCapture(str(video_path))
    selected_details = []
    thumbs = []
    for idx, record in enumerate(selected,1):
        fid = int(record['frame_id'])
        t = fid/FPS
        frame = extract_frame(cap, fid)
        ev = event_for_time(t)
        evname = ev['name'] if ev else '任务说明'
        over = add_chinese_overlay(frame, [f'AEIOU 选择 #{idx}  ·  {t:06.2f} 秒  ·  第 {fid} 帧', f'人类解释：{evname}'])
        path = THUMBS_DIR / f'selected_{idx:02d}_frame_{fid}.jpg'
        cv2.imwrite(str(path), over, [int(cv2.IMWRITE_JPEG_QUALITY), 91])
        thumbs.append(over)
        selected_details.append({
            'selection_index': idx,
            'frame_id': fid,
            'time_seconds': round(t,2),
            'human_event': evname,
            'scene_number': min(12, int(t//SCENE_SECONDS)+1),
            'record': record,
            'image': str(path.relative_to(ROOT)),
        })
    cap.release()

    # Contact sheet: 3 columns.
    cols = 3
    thumb_w, thumb_h = 400, 225
    rows = math.ceil(len(thumbs)/cols)
    sheet = np.full((rows*(thumb_h+38)+20, cols*thumb_w+20, 3), 245, np.uint8)
    for i, im in enumerate(thumbs):
        r,c = divmod(i,cols)
        resized = cv2.resize(im, (thumb_w-12, thumb_h-12), interpolation=cv2.INTER_AREA)
        y = 10+r*(thumb_h+38)
        x = 10+c*thumb_w
        sheet[y:y+thumb_h-12, x:x+thumb_w-12] = resized
        cv2.putText(sheet, f'#{i+1}  {selected_details[i]["time_seconds"]:06.2f}s  frame {selected_details[i]["frame_id"]}',
                    (x+4,y+thumb_h+12), cv2.FONT_HERSHEY_SIMPLEX, .52, (30,38,50), 1, cv2.LINE_AA)
    contact = ANALYSIS_DIR/'AEIOU选择关键帧总览.jpg'
    cv2.imwrite(str(contact), sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    # Selected-moments video: real clips around each chosen frame, deduplicate overlapping intervals.
    intervals = []
    for t in selected_times:
        a,b = max(0,t-0.65), min(DURATION,t+0.65)
        if intervals and a <= intervals[-1][1] + 0.25:
            intervals[-1][1] = max(intervals[-1][1],b)
        else:
            intervals.append([a,b])
    temp = VIDEO_DIR/'selected_temp.mp4'
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*'mp4v'), FPS, (W,H))
    cap = cv2.VideoCapture(str(video_path))
    sel_set = set(selected_ids)
    sel_counter = {fid:i+1 for i,fid in enumerate(selected_ids)}
    for a,b in intervals:
        start_f,end_f = int(a*FPS),int(b*FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES,start_f)
        for fid in range(start_f,end_f+1):
            ok,fr=cap.read()
            if not ok: break
            nearest_fid = min(selected_ids, key=lambda x: abs(x-fid))
            t=fid/FPS
            ev=event_for_time(nearest_fid/FPS)
            evname=ev['name'] if ev else '任务说明'
            fr=add_chinese_overlay(fr,[f'AEIOU 关键片段 #{sel_counter[nearest_fid]} · 中心选择帧 {nearest_fid}',f'对应故事事件：{evname}'],top=80)
            if abs(fid-nearest_fid)<=2:
                cv2.rectangle(fr,(4,4),(W-5,H-5),(30,30,240),8)
            writer.write(fr)
    cap.release(); writer.release()
    selected_video = VIDEO_DIR/'AEIOU选择的关键片段.mp4'
    subprocess.run(['ffmpeg','-y','-loglevel','error','-i',str(temp),'-c:v','libx264','-preset','medium','-crf','22','-pix_fmt','yuv420p','-movflags','+faststart',str(selected_video)],check=True)
    temp.unlink(missing_ok=True)

    source_chars = summary['source_characters']
    selected_count = len(selected)
    metrics = {
        'video_file': video_path.name,
        'duration_seconds': DURATION,
        'fps': FPS,
        'source_frames': len(records),
        'source_json_characters': source_chars,
        'selected_frames': selected_count,
        'record_compression_ratio': round(len(records)/selected_count,2),
        'selected_fraction_percent': round(selected_count/len(records)*100,4),
        'output_characters': summary['output_characters'],
        'elapsed_seconds': summary['elapsed_seconds'],
        'processing_frames_per_second': round(len(records)/summary['elapsed_seconds'],2),
        'events_total': len(EVENTS),
        'events_with_selected_frame_inside': sum(x['captured'] for x in event_rows),
        'scene_boundaries_total': len(boundaries),
        'boundaries_within_1s': sum(x['within_1s'] for x in boundary_rows),
        'boundaries_within_3s': sum(x['within_3s'] for x in boundary_rows),
        'selected_frame_ids': selected_ids,
        'important_note': '压缩输入只有从 MP4 像素提取的数值特征，不含事件名称；人类事件名称只在运行后用于解释结果。'
    }
    (ANALYSIS_DIR/'实验指标.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    (ANALYSIS_DIR/'事件覆盖明细.json').write_text(json.dumps(event_rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (ANALYSIS_DIR/'场景边界明细.json').write_text(json.dumps(boundary_rows,ensure_ascii=False,indent=2),encoding='utf-8')
    (ANALYSIS_DIR/'选择帧人类解释.json').write_text(json.dumps(selected_details,ensure_ascii=False,indent=2),encoding='utf-8')
    (DATASET_DIR/'故事事件真值表.json').write_text(json.dumps({'events':EVENTS,'scenes':SCENES},ensure_ascii=False,indent=2),encoding='utf-8')

    return metrics,event_rows,boundary_rows,selected_details,selected_video,contact


def create_readme(metrics,event_rows,boundary_rows):
    captured = metrics['events_with_selected_frame_inside']
    missing = [x['name'] for x in event_rows if not x['captured']]
    conclusion = (
        f"AEIOU 从 {metrics['source_frames']:,} 个真实视频帧的数值记录中保留了 {metrics['selected_frames']} 个原始帧记录，"
        f"覆盖了 {captured}/{metrics['events_total']} 个故事动作阶段。"
    )
    if missing:
        conclusion += ' 未直接保留的阶段：' + '、'.join(missing) + '。'
    else:
        conclusion += ' 11 个重要动作阶段均至少有一帧被保留。'
    readme = f'''# Experiment 006：完整视频文件 + 人类可读结果

## 先看结论

{conclusion}

这次不是虚构 10 万条数值。实验包中包含一段可直接播放的完整 MP4：

- `video/完整故事视频_小蓝送箱子.mp4`
- 时长：{DURATION/60:.1f} 分钟
- 帧率：{FPS} FPS
- 总帧数：{metrics['source_frames']:,}

视频讲的是一件普通人一眼就能看懂的事：

> 小蓝进入房间 → 拿起箱子 → 被闸门挡住 → 求助等待 → 闸门打开 → 穿过通道 → 把箱子交给小红 → 小红放进目标区 → 任务完成。

## 算法到底收到什么

程序先重新读取 MP4 的每一帧像素，再计算亮度、边缘、帧差、运动范围、颜色变化、变化位置等 15 个数值。生成：

- `dataset/实际视频逐帧特征_3600帧.json`

**这个 JSON 没有写入“小蓝拿箱子”“闸门打开”等事件名称。**
事件名称只保存在独立真值表里，算法运行完成后才用来向人解释所选帧对应什么。

## 实际运行数字

- 输入：{metrics['source_frames']:,} 个视频帧记录
- AEIOU保留：{metrics['selected_frames']} 个原始记录
- 记录压缩倍数：约 {metrics['record_compression_ratio']} 倍
- 保留比例：{metrics['selected_fraction_percent']}%
- 算法运行时间：{metrics['elapsed_seconds']:.3f} 秒
- 处理速度：约 {metrics['processing_frames_per_second']:,} 帧/秒
- 11 个重要动作阶段中，直接出现AEIOU选择帧的阶段：{captured}/11
- 11 个场景切换点中，1秒内命中：{metrics['boundaries_within_1s']}/11；3秒内命中：{metrics['boundaries_within_3s']}/11

## 普通人先打开这三个文件

1. `video/完整故事视频_小蓝送箱子.mp4`：完整原始视频。
2. `video/AEIOU选择的关键片段.mp4`：只播放算法选择帧附近的真实片段，红框出现时就是选择帧。
3. `analysis/人类可读实验报告.html`：浏览器打开，视频、图片、数字和解释都在一页里。

## 结果应该怎么理解

它说明：当完整视频被转换为连续帧变化数据后，AEIOU可以把数千帧缩成少量代表帧，并在多数故事动作阶段留下代表记录。

它还没有说明：AEIOU自己知道蓝色人物叫“小蓝”，或者理解“交付箱子”的语言含义。当前提取的是视频变化轨迹，事件名称来自实验完成后的人工对照。

## 文件结构

```text
Experiment_006_Real_Video_Story/
├── video/
│   ├── 完整故事视频_小蓝送箱子.mp4
│   └── AEIOU选择的关键片段.mp4
├── dataset/
│   ├── 实际视频逐帧特征_3600帧.json
│   └── 故事事件真值表.json
├── analysis/
│   ├── 人类可读实验报告.html
│   ├── AEIOU选择关键帧总览.jpg
│   ├── selected_keyframes/
│   ├── 实验指标.json
│   ├── 事件覆盖明细.json
│   ├── 场景边界明细.json
│   └── 选择帧人类解释.json
├── AEIOU_Local_Compressor_v2_1_JSON/
│   ├── input/
│   ├── output/
│   └── engine/json_adapter.py
└── scripts/
    ├── generate_story_video.py
    ├── extract_video_features.py
    └── analyze_video_result.py
```

## 复现

代码目录中已经保留输入和实际输出。直接执行：

```bash
cd AEIOU_Local_Compressor_v2_1_JSON
python run.py
```

视频生成、特征提取和结果解释脚本也一并放入 `scripts/`。
'''
    (ROOT/'README.md').write_text(readme,encoding='utf-8')


def html_escape(s):
    return str(s).replace('&','&amp;').replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')


def create_html(metrics,event_rows,boundary_rows,selected_details):
    event_tr=''.join(f"<tr><td>{x['event_id']}</td><td>{html_escape(x['name'])}</td><td>{x['start_s']:.0f}–{x['end_s']:.0f}s</td><td class={'ok' if x['captured'] else 'miss'}>{'抓到' if x['captured'] else '没直接抓到'}</td><td>{', '.join(map(str,x['selected_inside_frames'])) or '—'}</td></tr>" for x in event_rows)
    sel_tr=''.join(f"<tr><td>#{x['selection_index']}</td><td>{x['time_seconds']:.2f}s</td><td>{x['frame_id']}</td><td>{html_escape(x['human_event'])}</td><td><img src='../{x['image']}'></td></tr>" for x in selected_details)
    html=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><title>AEIOU 完整视频实验报告</title>
<style>
body{{font-family:system-ui,'Noto Sans CJK SC',sans-serif;background:#f3f6fa;color:#1d2733;margin:0;line-height:1.65}}main{{max-width:1100px;margin:0 auto;padding:28px}}h1{{font-size:34px;margin:0 0 8px}}h2{{margin-top:34px;border-left:6px solid #3975d5;padding-left:12px}}.hero{{background:#152335;color:white;padding:28px;border-radius:18px}}.hero p{{color:#d5e2f2}}.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:18px 0}}.card{{background:white;padding:18px;border-radius:14px;box-shadow:0 3px 14px #ccd4df66}}.num{{font-size:27px;font-weight:800;color:#245eb5}}video{{width:100%;border-radius:14px;background:#000}}img.sheet{{width:100%;border-radius:12px}}table{{width:100%;border-collapse:collapse;background:white;border-radius:12px;overflow:hidden}}th,td{{padding:10px;border-bottom:1px solid #e2e7ee;text-align:left;vertical-align:middle}}th{{background:#eaf0f8}}td img{{width:210px;border-radius:8px}}.ok{{color:#14733c;font-weight:700}}.miss{{color:#b13a2e;font-weight:700}}.note{{background:#fff8db;border:1px solid #ebcf66;padding:16px;border-radius:12px}}code{{background:#e9eef5;padding:2px 5px;border-radius:5px}}@media(max-width:800px){{.cards{{grid-template-columns:1fr 1fr}}td img{{width:130px}}}}
</style></head><body><main>
<section class="hero"><h1>AEIOU 完整视频动态轨迹实验</h1><p>先看视频，再看算法选中了哪些时刻。这个页面不要求读懂 JSON。</p></section>
<div class="cards"><div class="card"><div>完整视频</div><div class="num">{DURATION/60:.1f} 分钟</div></div><div class="card"><div>原始帧</div><div class="num">{metrics['source_frames']:,}</div></div><div class="card"><div>AEIOU保留</div><div class="num">{metrics['selected_frames']} 帧</div></div><div class="card"><div>动作覆盖</div><div class="num">{metrics['events_with_selected_frame_inside']}/11</div></div></div>
<h2>1. 完整原始视频</h2><p>故事：小蓝拿箱子，遇到闸门阻挡，等待闸门打开，再把箱子送给小红。</p><video controls preload="metadata" src="../video/完整故事视频_小蓝送箱子.mp4"></video>
<h2>2. AEIOU选择帧附近的真实片段</h2><p>下面不是重新编的动画，而是从上面完整视频中截出的实际片段。画面出现红框的时刻，就是AEIOU最终保留的帧。</p><video controls preload="metadata" src="../video/AEIOU选择的关键片段.mp4"></video>
<h2>3. 一张图看完所有选择结果</h2><img class="sheet" src="AEIOU选择关键帧总览.jpg">
<h2>4. 数字是什么意思</h2><div class="note"><b>算法输入中没有事件名称。</b>程序只把视频每帧转换成亮度、边缘、运动、颜色变化、变化位置等数值。运行结束后，本报告才根据时间把选择帧翻译成“小蓝拿箱子”“闸门打开”等人类能懂的描述。</div>
<ul><li>{metrics['source_frames']:,} 帧压缩为 {metrics['selected_frames']} 个原始帧记录，约 {metrics['record_compression_ratio']} 倍。</li><li>11个故事动作阶段中，{metrics['events_with_selected_frame_inside']} 个阶段内部至少保留了一帧。</li><li>算法本身运行 {metrics['elapsed_seconds']:.3f} 秒，约 {metrics['processing_frames_per_second']:,} 帧/秒。</li></ul>
<h2>5. 每个故事动作有没有被留下</h2><table><thead><tr><th>编号</th><th>人类看到的动作</th><th>时间</th><th>结果</th><th>被选帧号</th></tr></thead><tbody>{event_tr}</tbody></table>
<h2>6. AEIOU实际选择了哪些帧</h2><table><thead><tr><th>选择</th><th>时间</th><th>帧号</th><th>人类解释</th><th>画面</th></tr></thead><tbody>{sel_tr}</tbody></table>
<h2>7. 这次能证明什么，不能证明什么</h2><p><b>可以支持：</b>AEIOU能够处理从完整MP4逐帧提取的连续数值序列，把数千帧缩成少量代表帧，并在多数动作阶段留下记录。</p><p><b>还不能支持：</b>算法已经独立理解人物身份、箱子含义或“交付”语义。当前层面是动态变化轨迹选择，不是完整视觉语义识别。</p>
</main></body></html>'''
    (ANALYSIS_DIR/'人类可读实验报告.html').write_text(html,encoding='utf-8')


def copy_scripts():
    # Copy the complete runner as an audit/reproduction script and provide focused entry notes.
    shutil.copy2(Path(__file__), SCRIPTS_DIR/'run_complete_experiment.py')
    (SCRIPTS_DIR/'generate_story_video.py').write_text("# 完整实现见 run_complete_experiment.py 中 generate_video() 与 render_story_frame()。\n",encoding='utf-8')
    (SCRIPTS_DIR/'extract_video_features.py').write_text("# 完整实现见 run_complete_experiment.py 中 extract_features()。该函数重新读取 MP4 像素，不读取场景标签。\n",encoding='utf-8')
    (SCRIPTS_DIR/'analyze_video_result.py').write_text("# 完整实现见 run_complete_experiment.py 中 analyze_and_visualize()。\n",encoding='utf-8')


def checksums_and_zip():
    import hashlib
    rows=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name!='checksums.sha256':
            h=hashlib.sha256()
            with p.open('rb') as f:
                for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
            rows.append(f"{h.hexdigest()}  {p.relative_to(ROOT).as_posix()}")
    (ROOT/'checksums.sha256').write_text('\n'.join(rows)+'\n',encoding='utf-8')
    zip_path=Path('/mnt/data/Experiment_006_Real_Video_Story_Full.zip')
    if zip_path.exists(): zip_path.unlink()
    with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in ROOT.rglob('*'):
            if p.is_file(): z.write(p,Path(ROOT.name)/p.relative_to(ROOT))
    return zip_path


def main():
    reset_dirs()
    video_path, video_seconds = generate_video()
    json_path, records, feature_seconds = extract_features(video_path)
    summary, selected = configure_and_run()
    metrics,event_rows,boundary_rows,selected_details,selected_video,contact=analyze_and_visualize(video_path,records,summary,selected)
    metrics['video_generation_seconds']=round(video_seconds,3)
    metrics['feature_extraction_seconds']=round(feature_seconds,3)
    (ANALYSIS_DIR/'实验指标.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')
    create_readme(metrics,event_rows,boundary_rows)
    create_html(metrics,event_rows,boundary_rows,selected_details)
    copy_scripts()
    # Simple author note without resolving contact ambiguity.
    (ROOT/'AUTHOR.md').write_text('# 作者与架构归属\n\nDynamic AI Core / AEIOU 核心架构设计：海兹巴（中国辽宁鞍山）。\n\n本实验只在现有五维基础上增加视频帧 JSON 适配与人类可读展示，没有替换五维核心。\n',encoding='utf-8')
    zip_path=checksums_and_zip()
    print(json.dumps({'zip':str(zip_path),'metrics':metrics,'video_bytes':video_path.stat().st_size,'zip_bytes':zip_path.stat().st_size},ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
