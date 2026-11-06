#!/usr/bin/env python3
'''Script JSON -> MP4: AI backgrounds (SD-Turbo on CPU) + stickman + edge-tts voice + captions.'''
import asyncio, json, math, os, random, re, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 30
WORK, OUT = Path('work'), Path('output')
FONT = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
INK = (20, 20, 20)
BAR = 130
BASE_Y = 585


def sh(cmd):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-2000:])
        raise SystemExit('command failed: ' + str(cmd[0]))
    return r.stdout


def duration(path):
    out = sh(['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=nw=1:nk=1', path])
    return float(out.strip())


def load_pipe():
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        torch.set_num_threads(os.cpu_count() or 2)
        pipe = AutoPipelineForText2Image.from_pretrained('stabilityai/sd-turbo', torch_dtype=torch.float32, variant='fp16')
        pipe.set_progress_bar_config(disable=True)
        return pipe
    except Exception as e:
        print('!! image model unavailable, using fallback art:', repr(e))
        return None


def txt2img(pipe, prompt, w, h, seed):
    import torch
    g = torch.Generator('cpu').manual_seed(seed)
    out = pipe(prompt=prompt, num_inference_steps=2, guidance_scale=0.0, width=w, height=h, generator=g)
    return out.images[0].convert('RGB')


def fallback_bg(seed, w=768, h=432):
    rnd = random.Random(seed)
    top = tuple(rnd.randint(70, 150) for _ in range(3))
    bot = tuple(rnd.randint(20, 80) for _ in range(3))
    im = Image.new('RGB', (w, h))
    d = ImageDraw.Draw(im)
    for y in range(h):
        k = y / h
        d.line([(0, y), (w, y)], fill=tuple(int(top[i] * (1 - k) + bot[i] * k) for i in range(3)))
    for _ in range(7):
        x = rnd.randint(-80, w)
        bh = rnd.randint(80, 240)
        c = tuple(max(0, v - rnd.randint(10, 40)) for v in bot)
        d.rectangle([x, h - bh, x + rnd.randint(80, 220), h], fill=c)
    return im


def fallback_head():
    im = Image.new('RGBA', (400, 400), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.ellipse([60, 60, 340, 340], fill=(200, 200, 190, 255), outline=INK, width=8)
    d.polygon([(200, 200), (390, 260), (200, 280)], fill=(120, 110, 90, 255), outline=INK)
    d.ellipse([150, 150, 190, 190], fill=(255, 255, 255, 255), outline=INK, width=5)
    d.rectangle([20, 70, 380, 100], fill=INK)
    return im


def make_head(pipe, prompt, seed):
    if pipe is None:
        return fallback_head()
    try:
        img = txt2img(pipe, prompt, 512, 512, seed).convert('RGBA')
        for pt in [(0, 0), (511, 0), (0, 511), (511, 511)]:
            ImageDraw.floodfill(img, pt, (0, 0, 0, 0), thresh=60)
        box = img.getchannel('A').getbbox()
        return img.crop(box) if box else fallback_head()
    except Exception as e:
        print('!! head failed:', repr(e))
        return fallback_head()


def make_bg(pipe, prompt, seed):
    if pipe is None:
        return fallback_bg(seed)
    try:
        return txt2img(pipe, prompt, 768, 432, seed)
    except Exception as e:
        print('!! bg failed:', repr(e))
        return fallback_bg(seed)


async def synth(text, voice, path):
    import edge_tts
    await edge_tts.Communicate(text, voice).save(str(path))


PIPER_REPO = 'rhasspy/piper-voices'
PIPER_FILE = 'ru/ru_RU/dmitri/medium/ru_RU-dmitri-medium.onnx'


def tts(text, voice, i):
    out = WORK / f's{i}.mp3'
    for attempt in range(2):
        try:
            asyncio.run(synth(text, voice, out))
            return out
        except Exception as e:
            print('!! edge-tts failed:', repr(e)[:200], flush=True)
    from huggingface_hub import hf_hub_download
    model = hf_hub_download(PIPER_REPO, PIPER_FILE)
    hf_hub_download(PIPER_REPO, PIPER_FILE + '.json')
    wav = WORK / f'p{i}.wav'
    r = subprocess.run([sys.executable, '-m', 'piper', '--model', model, '--output_file', str(wav)], input=text, capture_output=True, text=True)
    if r.returncode:
        print(r.stderr[-1500:])
        raise SystemExit('piper failed')
    return wav


def sentences(text):
    return [s for s in re.split(r'(?<=[.!?]) +', text.strip()) if s]


def ken_burns(big, p, flip):
    bw, bh = big.size
    z = 1.12 + 0.18 * p
    cw = bw / z
    ch = cw * H / W
    fx = 0.42 + 0.16 * p if flip else 0.58 - 0.16 * p
    cx = min(max(bw * fx, cw / 2), bw - cw / 2)
    cy = min(max(bh * 0.5, ch / 2), bh - ch / 2)
    box = (int(cx - cw / 2), int(cy - ch / 2), int(cx + cw / 2), int(cy + ch / 2))
    return big.crop(box).resize((W, H), Image.BILINEAR)


def draw_caption(frame, text, f):
    d = ImageDraw.Draw(frame)
    d.rectangle([0, H - BAR, W, H], fill=(0, 0, 0))
    lines, cur = [], ''
    for w in text.split():
        t = (cur + ' ' + w).strip()
        if d.textlength(t, font=f) <= W - 80:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    y = H - BAR + (BAR - len(lines) * 40) // 2
    for ln in lines:
        d.text((W // 2, y), ln, font=f, fill=(255, 255, 255), anchor='ma')
        y += 40


def draw_stick(frame, head, cx, t, talking):
    d = ImageDraw.Draw(frame)
    bob = 7 * math.sin(t * 10) if talking else 2 * math.sin(t * 3)
    hip = (cx, BASE_Y - 170)
    neck = (cx, BASE_Y - 330 + bob * 0.4)
    d.ellipse([cx - 90, BASE_Y - 8, cx + 90, BASE_Y + 10], fill=(40, 40, 40))
    d.line([hip, (cx - 45, BASE_Y)], fill=INK, width=10)
    d.line([hip, (cx + 45, BASE_Y)], fill=INK, width=10)
    d.line([neck, hip], fill=INK, width=10)
    sy = neck[1] + 28
    wave = 24 * math.sin(t * 7) if talking else 0
    d.line([(cx, sy), (cx - 75, sy + 70 + wave)], fill=INK, width=10)
    d.line([(cx, sy), (cx + 75, sy + 60 - wave)], fill=INK, width=10)
    hh = 190
    hw = max(1, int(head.width * hh / head.height))
    h2 = head.resize((hw, hh), Image.LANCZOS)
    frame.paste(h2, (int(cx - hw / 2), int(neck[1] + 28 - hh + bob)), h2)


def render(ep, scenes, head, audio, out_mp4):
    ft = ImageFont.truetype(FONT, 46)
    fc = ImageFont.truetype(FONT, 32)
    cmd = ['ffmpeg', '-y', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
           '-i', str(audio), '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p',
           '-c:a', 'aac', '-b:a', '160k', '-shortest', str(out_mp4)]
    pr = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    shots, gt = [], 0.0
    for si, sc in enumerate(scenes):
        n = int(sc['dur'] * FPS)
        sents = sentences(sc['text'])
        weights = [len(s) for s in sents]
        total = float(sum(weights))
        cx = 360 if si % 2 == 0 else 920
        for f in range(n):
            t = f / FPS
            p = f / max(1, n - 1)
            frame = ken_burns(sc['bg'], p, si % 2 == 0)
            draw_stick(frame, head, cx, gt + t, t < sc['speech'])
            acc, idx = 0.0, 0
            for k, wgt in enumerate(weights):
                acc += wgt / total * sc['speech']
                idx = k
                if t <= acc:
                    break
            draw_caption(frame, sents[idx], fc)
            if si == 0 and t < 3.0:
                ImageDraw.Draw(frame).text((W // 2, 50), ep['title'], font=ft, fill=(255, 255, 255), anchor='mm', stroke_width=5, stroke_fill=INK)
            if t < 0.25:
                frame = Image.blend(Image.new('RGB', (W, H)), frame, t / 0.25)
            if f == int(1.5 * FPS):
                shots.append(frame.copy())
            pr.stdin.write(frame.tobytes())
        gt += sc['dur']
    pr.stdin.close()
    pr.wait()
    if shots:
        cols = 3
        rows = (len(shots) + cols - 1) // cols
        sheet = Image.new('RGB', (640 * cols, 360 * rows))
        for i, im in enumerate(shots):
            sheet.paste(im.resize((640, 360)), ((i % cols) * 640, (i // cols) * 360))
        sheet.save(OUT / 'preview.png')


def main(ep_path):
    ep = json.load(open(ep_path, encoding='utf-8'))
    WORK.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)
    pipe = load_pipe()
    head = make_head(pipe, ep['character']['head_prompt'], 7)
    head.save(WORK / 'head.png')
    scenes = []
    for i, sc in enumerate(ep['scenes']):
        print(f'scene {i + 1}/{len(ep["scenes"])}', flush=True)
        mp3 = WORK / f's{i}.mp3'
        wav = WORK / f's{i}.wav'
        mp3 = tts(sc['text'], ep['voice'], i)
        sh(['ffmpeg', '-y', '-i', mp3, '-af', 'apad=pad_dur=0.6', '-ar', '44100', '-ac', '1', wav])
        bg = make_bg(pipe, ep.get('style', '') + ', ' + sc['bg_prompt'], 100 + i)
        bg.save(WORK / f'bg{i}.png')
        big = bg.resize((int(W * 1.2), int(H * 1.2)), Image.LANCZOS)
        scenes.append({'text': sc['text'], 'dur': duration(wav), 'speech': duration(mp3), 'bg': big})
    lst = WORK / 'audio.txt'
    lst.write_text(''.join(f"file '{(WORK / f's{i}.wav').resolve()}'" + chr(10) for i in range(len(scenes))))
    audio = WORK / 'audio.wav'
    sh(['ffmpeg', '-y', '-f', 'concat', '-safe', '0', '-i', lst, '-c', 'copy', audio])
    render(ep, scenes, head, audio, OUT / 'video.mp4')
    print('done', flush=True)


if __name__ == '__main__':
    main(sys.argv[1])
