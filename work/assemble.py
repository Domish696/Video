"""Concatenate rendered shots, add the closing card + fade, mux the mix.
usage: assemble.py <shotdir> <out.mp4>"""
import os, sys, subprocess
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(__file__))
from lib import S, OUT_W, OUT_H
from edl import timeline

shotdir = sys.argv[1] if len(sys.argv) > 1 else S + '/shots'
out = sys.argv[2] if len(sys.argv) > 2 else S + '/out/preview.mp4'
os.makedirs(os.path.dirname(out), exist_ok=True)
tl, T = timeline()

with open(S + '/out/concat.txt', 'w') as f:
    for s, t, n in tl:
        f.write(f"file '{shotdir}/{s['id']}.mp4'\n")


def font(path, size, var):
    ft = ImageFont.truetype(path, size)
    try:
        ft.set_variation_by_name(var)
    except Exception:
        pass
    return ft


def spaced(draw, text, ft, tracking):
    w = 0
    for ch in text:
        w += draw.textlength(ch, font=ft) + tracking
    return w - tracking


def draw_spaced(draw, x, y, text, ft, tracking, fill):
    for ch in text:
        draw.text((x, y), ch, font=ft, fill=fill)
        x += draw.textlength(ch, font=ft) + tracking


card = Image.new('RGBA', (OUT_W, OUT_H), (0, 0, 0, 0))
d = ImageDraw.Draw(card)
f1 = font(S + '/fonts/Montserrat.ttf', 50, b'Light')
f2 = font(S + '/fonts/Montserrat.ttf', 27, b'Regular')
l1 = 'CAMPO DE MARTE'; l2 = 'DOMINGO AÉREO  ·  27.09.2026'
tr1, tr2 = 17, 7
w1 = spaced(d, l1, f1, tr1); w2 = spaced(d, l2, f2, tr2)
y1 = 1508
shadow = Image.new('RGBA', (OUT_W, OUT_H), (0, 0, 0, 0)); ds = ImageDraw.Draw(shadow)
for dd, a in ((ds, (0, 0, 0, 120)), (d, (255, 255, 255, 236))):
    draw_spaced(dd, (OUT_W - w1) / 2, y1, l1, f1, tr1, a)
    dd.line([(OUT_W / 2 - 34, y1 + 86), (OUT_W / 2 + 34, y1 + 86)], fill=a, width=2)
    draw_spaced(dd, (OUT_W - w2) / 2, y1 + 112, l2, f2, tr2, (a[0], a[1], a[2], int(a[3] * 0.9)))
shadow = shadow.filter(ImageFilter.GaussianBlur(6))
final = Image.alpha_composite(shadow, card)
final.save(S + '/out/card.png')

t_card = T - 2.55
t_fade = T - 0.85
vf = (f"[1:v]format=rgba,fade=t=in:st={t_card:.3f}:d=0.7:alpha=1,fade=t=out:st={t_fade + 0.05:.3f}:d=0.7:alpha=1[c];"
      f"[0:v][c]overlay=0:0:enable='gte(t,{t_card:.3f})',fade=t=out:st={t_fade:.3f}:d=0.85,format=yuv420p[v]")
cmd = ['ffmpeg', '-v', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', S + '/out/concat.txt',
       '-loop', '1', '-framerate', '24', '-t', f'{T:.3f}', '-i', S + '/out/card.png',
       '-i', S + '/mix/final_audio.wav',
       '-filter_complex', vf, '-map', '[v]', '-map', '2:a',
       '-c:v', 'libx264', '-preset', 'slow', '-crf', os.environ.get('CRF', '17'), '-profile:v', 'high', '-level', '4.1',
       '-pix_fmt', 'yuv420p', '-r', '24', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-colorspace', 'bt709',
       '-c:a', 'aac', '-b:a', '320k', '-ar', '48000', '-t', f'{T:.3f}', '-movflags', '+faststart', out]
subprocess.run(cmd, check=True)
print('wrote', out, T)
