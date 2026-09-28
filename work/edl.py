"""Edit decision list. Timeline at 24 fps, music grid 120 BPM (beat = 12 frames, bar = 2 s).

mode:
  native  - portrait source used full frame
  track   - 9:16 virtual camera following the detected aircraft (lead = offset of the crop centre
            relative to the subject, in crop widths; + puts space on the right)
  fixed   - static 9:16 window (cx normalised) with handheld jitter removed
  stack   - dual-camera clip: rear camera (what they saw) on top, selfie (them) below
"""

SHOTS = [
    # ---- HOOK: KC-390 Millennium low pass (outside the fence) -------------------------------
    dict(id='S01', src='WA0035', t_in=2.90, dur=4.50, mode='native', zoom=0.97, stab=0.0,
         grade=dict(expo=0.10, contrast=0.16, sat=1.04), note='KC-390 aproxima contra o sol e passa por cima'),
    # ---- PLACE: the crowd, the day ----------------------------------------------------------
    dict(id='S02', src='WA0052', t_in=0.40, dur=3.50, mode='fixed', cx=0.70, zoom=0.93, stab=1.0, sr_dn=0.20, face_mix=0.55,
         grade=dict(sat=1.03, contrast=0.14), note='multidão, caminhão de som, aeronaves estáticas'),
    dict(id='S03', src='WA0034', t_in=0.35, dur=2.50, mode='stack',
         grade=dict(sat=1.03, contrast=0.12), note='Rodolfo + tio Ferrari (câmera dupla)'),
    dict(id='S04', src='WA0032', t_in=7.45, dur=3.50, mode='stack',
         grade=dict(sat=1.03, contrast=0.12), note='Rodolfo, crianças acenando, a formação passa'),
    # ---- THE SHOW OPENS ---------------------------------------------------------------------
    dict(id='S05', src='WA0033', t_in=4.90, dur=4.00, mode='track', lead=-0.06,
         grade=dict(sat=1.04, contrast=0.16, dehaze=0.18), note='duplo-diamante chegando pela esquerda'),
    dict(id='S06', src='WA0082', t_in=1.75, dur=4.00, mode='track', lead=-0.10,
         grade=dict(sat=1.04, contrast=0.16, dehaze=0.3), note='solo: tonneau de quatro tempos'),
    dict(id='S07', src='WA0060', t_in=1.80, dur=4.00, mode='track', lead=-0.08,
         grade=dict(sat=1.04, contrast=0.16, dehaze=0.32), note='par: manobra clássica, mergulho'),
    # ---- PIVOT: verticals and a crossing ----------------------------------------------------
    dict(id='S08', src='WA0040', t_in=0.90, dur=3.50, mode='track', lead=0.0,
         grade=dict(sat=1.04, contrast=0.16, dehaze=0.15), note='solo sobe na vertical'),
    dict(id='S09', src='WA0064', t_in=5.95, dur=3.50, mode='track', lead=0.0,
         xkeys=[(6.9, 0.53), (7.5, 0.60), (8.0, 0.70), (8.9, 0.72), (9.5, 0.70)],
         grade=dict(sat=1.04, contrast=0.16, dehaze=0.12), note='cruzamento, A-29 colorido, subida vertical'),
    # ---- ESCALATION: formations -------------------------------------------------------------
    dict(id='S10', src='WA0070', t_in=9.40, dur=5.00, mode='track', lead=0.04,
         grade=dict(sat=1.04, contrast=0.16, dehaze=0.28), note='formação sobe e cresce no quadro'),
    dict(id='S11', src='WA0102', t_in=11.60, dur=3.00, mode='track', lead=0.08,
         grade=dict(sat=1.04, contrast=0.16, dehaze=0.15), note='seis A-29 em formação cerrada'),
    dict(id='S12', src='WA0044', t_in=16.40, dur=2.00, mode='track', lead=0.06,
         grade=dict(sat=1.04, contrast=0.16, dehaze=0.18), note='wing com troca: A-29 colorido gira sobre o outro'),
    # ---- APEX -------------------------------------------------------------------------------
    dict(id='S13', src='WA0046', t_in=17.00, dur=6.50, mode='track', lead=0.10,
         grade=dict(sat=1.04, contrast=0.16, dehaze=0.25), note='voo de dorso em formação (recorde mundial)'),
    dict(id='S14', src='WA0098', t_in=11.90, dur=2.75, mode='track', lead=0.06,
         grade=dict(sat=1.04, contrast=0.16, dehaze=0.18), note='diamante passa e mergulha'),
    # ---- AFTERGLOW --------------------------------------------------------------------------
    dict(id='S15', src='WA0054', t_in=22.05, dur=1.25, mode='fixed', cx=0.34, cy=0.56, zoom=0.80, stab=1.0, sr_dn=0.20, face_mix=0.55,
         cxkeys=[(22.0, 0.35), (22.9, 0.33), (23.3, 0.27)],
         grade=dict(sat=1.03, contrast=0.14), note='criança nos ombros ergue os braços'),
    dict(id='S16', src='WA0034', t_in=9.70, dur=2.00, mode='stack',
         grade=dict(sat=1.03, contrast=0.12), note='Rodolfo gargalha, A-29 passa atrás'),
    dict(id='S17', src='WA0032', t_in=16.15, dur=1.25, mode='stack',
         grade=dict(sat=1.03, contrast=0.12), note='Ferrari e Rodolfo, sinal de V'),
    dict(id='S18', src='WA0070', t_in=15.40, dur=3.25, mode='track', lead=-0.04,
         grade=dict(sat=1.04, contrast=0.16, dehaze=0.15), note='a formação se afasta no azul'),
]


def timeline():
    t = 0.0
    out = []
    for s in SHOTS:
        n = int(round(s['dur'] * 24))
        out.append((s, t, n))
        t += n / 24
    return out, t


if __name__ == '__main__':
    tl, T = timeline()
    for s, t, n in tl:
        print(f"{s['id']} {t:6.2f}-{t + n / 24:6.2f} ({n:3d}f) {s['src']} in={s['t_in']:.2f} {s['mode']:6s} {s['note']}")
    print('TOTAL', T)
