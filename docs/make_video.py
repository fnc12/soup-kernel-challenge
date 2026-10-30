#!/usr/bin/env python3
"""Builds the demo video: slides (HTML -> PNG, headless Chrome), narration (silero TTS over HTTP),
one mp4 (ffmpeg). Run on casablanca from the repository root:

    python3 docs/make_video.py --out video --speaker aidar --numbers results/t4-run5-packed-2repeats/result.json
"""

import argparse
import html
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

CSS = """
<meta charset="utf-8">
<style>
:root { --bg: #0f1117; --fg: #e8e6e3; --dim: #9aa0a6; --acc: #ffb454; --acc2: #7ad0ff; --code: #161a24; }
* { box-sizing: border-box; }
body { margin: 0; width: 1920px; height: 1080px; background: var(--bg); color: var(--fg);
       font-family: "DejaVu Sans", "Noto Sans", sans-serif; overflow: hidden; }
.slide { padding: 70px 100px; height: 1080px; display: flex; flex-direction: column; gap: 28px; }
h1 { font-size: 64px; margin: 0; letter-spacing: -0.5px; }
h1 small { display: block; font-size: 28px; color: var(--dim); font-weight: normal; margin-top: 10px; }
h2 { font-size: 52px; margin: 0; }
h2 .n { color: var(--acc); margin-right: 18px; }
p, li { font-size: 34px; line-height: 1.35; margin: 0; }
ul { margin: 0; padding-left: 44px; display: flex; flex-direction: column; gap: 14px; }
.row { display: flex; gap: 48px; align-items: flex-start; }
.col { flex: 1; min-width: 0; display: flex; flex-direction: column; gap: 22px; }
table { border-collapse: collapse; font-size: 34px; }
th, td { padding: 12px 28px; text-align: left; border-bottom: 1px solid #2a2f3a; }
th { color: var(--dim); font-weight: normal; font-size: 28px; }
td.num { font-variant-numeric: tabular-nums; }
.big { color: var(--acc); font-weight: bold; }
pre { background: var(--code); border-radius: 14px; padding: 22px 26px; font-size: 20px;
      line-height: 1.4; margin: 0; font-family: "DejaVu Sans Mono", monospace; overflow: hidden;
      border-left: 6px solid var(--acc2); white-space: pre; max-height: 740px; width: 100%; }
.col.code { flex: 0 0 900px; }
.foot { margin-top: auto; color: var(--dim); font-size: 26px; display: flex; justify-content: space-between; }
.bars { display: flex; flex-direction: column; gap: 18px; font-size: 30px; }
.bar { display: flex; align-items: center; gap: 20px; }
.bar b { width: 420px; text-align: right; font-weight: normal; color: var(--fg); }
.bar i { display: block; height: 40px; background: var(--acc2); border-radius: 6px; }
.bar.pad i { background: var(--acc); }
.bar span { color: var(--dim); }
.checks { font-size: 30px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px 50px; }
.checks div::before { content: "✓ "; color: #7bd88f; }
</style>
"""


def slide(body, foot_left="Студия Евгения Захарова · Soup Kernel Challenge · Decentrathon 6.0", foot_right=""):
    return f"<!doctype html><html><head>{CSS}</head><body><div class='slide'>{body}" \
           f"<div class='foot'><span>{foot_left}</span><span>{foot_right}</span></div></div></body></html>"


def code(path, marker, n, width=96):
    """`n` lines of `path` from the first line containing `marker`, cut to `width` columns."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    start = next(i for i, line in enumerate(lines) if marker in line)
    picked = [line[:width] for line in lines[start:start + n]]
    return "<pre>" + html.escape("\n".join(picked)) + "</pre>"


def num(n):
    return f"{n:,}".replace(",", "\u202f")


def build_scenes(n):
    """Returns [(slide html, narration)]. n: numbers from result.json."""
    b, s, r = n["base"], n["solution"], n["ratios"]
    speed, mem = r["speedup"], r["memory_ratio"]
    scenes = []
    scenes.append((slide(f"""
<h1>Soup Kernel Challenge<small>Triton-кернелы для soup train на Kaggle T4 · Студия Евгения Захарова</small></h1>
<div class='row'><div class='col'>
<table>
<tr><th></th><th>База Soup 0.77.0</th><th>Решение</th><th></th></tr>
<tr><td>Вся команда soup train</td><td class='num'>{b['wall_s']:.1f} с</td><td class='num'>{s['wall_s']:.1f} с</td><td class='big'>{speed:.2f}×</td></tr>
<tr><td>Пик памяти GPU</td><td class='num'>{num(b['gpu_peak_mib'])} МиБ</td><td class='num'>{num(s['gpu_peak_mib'])} МиБ</td><td class='big'>{mem:.2f}×</td></tr>
<tr><td>Eval loss адаптера</td><td class='num'>{b['eval_loss']:.5f}</td><td class='num'>{s['eval_loss']:.5f}</td><td>{r['quality_delta_pct']:+.3f}%</td></tr>
</table>
<p>Qwen2.5-1.5B-Instruct · NF4 · LoRA r=16 · batch 4 × 512 · 64 шагов · один Tesla T4</p>
<p>Скрипты организаторов, протокол жюри: два прогона на сторону, все 10 проверок пройдены.</p>
</div></div>"""),
        f"Это решение Студии Евгения Захарова для Soup Kernel Challenge. Задача: soup train дообучает "
        f"Qwen два с половиной на четыре бита с LoRA на казахских инструкциях на бесплатной Kaggle T4. "
        f"База занимает около девяноста секунд и четырнадцать гигабайт памяти из пятнадцати. "
        f"Наш результат на тех же скриптах организаторов: в полтора раза быстрее и в три раза меньше памяти, "
        f"все десять проверок пройдены, качество адаптера то же."))
    scenes.append((slide("""
<h2><span class='n'>1</span>Куда уходил шаг базы на T4</h2>
<div class='row'><div class='col'>
<div class='bars'>
<div class='bar'><b>поэлементные операции и касты</b><i style='width:620px'></i><span>44%</span></div>
<div class='bar'><b>матмулы</b><i style='width:510px'></i><span>36%</span></div>
<div class='bar'><b>softmax по словарю 151 936</b><i style='width:100px'></i><span>7%</span></div>
<div class='bar'><b>attention</b><i style='width:70px'></i><span>5%</span></div>
<div class='bar'><b>распаковка NF4</b><i style='width:70px'></i><span>5%</span></div>
<div class='bar pad'><b>паддинг в батчах</b><i style='width:590px'></i><span>42% позиций</span></div>
</div>
<ul>
<li>bitsandbytes после каждого 4-битного матмула кастит результат обратно в fp32</li>
<li>логиты 4 × 512 × 151 936: fp16, fp32-копия, log-softmax, ещё два прохода TRL и Soup</li>
<li>батч добивается до самой длинной строки: средняя строка 207 токенов, максимум 512</li>
</ul></div></div>"""),
        "Куда уходил шаг базы. Сорок четыре процента на поэлементные операции и приведения типов: "
        "bitsandbytes после каждого четырёхбитного матмула переводит результат обратно в fp32. "
        "Логиты по словарю в сто пятьдесят две тысячи токенов занимают гигабайты. И главное: батч "
        "добивается паддингом до самой длинной строки, а строки короткие, поэтому сорок два процента "
        "вычислений шли впустую."))
    scenes.append((slide("""
<h2><span class='n'>2</span>Слитый выходной слой и cross-entropy по кускам</h2>
<div class='row'><div class='col'>
<ul>
<li>матмул головы по 256 позиций, логиты куска живут только внутри куска</li>
<li>один кернел за один проход: log-sum-exp, потеря, градиент логитов, энтропия, argmax</li>
<li>градиент по hidden считается тут же: ни одного полного тензора логитов в памяти</li>
<li>колонки entropy, accuracy, num_tokens и построчные потери rewind — те же формулы</li>
</ul></div>
<div class='col code'>""" + code("solution/cross_entropy.py", "    row = tl.program_id(0)", 22) + "</div></div>"),
        "Что мы сделали, семь Triton-кернелов. Первый: слитый выходной слой с cross-entropy по "
        "кускам строк. Логиты целиком в памяти не появляются; один кернел за один проход по куску "
        "считает log-sum-exp, потерю, градиент, энтропию и точность, те самые колонки, которые "
        "логирует TRL и пишет rewind Soup."))
    scenes.append((slide("""
<h2><span class='n'>3</span>Распаковка NF4 в один проход и склеенные веса</h2>
<div class='row'><div class='col'>
<ul>
<li>bitsandbytes: два кернела на проекцию (absmax, веса) + cuBLAS, и так же в backward</li>
<li>наш кернел: оба уровня квантования сразу, одна слитная запись fp16-строк</li>
<li>q | k | v → одна матрица 2048 × 1536, gate | up → одна 17 920 × 1536</li>
<li>одна распаковка и один GEMM вместо трёх узких; исходники освобождают память</li>
</ul></div>
<div class='col code'>""" + code("solution/quant.py", "    pid = tl.program_id(0)", 18) + "</div></div>"),
        "Второй: распаковка NF4 в один проход, оба уровня квантования сразу, и для прямого, и для "
        "обратного прохода. Веса q, k, v склеены в одну матрицу, gate и up тоже: одна распаковка и "
        "один GEMM вместо трёх узких."))
    scenes.append((slide("""
<h2><span class='n'>4</span>SwiGLU, RMSNorm, RoPE, fp16 между матмулами</h2>
<div class='row'><div class='col'>
<ul>
<li>двумерный SwiGLU поверх склеенного gate|up-буфера, backward пишет один [R, 2I]-градиент</li>
<li>RMSNorm: fp32-статистика, fp16 на выходе, backward из входа и одного скаляра на строку</li>
<li>RoPE для q и k одним кернелом в каждую сторону вместо цепочки из десяти операций</li>
<li>активации остаются в fp16, residual-поток fp32; таблица эмбеддингов снова в fp16, в котором Soup её загрузил</li>
</ul></div>
<div class='col code'>""" + code("solution/rmsnorm.py", "def rmsnorm_fwd_kernel", 14) + "</div></div>"),
        "Дальше: двумерный SwiGLU поверх склеенного буфера, слитый RMSNorm с fp16 на выходе, RoPE "
        "одним кернелом. Активации между матмулами остаются в fp16, таблица эмбеддингов возвращена в "
        "fp16, в котором Soup её и загрузил."))
    scenes.append((slide("""
<h2><span class='n'>5</span>Паддинг убран внутри батча</h2>
<div class='row'><div class='col'>
<ul>
<li>после эмбеддингов остаются только реальные токены: [1, N_real, D]</li>
<li>нормы, проекции, LoRA, MLP и выходной слой считают по ним</li>
<li>attention: причинный SDPA по каждой строке на её токенах, без маски и блоков паддинга</li>
<li>строки между батчами не смешиваются; маски dropout у LoRA те же, что у базы</li>
</ul></div>
<div class='col code'>""" + code("solution/packed.py", "def per_row_attention", 20) + "</div></div>"),
        "И самое весомое по скорости: паддинг убран внутри батча. После эмбеддингов остаются только "
        "реальные токены, все слои считают по ним, а attention идёт по каждой строке отдельно, "
        "причинно, без маски и без блоков паддинга. Строки между батчами не смешиваются, маски "
        "dropout у LoRA те же, что у базы."))
    checks = "".join(f"<div>{c}</div>" for c in [
        "все прогоны завершились", "без готовых кернел-библиотек", "свои @triton.jit работают в обучении",
        "доля своих кернелов ≥ 5%", "та же задача и выходы Soup", "кривая потерь как у базы",
        "качество: eval loss в допуске", "не хуже базы по времени и памяти", "чистый замер на T4",
        "79 регрессионных тестов Soup"])
    scenes.append((slide(f"""
<h2><span class='n'>6</span>Замер по протоколу жюри</h2>
<div class='row'><div class='col'>
<table>
<tr><th></th><th>База</th><th>Решение</th><th></th></tr>
<tr><td>soup train, среднее двух прогонов</td><td class='num'>{b['wall_s']:.1f} с</td><td class='num'>{s['wall_s']:.1f} с</td><td class='big'>{speed:.2f}×</td></tr>
<tr><td>пик памяти, nvidia-smi</td><td class='num'>{num(b['gpu_peak_mib'])} МиБ</td><td class='num'>{num(s['gpu_peak_mib'])} МиБ</td><td class='big'>{mem:.2f}×</td></tr>
<tr><td>eval loss</td><td class='num'>{b['eval_loss']:.5f}</td><td class='num'>{s['eval_loss']:.5f}</td><td>{r['quality_delta_pct']:+.3f}%</td></tr>
</table>
<p>Кривая потерь: первый шаг в пределах 0.01%, в среднем 0.04%. Режим SOLUTION_RECOMPUTE=1: память 3.9×, время 1.4×.</p>
<div class='checks'>{checks}</div>
</div></div>"""),
        f"Замер по протоколу жюри, два прогона на сторону: база {b['wall_s']:.0f} секунд, решение "
        f"{s['wall_s']:.0f}, пик памяти {b['gpu_peak_mib']} мегабайт против {s['gpu_peak_mib']}. "
        "Кривая потерь в пределах четырёх сотых процента от базы, eval loss совпадает. Переключатель "
        "пересчёта MLP даёт четырёхкратную экономию памяти ценой десяти процентов времени."))
    scenes.append((slide("""
<h2><span class='n'>7</span>Ограничения и что дальше</h2>
<div class='row'><div class='col'>
<ul>
<li>attention остаётся на SDPA PyTorch, по строкам; свой кернел по упакованным строкам — следующий шаг</li>
<li>распаковка NF4 отдельно от GEMM: слитый вариант должен обогнать cuBLAS на T4</li>
<li>патчи рассчитаны на Qwen2 + NF4 (блоки 64, двойное квантование) + обычную LoRA PEFT</li>
<li>в репозитории: код, тест каждого кернела против PyTorch и bitsandbytes, сырые результаты всех прогонов</li>
</ul>
<p>github.com/BAITC-Hacks/hack-a8ca734b-solo-ai-team</p>
</div></div>"""),
        "Что дальше: свой кернел attention по упакованным строкам и слитая распаковка с GEMM. Код, "
        "тесты каждого кернела и сырые результаты в репозитории. Спасибо."))
    return scenes


_V4 = {}


def tts(text, speaker, out_wav, url):
    """Narration: silero v4_ru (native Russian voices aidar, baya, kseniya, xenia, eugene) loaded
    through torch.hub when `url` is "v4_ru"; otherwise an HTTP silero service."""
    if url == "v4_ru":
        import torch

        if "model" not in _V4:
            model, _ = torch.hub.load("snakers4/silero-models", "silero_tts", language="ru",
                                      speaker="v4_ru", trust_repo=True)
            _V4["model"] = model
        audio = _V4["model"].apply_tts(text=text, speaker=speaker, sample_rate=48000,
                                       put_accent=True, put_yo=True)
        import wave

        pcm = (audio.clamp(-1, 1) * 32767).to(torch.int16).numpy().tobytes()
        with wave.open(str(out_wav), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(48000)
            w.writeframes(pcm)
        return
    req = urllib.request.Request(url, data=json.dumps({"text": text, "speaker": speaker,
                                                       "sample_rate": 48000}).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        Path(out_wav).write_bytes(resp.read())


def duration(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                          "csv=p=0", str(path)], capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="video")
    ap.add_argument("--speaker", default="aidar")
    ap.add_argument("--tts-url", default="v4_ru", help='"v4_ru" (torch.hub) or a silero HTTP endpoint')
    ap.add_argument("--numbers", default="results/t4-run5-packed-2repeats/result.json")
    ap.add_argument("--chrome", default="google-chrome")
    ap.add_argument("--pad", type=float, default=0.8, help="seconds of silence after each scene")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    numbers = json.load(open(args.numbers))["summary"]
    scenes = build_scenes(numbers)
    parts = []
    for i, (page, text) in enumerate(scenes, 1):
        h = out / f"scene{i}.html"
        png = out / f"scene{i}.png"
        wav = out / f"scene{i}.wav"
        h.write_text(page, encoding="utf-8")
        subprocess.run([args.chrome, "--headless=new", "--disable-gpu", "--no-sandbox",
                        "--hide-scrollbars", "--force-device-scale-factor=1",
                        "--window-size=1920,1080", f"--screenshot={png}", f"file://{h.resolve()}"],
                       check=True, capture_output=True)
        tts(text, args.speaker, wav, args.tts_url)
        d = duration(wav) + args.pad
        seg = out / f"scene{i}.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-loop", "1", "-framerate", "25", "-i", str(png),
                        "-i", str(wav), "-filter_complex", f"[1:a]apad=pad_dur={args.pad}[a]",
                        "-map", "0:v", "-map", "[a]", "-t", f"{d:.3f}", "-c:v", "libx264", "-preset",
                        "medium", "-crf", "20", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "160k",
                        "-shortest", str(seg)], check=True)
        parts.append(seg)
        print(f"scene {i}: {d:.1f} s", file=sys.stderr)
    lst = out / "list.txt"
    lst.write_text("".join(f"file '{p.resolve()}'\n" for p in parts))
    final = out / "soup-kernel-challenge.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst), "-c",
                    "copy", str(final)], check=True)
    print(f"{final}: {duration(final):.1f} s, {os.path.getsize(final) / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
