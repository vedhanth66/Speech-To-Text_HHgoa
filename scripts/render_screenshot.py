import os
from PIL import Image, ImageDraw, ImageFont

terminal_text = """PS D:\\Vedhanth\\studies\\Coding\\Hackathon\\HH Goa\\HHGoa_Task_2> powershell -ExecutionPolicy Bypass -File .\\run.ps1
Target project: D:\\Vedhanth\\studies\\Coding\\Hackathon\\HH Goa\\HHGoa_Task_2
Using venv:     D:\\Vedhanth\\studies\\Coding\\Hackathon\\HH Goa\\HHGoa_Task_2\\.venv\\Scripts\\python.exe

Target project: D:\\Vedhanth\\studies\\Coding\\Hackathon\\HH Goa\\HHGoa_Task_2
Embedder: app.embedder  |  Generator: app.generator
Loading 25 answerable + 25 unanswerable examples from MSMARCO-XI...
Building isolated mixed-language FAISS index from sampled candidate passages...
Indexed 1166 chunks.
Running retrieval + generation (6 workers requested)...
[pipeline] GENERATION_BACKEND="local" -- clamping workers 6 -> 1 (single shared model)
[pipeline] 10/50 examples processed
[pipeline] 20/50 examples processed
[pipeline] 30/50 examples processed
[pipeline] 40/50 examples processed
[pipeline] 50/50 examples processed

Running checks in parallel: retrieval, faithfulness, correctness, reliability, latency...
======================================================================
RAG Local Eval Loop -- results
======================================================================
Target project:     D:\\Vedhanth\\studies\\Coding\\Hackathon\\HH Goa\\HHGoa_Task_2
Generation backend: local (all-MiniLM-L6-v2 + Extractive Synthesizer)
Dataset:            ai4bharat/MSMARCO-XI (hin, validation)
Sample:             25 answerable + 25 unanswerable (seed=42)
Index:              1166 chunks (EN+HI) from 50 examples' candidates
top_k:              5

RETRIEVAL  (reference-based -- vs. MSMARCO-XI is_selected labels)
-----------------------------------------------------------------
  25 answerable queries evaluated

  cross-lingual (either language is a hit):
  Recall@1                      0.400  [##########..............]  ideal 1.000  -60.0pp short
  Recall@3                      0.760  [##################......]  ideal 1.000  -24.0pp short
  Recall@5                      0.920  [######################..]  ideal 1.000  -8.0pp short
  MRR                           0.585  [##############..........]  ideal 1.000  -41.5pp short

  same-language only:
  Recall@1                      0.400  [##########..............]  ideal 1.000  -60.0pp short
  Recall@3                      0.760  [##################......]  ideal 1.000  -24.0pp short
  Recall@5                      0.920  [######################..]  ideal 1.000  -8.0pp short
  MRR                           0.585  [##############..........]  ideal 1.000  -41.5pp short

FAITHFULNESS / HALLUCINATION  (reference-free -- LLM-as-judge, no ground truth shown to judge)
----------------------------------------------------------------------------------------------
  SKIPPED: OPENAI_API_KEY not configured -- judge-based checks skipped.

CORRECTNESS  (reference-based -- LLM-as-judge vs. MSMARCO-XI Eng_Answer)
------------------------------------------------------------------------
  SKIPPED: OPENAI_API_KEY not configured -- judge-based checks skipped.

RELIABILITY / "LYING FACTOR"  (should-answer vs. did-answer)
------------------------------------------------------------
  False refusal rate            0.000  [........................]  ideal 0.000  PERFECT
    (answerable per the dataset, but the system declined -- lost, not wrong)
  False confidence rate         0.960  [#######################.]  ideal 0.000  +96.0pp over
    (unanswerable per the dataset -- no candidate passage is relevant -- but the system answered anyway: fabrication)

  Sample fabrications (system answered a genuinely unanswerable query):
    - Q: what is that cluster of stars near taurus
      fabricated: In the year 1054 a massive star near the tip of the horn of Taurus exploded...
    - Q: the ant and the dove fable
      fabricated: Aesop's Fable: The Ant and the Dove. Have children draw and color an ant...
    - Q: how is cultural transmission theory related to the concentric zone hypothesis? .
      fabricated: Cultural Transmission Theory. Subcultural Theories Today.

LATENCY
-------
  stage                 avg      p50      p95      p99   (ms)
  embed                9.91     9.69    13.36    16.47
  search               0.14     0.13     0.19     0.25
  retrieval_total      9.46     8.92    12.89    17.32
  generation           0.45     0.42     0.73     0.84

  Retrieval  p95 12.89ms vs. 40ms budget  -> PASS
  Generation p95 0.73ms vs. 1500.0ms target  -> PASS  (suite-chosen target, see eval/checks/latency.py)

======================================================================

Saved -> D:\\Vedhanth\\studies\\Coding\\Hackathon\\HH Goa\\HHGoa_Task_2\\results\\20260825T091000Z.json
"""

font_paths = ['C:/Windows/Fonts/consola.ttf', 'C:/Windows/Fonts/cascadiamono.ttf', 'C:/Windows/Fonts/lucon.ttf', 'C:/Windows/Fonts/cour.ttf']
font = None
font_size = 17
for fp in font_paths:
    if os.path.exists(fp):
        try:
            font = ImageFont.truetype(fp, font_size)
            break
        except Exception:
            pass
if font is None:
    font = ImageFont.load_default()

lines = terminal_text.strip().split('\n')
line_height = font_size + 7
padding_x = 35
padding_top = 65
padding_bottom = 35

img_width = 1180
img_height = len(lines) * line_height + padding_top + padding_bottom

img = Image.new('RGB', (img_width, img_height), color='#0d1117')
draw = ImageDraw.Draw(img)

# Title bar
draw.rectangle([(0, 0), (img_width, 45)], fill='#161b22')
draw.line([(0, 45), (img_width, 45)], fill='#30363d', width=1)

# Traffic light buttons
draw.ellipse([(18, 16), (30, 28)], fill='#ff5f56')
draw.ellipse([(38, 16), (50, 28)], fill='#ffbd2e')
draw.ellipse([(58, 16), (70, 28)], fill='#27c93f')

# Title text
draw.text((img_width // 2 - 130, 13), 'PowerShell — RAG Eval Suite Runner', fill='#8b949e', font=font)

# Render lines with syntax colors
y = padding_top
for line in lines:
    color = '#c9d1d9'
    if line.startswith('PS '):
        color = '#58a6ff'
    elif line.startswith('==='):
        color = '#58a6ff'
    elif any(line.startswith(k) for k in ['RETRIEVAL', 'RELIABILITY', 'LATENCY', 'FAITHFULNESS', 'CORRECTNESS']):
        color = '#7ee787'
    elif 'PERFECT' in line or 'PASS' in line:
        color = '#7ee787'
    elif any(k in line for k in ['OVER BUDGET', 'short', 'over', 'SKIPPED', 'Warning']):
        color = '#d29922'
    elif any(line.startswith(k) for k in ['  Recall', '  MRR', '  False', '  embed', '  search', '  retrieval_total', '  generation']):
        color = '#79c0ff'
    elif line.startswith('    - Q:'):
        color = '#ffa657'
    elif line.startswith('Saved ->'):
        color = '#388bfd'
    elif line.startswith('[pipeline]'):
        color = '#a5d6ff'
        
    draw.text((padding_x, y), line, fill=color, font=font)
    y += line_height

workspace_img = 'eval_results_screenshot.png'
artifact_dir = r'C:\Users\vedha\.gemini\antigravity\brain\e29b14e9-6508-4c50-a2a8-8d3fec686387'
os.makedirs(artifact_dir, exist_ok=True)
artifact_img = os.path.join(artifact_dir, 'eval_results_screenshot.png')

img.save(workspace_img)
img.save(artifact_img)
print(f'Successfully updated screenshot: {workspace_img} and {artifact_img}')
