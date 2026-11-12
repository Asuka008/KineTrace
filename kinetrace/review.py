"""Build a portable review sheet from a saved compare-top1 report."""
import base64
import json
import random
import re
from html import escape

from .data import output_file


def video_player(video_dir, video_id, label):
    if not re.fullmatch(r"[\w-]+", video_id):
        raise ValueError(f"Invalid video ID: {video_id!r}")
    data = base64.b64encode((video_dir / f"{video_id}.mp4").read_bytes()).decode("ascii")
    return (f"<figure><figcaption>视频 {label}</figcaption>"
            f'<video controls preload="none" src="data:video/mp4;base64,{data}"></video></figure>')


def write_review(args):
    if args.out.suffix.lower() != ".html" or args.out.resolve() == args.input.resolve():
        raise ValueError("Choose a separate .html file for the review")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    pool = [row for row in report["changes"]
            if row["outcome"] == "lost" and row["query_id"] not in args.exclude]
    if not pool:
        raise ValueError("No remaining lost cases in this comparison")
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.limit, len(pool)))
    with output_file(args.out) as page:
        page.write('''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>KineTrace 视频复核</title><style>
body{font:16px/1.5 system-ui;margin:2em auto;padding:0 1em;max-width:960px}
section{border-top:1px solid #aaa;padding:1em 0}.pair{display:flex;gap:1em;flex-wrap:wrap}
figure{flex:1 1 280px;min-width:0;margin:0}video{width:100%;background:#111}
select,button{font:inherit;padding:.3em}details,label{display:block;margin:.6em 0}
</style><h1>KineTrace 视频复核</h1>
<p>根据文字描述判断哪段视频符合，再展开查看原始排名。关闭页面前可下载复核结果。</p>''')
        for row in sample:
            qid = escape(row["query_id"], quote=True)
            clips = [row["clip_top"], row["qwen_top"]]
            rng.shuffle(clips)
            a, b = clips
            page.write(f'<section data-query="{qid}" data-a="{escape(a)}" data-b="{escape(b)}">'
                       f'<h2>{qid} · {escape(row["query"])}</h2><div class="pair">')
            page.write(video_player(args.videos, a, "A"))
            page.write(video_player(args.videos, b, "B"))
            page.write('''</div><label>符合描述的片段 <select>
<option value="">尚未评判</option><option value="A">仅 A</option><option value="B">仅 B</option>
<option value="both">两段都符合</option><option value="neither">两段都不符合</option>
<option value="audio_required">需要声音才能判断</option>
<option value="uncertain">无法判断</option></select></label>''')
            page.write(f'<details><summary>查看视频编号与排名</summary>A = {escape(a)}，B = {escape(b)}。'
                       f'标注目标 {escape(row["target"])} 从第 {int(row["clip_rank"])} 降至第 '
                       f'{int(row["qwen_rank"])}，Qwen 首位为 {escape(row["qwen_top"])}。</details></section>')
        page.write('''<button id="save">下载复核结果</button><script>
document.getElementById("save").onclick = () => {
  const rows = Array.from(document.querySelectorAll("section"), s => ({
    query_id: s.dataset.query, video_a: s.dataset.a, video_b: s.dataset.b,
    judgment: s.querySelector("select").value || null
  }));
  const link = document.createElement("a");
  link.href = URL.createObjectURL(new Blob([JSON.stringify(rows, null, 2)], {type:"application/json"}));
  link.download = "review-judgments.json";
  link.click();
  setTimeout(() => URL.revokeObjectURL(link.href), 1000);
};
</script></html>''')
    return {"review": str(args.out), "eligible": len(pool), "query_ids": [r["query_id"] for r in sample]}
