"""Generate a dependency-free HTML report from the compact movement database."""

from __future__ import annotations

import html
import json
import sqlite3
from pathlib import Path


def generate_report(database: Path, output: Path) -> None:
    connection = sqlite3.connect(database)
    videos = connection.execute("SELECT id,source_path,fps FROM videos ORDER BY id").fetchall()
    run = connection.execute(
        "SELECT details_json FROM model_runs ORDER BY id DESC LIMIT 1"
    ).fetchone()
    series = []
    coverage_rows = []
    all_near_speeds: list[float] = []
    all_far_speeds: list[float] = []
    all_ball_speeds: list[float] = []
    for video_id, source, fps in videos:
        rows = connection.execute(
            "SELECT frame_index,near_clean_x,near_clean_y,far_clean_x,far_clean_y,"
            "ball_clean_x,ball_clean_y,near_speed_mps,far_speed_mps,ball_speed_pxps "
            "FROM coordinates WHERE video_id=? ORDER BY frame_index",
            (video_id,),
        ).fetchall()
        sampled = rows[:: max(1, len(rows) // 500)]
        series.append({"id": video_id, "source": source, "fps": fps, "points": sampled})
        all_near_speeds.extend(float(row[7]) for row in rows if row[7] is not None)
        all_far_speeds.extend(float(row[8]) for row in rows if row[8] is not None)
        all_ball_speeds.extend(float(row[9]) for row in rows if row[9] is not None)
        total = max(len(rows), 1)
        coverage_rows.append(
            (
                video_id,
                Path(source).name,
                len(rows),
                sum(row[1] is not None for row in rows) / total,
                sum(row[3] is not None for row in rows) / total,
                sum(row[5] is not None for row in rows) / total,
            )
        )
    connection.close()
    metrics = json.loads(run[0]) if run else None
    cards = "<div class='card'><b>Model</b><span>not trained yet</span></div>"
    if metrics:
        player_speeds = all_near_speeds + all_far_speeds
        cards = "".join(
            f"<div class='card'><b>{label}</b><span>{value}</span></div>"
            for label, value in (
                ("Future horizon", f"{metrics['horizon_seconds']:.2f} s"),
                ("Model error", f"{metrics['model_mae_m']:.3f} m"),
                ("90th percentile", f"{metrics.get('model_p90_error_m', 0.0):.3f} m"),
                ("Velocity baseline", f"{metrics['baseline_mae_m']:.3f} m"),
                (
                    "vs. velocity",
                    f"{metrics.get('improvement_over_velocity_pct', 0.0):+.1f}%",
                ),
                ("Validation windows", str(metrics["validation_samples"])),
                (
                    "Peak player speed",
                    "N/A" if not player_speeds else f"{max(player_speeds):.2f} m/s",
                ),
                (
                    "Peak ball image speed",
                    "N/A" if not all_ball_speeds else f"{max(all_ball_speeds):.0f} px/s",
                ),
            )
        )
    source_bytes = sum(
        Path(source).stat().st_size for _, source, _ in videos if Path(source).is_file()
    )
    cards += (
        f"<div class='card'><b>Coordinate database</b><span>{database.stat().st_size / 1024:.0f} KB</span></div>"
        f"<div class='card'><b>Source video referenced</b><span>{source_bytes / 1024 / 1024:.1f} MB</span></div>"
    )
    table = "".join(
        f"<tr><td>{video_id}</td><td>{html.escape(name)}</td><td>{frames}</td>"
        f"<td>{near:.1%}</td><td>{far:.1%}</td><td>{ball:.1%}</td></tr>"
        for video_id, name, frames, near, far, ball in coverage_rows
    )
    payload = json.dumps(series).replace("</", "<\\/")
    prediction_payload = json.dumps(
        None if metrics is None else metrics.get("latest_prediction")
    ).replace("</", "<\\/")
    document = f"""<!doctype html><html><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>Tennis Movement Analyzer</title><style>
body{{margin:0;background:#07140d;color:#e9fff0;font:15px system-ui,sans-serif}}main{{max-width:1050px;margin:auto;padding:32px}}
h1{{font-size:34px;margin-bottom:4px}}.sub{{color:#9fc2aa}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:12px;margin:24px 0}}
.card,section{{background:#10251a;border:1px solid #244a32;border-radius:12px;padding:18px}}.card span{{display:block;font-size:25px;margin-top:8px;color:#89f0aa}}
canvas{{width:100%;height:430px;background:#173924;border-radius:8px}}#speed{{height:240px;background:#0b1b12}}table{{width:100%;border-collapse:collapse}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #244a32}}
.legend i{{display:inline-block;width:11px;height:11px;border-radius:50%;margin:0 6px 0 14px}}</style></head><body><main>
<h1>Tennis Movement Analyzer</h1><div class='sub'>Compact coordinates, player speed, ball image speed, and 0.5-second movement prediction.</div>
<div class='cards'>{cards}</div><section><h2>Top-down movement</h2><div class='legend'><i style='background:#47e675'></i>near player <i style='background:#ffad33'></i>far player</div><canvas id='court' width='960' height='430'></canvas></section>
<section style='margin-top:16px'><h2>Movement speed</h2><div class='legend'><i style='background:#47e675'></i>near m/s <i style='background:#ffad33'></i>far m/s</div><canvas id='speed' width='960' height='240'></canvas></section>
<section style='margin-top:16px'><h2>Extraction coverage</h2><table><tr><th>ID</th><th>Video</th><th>Frames</th><th>Near</th><th>Far</th><th>Ball</th></tr>{table}</table></section>
<script>const data={payload},prediction={prediction_payload};const c=document.getElementById('court'),x=c.getContext('2d');
function p(a,b){{return [70+a/1665*820,25+b/3506*380]}}x.strokeStyle='#d9efe0';x.strokeRect(...p(286,561),1093/1665*820,2374/3506*380);
for(const [off,color] of [[1,'#47e675'],[3,'#ffad33']]){{x.strokeStyle=color;x.lineWidth=2;for(const v of data){{x.beginPath();let started=false;for(const r of v.points)if(r[off]!=null){{const q=p(r[off],r[off+1]);started?x.lineTo(q[0],q[1]):x.moveTo(q[0],q[1]);started=true}}x.stroke()}}}}
if(prediction)for(const [side,color] of [['near','#47e675'],['far','#ffad33']]){{const q=p(prediction[side].court_x,prediction[side].court_y);x.strokeStyle=color;x.lineWidth=3;x.beginPath();x.arc(q[0],q[1],9,0,Math.PI*2);x.stroke();x.fillStyle='#fff';x.fillText('predicted',q[0]+12,q[1])}}
const s=document.getElementById('speed'),g=s.getContext('2d');g.strokeStyle='#355844';g.beginPath();for(let y=30;y<230;y+=40){{g.moveTo(35,y);g.lineTo(950,y)}}g.stroke();
for(const [off,color] of [[7,'#47e675'],[8,'#ffad33']]){{const pts=data.flatMap(v=>v.points).filter(r=>r[off]!=null),n=Math.max(pts.length-1,1);g.beginPath();g.strokeStyle=color;g.lineWidth=2;pts.forEach((r,i)=>{{const px=35+i/n*915,py=220-Math.min(r[off],12)/12*190;i?g.lineTo(px,py):g.moveTo(px,py)}});g.stroke()}}</script>
</main></body></html>"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
