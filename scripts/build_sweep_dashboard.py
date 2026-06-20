#!/usr/bin/env python3
"""Parse the lift-gate sweep logs and generate a self-contained HTML dashboard.

Reads outputs/lift_sweep_h0XX.log (one per minimal_height value), extracts the
per-iteration training series, downsamples, computes summary success indicators,
and writes a single offline HTML file with the data embedded inline plus
interactive SVG charts. Reproducible: re-run after new sweeps.
"""
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RUNS = [("0.04", "lift_sweep_h004.log"), ("0.06", "lift_sweep_h006.log"), ("0.08", "lift_sweep_h008.log")]

# Labels we pull out of each "Learning iteration" block.
FIELDS = {
    "mean_reward": re.compile(r"Mean reward:\s*([-0-9.eE]+)"),
    "reaching": re.compile(r"Episode_Reward/reaching_object:\s*([-0-9.eE]+)"),
    "lifting": re.compile(r"Episode_Reward/lifting_object:\s*([-0-9.eE]+)"),
    "goal": re.compile(r"Episode_Reward/object_goal_tracking:\s*([-0-9.eE]+)"),
    "position_error": re.compile(r"Metrics/object_pose/position_error:\s*([-0-9.eE]+)"),
}
ITER_RE = re.compile(r"Learning iteration\s+(\d+)/")


def parse_log(path):
    series = {k: [] for k in FIELDS}
    iters = []
    cur = None
    for line in path.read_text(errors="ignore").splitlines():
        m = ITER_RE.search(line)
        if m:
            if cur is not None:
                iters.append(cur["_it"])
                for k in FIELDS:
                    series[k].append(cur.get(k, None))
            cur = {"_it": int(m.group(1))}
            continue
        if cur is None:
            continue
        for k, rx in FIELDS.items():
            mm = rx.search(line)
            if mm:
                cur[k] = float(mm.group(1))
    if cur is not None:
        iters.append(cur["_it"])
        for k in FIELDS:
            series[k].append(cur.get(k, None))
    return iters, series


def downsample(xs, ys, step):
    return xs[::step], ys[::step]


def safe(vals):
    return [v for v in vals if v is not None]


def main():
    data = {}
    summary = {}
    for gate, fname in RUNS:
        p = REPO / "outputs" / fname
        if not p.exists():
            print(f"WARN missing {p}", file=sys.stderr)
            continue
        iters, series = parse_log(p)
        step = max(1, len(iters) // 300)
        di, _ = downsample(iters, iters, step)
        ds = {k: downsample(iters, series[k], step)[1] for k in FIELDS}
        data[gate] = {"iter": di, **ds}
        lift = safe(series["lifting"])
        reach = safe(series["reaching"])
        goal = safe(series["goal"])
        rew = safe(series["mean_reward"])
        pe = safe(series["position_error"])
        final_lift = sum(safe(series["lifting"][-20:])) / max(1, len(safe(series["lifting"][-20:])))
        summary[gate] = {
            "lift_max": max(lift) if lift else 0,
            "lift_final": final_lift,
            "reach_max": max(reach) if reach else 0,
            "goal_max": max(goal) if goal else 0,
            "reward_peak": max(rew) if rew else 0,
            "reward_final": rew[-1] if rew else 0,
            "pe_best": min(pe) if pe else 0,
            "lifted": (max(lift) if lift else 0) > 0.05 and final_lift > 0.02,
        }
    blob = json.dumps({"data": data, "summary": summary})
    html = TEMPLATE.replace("/*DATA*/", blob)
    out = REPO / "docs" / "learning" / "lift-gate-sweep-dashboard.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html)} bytes)")
    # also mirror to the main repo for easy viewing
    main_out = Path.home() / "isaaclab-workspace" / "docs" / "learning" / "lift-gate-sweep-dashboard.html"
    if main_out.parent.exists():
        main_out.write_text(html)
        print(f"mirrored to {main_out}")


TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lift-Gate Sweep — Data Analysis Dashboard</title>
<style>
  :root{--bg:#0e1116;--panel:#161b22;--ink:#e6edf3;--mut:#9aa7b4;--acc:#58a6ff;--acc2:#7ee787;--warn:#ffa657;--red:#ff7b72;--line:#2a313c;--code:#0b0f14;--eqbg:#0c131c}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);font:16px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
  .wrap{max-width:1000px;margin:0 auto;padding:40px 22px 100px}
  .kicker{color:var(--acc);font-weight:700;letter-spacing:.13em;text-transform:uppercase;font-size:12px}
  h1{font-size:31px;line-height:1.15;margin:.2em 0 .1em}
  .deck{color:var(--mut);font-size:17px;margin:0 0 18px;max-width:70ch}
  .meta{color:var(--mut);font-size:13px;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:10px 0;margin:14px 0 22px;display:flex;gap:22px;flex-wrap:wrap}
  .meta b{color:var(--ink)}
  h2{font-size:21px;margin:40px 0 12px;padding-top:12px;border-top:1px solid var(--line)}
  p{margin:0 0 14px;max-width:74ch}.mut{color:var(--mut)}
  code{background:var(--code);padding:.1em .4em;border-radius:5px;font-size:.88em;font-family:"SF Mono",ui-monospace,Menlo,monospace;color:var(--acc2)}
  .finding{border-left:3px solid var(--acc2);background:#10261a;padding:14px 18px;border-radius:0 8px 8px 0;margin:18px 0}
  .finding b{color:var(--acc2)}
  .note{border-left:3px solid var(--acc);background:#11202e;padding:13px 18px;border-radius:0 8px 8px 0;margin:18px 0}
  table{border-collapse:collapse;width:100%;margin:16px 0;font-size:14.5px}
  th,td{border:1px solid var(--line);padding:9px 12px;text-align:left}
  th{background:var(--panel);color:var(--acc)}
  tr.best td{background:#10261a}
  .cards{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:18px 0}
  @media(max-width:680px){.cards{grid-template-columns:1fr}}
  .card{border:1px solid var(--line);border-radius:12px;padding:16px;background:var(--panel)}
  .card h3{margin:0 0 8px;font-size:16px;display:flex;justify-content:space-between;align-items:center}
  .card .badge{font-size:11px;padding:2px 8px;border-radius:999px}
  .card .ok{background:#15311f;color:#7ee787}.card .no{background:#3a1a1a;color:#ff9e9e}
  .card .row{display:flex;justify-content:space-between;font-size:13px;padding:3px 0;color:var(--mut)}
  .card .row b{color:var(--ink);font-family:"SF Mono",ui-monospace,monospace}
  .controls{display:flex;gap:18px;flex-wrap:wrap;align-items:center;background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:14px 18px;margin:18px 0}
  .controls .grp{display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .controls label{font-size:13px;color:var(--mut)}
  .controls button{font:inherit;cursor:pointer;border:1px solid var(--line);background:#1b2230;color:var(--mut);border-radius:8px;padding:5px 12px;font-size:12.5px}
  .controls input[type=range]{accent-color:var(--acc);width:140px}
  .grid2{display:grid;grid-template-columns:1fr 1fr;gap:16px}
  @media(max-width:680px){.grid2{grid-template-columns:1fr}}
  .chartbox{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px}
  .chartbox h4{margin:2px 0 4px;font-size:14px;color:var(--ink)}
  .chartbox .sub{font-size:11.5px;color:var(--mut);margin-bottom:4px}
  svg{width:100%;height:auto;background:var(--eqbg);border-radius:8px}
  .foot{margin-top:50px;color:var(--mut);font-size:13px;border-top:1px solid var(--line);padding-top:16px}
  .swatch{display:inline-block;width:11px;height:11px;border-radius:3px;vertical-align:middle;margin-right:5px}
</style>
</head>
<body>
<div class="wrap">
  <div class="kicker">Data analysis · Franka Lift</div>
  <h1>Lift-Gate Parameter Sweep</h1>
  <p class="deck">Does an easier lift threshold actually teach the policy to lift? We swept the <code>minimal_height</code> gate — the height the cube must clear before the big lift bonus and goal-tracking reward unlock — across three values, holding everything else fixed, and measured success on weight-independent indicators.</p>
  <div class="meta">
    <span><b>Task</b> Isaac-Lift-Cube-Franka</span>
    <span><b>Controlled</b> seed 42 · 128 envs · 1200 iters · PPO (rsl_rl)</span>
    <span><b>Swept</b> minimal_height &#8712; {0.04, 0.06, 0.08} m</span>
    <span><b>Hardware</b> laptop RTX 4050 (6 GB)</span>
  </div>

  <div class="finding" id="finding"></div>

  <h2>Success indicators per gate</h2>
  <p class="mut">Reward totals aren't comparable across configs (Ch. 04), so we read behavior off these: <b>lift</b> = did the cube clear the gate and stay up (<code>lifting_object</code>), <b>reach</b> = did it find the cube, <b>carry</b> = best object&#8594;goal distance.</p>
  <div class="cards" id="cards"></div>

  <h2>Learning curves</h2>
  <div class="controls">
    <div class="grp" id="gateToggles"><label>show gates:</label></div>
    <div class="grp"><label for="sm">smoothing</label><input type="range" id="sm" min="0" max="0.9" step="0.05" value="0.6"><span id="smv" class="mut" style="font-family:monospace;font-size:12px">0.60</span></div>
  </div>
  <div class="grid2">
    <div class="chartbox"><h4>Lift success &#8212; <code>lifting_object</code> reward</h4><div class="sub">the key signal: &gt;0 and sustained = the cube is being lifted past the gate</div><svg id="c_lifting" viewBox="0 0 460 220"></svg></div>
    <div class="chartbox"><h4>Mean reward</h4><div class="sub">note the rise&#8594;collapse pattern (Ch. 04 pathology)</div><svg id="c_mean_reward" viewBox="0 0 460 220"></svg></div>
    <div class="chartbox"><h4>Object&#8594;goal position error (m)</h4><div class="sub">lower = cube carried closer to the goal</div><svg id="c_position_error" viewBox="0 0 460 220"></svg></div>
    <div class="chartbox"><h4>Reaching &#8212; <code>reaching_object</code> reward</h4><div class="sub">stage 1: finding/approaching the cube</div><svg id="c_reaching" viewBox="0 0 460 220"></svg></div>
  </div>

  <h2>What it means</h2>
  <div class="note" id="interp"></div>

  <div class="foot">
    Generated from <code>outputs/lift_sweep_h0XX.log</code> by <code>scripts/build_sweep_dashboard.py</code> &#183; self-contained &amp; offline.<br>
    Pairs with the learning series: staged/gated reward (Ch. 09), the honest-metric trap (Ch. 04), exploration &amp; difficulty (Ch. 10).
  </div>
</div>
<script>
const PAYLOAD = /*DATA*/;
const DATA = PAYLOAD.data, SUM = PAYLOAD.summary;
const GATES = Object.keys(DATA);
const COLORS = {"0.04":"#7ee787","0.06":"#58a6ff","0.08":"#ffa657"};
const shown = {}; GATES.forEach(g=>shown[g]=true);
let beta = 0.6;

function ema(arr,b){let out=[],s=null;for(const v of arr){if(v==null){out.push(s);continue;}s=(s==null)?v:b*s+(1-b)*v;out.push(s);}return out;}

// ---- finding + cards + interp ----
function fmt(x,d=3){return (x>=0?'':'')+Number(x).toFixed(d);}
(function(){
  // best gate = highest sustained lift
  let best=GATES[0];GATES.forEach(g=>{if(SUM[g].lift_final>SUM[best].lift_final)best=g;});
  document.getElementById('finding').innerHTML =
    '<b>Key finding:</b> only the easiest gate <b>minimal_height = '+best+' m</b> learned to <b>sustain lifting</b> '+
    '(final lift reward '+fmt(SUM[best].lift_final)+' vs ~0 for the harder gates), and it also carried the cube closest to the goal '+
    '(best position error '+fmt(SUM[best].pe_best)+' m). Raising the gate monotonically degraded lifting: the higher the bar, '+
    'the rarer the cube cleared it, so the big lift bonus almost never fired and the policy regressed to mere reaching.';
  // cards
  const cards=document.getElementById('cards');
  GATES.forEach(g=>{
    const s=SUM[g], lifted=s.lifted;
    const div=document.createElement('div');div.className='card';
    div.innerHTML='<h3><span><span class="swatch" style="background:'+COLORS[g]+'"></span>gate '+g+' m</span>'+
      '<span class="badge '+(lifted?'ok':'no')+'">'+(lifted?'lifted':'no lift')+'</span></h3>'+
      '<div class="row"><span>lift (max)</span><b>'+fmt(s.lift_max)+'</b></div>'+
      '<div class="row"><span>lift (final)</span><b>'+fmt(s.lift_final)+'</b></div>'+
      '<div class="row"><span>reach (max)</span><b>'+fmt(s.reach_max)+'</b></div>'+
      '<div class="row"><span>carry: best pos err</span><b>'+fmt(s.pe_best)+' m</b></div>'+
      '<div class="row"><span>mean reward peak</span><b>'+fmt(s.reward_peak,2)+'</b></div>';
    cards.appendChild(div);
  });
  document.getElementById('interp').innerHTML =
    'This is the staged-reward gate of Ch. 09 in action. The lift bonus only pays when the cube clears <code>minimal_height</code>; '+
    'if the bar is too high for the policy to reach early on, that stage of the staircase never lights up, later stages (carry-to-goal) '+
    'stay at zero, and PPO has nothing to climb toward a real pick-and-place &#8212; so it settles for the small reaching reward. '+
    'The <b>rise&#8594;collapse</b> in mean reward (all three) is the Ch. 04 instability/regression pattern: brief lifts early, then the '+
    'policy abandons the unrewarding hard behavior. Takeaway: on a 6&#8239;GB / 1200-iter budget, keep the gate easy (0.04) &#8212; or '+
    'add curriculum / more envs / more iterations (Ch. 06, 10) before raising the difficulty.';
})();

// ---- gate toggles ----
const gt=document.getElementById('gateToggles');
GATES.forEach(g=>{const b=document.createElement('button');b.style.borderColor=COLORS[g];b.dataset.g=g;
  b.innerHTML='<span class="swatch" style="background:'+COLORS[g]+'"></span>'+g+' m';b.style.color=COLORS[g];
  b.onclick=()=>{shown[g]=!shown[g];b.style.opacity=shown[g]?1:0.4;draw();};gt.appendChild(b);});
const smEl=document.getElementById('sm'),smv=document.getElementById('smv');
smEl.oninput=()=>{beta=parseFloat(smEl.value);smv.textContent=beta.toFixed(2);draw();};

function drawChart(id,key,opts){
  opts=opts||{};
  const svg=document.getElementById(id);const W=460,H=220,pl=44,pr=12,pb=26,pt=12;
  // collect y-range over shown gates
  let lo=Infinity,hi=-Infinity,maxIt=0;
  GATES.forEach(g=>{if(!shown[g])return;const ys=ema(DATA[g][key],beta);ys.forEach(v=>{if(v==null)return;if(v<lo)lo=v;if(v>hi)hi=v;});maxIt=Math.max(maxIt,DATA[g].iter[DATA[g].iter.length-1]);});
  if(lo===Infinity){svg.innerHTML='';return;}
  if(opts.zeroFloor&&lo>0)lo=0; const pad=(hi-lo)*0.08||0.1; lo-=pad;hi+=pad;
  const X=it=>pl+(it/maxIt)*(W-pl-pr);
  const Y=v=>pt+(1-(v-lo)/(hi-lo))*(H-pt-pb);
  let s='';
  // gridlines
  for(let k=0;k<=3;k++){const yv=lo+(hi-lo)*k/3;const y=Y(yv);s+='<line x1="'+pl+'" y1="'+y+'" x2="'+(W-pr)+'" y2="'+y+'" stroke="#1c2430"/>';s+='<text x="'+(pl-5)+'" y="'+(y+3)+'" fill="#6e7781" font-size="9" text-anchor="end">'+yv.toFixed(2)+'</text>';}
  s+='<text x="'+pl+'" y="'+(H-7)+'" fill="#6e7781" font-size="9">0</text><text x="'+(W-pr)+'" y="'+(H-7)+'" fill="#6e7781" font-size="9" text-anchor="end">'+maxIt+' iters</text>';
  GATES.forEach(g=>{if(!shown[g])return;const xs=DATA[g].iter,ys=ema(DATA[g][key],beta);let pts='';for(let i=0;i<xs.length;i++){if(ys[i]==null)continue;pts+=(pts?' ':'')+X(xs[i]).toFixed(1)+','+Y(ys[i]).toFixed(1);}s+='<polyline fill="none" stroke="'+COLORS[g]+'" stroke-width="2" points="'+pts+'"/>';});
  svg.innerHTML=s;
}
function draw(){
  drawChart('c_lifting','lifting',{zeroFloor:true});
  drawChart('c_mean_reward','mean_reward',{});
  drawChart('c_position_error','position_error',{});
  drawChart('c_reaching','reaching',{zeroFloor:true});
}
draw();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
