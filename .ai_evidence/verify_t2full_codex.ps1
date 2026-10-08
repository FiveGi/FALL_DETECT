$ErrorActionPreference = 'Stop'
function Avg($v) { ($v | Measure-Object -Average).Average }
$p = 'training/data/stage1/results_pinned.jsonl'
$rows = @{}; $i=0
Get-Content $p | ForEach-Object { $i++; $r=$_ | ConvertFrom-Json; if($rows.ContainsKey($r.name)){throw "Duplicate $($r.name)"}; $rows[$r.name]=@{r=$r;line=$i} }
'Pinned cached replay: threshold .65, nightaug_s44 pose; owner 75 falls/13 negatives, multi 25; 8 phases (not independent clips). URFD half B 28 falls/20 ADL; GMDCSA val 16 ADL; simulated night 60 falls/40 ADL x 3 draws. Denominators: stage1_eval_pinned.py:2-6; plan_final.md:27-34.'
foreach($s in 45,46,47) {
 $a=$rows["T1_ctrl_s${s}_065"]; $b=$rows["T2_truncfull_s${s}_065"]
 "seed $s source ${p}:$($a.line),$($b.line)"
 foreach($k in 'owner_caught','owner_fa','owner_multi','halfB_falls','halfB_adl_clean','val_adl_clean') { '{0}: {1:F3} -> {2:F3}; delta {3:F3}' -f $k,(Avg $a.r.$k),(Avg $b.r.$k),((Avg $b.r.$k)-(Avg $a.r.$k)) }
}
$a=$rows['N1_nightaug44_deployed065']; $b=$rows['T2F_ens_065']
"ensemble minus deployed-classifier (same pose), source ${p}:$($a.line),$($b.line)"
foreach($k in 'owner_caught','owner_fa','owner_multi','halfB_falls','halfB_adl_clean','val_adl_clean','night_falls','night_fa') { '{0}: {1:F3} -> {2:F3}; delta {3:F3}' -f $k,(Avg $a.r.$k),(Avg $b.r.$k),((Avg $b.r.$k)-(Avg $a.r.$k)) }
$d=@{}
foreach($f in 'track_metric_t2.txt','track_metric_step1b.txt','track_metric_t2f_ens.txt') {
 $i=0; Get-Content "training/data/multi_diag_v2/$f" | ForEach-Object { $i++; $r=$_ | ConvertFrom-Json; $key=Split-Path $r.model_dir -Leaf; if($f -match 'ens'){$key='ensemble'}; $d[$key]=$r; if($key -match 'models|t1_ctrl|truncfull|ensemble'){ "$f`:$i $key correct=$(Avg $r.correct) wrong=$(Avg $r.wrong) scored=$($r.n_scored) excluded=$($r.n_excluded) phases=$($r.phases)" } }
}
foreach($s in 45,46,47) { "D3 delta seed $s correct=$((Avg $d["t2_truncfull_s$s"].correct)-(Avg $d["t1_ctrl_s$s"].correct)) wrong=$((Avg $d["t2_truncfull_s$s"].wrong)-(Avg $d["t1_ctrl_s$s"].wrong))" }
'Near-distance counts absent from pinned JSONL and D3 files: not verified. No model replay or timing rerun.'
'CPU arithmetic from pipeline_timing.txt:6-8 (11 segments,252 frames,4 threads): median +{0:F2}%, p95 +{1:F2}%' -f ((86.8/57.1-1)*100),((145.2/65-1)*100)
