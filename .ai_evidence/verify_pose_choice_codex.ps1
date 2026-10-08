$ErrorActionPreference='Stop'
function Avg($v){($v | Measure-Object -Average).Average}
$p='training/data/stage1/results_pinned.jsonl'; $i=0
'Counts: stage1_eval_pinned.py:3-6: owner 75 falls/13 negatives, multi25,8 phases; URFD halfB28 falls/20 ADL, GMDCSA val16 ADL,8 day caches; simulated night60 falls/40 ADL,3 draws. Stored URFD/val means inspected, not rebuilt from caches.'
Get-Content $p | ForEach-Object {
$i++; $r=$_|ConvertFrom-Json
if($r.name -in @('P1_posneg44_deployed065','N1_nightaug44_deployed065','T2_truncfull_s45_rule','P2_posneg44_t2fs45_070','parity_deployed_v2')){
"${p}:$i $($r.name) pose=$($r.pose) threshold=$($r.threshold)"
"hBf=$($r.halfB_falls) ADL=$($r.halfB_adl_clean) val=$($r.val_adl_clean) nF=$(Avg $r.night_falls) nFA=$(Avg $r.night_fa) owner=$(Avg $r.owner_caught) oFA=$(Avg $r.owner_fa) multi=$(Avg $r.owner_multi)"
"five-listed-gates=$(($r.halfB_falls -ge 21.8)-and($r.halfB_adl_clean -ge 16.1)-and($r.val_adl_clean -ge 6.1)-and((Avg $r.night_fa) -le (13/3))-and((Avg $r.owner_fa) -le 5))"
}}
foreach($f in 'track_metric_posneg44.txt','track_metric_posneg44_t2fs45.txt','track_metric_s44_t2fs45_070.txt','track_metric_t2.txt'){
$i=0; Get-Content "training/data/multi_diag_v2/$f" | ForEach-Object {$i++;$r=$_|ConvertFrom-Json;if($f -ne 'track_metric_t2.txt' -or $i -eq 1){"${f}:$i correct=$(Avg $r.correct) wrong=$(Avg $r.wrong) scored=$($r.n_scored) excluded=$($r.n_excluded) phases=$($r.phases)"}}
}
'No inference, timing, or live tests run. Plan_final.md:31 object promotion gate <=72; :36-37 exact CPU gate remains separate.'
'Nightaug minus posneg T2F: owner +3.5/75, multi +2.75/25, simulated-night +16/3=5.333333/60 (5.4 is rounded-input subtraction); D3 correct +0.5/14, wrong +2.375/14.'
'Night simulation ranges: nightaug T2F and posneg T2F follow:'
Get-Content $p | ForEach-Object {$r=$_|ConvertFrom-Json;if($r.name -in @('T2_truncfull_s45_rule','P2_posneg44_t2fs45_070')){"$($r.name) falls=$($r.night_falls -join ',') FA=$($r.night_fa -join ',')"}}