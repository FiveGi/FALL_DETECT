$ErrorActionPreference='Stop'
function Mean($v) { ($v | Measure-Object -Average).Average }
$root='training/data/stage1'
$b=Get-Content training/data/day_first/owner_size_buckets_v4.json -Raw | ConvertFrom-Json
$truth=@{}
$inc=Get-Content test_result/incidents/incidents.json -Raw -Encoding UTF8 | ConvertFrom-Json
foreach($clip in $inc.clips.PSObject.Properties) { foreach($r in $clip.Value) {
 if($r.label) { if($r.label -ne 'out_of_domain') { $truth["$($clip.Name)#$($r.segment)"]=($r.label -eq 'fall') } }
 elseif($r.antigravity -and !$r.antigravity.error -and !$r.antigravity.out_of_domain) { $truth["$($clip.Name)#$($r.segment)"]=[bool]$r.antigravity.fall }
} }
$fall=@($truth.Keys | Where-Object {$truth[$_]}); $neg=@($truth.Keys | Where-Object {!$truth[$_]})
if(@($b.PSObject.Properties).Count -ne 75 -or @($fall | Where-Object {!$b.PSObject.Properties[$_]}).Count) { throw 'Bucket/truth mismatch' }
"Command: Invoke-Command -ScriptBlock ([scriptblock]::Create((Get-Content .ai_evidence/codex_t2_arithmetic.ps1 -Raw)))"
"Owner: $($fall.Count) fall segments + $($neg.Count) negatives, $( @($truth.Keys | ForEach-Object { ($_ -split '#')[0] } | Sort-Object -Unique).Count ) source clips; 8 phases x 7 models = 56 JSONs."
foreach($bucket in 'near','medium','far','unknown') { "$bucket count=$(@($b.PSObject.Properties | Where-Object {$_.Value.bucket -eq $bucket}).Count)" }
$rows=@{}; $refs=@{}
foreach($log in 't1_eval.log','t2_eval.log') { $n=0; foreach($line in Get-Content "$root/$log") { $n++; if($line -match '^\{"name":') { $r=$line | ConvertFrom-Json; if($r.name -match '^(T1_ctrl|T2_trunc|T2_truncfull)_s4[567]_065$') { if($rows.ContainsKey($r.name)) {throw 'Duplicate row'}; $rows[$r.name]=$r; $refs[$r.name]="$root/${log}:$n" } } } }
$multiKeys=@(Get-Content ../.ai_evidence/multi_person_genuine.txt | Where-Object {$_ -and !$_.StartsWith('#')})
$means=@{}
foreach($name in ($rows.Keys | Sort-Object)) {
 $r=$rows[$name]; $sums=@{near=0;medium=0;far=0;unknown=0}
 for($p=0;$p -lt 8;$p++) {
  $f="$root/${name}_pinned_owner$p.json"; $j=Get-Content $f -Raw | ConvertFrom-Json
  if($j.profile.threshold -ne 0.65 -or $j.profile.roi_phase -ne $p -or $j.profile.pose_model -notmatch 'nightaug_s44') { throw "Profile mismatch $f" }
  foreach($k in $truth.Keys) { if(!$j.segments.PSObject.Properties[$k]) {throw "Missing $k in $f"} }
  $caught=@($fall | Where-Object {@($j.segments.$_.alerts_at).Count -gt 0}).Count
  $fa=@($neg | Where-Object {@($j.segments.$_.alerts_at).Count -gt 0}).Count
  $multi=@($multiKeys | Where-Object {@($j.segments.$_.alerts_at).Count -gt 0}).Count
  if($caught -ne $r.owner_caught[$p] -or $fa -ne $r.owner_fa[$p] -or $multi -ne $r.owner_multi[$p]) {throw "Log/JSON mismatch $f"}
  foreach($k in $fall) { if(@($j.segments.$k.alerts_at).Count -gt 0) { $sums[$b.$k.bucket]++ } }
 }
 $means[$name]=@{}; foreach($bucket in 'near','medium','far','unknown') { $means[$name][$bucket]=$sums[$bucket]/8 }
 "$name near/medium/far=$($sums.near/8)/$($sums.medium/8)/$($sums.far/8) caught=$(Mean $r.owner_caught) FA=$(Mean $r.owner_fa) multi=$(Mean $r.owner_multi); $($refs[$name]); JSON lines 2-21 profile,22+ segments"
}
foreach($metric in 'halfB_falls','halfB_adl_clean','val_adl_clean','owner_caught','owner_fa','owner_multi','night_falls','night_fa') {
 $d=@(45..47 | ForEach-Object {(Mean $rows["T2_trunc_s${_}_065"].$metric)-(Mean $rows["T1_ctrl_s${_}_065"].$metric)})
 "$metric paired deltas=$($d -join ',') mean=$(Mean $d)"
}
foreach($s in 45..47) { "$s bucket deltas=" + ((@('near','medium','far','unknown') | ForEach-Object {$means["T2_trunc_s${s}_065"][$_]-$means["T1_ctrl_s${s}_065"][$_]}) -join ',') }
'TRUNC-full s45 bucket deltas=' + ((@('near','medium','far','unknown') | ForEach-Object {$means['T2_truncfull_s45_065'][$_]-$means['T1_ctrl_s45_065'][$_]}) -join ',')
'PASS: 56 JSON profiles, complete truth keys, all phase-level caught/FA/multi agree with 7 log rows; bucket map covers all 75 falls.'
'Limits: summary arithmetic only for URFD/day/night; no inference, recipe validation, queue inspection or training performed.'
'Counts per requested verdict: URFD halfB 28 falls/20 ADL; val 16 ADL; simulated night 60 falls/40 ADL x3 noise draws; multi 25 designated segments (scorer list ../.ai_evidence/multi_person_genuine.txt:1; separate from bucket-map multi flags).'
'Provenance: training/t1_eval.sh:12 and training/t2_eval.sh:10; training/measure/stage1_eval_pinned.py:57-76; training/measure/score_incidents.py:26-59; training/data/day_first/owner_size_buckets_v4.json:1.'
