$ErrorActionPreference='Stop'
function Mean($v) { ($v|Measure-Object -Average).Average }
$p='training/data/stage1/results_pinned.jsonl'; $ln=0; $rows=@()
Get-Content $p | ForEach-Object {
 $ln++; $r=$_|ConvertFrom-Json
 if($r.name -match '^(T2_truncfull|T2FEMA|T2FEMA4)_s(45|46|47|48|49|50)_rule$') {
  $arm=$Matches[1]; $seed=[int]$Matches[2]
  $pass=($r.halfB_falls -ge 21.8 -and $r.halfB_adl_clean -ge 16.1 -and $r.val_adl_clean -ge 6.1 -and (Mean $r.night_fa) -le (13.0/3) -and (Mean $r.owner_fa) -le 5)
  $rows += [pscustomobject]@{arm=$arm; seed=$seed; pass=$pass; caught=(Mean $r.owner_caught); threshold=$r.threshold}
  "$p`:$ln $($r.name) pass=$pass owner=$(Mean $r.owner_caught) multi=$(Mean $r.owner_multi) threshold=$($r.threshold) dayPhases=$($r.n_day) ownerPhases=$($r.n_owner)"
 }
}
foreach($arm in 'T2_truncfull','T2FEMA','T2FEMA4') {
 $a=@($rows|Where-Object arm -eq $arm); if($a.Count -ne 6){throw 'Expected six rows'}
 $m=$a.caught|Measure-Object -Minimum -Maximum -Average; $t=$a.threshold|Measure-Object -Minimum -Maximum
 "$arm passes=$(@($a|Where-Object pass).Count)/6 seeds=$((($a|Where-Object pass).seed)-join ',') mean=$($m.Average) range=$($m.Maximum-$m.Minimum) min=$($m.Minimum) max=$($m.Maximum) threshold=$($t.Minimum)..$($t.Maximum)"
}
foreach($seed in 45..50) { $plain=$rows|Where-Object {$_.arm -eq 'T2_truncfull' -and $_.seed -eq $seed}; $ema=$rows|Where-Object {$_.arm -eq 'T2FEMA' -and $_.seed -eq $seed}; "paired seed=$seed owner delta=$($ema.caught-$plain.caught)" }
$files=@(Get-ChildItem training/data/multi_diag_v2/track_metric_t2fema*_s*.txt); if($files.Count -ne 12){throw 'Expected 12 D3 files'}
$files+=Get-Item training/data/multi_diag_v2/track_metric_s44_t2fs45_070.txt
foreach($f in $files) { $d=Get-Content $f.FullName|ConvertFrom-Json; "$($f.Name):1 threshold=$($d.threshold) scored=$($d.n_scored) excluded=$($d.n_excluded) phases=$($d.phases) correct=$(Mean $d.correct) wrong=$(Mean $d.wrong)" }
'Counts: URFD halfB 28 falls/20 ADL; GMDCSA val16 ADL; simulated night60 falls/40 ADL x3; owner75 falls/13 negatives, multi25, 8 phases. Artifact recomputation only, 0 clips rerun.'
