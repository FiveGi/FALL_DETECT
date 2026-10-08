$ErrorActionPreference = 'Stop'
function Mean($v) { ($v | Measure-Object -Average).Average }
$path = 'training/data/stage1/results_pinned.jsonl'
$lines = Get-Content $path
$rows = @(); $ln = 0
foreach ($line in $lines) {
    $ln++; $r = $line | ConvertFrom-Json
    if ($r.name -eq 'parity_deployed_v2') { $baseline = Mean $r.owner_caught }
    if ($r.name -match '^T2_truncfull_s(45|46|47|48|49|50)_rule$') {
        $seed = [int]$Matches[1]; $fail = @()
        if ($r.halfB_falls -lt 21.8) { $fail += 'falls' }
        if ($r.halfB_adl_clean -lt 16.1) { $fail += 'ADL' }
        if ($r.val_adl_clean -lt 6.1) { $fail += 'val' }
        if ((Mean $r.night_fa) -gt (13.0/3)) { $fail += 'nightFA' }
        if ((Mean $r.owner_fa) -gt 5) { $fail += 'ownerFA' }
        $rows += [pscustomobject]@{seed=$seed; fails=$fail; caught=(Mean $r.owner_caught); multi=(Mean $r.owner_multi)}
        "$path`:$ln seed=$seed threshold=$($r.threshold) falls/ADL/val=$($r.halfB_falls)/$($r.halfB_adl_clean)/$($r.val_adl_clean) nightFA=$(Mean $r.night_fa) ownerFA=$(Mean $r.owner_fa) owner=$(Mean $r.owner_caught) multi=$(Mean $r.owner_multi) failures=$($fail -join ',') day=$($r.n_day) ownerPhases=$($r.n_owner)"
    }
}
if ($rows.Count -ne 6) { throw 'Expected exactly six primary rows' }
"New passes=$(@($rows | Where-Object { $_.seed -ge 48 -and $_.fails.Count -eq 0 }).Count)/3; total passes=$(@($rows | Where-Object { $_.fails.Count -eq 0 }).Count)/6; ADL failures=$(@($rows | Where-Object { $_.fails -contains 'ADL' }).Count)/6; production owner=$baseline; owner above production=$(@($rows | Where-Object { $_.caught -gt $baseline }).Count)/6"
foreach ($seed in 48..50) {
    $file = "training/data/multi_diag_v2/track_metric_t2f_s$seed.txt"
    $d = Get-Content $file | ConvertFrom-Json
    "$file`:1 threshold=$($d.threshold) scored=$($d.n_scored) excluded=$($d.n_excluded) phases=$($d.phases) correct=$(Mean $d.correct) wrong=$(Mean $d.wrong)"
}
