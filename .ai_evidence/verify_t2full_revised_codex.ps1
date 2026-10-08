$ErrorActionPreference = 'Stop'
$p='training/data/stage1/results_pinned.jsonl'
$rows=@(Get-Content $p | ForEach-Object { $_ | ConvertFrom-Json })
function Avg($v) { ($v | Measure-Object -Average).Average }
$base=$rows | Where-Object name -eq 'parity_deployed'
$limit=(Avg $base.night_fa)+1
"Night FA limit baseline+1 = $limit; source ${p}:1; plan_final.md:32-33."
'URFD half-B 28 falls/20 ADL; GMDCSA val 16 ADL (8 caches); simulated night 60 falls/40 ADL x3; owner 75 falls/13 negatives, multi25 (8 phases). Cached metrics only, 0 clips rerun.'
for($i=0;$i -lt $rows.Count;$i++) {
 $r=$rows[$i]; if($r.name -notmatch '^T2_truncfull_s(45|46|47)_(065|rule)$'){continue}
 $fail=@(); if($r.halfB_falls -lt 21.8){$fail+='falls'}; if($r.halfB_adl_clean -lt 16.1){$fail+='ADL'}; if($r.val_adl_clean -lt 6.1){$fail+='val'}; if((Avg $r.night_fa) -gt $limit){$fail+='nightFA'}; if((Avg $r.owner_fa) -gt 5){$fail+='ownerFA'}
 "${p}:$($i+1) $($r.name): $($r.halfB_falls)/$($r.halfB_adl_clean)/$($r.val_adl_clean)/$(Avg $r.night_fa)/$(Avg $r.owner_fa); owner=$(Avg $r.owner_caught),multi=$(Avg $r.owner_multi); failed=$($fail -join ',')"
}
'Only s45_rule passes these five numeric checks; full scorecard/CPU/promotion not verified.'
'Counts provenance: training/measure/stage1_eval_pinned.py:2-6; training/data/multi_diag_v2/plan_final.md:27-34.'
'Command: & ([scriptblock]::Create((Get-Content -Raw .ai_evidence/verify_t2full_revised_codex.ps1)))'