$ErrorActionPreference = 'Stop'
$days=((Get-Content training/data/stage1/nightaug_s44_cache_lists.txt -TotalCount 1) -split '\s+')[1..8]
foreach($d in $days) {
    $names=@(Get-ChildItem -LiteralPath $d -Filter 'urfd_adl__*.npz' | Where-Object { $_.Name -match 'adl-(\d+)' -and ([math]::Floor(([int]$Matches[1]-1)/2)%2 -eq 1) })
    if($names.Count -ne 20){throw "Unexpected half-B count: $d"}
    "$d : halfB=$($names.Count)"
}
$lines=Get-Content training/data/multi_diag_v2/urfd_adl_fa_by_clip.txt
$freq=@{}
foreach($l in $lines[0..6]) {
    $m=[regex]::Matches($l,"'([^']+)': (\d+)")
    $sum=0
    foreach($v in $m){$sum += [int]$v.Groups[2].Value; $freq[$v.Groups[1].Value]++}
    if($l -notmatch "FA clip-phases $sum "){throw 'Total mismatch'}
    '{0}: sum={1}; clean={2}; ADLgate={3}' -f ($l -split ' ')[0],$sum,(20-$sum/8),((20-$sum/8) -ge 16.1)
}
$freq.GetEnumerator() | Sort-Object Name | ForEach-Object { "$($_.Name): configs=$($_.Value)" }
'nightaug baseline margin={0}' -f (20-25/8-16.1)
