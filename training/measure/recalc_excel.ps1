# Recalculate a workbook with the real Excel and report any formula errors.
#
# The xlsx skill ships scripts/recalc.py, which drives LibreOffice over a Unix socket and
# cannot run on Windows (socket.AF_UNIX does not exist there). Excel itself is installed on
# this machine, so it does the job -- and it is the application the file will actually be
# opened in, which makes it the better oracle anyway.
#
# Writes cached values back into the file, so pandas and openpyxl(data_only=True) can read the
# results instead of None.
param([Parameter(Mandatory=$true)][string]$Path)

$full = (Resolve-Path $Path).Path
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
$errors = @()
$formulaCount = 0
try {
    $wb = $excel.Workbooks.Open($full)
    $excel.CalculateFullRebuild()
    foreach ($ws in $wb.Worksheets) {
        $used = $ws.UsedRange
        if ($null -eq $used) { continue }
        # -4123 = xlCellTypeFormulas; 16 = xlErrors
        try {
            $f = $used.SpecialCells(-4123)
            if ($f) { $formulaCount += $f.Count }
        } catch {}
        try {
            $bad = $used.SpecialCells(-4123, 16)
            if ($bad) {
                foreach ($c in $bad) {
                    $errors += "$($ws.Name)!$($c.Address($false,$false)) = $($c.Text)"
                }
            }
        } catch {}
    }
    $wb.Save()
    $wb.Close($true)
} finally {
    $excel.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel) | Out-Null
    [GC]::Collect()
}
"formulas: $formulaCount"
if ($errors.Count -eq 0) {
    "status: success - no formula errors"
} else {
    "status: errors_found ($($errors.Count))"
    $errors | Select-Object -First 40 | ForEach-Object { "  $_" }
}
