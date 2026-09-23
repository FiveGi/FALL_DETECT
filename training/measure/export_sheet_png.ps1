# Screenshot a worksheet range through Excel, so the layout can be looked at rather than
# assumed. The validator checks colour; nothing but eyes checks label collisions, column
# widths and a chart that overlaps its own table.
param(
  [Parameter(Mandatory=$true)][string]$Path,
  [Parameter(Mandatory=$true)][string]$Sheet,
  [Parameter(Mandatory=$true)][string]$Range,
  [Parameter(Mandatory=$true)][string]$Out
)
Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$full = (Resolve-Path $Path).Path
$excel = New-Object -ComObject Excel.Application
$excel.Visible = $false
$excel.DisplayAlerts = $false
try {
    $wb = $excel.Workbooks.Open($full)
    $ws = $wb.Worksheets.Item($Sheet)
    $ws.Range($Range).CopyPicture(1, 2)   # xlScreen, xlBitmap
    Start-Sleep -Milliseconds 700
    $img = [System.Windows.Forms.Clipboard]::GetImage()
    if ($null -eq $img) { throw "clipboard held no image" }
    $img.Save($Out, [System.Drawing.Imaging.ImageFormat]::Png)
    "saved $Out  ($($img.Width)x$($img.Height))"
    $wb.Close($false)
} finally {
    $excel.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel) | Out-Null
    [GC]::Collect()
}
