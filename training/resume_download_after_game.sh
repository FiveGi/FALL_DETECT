#!/bin/sh
# Resume the COCO person-free download when the game (TOTClient) is closed and >= 12 GB is free.
until [ -z "$(tasklist //FI "IMAGENAME eq TOTClient-Win64-Shipping.exe" //NH | grep -i TOTClient)" ] && \
      [ "$(powershell -NoProfile -c "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1MB*10)" | tr -d '\r')" -ge 120 ]; do sleep 60; done
cd D:/project/PROJECT/datasets/posneg && python download.py >> download.log 2>&1
