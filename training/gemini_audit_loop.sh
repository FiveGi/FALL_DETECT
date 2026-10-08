#!/bin/sh
# Gemini's half of the agreed dual audit of the 1,713 CLS-ADAPT OF-Syn clips (Claude's half after the
# quota reset). 4 sheets per call; answers in .ai_evidence/ofsyn_audit_all/gemini_*.txt.
cd "D:/project/PROJECT/Backend-Elderly-Surveillance-main"
D=D:/project/PROJECT/.ai_evidence/ofsyn_audit_all
until grep -q "^ok" training/data/ofsyn_sheets_all.log 2>/dev/null; do sleep 30; done
tools/ask_antigravity.sh "No file edits. Codex froze TRUNC-v1 progressive-mask parameters: 50% static / 50% progressive; onset uniform over frames 0-10; one joint masked at onset, one more per frame through frame 14; cap 4 or 6 joints (frame edge, equiprobable) or 3 (occlusion). AGREE or DISAGREE + one line reason." > $D/gemini_P4.txt 2>&1
for name in lie fall; do
  ls $D/${name}_*.jpg | sort > /tmp/sheets_$name.txt
  split -l 4 /tmp/sheets_$name.txt /tmp/batch_${name}_
  for b in /tmp/batch_${name}_*; do
    files=$(cat $b | sed 's#.*/##' | tr '\n' ' ')
    out=$D/gemini_$(basename $b).txt
    [ -s $out ] && continue
    tools/ask_antigravity.sh "Audit, no file edits. In D:/project/PROJECT/.ai_evidence/ofsyn_audit_all/ open these contact sheets: $files . Each cell = one AI-generated clip, 4 frames left to right, yellow tag (L = labelled 'deliberate lie_down', F = labelled 'accidental fall'); index.txt maps tags to clips. List ONLY the tags to EXCLUDE from training a fall-vs-deliberate-lie-down classifier, each with 3-5 words: label wrong, never lies down, body out of frame, morphing body, unclear multiple people. If you cannot open a sheet, say so explicitly -- do not guess. Plain lines 'TAG: reason'." > $out 2>&1
  done
done
echo done > $D/gemini_audit.done
