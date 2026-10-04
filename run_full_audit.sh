#!/usr/bin/env bash
# Wrapper: wait for MIT-BIH download to finish, then run the full audit.
# Designed to survive PC-lock: writes everything to log files under
# results/audit/ so we have a paper trail even if notifications drop.

set -o pipefail

ROOT="D:/Trần An Khánh/E14/robustness"
MITDB="D:/Trần An Khánh/E14/mitdb"
WESAD="C:/Users/Lenovo/Downloads/wesad_pkls"
LOG_DIR="$ROOT/results/audit"

export PYTHONIOENCODING=utf-8
export WESAD_ROOT="$WESAD"
export MITBIH_ROOT="$MITDB"

mkdir -p "$LOG_DIR"

phase_log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_DIR/run_full_audit.log"; }

phase_log "=== Phase 1: waiting for MIT-BIH download ==="
# MIT-BIH Arrhythmia DB has 48 records * 3 files (.hea/.dat/.atr) = 144 files.
# Plus some ancillary files. Wait until we see at least 130 files.
until [ "$(ls -1 "$MITDB" 2>/dev/null | wc -l)" -gt 130 ]; do
  count=$(ls -1 "$MITDB" 2>/dev/null | wc -l)
  phase_log "  mitdb file count: $count / ~144"
  sleep 30
done
phase_log "  mitdb download complete: $(ls -1 "$MITDB" | wc -l) files"

phase_log "=== Phase 2: rebuild feature cache from real data ==="
# Delete any stale cache; _load_dataset will rebuild from real WESAD+MIT-BIH.
rm -f "$ROOT/../stage3_hb1_dataset.csv"
cd "$ROOT" || exit 2
python -c "
import warnings; warnings.simplefilter('ignore')
from train_stage3_models import _load_dataset
df = _load_dataset()
print('feature cache built:', df.shape, 'columns:', len(df.columns))
print('by dataset:', df.groupby('dataset').size().to_dict())
print('by label:',   df.groupby('label').size().to_dict())
print('subjects:',   df['subject_id'].nunique())
" 2>&1 | tee -a "$LOG_DIR/run_full_audit.log"

phase_log "=== Phase 3: run full audit (Steps 0-9, --fast mode) ==="
# --fast keeps replicates modest so the whole thing finishes in tens of
# minutes on real data. Drop --fast for the paper-final run overnight.
python -m audit.run_all --fast 2>&1 | tee -a "$LOG_DIR/run_full_audit.log"
audit_exit=$?

phase_log "=== Phase 4: summary ==="
phase_log "  audit exit: $audit_exit"
phase_log "  artifacts in $LOG_DIR:"
ls -1 "$LOG_DIR" | sed 's/^/    /' | tee -a "$LOG_DIR/run_full_audit.log"

phase_log "=== DONE ==="
exit $audit_exit
