#!/usr/bin/env bash
cd ~/krsl2speech
until grep -q AUDIT-SCORING-DONE output/audit.log; do sleep 120; done
.venv/bin/python scripts/audit_report.py --out output/audit_report > output/audit_report.log 2>&1
.venv/bin/python scripts/build_audit_viewer.py --per-source 20 >> output/audit_report.log 2>&1
echo AUDIT-ALL-DONE >> output/audit_report.log
