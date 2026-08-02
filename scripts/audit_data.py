from pathlib import Path
import json

from src.data_pipeline import audit_inputs


root = Path(__file__).resolve().parents[1]
report = audit_inputs(root)
output = root / "artifacts" / "data"
output.mkdir(parents=True, exist_ok=True)
with (output / "g0_report.json").open("w", encoding="utf-8") as handle:
    json.dump(report, handle, ensure_ascii=False, indent=2)
print(json.dumps({"passed": report["passed"], "checks": report["checks"], "metadata": report["metadata"]}, ensure_ascii=False, indent=2))
