import json
import os
from paths import a3_stochastic_relative_path

db_path = str(a3_stochastic_relative_path / "automation" / "deformation_path_db.json")

with open(db_path, "r") as f:
    data = json.load(f)

updated = False
paths = data.get("deformation_paths", {})
for key, value in paths.items():
    if value.startswith("/bbysvm"):
        paths[key] = "/" + value
        updated = True

if updated:
    with open(db_path, "w") as f:
        json.dump(data, f, indent=4)
    print("Fixed single slash issues in JSON.")
else:
    print("No single slash issues found.")
