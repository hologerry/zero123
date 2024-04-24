import json


valid_paths_json = "view_release/valid_paths.json"
with open(valid_paths_json) as f:
    valid_paths = json.load(f)

print(type(valid_paths), len(valid_paths))
print(valid_paths[:10])
