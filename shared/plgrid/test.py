import argparse
import json
import sys

def main(params: dict):
    print("=== PARAMS RECEIVED ===")
    for k, v in params.items():
        print(f"{k} = {v}")

parser = argparse.ArgumentParser(description="Universal parameter handler")

parser.add_argument(
    "--json",
    type=str,
    help='JSON string, e.g. \'{"start":3,"end":10,"model":"resnet"}\''
)

args, unknown = parser.parse_known_args()

params = {}

if args.json:
    try:
        params.update(json.loads(args.json))
    except json.JSONDecodeError as e:
        print("Invalid JSON:", e)
        sys.exit(1)

key = None
for item in unknown:
    if item.startswith("--"):
        key = item.lstrip("-")
        params[key] = True
    else:
        if key is None:
            continue
        params[key] = item
        key = None

main(params)
