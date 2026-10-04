"""Record aggregate acquisition status monthly so the public schedule stays active."""
import base64
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="outputs/continuous_status.json")
    args = parser.parse_args()
    status = json.loads(Path(args.summary).read_text())
    if status["state"] not in {"building", "waiting_for_legacy_controller"}:
        return
    repo = os.environ["GITHUB_REPOSITORY"]
    path = "docs/vibiomir_continuous_status.json"
    endpoint = f"repos/{repo}/contents/{path}"
    read = subprocess.run(["gh", "api", endpoint], capture_output=True, text=True)
    previous = json.loads(read.stdout) if read.returncode == 0 else None
    if previous:
        old = json.loads(base64.b64decode(previous["content"]))
        last = datetime.fromisoformat(old["recorded_at"])
        if (datetime.now(timezone.utc)-last).days < 28:
            return
    status.pop("control_ref", None)
    status["recorded_at"] = datetime.now(timezone.utc).isoformat()
    body = {"message":"docs: record continuous CPU crawl status", "content":base64.b64encode(
            (json.dumps(status, indent=2)+"\n").encode()).decode()}
    if previous:
        body["sha"] = previous["sha"]
    subprocess.run(["gh", "api", "--method", "PUT", endpoint, "--input", "-"],
                   input=json.dumps(body), text=True, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
