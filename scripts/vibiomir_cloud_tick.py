"""Entrypoint for the continuously scheduled cloud CPU crawl controller."""
import argparse
import json
import logging
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--legacy-version", type=int, default=10)
    parser.add_argument("--root", required=True)
    parser.add_argument("--workflow-url", required=True)
    parser.add_argument("--summary", default="outputs/continuous_status.json")
    args = parser.parse_args()
    # An encrypted Actions secret supplies OAuth refresh credentials. Only the
    # ephemeral runner's credential file receives them; never a crawl artifact.
    credentials = os.environ.pop("KAGGLE_OAUTH_CREDENTIALS", None)
    if os.environ.get("GITHUB_ACTIONS") == "true" and not credentials:
        raise RuntimeError("Missing encrypted Actions secret KAGGLE_OAUTH_CREDENTIALS")
    if credentials:
        value = json.loads(credentials)
        if value.get("username") != args.owner or not value.get("refresh_token"):
            raise ValueError("OAuth secret belongs to a different Kaggle account")
        path = Path.home()/".kaggle/credentials.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(mode=0o600)
        path.chmod(0o600)
        path.write_text(json.dumps(value))
    from kaggle import api
    from src.data.kaggle_continuous import continuous_tick
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    summary = Path(args.summary)
    summary.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = continuous_tick(Path(__file__).resolve().parents[1], args.owner, args.root, api,
                                 legacy_version=args.legacy_version, workflow_url=args.workflow_url)
    except Exception as error:
        result = {"state":"retry_on_next_tick", "error_type":type(error).__name__, "compute":"CPU"}
        summary.write_text(json.dumps(result, indent=2))
        logging.error("Cloud tick failed (%s); the next scheduled tick will retry", type(error).__name__)
        raise SystemExit(1) from None
    summary.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
