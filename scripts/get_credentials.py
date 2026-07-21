#!/usr/bin/env python3
"""Fetch a Kowalski JWT token and save it to the ml4em .env file.

Uses only the Python standard library — no conda environment or container
needed. Run this on the MSI login node after cloning the repo.

Usage:
    python3 get_credentials.py                # password input is hidden
    python3 get_credentials.py --show-password  # password is visible (useful when pasting)
"""

import getpass
import json
import os
import sys
import urllib.request
from pathlib import Path

HOST = "melman.caltech.edu"
ENV_FILE = Path(f"/scratch.global/{os.environ['USER']}/ml4em_data/.env")


def main() -> None:
    show_password = "--show-password" in sys.argv

    print("Enter your Kowalski (ZTF) credentials.")
    username = input("Username: ")
    password = input("Password: ") if show_password else getpass.getpass("Password: ")

    payload = json.dumps({"username": username, "password": password}).encode()
    req = urllib.request.Request(
        f"https://{HOST}/api/auth",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise SystemExit(f"Authentication failed ({exc.code}):\n{body}") from exc

    if "data" not in data or "token" not in data.get("data", {}):
        msg = data.get("message", json.dumps(data, indent=2))
        raise SystemExit(f"Server error: {msg}")

    token = data["data"]["token"]

    ENV_FILE.parent.mkdir(parents=True, exist_ok=True)
    ENV_FILE.write_text(f"ML4EM_ZTF_TOKEN={token}\n")
    ENV_FILE.chmod(0o600)

    print(f"\nToken saved to {ENV_FILE}")
    print(f"To update your credentials at any time, re-run this script.")


if __name__ == "__main__":
    main()
