#!/usr/bin/env bash
# Show paper-trading account + trades. Works from any folder.
cd "$(dirname "$0")" && exec .venv/bin/python -W ignore -m src.status "$@"
