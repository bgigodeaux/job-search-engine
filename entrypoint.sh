#!/usr/bin/env bash
set -e
uvicorn app.api:app --host 0.0.0.0 --port