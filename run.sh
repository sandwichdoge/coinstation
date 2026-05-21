#!/usr/bin/env bash
set -euo pipefail

cleanup() {
  echo ""
  echo "Shutting down..."
  docker compose down
  exit 0
}
trap cleanup SIGINT SIGTERM

echo "Building and starting services..."
docker compose up --build --detach

echo "Waiting for app to be ready on http://localhost:8080 ..."
until curl -s -o /dev/null http://localhost:8080; do
  sleep 0.5
done

open http://localhost:8080
echo "App is ready. Press Ctrl+C to stop."

docker compose logs --follow
