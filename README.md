# h3-live-worker

Runpod Serverless worker image for the h3-live interactive livestream. One job runs one livestream.

- Environment pinned to the GPU pod environment of 2026-10-13: Ubuntu 24.04 / Python 3.12.3,
  torch 2.14.1+cu130, ComfyUI v0.39.0 (`b0b7435`), custom nodes by commit (`custom_nodes.txt`),
  every Python package by version (`requirements-comfy.txt`, installed with `--no-deps`).
- Models are not in the image: the endpoint's cached model `GloriaWang23/h3-live-runtime`
  is read from `/runpod-volume/huggingface-cache`.
- Livestream code is not in the image either: each job downloads the current code bundle from
  the h3-live cloud console and runs its `pod/serverless_job.sh`.

Job input: `{"control_url": "...", "pod_token": "...", "session_id": "..."}` (submitted by the console).
