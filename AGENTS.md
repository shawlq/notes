# AGENTS.md

## Cursor Cloud specific instructions

### Codebase overview

This is an **infrastructure-as-scripts** repository for managing LLM inference services on a GPU server ("Spark"). It contains:

- `scripts/` — Bash scripts for starting Docker containers (SGLang, vLLM, Open WebUI, Code Server, Tailscale VPN)
- `docs/` — Markdown documentation (benchmarks, deployment guides, study plans)

There is no application source code, no package manager, no build system, and no test framework.

### Linting

The primary development tool is **shellcheck** for linting bash scripts:

```bash
# Lint all scripts
shellcheck scripts/*.sh

# Validate bash syntax
bash -n scripts/*.sh
```

### Running services

All services run as Docker containers requiring NVIDIA GPUs and pre-downloaded model weights. They **cannot** be started in the Cloud Agent environment (no GPU, no model files). The scripts are designed for the target GPU server.

### Key notes

- No `package.json`, `requirements.txt`, `Makefile`, or equivalent — dependencies are system-level only (`shellcheck`).
- Scripts use `docker run` with `--gpus all` and host-mounted model paths (`/data/models/modelscope`).
- vLLM scripts require a `HF_TOKEN` environment variable.
- Open WebUI connects to the inference backend at `host.docker.internal:30000`.
