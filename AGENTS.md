# MLOps eKYC repository guidelines

This project classifies simulated MIDV-2020 document images as `ID_Card` or `Passport`. Reviewers label image quality as `Clear`, `Blur`, `Glare`, or `Occluded`. See [README.md](README.md) for entry points, [LOCAL-DEVELOPMENT.md](LOCAL-DEVELOPMENT.md) for setup, and [WORKFLOW.md](WORKFLOW.md) for Assignment 13 acceptance criteria.

## Where to make changes

- `data_pipeline/` validates and samples images and builds manifests. `prepared_midv.py` uses the template assignments in `midv-split.json`.
- `labeling/` contains the Label Studio task, export, and manifest tools and the `quality.xml` and `hitl.xml` label schemas. See [labeling/README.md](labeling/README.md).
- `modeling/` trains, evaluates, packages, and serves models. `modeling.workflow` provides the candidate training and release checks, and `modeling.score` serves predictions.
- `demo/` contains the HTTP server, static pages, SQLite and SSE monitoring, local training, and review flows. `demo.server` is the web entry point.
- `azure/` contains the Azure ML job and deployment definitions, staging tools, benchmarks, and telemetry queries. See [azure/README.md](azure/README.md).

## Develop and verify

Run commands from the repository root.

- Start the stack with `docker compose up --build -d --wait` and open `http://127.0.0.1:8081`. Predictions require a model bundle or a scoring endpoint. Use `.env.example` for supported configuration keys.
- Use `compose.dev.yaml` with `compose.yaml` to mount source files during development. Reload the page for static asset changes, restart `app` for Python changes, and rebuild after dependencies change. Check for active training runs before restarting the app container.
- Enable the `review` profile for Label Studio. Use `compose.azure.yaml` with `compose.yaml` for the Azure-enabled app. Keep the same Compose files when running commands against an existing stack.
- Run one-off commands with `docker compose run --rm app python -m <module>`. For direct execution on Linux, use Python 3.11 in a virtual environment and install `requirements.txt`.
- Match surrounding code. Use four-space indentation, `snake_case` for Python functions and modules, and `PascalCase` for classes.
- Reproduce CI with `python -m compileall -q azure data_pipeline demo labeling modeling`. Exercise the affected flow separately. Use `python -m demo.build` to run the same compile check and save its log under `runs/builds/` by default. Local tests and `checks/` are ignored by Git, so confirm their availability before relying on them.

## Data, review, and releases

- Treat source data under `Data/` as read-only. Use a fresh output directory under `runs/` or outside the checkout and preserve each run's manifest, provenance, and evaluation evidence. Keep datasets, checkpoints, logs, generated artifacts, and credentials out of Git.
- Split by document template. Within an evaluation version, preserve validation and test membership and labels across review rounds. Select candidates on validation and evaluate the locked candidate on held-out test data.
- Use human-reviewed labels for new training data. Keep audit-only images out of training manifests and apply the documented allowlist and approval flow before publishing a revised manifest.
- Use `python -m modeling.workflow check-release --run <run>` before registering or staging a candidate. Follow the gates in `WORKFLOW.md`, deploy staging at 0% traffic, and record predictions and benchmark results before routing traffic.
- Update `.env.example`, the relevant Compose files, and setup documentation together when changing configuration. Keep actual credentials in local configuration.

## Commits and pull requests

Use short imperative commit subjects. In pull requests, describe the behavior change, list the checks run, link relevant issues, and include screenshots for visible demo changes.
