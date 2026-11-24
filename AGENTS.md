# MLOps eKYC

## Objective

Build and verify the eKYC pipeline for the course rubric. Use MIDV-2020 clips first.
Use the repository's `Data/` directory as the local read-only source.
Tools should accept an explicit source path. Do not commit archives, extracted
images, model binaries, cloud credentials, or real identity documents.

Classify ID_Card and Passport. Map *_id to ID_Card and *_passport to Passport,
excluding rus_internalpassport from the first dataset. Group splits by document
template, and sample separated frames from clips. Derive quality ground truth
from Label Studio review or explicitly tracked train-only augmentations.
Quality labels are Clear, Blur, Glare and Occluded. Review-only Outside_Scope
is excluded from the first training dataset.

## Communication

The primary assistant communicates only with the user and the Orchestrator.
All requests, updates, and findings between the primary assistant and terminal
workers go through the Orchestrator.
The Orchestrator alone dispatches and coordinates terminal workers. Workers
report to the Orchestrator, who returns verified outcomes to the primary
assistant.

Terminal worker launchers must enforce role-specific write permissions,
deny direct network access and command escalation, and prevent workers
from controlling the team's tmux sessions. Workers request operations
outside those permissions through their role checkpoints. The Orchestrator
runs authorized privileged operations and records the result. Verify both
allowed and denied operations for each role before using a launcher.

## Maintaining user instructions

The primary assistant records every project rule the user sets in this file
as soon as it is given. Update the relevant section when the user changes
or withdraws a rule so this file reflects the current instructions.

Read this file at the start of each session and after context compaction.
Check the relevant sections when the user gives new direction, before
dispatching or handing off work, at milestones, and before reporting
completion. The primary assistant relays changes that affect terminal
workers through the Orchestrator.

## Terminal roles

- Orchestrator owns integration, handoffs, verification and blockers. Check
  active tmux panes before restarting workers. Give each worker exclusive
  file ownership and resolve interface disagreements before integration.
- Data owns ETL, sampling and splits, Label Studio project configuration, and
  reviewed JSON ingestion under data_pipeline/ and labeling/.
- Model owns PyTorch/MLflow training, artifact format and model serving code
  under modeling/.
- Azure owns Azure ML definitions, demo client, telemetry and cloud benchmark
  helpers under azure/ and demo/. Use placeholders or environment variables for
  subscription settings and credentials. Coordinate resource changes with the
  orchestrator.

## Continuation

Read `.agent-state/orchestrator.md` and your role checkpoint before each new
session and after context compaction. Each role writes its own checkpoint
after milestones and before yielding, with completed work, files owned, test
results, blockers and an executable next step. Keep coordinating until all
independent work is complete. Verify changes on disk before relying on a
checkpoint. Resume the same role after context reset instead of creating a
duplicate worker. If the process exits, the launcher preserves the pane and
shows how to restart the role from its checkpoint. An explicit stop from the
user takes precedence.

Each role takes only its latest assignment from the checkpoint or the
Orchestrator and reports ready when no assignment is open. The Orchestrator
checks all role checkpoints regularly and verifies work in tmux before
recording handoffs. It owns integration, project documentation and reporting,
and leaves paths owned by active workers to those workers.

Use the existing local archive to verify a small slice. Never infer quality
labels from directory names. Keep local test results and Azure evidence
distinct. Record missing quota, CLI, data or credentials honestly.

Automatic compaction is configured in `.codex/config.toml` for project CLI
sessions. Keep tool output bounded and save full logs on disk. Before manual
compaction, checkpoint the active process, output path, ownership and the next
command. Compact an idle pane only. After compaction, check the running process
and outputs before relaunching any job.

## Autonomous operation

The user delegates routine in-scope decisions and execution to the orchestrator.
Proceed with authorized project work without asking the user to reconfirm
routine decisions or actions. Follow any approval required by the execution
environment.
Launch every project Codex CLI role with approval policy `never` so routine
work runs unattended. Keep role-specific sandbox restrictions for workers.
Worker shells have scoped write paths, disabled networking and web search,
and a minimal environment. Workers write their own live checkpoint at
`runs/workers/ROLE/checkpoint.md`. Canonical `.agent-state` is read-only
for workers. If a worker operation is denied, record the exact operation and
reason in the live checkpoint for the Orchestrator.
The orchestrator checks process state, assigns follow-ups, retries recoverable
failures, runs integration, and records unresolved external dependencies. Use
the project's approved Azure resources and budget when credentials and quota are
available. Preserve unrelated resources and files. Limit dependency installs
to project environments. After a restart, inspect existing files before editing,
since interrupted workers may have already written changes.

## Container runtime

For Label Studio, use Docker if its daemon is reachable, or the installed
rootless Podman with the same official OCI image. Let the runtime pull the
image on demand and persist project data. Assign the Data worker to verify
startup and the health endpoint. If a step explicitly requires Docker and no
usable Docker runtime exists, the orchestrator owns installation from the
distribution's package manager when noninteractive privilege is available.
On Fedora use the distro moby-engine package and enable the docker service.
Do not wait on a sudo password prompt. If installation cannot proceed, use
Podman where compatible and record only Docker-specific blockers.

## Release verification

The orchestrator owns the acceptance matrix and may assign independent tests
to idle workers. Re-run documented entry points from a clean project-local
environment and keep reproducible commands, versions, and raw results.
Cover data corruption, duplicates, split leakage, missing classes or labels,
canceled or conflicting annotations, invalid and oversized image requests,
threshold boundaries, network and storage errors, concurrent or repeated review
requests, and retry/restart behavior where applicable. Add regression tests for
failures found and re-run after fixes. Verify Word and PowerPoint are readable
and agree with the measured evidence. Record each scenario as pass, fail, or
blocked. Human approval of labels and Azure tests require actual evidence.
Do not mark the full release verified while any required check is blocked.

## Human-in-the-loop acceptance

The demo must let a reviewer see and act on an image queued by a real
prediction with needs_review or by the Report incorrect prediction action.
Persist the allowed mock image and event metadata idempotently. Provide
documented commands to send the queue batch to Label Studio, enter a corrected
document type and quality (or Outside_Scope), and import the reviewed export.
Record approval with reviewer identity and export hash before creating a new
manifest with lineage. A manually submitted training job consumes the new
manifest. Preserve local evidence of the full cycle, including queue, labeling
project and before/after labels. Test duplicates, conflicting and canceled
reviews, rejected inputs, storage failures, and Outside_Scope exclusion.
Tests may use fixtures, but evidence for actual human review requires a real
human, and cloud evidence requires actual Azure execution.

## Model quality acceptance

Target 85–95% held-out document classification accuracy, with 85% as the
minimum acceptance threshold. Report macro-F1, per-class recall, confusion
matrix and sample counts alongside accuracy. Keep the split fixed by template.
Use validation only to select checkpoint, fine-tuning configuration and
calibration or review thresholds. Compare pretrained frozen and fine-tuned
MobileNetV3. If accuracy is below target, diagnose class errors and data
coverage and improve training using train/validation only. Report the actual
held-out result and mark acceptance failed if the target is not attained.
API confidence is a per-image score. Report its calibration and retained
coverage if using an abstention threshold. Evaluate the quality head
separately on human-reviewed labels and mark it pending if unavailable.
Keep manual review-triggered retraining outputs isolated from the selected
baseline. Deploy a retrained candidate only after validation.

## Additional terminal workers

The user authorizes the orchestrator to open additional visible Kitty/tmux
windows and start Codex workers for independent tests, fine-tuning, or
integration when useful. Reuse an idle worker first. Give each additional
worker a unique task, exclusive output paths, and a checkpoint in
.agent-state/. Coordinate CPU, RAM, GPU and Azure budget before concurrent
training runs. Keep validation-based model selection centralized. Expose each
extra worker in its own Kitty window via a grouped tmux session. Record the
window, purpose and next step in the orchestrator checkpoint. Reclaim finished
workers when their logs and artifacts have been saved.

## Verify terminal dispatch

A message visible in a Codex input field is still a draft. For an idle worker,
send the text and wait for it to appear before pressing Enter. Then verify a
new Working indicator, tool action, or fresh completed turn in the target
pane before recording the task as dispatched. If the text remains in the
input field after the paste settles, press Enter again. Avoid sending raw
keystrokes to a worker while it is running. Write active-task updates to its
checkpoint and verify delivery at the next handoff.

## Status reports

Use a table for each project status report. Include each active task's owner,
current state, what it is doing, and its direct job or artifact link when
available. Report cloud statuses from Azure and agent statuses from their
terminal windows. For a task without a direct web link, give its terminal
window name or link to the local checkpoint.
