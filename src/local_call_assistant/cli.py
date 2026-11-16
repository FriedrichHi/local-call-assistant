"""Command line: ``lca status | recover | ingest | ingest-dir | process | retry``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from local_call_assistant.config import Config, config_path, load_config
from local_call_assistant.integrations.claude_cli import claude_handler
from local_call_assistant.integrations.hermes import analysis_handler
from local_call_assistant.integrations.obsidian import note_handler
from local_call_assistant.models import PROCESSABLE_STATUSES
from local_call_assistant.processing import STAGE_ORDER, Pipeline
from local_call_assistant.storage.ingest import ingest_directory, ingest_file
from local_call_assistant.storage.spool import RecordingSpool, SpoolError
from local_call_assistant.transcription import transcription_handler


def build_pipeline(config: Config) -> Pipeline:
    analysis = (claude_handler(config.claude) if config.claude.enabled
                else analysis_handler(config.hermes))  # each skips when disabled
    handlers = {
        "transcription": transcription_handler(config.transcription),
        "analysis": analysis,
        "note": note_handler(config.obsidian),
    }
    return Pipeline(RecordingSpool(config.runtime_root), handlers, config.max_attempts)


def cmd_status(config: Config, args: argparse.Namespace) -> int:
    spool = RecordingSpool(config.runtime_root)
    rows = [m.to_dict() for m in spool.iter_manifests()]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"runtime root: {config.runtime_root}")
    print(f"config file:  {config_path(config.runtime_root)}")
    for row in rows:
        stages = " ".join(f"{name}={row['stages'].get(name, {}).get('status', 'pending')}"
                          for name in STAGE_ORDER)
        print(f"{row['job_id']}  {row['status']:<11} {row['started_at'][:19]}  {stages}")
    return 0


def cmd_recover(config: Config, args: argparse.Namespace) -> int:
    pipeline = build_pipeline(config)
    sealed = pipeline.spool.recover_incomplete()
    stages = pipeline.recover_stages()
    for manifest in sealed:
        print(f"recovered recording {manifest.job_id} ({manifest.audio_bytes} audio bytes)")
    for job_id in stages:
        print(f"marked interrupted stage(s) failed for {job_id}")
    return 0


def cmd_ingest(config: Config, args: argparse.Namespace) -> int:
    spool = RecordingSpool(config.runtime_root)
    min_age = config.ingest_min_age_seconds if args.min_age is None else args.min_age
    if args.command == "ingest-dir":
        results = ingest_directory(spool, Path(args.directory), recursive=args.recursive,
                                   min_age_seconds=min_age)
    else:
        results = [ingest_file(spool, Path(p), min_age_seconds=min_age) for p in args.paths]
    for result in results:
        suffix = f" -> {result.job_id}" if result.job_id else ""
        print(f"{result.outcome:<9} {result.path.name}{suffix}")
    return 0


def cmd_process(config: Config, args: argparse.Namespace) -> int:
    pipeline = build_pipeline(config)
    pipeline.spool.recover_incomplete()
    pipeline.recover_stages()
    job_ids = args.jobs or [m.job_id for m in pipeline.spool.iter_manifests()
                            if m.status in PROCESSABLE_STATUSES]
    failed = False
    for job_id in job_ids:
        manifest = pipeline.process(job_id)
        summary = {name: str(manifest.stage(name).status) for name in STAGE_ORDER}
        failed |= "failed" in summary.values()
        print(f"{job_id}  " + " ".join(f"{k}={v}" for k, v in summary.items()))
    return 1 if failed else 0


def cmd_retry(config: Config, args: argparse.Namespace) -> int:
    build_pipeline(config).retry(args.job, args.stage)
    print(f"{args.job}: {args.stage} reset to pending; run 'lca process {args.job}'")
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="lca", description="Local Call Assistant")
    root.add_argument("--home", help="runtime root (default: per-user app data folder)")
    sub = root.add_subparsers(dest="command", required=True)
    status = sub.add_parser("status", help="list recordings and stage states")
    status.add_argument("--json", action="store_true")
    sub.add_parser("recover", help="seal interrupted recordings and stages")
    ingest = sub.add_parser("ingest", help="copy audio files into the spool")
    ingest.add_argument("paths", nargs="+")
    ingest_dir = sub.add_parser("ingest-dir", help="ingest every audio file in a folder")
    ingest_dir.add_argument("directory")
    ingest_dir.add_argument("--recursive", action="store_true")
    for command in (ingest, ingest_dir):
        command.add_argument("--min-age", type=float, default=None,
                             help="skip files modified fewer than this many seconds ago")
    process = sub.add_parser("process", help="run pending stages")
    process.add_argument("jobs", nargs="*")
    retry = sub.add_parser("retry", help="reset a failed or skipped stage")
    retry.add_argument("job")
    retry.add_argument("--stage", choices=STAGE_ORDER, default="transcription")
    return root


COMMANDS = {"status": cmd_status, "recover": cmd_recover, "ingest": cmd_ingest,
            "ingest-dir": cmd_ingest, "process": cmd_process, "retry": cmd_retry}


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    config = load_config(Path(args.home) if args.home else None)
    try:
        return COMMANDS[args.command](config, args)
    except (KeyError, ValueError, OSError, SpoolError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
