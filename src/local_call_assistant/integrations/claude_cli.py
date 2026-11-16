"""Analyse a transcript with the Claude Code CLI on this machine.

Runs ``claude -p`` signed in with the user's Claude subscription. The call is
isolated:

- no tools, no MCP servers, no user/project/local settings (so no hooks or
  plugins that could copy the transcript elsewhere), no saved session, no
  slash commands, and a fresh empty working directory;
- the child gets an allowlisted environment only, so nothing inherited can
  switch it to API billing or another provider, redirect its endpoint, add
  headers, or load plugins (for example ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL,
  CLAUDE_CODE_USE_BEDROCK or CLAUDE_CODE_PLUGIN_DIRS never reach it);
- the native executable is run directly. A Windows .cmd/.bat launcher is
  never executed, because cmd.exe would reinterpret the arguments; npm's
  launcher is resolved to the claude.exe it wraps;
- the transcript goes in on standard input, never on the command line, from
  a temporary file that is deleted when the call ends.

Error messages are fixed texts and never include CLI output.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from local_call_assistant.config import ClaudeConfig
from local_call_assistant.integrations.analysis import (
    SYSTEM_PROMPT, transcript_text, user_message, write_analysis,
)
from local_call_assistant.processing import StageContext, StageSkipped

PROVIDER = "claude-cli"
ISOLATION_FLAGS = (
    "-p", "--output-format", "json",
    "--tools", "",
    "--strict-mcp-config",
    "--setting-sources", "",
    "--no-session-persistence",
    "--disable-slash-commands",
)
# Operating-system basics the CLI needs to run and to find its own sign-in.
ENV_ALLOWLIST = frozenset({
    "PATH", "PATHEXT", "SYSTEMROOT", "SYSTEMDRIVE", "WINDIR", "COMSPEC", "TEMP", "TMP",
    "TMPDIR", "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "APPDATA", "LOCALAPPDATA",
    "PROGRAMDATA", "PROGRAMFILES", "PROGRAMFILES(X86)", "PROGRAMW6432", "USERNAME", "USER",
    "LOGNAME", "OS", "PROCESSOR_ARCHITECTURE", "NUMBER_OF_PROCESSORS", "LANG", "TERM",
    "XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "XDG_RUNTIME_DIR",
})
ENV_ALLOWED_PREFIXES = ("LC_",)
BATCH_SUFFIXES = (".cmd", ".bat")
NPM_NATIVE_EXECUTABLE = Path("node_modules", "@anthropic-ai", "claude-code", "bin", "claude.exe")


class ClaudeError(RuntimeError):
    pass


def resolve_executable(name: str) -> str:
    found = shutil.which(name)
    if found is None:
        raise ClaudeError("the claude command was not found on PATH")
    if not found.lower().endswith(BATCH_SUFFIXES):
        return found
    native = Path(found).parent / NPM_NATIVE_EXECUTABLE
    if native.is_file():
        return str(native)
    raise ClaudeError("claude resolves to a .cmd/.bat launcher; set claude.command to "
                      "the claude executable itself")


def build_command(config: ClaudeConfig) -> list[str]:
    command = [resolve_executable(config.command[0]), *config.command[1:],
               *ISOLATION_FLAGS, "--system-prompt", SYSTEM_PROMPT]
    if config.model:
        command += ["--model", config.model]
    return command


def child_environment() -> dict[str, str]:
    return {k: v for k, v in os.environ.items()
            if k.upper() in ENV_ALLOWLIST or k.upper().startswith(ENV_ALLOWED_PREFIXES)}


def analyse(config: ClaudeConfig, job_id: str, text: str) -> str:
    command = build_command(config)
    # The message is handed over as a prefilled, self-deleting temporary file
    # rather than through input=: on Windows, subprocess writes input= before
    # its timed wait, so a child that never reads stdin would block us forever.
    # The empty working directory is left behind if a stray descendant still
    # holds it open (Windows), so cleanup never masks the fixed error texts.
    with tempfile.TemporaryDirectory(prefix="lca-claude-",
                                     ignore_cleanup_errors=True) as empty_directory, \
            tempfile.TemporaryFile(prefix="lca-claude-stdin-") as stdin_file:
        stdin_file.write(user_message(job_id, text).encode("utf-8"))
        stdin_file.flush()
        stdin_file.seek(0)
        try:
            completed = subprocess.run(
                command, stdin=stdin_file, capture_output=True,
                text=True, encoding="utf-8", errors="replace", cwd=empty_directory,
                env=child_environment(), timeout=config.timeout_seconds, check=False)
        except subprocess.TimeoutExpired:
            raise ClaudeError("claude did not finish within the timeout") from None
        except OSError:
            raise ClaudeError("claude could not be started") from None
    if completed.returncode != 0:
        raise ClaudeError(f"claude exited with code {completed.returncode}")
    return _result_of(completed.stdout)


def claude_handler(config: ClaudeConfig):
    def handle(context: StageContext) -> dict[str, Any]:
        if not config.enabled:
            raise StageSkipped("Claude analysis is disabled in the configuration")
        markdown = analyse(config, context.manifest.job_id, transcript_text(context))
        return write_analysis(context, markdown, PROVIDER, config.model)
    return handle


def _result_of(stdout: str) -> str:
    try:
        document = json.loads(stdout)
    except ValueError:
        raise ClaudeError("claude output was not the expected JSON") from None
    if not isinstance(document, dict) or document.get("is_error"):
        raise ClaudeError("claude reported an error")
    result = document.get("result")
    if not isinstance(result, str) or not result.strip():
        raise ClaudeError("claude returned an empty analysis")
    return result
