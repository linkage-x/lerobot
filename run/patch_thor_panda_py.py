#!/usr/bin/env python3
"""Prepare a local panda-py source tree for explicitly restarted Thor FR3 control.

Only the inspected upstream recover() implementation is accepted. The capability
is exported from the compiled extension after native automatic recovery is
removed. No package is installed and no robot is contacted by this helper.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re


ORIGINAL_RECOVER = '''void Panda::recover() {
  auto state = robot_->readOnce();
  if (state.current_errors || state.robot_mode == franka::RobotMode::kReflex ||
      state.robot_mode == franka::RobotMode::kOther) {
    _log("warning",
         "Irregular state detected. Attempting automatic error recovery.");
    robot_->automaticErrorRecovery();
  }
}'''
PATCHED_RECOVER = '''void Panda::recover() {
  auto state = robot_->readOnce();
  if (state.current_errors || state.robot_mode == franka::RobotMode::kReflex ||
      state.robot_mode == franka::RobotMode::kOther) {
    throw std::runtime_error("FR3 error must be cleared in Desk before retry");
  }
}'''
MODULE_DECLARATION = "PYBIND11_MODULE(_core, m) {"
CAPABILITY = '  m.attr("FR3_NO_AUTOMATIC_ERROR_RECOVERY") = true;'
ORIGINAL_STOP_BINDING = '.def("stop_controller", &Panda::stopController)'
PATCHED_STOP_BINDING = '.def("stop_controller", &Panda::stopController, py::call_guard<py::gil_scoped_release>())'
ORIGINAL_STOP_STATE = "current_controller_->stop(state_, model_);"
PATCHED_STOP_STATE = "current_controller_->stop(getState(), model_);"
AUTOMATIC_RECOVERY_CALL = re.compile(r"\bautomaticErrorRecovery\s*\(")
RECOVER_FUNCTION = re.compile(r"void Panda::recover\(\) \{.*?^\}", re.MULTILINE | re.DOTALL)


def prepare_sources(source: Path) -> dict[Path, str]:
    """Validate every relevant input before returning any files to write."""
    native_root = source / "src"
    panda_path = native_root / "panda.cpp"
    core_path = native_root / "_core.cpp"
    if not panda_path.is_file() or not core_path.is_file():
        raise ValueError("Expected a panda-py source checkout containing src/panda.cpp and src/_core.cpp")
    panda = panda_path.read_text()
    core = core_path.read_text()
    functions = RECOVER_FUNCTION.findall(panda)
    if len(functions) != 1 or functions[0] not in {ORIGINAL_RECOVER, PATCHED_RECOVER}:
        raise ValueError("Unknown Panda::recover implementation; inspect upstream changes before patching")
    # Copy state through Panda's existing mutex during controller shutdown.
    if panda.count(ORIGINAL_STOP_STATE) == 1:
        if "FR3_NO_AUTOMATIC_ERROR_RECOVERY" in core:
            raise ValueError("Existing native capability marker lacks the synchronized stop-state patch")
        panda = panda.replace(ORIGINAL_STOP_STATE, PATCHED_STOP_STATE, 1)
    elif panda.count(PATCHED_STOP_STATE) != 1:
        raise ValueError("Unknown native controller stop-state implementation")
    # stopController joins the native thread, whose error logging acquires the
    # GIL. Release it around the join so fault shutdown cannot deadlock Python.
    if core.count(ORIGINAL_STOP_BINDING) == 1:
        if "FR3_NO_AUTOMATIC_ERROR_RECOVERY" in core:
            raise ValueError("Existing native capability marker lacks the shutdown binding patch")
        core = core.replace(ORIGINAL_STOP_BINDING, PATCHED_STOP_BINDING, 1)
    elif core.count(PATCHED_STOP_BINDING) != 1:
        raise ValueError("Unknown stop_controller native binding; inspect source before patching")
    if "FR3_NO_AUTOMATIC_ERROR_RECOVERY" in core:
        if functions[0] != PATCHED_RECOVER or core.count(CAPABILITY) != 1:
            raise ValueError("Existing capability marker does not match the inspected native patch")
        if core.count("FR3_NO_AUTOMATIC_ERROR_RECOVERY") != 1:
            raise ValueError("Unknown duplicate native capability marker")
    else:
        if core.count(MODULE_DECLARATION) != 1:
            raise ValueError("Unknown panda-py extension module declaration")
        core = core.replace(MODULE_DECLARATION, MODULE_DECLARATION + "\n" + CAPABILITY, 1)
    panda = panda.replace(functions[0], PATCHED_RECOVER, 1)
    if "#include <stdexcept>" not in panda:
        if "#include" not in panda:
            raise ValueError("Cannot locate native includes")
        panda = panda.replace("#include", "#include <stdexcept>\n#include", 1)
    replacements = {panda_path: panda, core_path: core}
    for candidate in native_root.rglob("*"):
        if candidate.suffix not in {".cpp", ".cc", ".cxx", ".h", ".hpp"} or not candidate.is_file():
            continue
        content = replacements.get(candidate, candidate.read_text())
        if AUTOMATIC_RECOVERY_CALL.search(content):
            raise ValueError(f"Additional native automaticErrorRecovery call remains in {candidate}; refusing marker")
    return replacements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local panda-py source checkout")
    parser.add_argument("--check", action="store_true", help="Require an already patched source checkout without editing")
    args = parser.parse_args(argv)
    try:
        replacements = prepare_sources(args.source)
        if args.check:
            if any(path.read_text() != text for path, text in replacements.items()):
                raise ValueError("Native source patch has not yet been applied")
        else:
            for path, text in replacements.items():
                path.write_text(text)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")
    print("Native panda-py automatic recovery disabled and synchronized shutdown patched; rebuild its aarch64 wheel before F.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
