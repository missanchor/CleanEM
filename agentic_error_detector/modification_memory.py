"""
Modification memory and refinement logging utilities.
"""
import os
import json
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TextIO, Tuple, Any


@dataclass
class ModificationMemory:
    entries: List[Dict[str, str]] = field(default_factory=list)
    max_entries: int = 20

    def add(self, rule_type: str, rule_name: str, action: str, reason: str, new_rule_str: Optional[str] = None) -> None:
        if len(self.entries) >= self.max_entries:
            self.entries = self._summarize_oldest(5) + self.entries[5:]
        entry: Dict[str, str] = {
            "type": rule_type,
            "name": rule_name,
            "action": action,
            "reason": reason,
        }
        if new_rule_str is not None:
            entry["new_rule"] = new_rule_str
        self.entries.append(entry)

    def _summarize_oldest(self, n: int) -> List[Dict[str, str]]:
        old = self.entries[:n]
        names = [e.get("name", "") for e in old]
        summary = f"[{n} prior modifications: {', '.join(names)}]"
        return [{"type": "summary", "name": "", "action": "", "reason": summary}]

    def to_context(self) -> str:
        if not self.entries:
            return "# Modification History\nNo modifications yet."
        lines: List[str] = ["# Modification History"]
        for e in self.entries[-10:]:
            if e.get("type") == "summary":
                lines.append(f"- {e.get('reason', '')}")
            else:
                lines.append(f"- [{e.get('type','')}/{e.get('name','')}] {e.get('action','')}: {e.get('reason','')}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.entries = []


class RefinementLogger:
    def __init__(self) -> None:
        self.log_file: Optional[TextIO] = None
        self.log_path: Optional[str] = None
        self.run_id: Optional[str] = None

    def start(self, output_dir: str, dataset_name: Optional[str] = None) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = timestamp
        log_dir = os.path.join(output_dir, "refinement_logs")
        os.makedirs(log_dir, exist_ok=True)
        if dataset_name:
            filename = f"{timestamp}_{dataset_name}_clean_dirty_rule_generation.log"
        else:
            filename = f"{timestamp}_clean_dirty_rule_generation.log"
        self.log_path = os.path.join(log_dir, filename)
        self.log_file = open(self.log_path, "w", encoding="utf-8")
        self._write_header()

    def _write_header(self) -> None:
        if not self.log_file:
            return
        self.log_file.write("=" * 60 + "\n")
        self.log_file.write("REFINEMENT AUDIT LOG\n")
        self.log_file.write("=" * 60 + "\n")
        self.log_file.write(f"Run ID: {self.run_id}\n")
        self.log_file.write(f"Started: {datetime.now().isoformat()}\n")
        self.log_file.write("=" * 60 + "\n\n")
        self.log_file.flush()

    def _log(self, msg: str) -> None:
        if not self.log_file:
            return
        self.log_file.write(msg + "\n")
        self.log_file.flush()

    def begin_column_refinement(
        self,
        column: str,
        round_num: int,
        stage: str,
        clean_rules: Dict[str, str],
        dirty_rules: Dict[str, str],
    ) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log(f"[{ts}] --- Column: {column} | Round: {round_num} | Stage: {stage} ---")
        self._log(f"  Clean rules before: {json.dumps(clean_rules, ensure_ascii=False)}")
        self._log(f"  Dirty rules before: {json.dumps(dirty_rules, ensure_ascii=False)}")

    def end_column_refinement(
        self,
        clean_rules: Dict[str, str],
        dirty_rules: Dict[str, str],
    ) -> None:
        self._log(f"  Clean rules after: {json.dumps(clean_rules, ensure_ascii=False)}")
        self._log(f"  Dirty rules after: {json.dumps(dirty_rules, ensure_ascii=False)}")
        self._log("  --- End of column refinement ---")

    def set_prompt(self, prompt: str) -> None:
        self._log("  [PROMPT]")
        for line in prompt.splitlines():
            self._log("    " + line)
        self._log("  [/PROMPT]")

    def set_response(self, response: str) -> None:
        self._log("  [RESPONSE]")
        for line in response.splitlines():
            self._log("    " + line)
        self._log("  [/RESPONSE]")

    def add_modification(
        self,
        rule_type: str,
        rule_name: str,
        old_rule: str,
        new_rule: str,
        modification_type: str,
        reason: str,
    ) -> None:
        self._log(f"  [MODIFICATION] {modification_type}")
        self._log(f"    Type: {rule_type}, Name: {rule_name}")
        self._log(f"    Old: {old_rule}")
        self._log(f"    New: {new_rule}")
        self._log(f"    Reason: {reason}")

    def log_error(self, column: str, error_type: str, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log(f"[{ts}] [ERROR] Column: {column} | Type: {error_type}")
        self._log(f"  Message: {message}")

    def log_round_summary(
        self,
        column: str,
        round_num: int,
        conflict_rate: float,
        gap_rate: float,
        num_modifications: int,
    ) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log(f"[{ts}] [SUMMARY] Column: {column} | Round: {round_num}")
        self._log(f"  Conflict rate: {conflict_rate:.4f}")
        self._log(f"  Gap rate: {gap_rate:.4f}")
        self._log(f"  Modifications: {num_modifications}")

    def log_rejection(
        self,
        rule_type: str,
        rule_name: str,
        old_rule: str,
        new_rule: str,
        violation_rate: float,
        reason: str,
    ) -> None:
        self._log("  [REJECTED] Rule rejected for excessive violations")
        self._log(f"    Type: {rule_type}, Name: {rule_name}")
        self._log(f"    Violation rate: {violation_rate * 100:.1f}% (>50% threshold)")
        self._log(f"    Old: {old_rule}")
        self._log(f"    Proposed new: {new_rule}")
        self._log(f"    Reason: {reason}")

    def log_initial_rules(
        self,
        clean_rules: Dict[str, List[Tuple[str, str]]],
        dirty_rules: Dict[str, List[Tuple[str, str]]],
        clean_prompts: Optional[Dict[str, Dict[str, Any]]] = None,
        dirty_prompts: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._log("=" * 60)
        self._log(f"[{ts}] INITIAL RULE GENERATION")
        self._log("=" * 60)
        self._log("--- CLEAN RULES (Quality Pillars) ---")
        if clean_rules:
            for column, rules in clean_rules.items():
                self._log(f"  Column: {column}")
                for pillar_name, rule in rules:
                    self._log(f"    [{pillar_name}]")
                    self._log(f"      Rule: {rule}")
                    if clean_prompts and column in clean_prompts:
                        col_prompts = clean_prompts[column]
                        if pillar_name in col_prompts:
                            prompt_info = col_prompts[pillar_name]
                            prompt_text = prompt_info.get("prompt")
                            response_text = prompt_info.get("response")
                            if prompt_text:
                                self.set_prompt(prompt_text)
                            if response_text:
                                self.set_response(response_text)
        self._log("--- DIRTY RULES (Agent Rules) ---")
        if dirty_rules:
            for column, rules in dirty_rules.items():
                self._log(f"  Column: {column}")
                for agent_name, rule in rules:
                    self._log(f"    [{agent_name}]")
                    self._log(f"      Rule: {rule}")
                    if dirty_prompts and column in dirty_prompts:
                        col_prompts = dirty_prompts[column]
                        if agent_name in col_prompts:
                            prompt_info = col_prompts[agent_name]
                            prompt_text = prompt_info.get("prompt")
                            response_text = prompt_info.get("response")
                            if prompt_text:
                                self.set_prompt(prompt_text)
                            if response_text:
                                self.set_response(response_text)
        else:
            self._log("  No dirty rules generated")
        self._log("=" * 60)

    def close(self, summary: Optional[Dict[str, Any]] = None) -> None:
        if self.log_file:
            self._log("=" * 60)
            self._log(f"Run ID: {self.run_id}")
            self._log(f"Finished: {datetime.now().isoformat()}")
            if summary is not None:
                self._log(f"Summary: {json.dumps(summary, ensure_ascii=False)}")
            self.log_file.close()
            self.log_file = None

    @property
    def path(self) -> str:
        return self.log_path or ""


_logger: Optional[RefinementLogger] = None


def get_logger() -> RefinementLogger:
    global _logger
    if _logger is None:
        _logger = RefinementLogger()
    return _logger


def start_logger(output_dir: str = "agentic_error_detector/results", dataset_name: Optional[str] = None) -> RefinementLogger:
    logger = get_logger()
    logger.start(output_dir, dataset_name)
    return logger


def stop_logger(summary: Optional[Dict[str, Any]] = None) -> None:
    global _logger
    if _logger is not None:
        _logger.close(summary)
        _logger = None
