"""
Modification memory and refinement logging utilities.
"""
import os
import json
import threading
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional, TextIO, Tuple, Any


@dataclass
class ModificationMemory:
    entries: List[Dict[str, Any]] = field(default_factory=list)
    max_entries: int = 20

    def add(self, rule_type: str, rule_name: str, action: str, reason: str, new_rule_str: Optional[str] = None,
            metrics: Optional[Dict[str, float]] = None, round_num: Optional[int] = None) -> None:
        if len(self.entries) >= self.max_entries:
            self.entries = self._summarize_oldest(5) + self.entries[5:]
        entry: Dict[str, Any] = {
            "type": rule_type,
            "name": rule_name,
            "action": action,
            "reason": reason,
        }
        if new_rule_str is not None:
            entry["new_rule"] = new_rule_str
        if metrics:
            entry["metrics"] = metrics
        if round_num is not None:
            entry["round"] = round_num
        self.entries.append(entry)

    def _summarize_oldest(self, n: int) -> List[Dict[str, str]]:
        old = self.entries[:n]
        names = [e.get("name", "") for e in old]
        summary = f"[{n} prior modifications: {', '.join(names)}]"
        return [{"type": "summary", "name": "", "action": "", "reason": summary}]

    def add_round_summary(self, round_num: int, conflict_rate: float, gap_rate: float) -> None:
        if len(self.entries) >= self.max_entries:
            self.entries = self._summarize_oldest(5) + self.entries[5:]
        self.entries.append({
            "type": "round_summary",
            "name": "",
            "action": "summary",
            "reason": "",
            "round": round_num,
            "metrics": {
                "conflict_rate": conflict_rate,
                "gap_rate": gap_rate
            }
        })

    def to_context(self) -> str:
        if not self.entries:
            return "# Refinement History\nNo modifications yet."
        round_summaries: Dict[int, Dict[str, float]] = {}
        round_mods: Dict[int, List[Dict[str, Any]]] = {}
        for e in self.entries:
            if e.get("type") == "round_summary":
                round_num = e.get("round")
                if isinstance(round_num, int):
                    round_summaries[round_num] = e.get("metrics", {}) or {}
            elif e.get("type") == "summary":
                continue
            else:
                round_num = e.get("round")
                if isinstance(round_num, int):
                    round_mods.setdefault(round_num, []).append(e)
        if not round_summaries:
            return "# Refinement History\nNo round summaries available."
        sorted_rounds = sorted(round_summaries.keys())
        if len(sorted_rounds) > 10:
            sorted_rounds = sorted_rounds[-10:]
        
        lines: List[str] = ["# Refinement History"]
        
        # Add a clear instruction for the LLM
        lines.append("\n**CRITICAL: Do NOT propose any rules that have been REJECTED in previous rounds.**\n")
        
        prev_conflict: Optional[float] = None
        prev_gap: Optional[float] = None
        for round_num in sorted_rounds:
            metrics = round_summaries.get(round_num, {})
            conflict_rate = metrics.get("conflict_rate")
            gap_rate = metrics.get("gap_rate")
            if conflict_rate is not None and prev_conflict is not None:
                conflict_str = f"{prev_conflict:.4f}->{conflict_rate:.4f}"
            elif conflict_rate is not None:
                conflict_str = f"{conflict_rate:.4f}"
            else:
                conflict_str = "N/A"
            if gap_rate is not None and prev_gap is not None:
                gap_str = f"{prev_gap:.4f}->{gap_rate:.4f}"
            elif gap_rate is not None:
                gap_str = f"{gap_rate:.4f}"
            else:
                gap_str = "N/A"
            mods = round_mods.get(round_num, [])
            if mods:
                change_items: List[str] = []
                for m in mods:
                    action = m.get("action", "")
                    rule_str = m.get("new_rule") or ""
                    if rule_str:
                        if len(rule_str) > 120:
                            rule_str_disp = rule_str[:117] + "..."
                        else:
                            rule_str_disp = rule_str
                        base = f"{m.get('type','')}/{m.get('name','')} {action}: {rule_str_disp}"
                    else:
                        base = f"{m.get('type','')}/{m.get('name','')} {action}"
                    if action == "rejected":
                        reason = m.get("reason") or ""
                        metrics = m.get("metrics") or {}
                        violation_rate = metrics.get("violation_rate")
                        reason_parts: List[str] = []
                        if reason:
                            reason_parts.append(reason)
                        if isinstance(violation_rate, float):
                            reason_parts.append(f"violation_rate={violation_rate:.4f}")
                        if reason_parts:
                            base = f"{base} | reason=" + ", ".join(reason_parts)
                    change_items.append(base)
                changes_str = "; ".join(change_items)
            else:
                changes_str = "no rule changes"
            lines.append(
                f"- [round {round_num}] changes: {changes_str}; "
                f"conflict={conflict_str}, gap={gap_str}"
            )
            if conflict_rate is not None:
                prev_conflict = conflict_rate
            if gap_rate is not None:
                prev_gap = gap_rate
        return "\n".join(lines)

    def clear(self) -> None:
        self.entries = []


class RefinementLogger:
    def __init__(self) -> None:
        self.log_file: Optional[TextIO] = None
        self.log_path: Optional[str] = None
        self.run_id: Optional[str] = None
        self._lock = threading.RLock()

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
        with self._lock:
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
        if not self.log_file:
            return
        with self._lock:
            self.log_file.write("  [PROMPT]\n")
            for line in prompt.splitlines():
                self.log_file.write("    " + line + "\n")
            self.log_file.write("  [/PROMPT]\n")
            self.log_file.flush()

    def set_response(self, response: str) -> None:
        if not self.log_file:
            return
        with self._lock:
            self.log_file.write("  [RESPONSE]\n")
            for line in response.splitlines():
                self.log_file.write("    " + line + "\n")
            self.log_file.write("  [/RESPONSE]\n")
            self.log_file.flush()

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
        self._log(f"  [REJECTION] {rule_name}")
        self._log(f"    Type: {rule_type}")
        self._log(f"    Old: {old_rule}")
        self._log(f"    New: {new_rule}")
        self._log(f"    Violation Rate: {violation_rate:.4f}")
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
