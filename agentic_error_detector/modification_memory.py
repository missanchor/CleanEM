"""
Modification Memory for tracking rule refinement history.
Writes timestamped log files for debugging and audit.
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, TextIO, Tuple
from dataclasses import dataclass, field


# Backward compatible class (still used by ConflictResolver/GapResolver)
@dataclass
class ModificationMemory:
    """
    Tracks rule modifications for LLM context.
    The detailed logging is now handled by RefinementLogger.
    """
    entries: List[Dict[str, str]] = field(default_factory=list)
    MAX_ENTRIES: int = 20

    def add(self, rule_type: str, rule_name: str, action: str, reason: str,
            new_rule_str: str = None):
        """Add a modification entry."""
        if len(self.entries) >= self.MAX_ENTRIES:
            self.entries = self._summarize_oldest(5) + self.entries[5:]

        entry = {
            'type': rule_type,
            'name': rule_name,
            'action': action,
            'reason': reason
        }
        if new_rule_str:
            entry['new_rule'] = new_rule_str

        self.entries.append(entry)

    def _summarize_oldest(self, n: int) -> List[Dict[str, str]]:
        """Summarize the oldest n entries."""
        old = self.entries[:n]
        summary = f"[{n} prior modifications: {', '.join([e['name'] for e in old])}]"
        return [{'type': 'summary', 'name': '', 'action': '', 'reason': summary}]

    def to_context(self) -> str:
        """Convert to string for LLM context."""
        if not self.entries:
            return "# Modification History\nNo modifications yet."

        lines = ["# Modification History"]
        for e in self.entries[-10:]:
            if e['type'] == 'summary':
                lines.append(f"- {e['reason']}")
            else:
                lines.append(f"- [{e['type']}/{e['name']}] {e['action']}: {e['reason']}")
        return "\n".join(lines)

    def clear(self):
        """Clear all entries."""
        self.entries = []


"""
Modification Memory for tracking rule refinement history.
Writes timestamped log files for debugging and audit.
"""
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, TextIO, Tuple
from dataclasses import dataclass, field


# Backward compatible class (still used by ConflictResolver/GapResolver)
@dataclass
class ModificationMemory:
    """
    Tracks rule modifications for LLM context.
    The detailed logging is now handled by RefinementLogger.
    """
    entries: List[Dict[str, str]] = field(default_factory=list)
    MAX_ENTRIES: int = 20

    def add(self, rule_type: str, rule_name: str, action: str, reason: str,
            new_rule_str: str = None):
        """Add a modification entry."""
        if len(self.entries) >= self.MAX_ENTRIES:
            self.entries = self._summarize_oldest(5) + self.entries[5:]

        entry = {
            'type': rule_type,
            'name': rule_name,
            'action': action,
            'reason': reason
        }
        if new_rule_str:
            entry['new_rule'] = new_rule_str

        self.entries.append(entry)

    def _summarize_oldest(self, n: int) -> List[Dict[str, str]]:
        """Summarize the oldest n entries."""
        old = self.entries[:n]
        summary = f"[{n} prior modifications: {', '.join([e['name'] for e in old])}]"
        return [{'type': 'summary', 'name': '', 'action': '', 'reason': summary}]

    def to_context(self) -> str:
        """Convert to string for LLM context."""
        if not self.entries:
            return "# Modification History\nNo modifications yet."

        lines = ["# Modification History"]
        for e in self.entries[-10:]:
            if e['type'] == 'summary':
                lines.append(f"- {e['reason']}")
            else:
                lines.append(f"- [{e['type']}/{e['name']}] {e['action']}: {e['reason']}")
        return "\n".join(lines)

    def clear(self):
        """Clear all entries."""
        self.entries = []


class ColumnRefinementRecord:
    """Record for a single column's refinement step."""

    def __init__(self, column: str, round_num: int, stage: str):
        self.column = column
        self.round = round_num
        self.stage = stage  # 'conflict_resolution' or 'gap_resolution'
        self.timestamp = datetime.now().isoformat()
        self.prompt: Optional[str] = None
        self.llm_response: Optional[str] = None
        self.clean_rules_before: Dict[str, str] = {}
        self.dirty_rules_before: Dict[str, str] = {}
        self.clean_rules_after: Dict[str, str] = {}
        self.dirty_rules_after: Dict[str, str] = {}
        self.modifications: List[Dict] = []

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "column": self.column,
            "round": self.round,
            "stage": self.stage,
            "prompt": self.prompt,
            "llm_response": self.llm_response,
            "clean_rules_before": self.clean_rules_before,
            "dirty_rules_before": self.dirty_rules_before,
            "clean_rules_after": self.clean_rules_after,
            "dirty_rules_after": self.dirty_rules_after,
            "modifications": self.modifications
        }


class RefinementLogger:
    """
    Logger for refinement process.
    Writes plain text logs with timestamp in filename.
    All columns in a single log file.
    """

    def __init__(self):
        self.log_file: Optional[TextIO] = None
        self.log_path: Optional[str] = None
        self.run_id: str = ""
        self.current_column: str = ""

    def start(self, output_dir: str = "agentic_error_detector/results", dataset_name: str = None):
        """Start a new logging session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = timestamp

        log_dir = os.path.join(output_dir, "refinement_logs")
        os.makedirs(log_dir, exist_ok=True)

        if dataset_name:
            self.log_path = os.path.join(log_dir, f"{timestamp}_{dataset_name}_clean_dirty_rule_generation.log")
        else:
            self.log_path = os.path.join(log_dir, f"{timestamp}_clean_dirty_rule_generation.log")

        self.log_file = open(self.log_path, "w", encoding="utf-8")

        self.log_file.write(f"{'='*60}\n")
        self.log_file.write(f"REFINEMENT AUDIT LOG\n")
        self.log_file.write(f"{'='*60}\n")
        self.log_file.write(f"Run ID: {self.run_id}\n")
        self.log_file.write(f"Started: {datetime.now().isoformat()}\n")
        self.log_file.write(f"{'='*60}\n\n")
        self.log_file.flush()

    def _log(self, msg: str):
        """Write a plain text message to log."""
        if self.log_file:
            self.log_file.write(msg + "\n")
            self.log_file.flush()

    def log_phase(self, phase: str, column: str, round_num: int = None):
        """Log the start of a refinement phase."""
        ts = datetime.now().strftime("%H:%M:%S")
        round_str = f" round {round_num}" if round_num is not None else ""
        self._log(f"\n[{ts}] === PHASE: {phase} | Column: {column}{round_str} ===")

    def begin_column_refinement(
        self,
        column: str,
        round_num: int,
        stage: str,
        clean_rules: Dict[str, str],
        dirty_rules: Dict[str, str]
    ):
        """Begin recording a refinement step for a column."""
        self.current_column = column
        ts = datetime.now().strftime("%H:%M:%S")
        self._log(f"\n[{ts}] --- Column: {column} | Round: {round_num} | Stage: {stage} ---")
        self._log(f"  Clean rules before: {clean_rules}")
        self._log(f"  Dirty rules before: {dirty_rules}")

    def set_prompt(self, prompt: str):
        """Log the LLM prompt."""
        self._log(f"\n\n  [PROMPT]\n{prompt}\n  [/PROMPT]")

    def set_response(self, response: str):
        """Log the LLM response."""
        self._log(f"\n  [RESPONSE]\n{response}\n  [/RESPONSE]")

    def add_modification(
        self,
        rule_type: str,
        rule_name: str,
        old_rule: str,
        new_rule: str,
        modification_type: str,
        reason: str
    ):
        """Log a modification."""
        self._log(f"  [MODIFICATION] {modification_type}")
        self._log(f"    Type: {rule_type}, Name: {rule_name}")
        self._log(f"    Old: {old_rule}")
        self._log(f"    New: {new_rule}")
        self._log(f"    Reason: {reason}")

    def end_column_refinement(
        self,
        clean_rules: Dict[str, str],
        dirty_rules: Dict[str, str]
    ):
        """End recording for current column."""
        self._log(f"  Clean rules after: {clean_rules}")
        self._log(f"  Dirty rules after: {dirty_rules}")
        self._log(f"  --- End of column refinement ---\n")
        self.current_column = ""

    def log_error(self, column: str, error_type: str, message: str):
        """Log an error."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._log(f"\n[{ts}] [ERROR] Column: {column} | Type: {error_type}")
        self._log(f"  Message: {message}")

    def log_round_summary(
        self,
        column: str,
        round_num: int,
        conflict_rate: float,
        gap_rate: float,
        num_modifications: int,
        converged: bool = True
    ):
        """Log summary at end of each round."""
        ts = datetime.now().strftime("%H:%M:%S")
        status = "CONVERGED" if converged else "CONTINUING"
        self._log(f"\n[{ts}] [SUMMARY] Column: {column} | Round: {round_num}")
        self._log(f"  Conflict rate: {conflict_rate:.4f}")
        self._log(f"  Gap rate: {gap_rate:.4f}")
        self._log(f"  Modifications: {num_modifications}")
        self._log(f"  Status: {status}")

    def log_rejection(
        self,
        rule_type: str,
        rule_name: str,
        old_rule: str,
        new_rule: str,
        violation_rate: float,
        reason: str
    ):
        """Log a rule rejection due to excessive violation rate."""
        self._log(f"\n  [REJECTED] Rule rejected for excessive violations")
        self._log(f"    Type: {rule_type}, Name: {rule_name}")
        self._log(f"    Violation rate: {violation_rate*100:.1f}% (>50% threshold)")
        self._log(f"    Old: {old_rule}")
        self._log(f"    Proposed new: {new_rule}")
        self._log(f"    Reason: {reason}")

    def log_initial_rules(
        self,
        clean_rules: Dict[str, List[Tuple[str, str]]],
        dirty_rules: Dict[str, List[Tuple[str, str]]],
        clean_prompts: Dict[str, Dict[str, str]] = None,
        dirty_prompts: Dict[str, Dict[str, str]] = None
    ):
        """Log the initial rule generation prompts and responses."""
        ts = datetime.now().strftime("%H:%M:%S")
        self._log(f"\n{'='*60}")
        self._log(f"[{ts}] INITIAL RULE GENERATION")
        self._log(f"{'='*60}\n")

        # Log clean rules
        self._log("--- CLEAN RULES (Quality Pillars) ---")
        if clean_rules:
            for column, rules in clean_rules.items():
                self._log(f"\n  Column: {column}")
                for pillar_name, rule in rules:
                    self._log(f"    [{pillar_name}]")
                    self._log(f"      Rule: {rule}")
                    # Log prompt if available
                    if clean_prompts and column in clean_prompts:
                        col_prompts = clean_prompts[column]
                        if pillar_name in col_prompts:
                            prompt_info = col_prompts[pillar_name]
                            if 'prompt' in prompt_info:
                                self._log(f"\n\n      [PROMPT]\n{prompt_info['prompt']}\n      [/PROMPT]")
                            if 'response' in prompt_info:
                                self._log(f"      [RESPONSE]\n{prompt_info['response']}\n      [/RESPONSE]")
        else:
            self._log("  No clean rules generated")

        # Log dirty rules
        self._log("\n--- DIRTY RULES (Error Detection Agents) ---")
        if dirty_rules:
            for column, rules in dirty_rules.items():
                self._log(f"\n  Column: {column}")
                for agent_name, rule in rules:
                    self._log(f"    [{agent_name}]")
                    self._log(f"      Rule: {rule}")
                    # Log prompt if available
                    if dirty_prompts and column in dirty_prompts:
                        col_prompts = dirty_prompts[column]
                        if agent_name in col_prompts:
                            prompt_info = col_prompts[agent_name]
                            if 'prompt' in prompt_info:
                                self._log(f"\n\n      [PROMPT]\n{prompt_info['prompt']}\n      [/PROMPT]")
                            if 'response' in prompt_info:
                                self._log(f"      [RESPONSE]\n{prompt_info['response']}\n      [/RESPONSE]")
        else:
            self._log("  No dirty rules generated")

        self._log(f"\n{'='*60}\n")

    def close(self, summary: Dict = None):
        """Close the log file."""
        if self.log_file:
            self._log(f"\n{'='*60}")
            self._log(f"Run ID: {self.run_id}")
            self._log(f"Finished: {datetime.now().isoformat()}")
            if summary:
                self._log(f"Summary: {summary}")
            self._log(f"{'='*60}")
            self.log_file.close()
            self.log_file = None

    @property
    def path(self) -> str:
        """Get the log file path."""
        return self.log_path


# Global logger instance
_logger: Optional[RefinementLogger] = None


def get_logger() -> RefinementLogger:
    """Get the global logger instance."""
    global _logger
    if _logger is None:
        _logger = RefinementLogger()
    return _logger


def start_logger(output_dir: str = "agentic_error_detector/results", dataset_name: str = None) -> RefinementLogger:
    """Start a new logging session."""
    logger = get_logger()
    logger.start(output_dir, dataset_name)
    return logger


def stop_logger(summary: Dict = None):
    """Stop the current logging session."""
    global _logger
    if _logger:
        _logger.close(summary)
        _logger = None
