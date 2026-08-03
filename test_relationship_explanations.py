import unittest

import pandas as pd

import evaluate_tableeg_semantic_correctness as tableeg_eval
import main
from cleanem_models import EvidenceContribution, EvidenceObservation


class RelationshipExplanationTests(unittest.TestCase):
    def test_agent_pattern_rule_scale_only_changes_agent_pattern_evidence(self):
        agent_pattern = EvidenceObservation(
            target_scope="cell",
            target_key="0::code",
            source_id="pattern_mismatch",
            family="pattern",
            polarity="dirty",
            strength=0.8,
            hard=False,
            reason_code="pattern_mismatch",
            metadata={"rule_pool_source": "agentic_clean_rule_pool"},
        )
        deterministic_pattern = EvidenceObservation(
            target_scope="cell",
            target_key="0::code",
            source_id="regex_fail",
            family="pattern",
            polarity="dirty",
            strength=0.6,
            hard=False,
            reason_code="regex_fail",
        )
        counts = main._scale_agent_pattern_rule_evidence(
            {},
            {"0::code": [agent_pattern, deterministic_pattern]},
            0.5,
        )
        self.assertAlmostEqual(agent_pattern.strength, 0.4)
        self.assertAlmostEqual(deterministic_pattern.strength, 0.6)
        self.assertEqual(counts, {"cell": 1, "dirty": 1})
        self.assertEqual(
            agent_pattern.metadata["agent_pattern_rule_scale"],
            0.5,
        )

    def test_pattern_mismatch_keeps_each_failed_agent_rule(self):
        df = pd.DataFrame({"code": ["12345", "12x45", "99999"]})
        metadata = {
            "code": {
                "type": "categorical",
                "relationship_profiles": [],
            }
        }
        rule_pool = {
            "code": [
                {
                    "agent": "pattern_consistency",
                    "family": "pattern",
                    "rule_name": "pattern_consistency_0",
                    "rule_str": (
                        "lambda value, row=None: "
                        "str(value).strip().isdigit()"
                    ),
                    "rule_func": (
                        lambda value, row=None: str(value).strip().isdigit()
                    ),
                }
            ]
        }
        value_registry = main._build_value_registry(df, metadata)
        cell_registry = main._build_cell_registry(
            df,
            metadata,
            value_registry,
        )
        _, cell_observations = main._extract_evidence(
            df,
            metadata,
            rule_pool,
            value_registry,
            cell_registry,
            enable_canonicalization_evidence=False,
            enable_evidence_gating=False,
        )
        mismatch = next(
            item
            for item in cell_observations["1::code"]
            if item.source_id == "pattern_mismatch"
        )
        self.assertEqual(mismatch.metadata["evaluated_rule_count"], 1)
        self.assertEqual(mismatch.metadata["failed_rule_count"], 1)
        self.assertEqual(
            mismatch.metadata["failed_rules"][0]["rule_name"],
            "pattern_consistency_0",
        )
        self.assertEqual(
            mismatch.metadata["failed_rules"][0]["agent"],
            "pattern_consistency",
        )
        self.assertIn(
            "isdigit",
            mismatch.metadata["failed_rules"][0]["rule_str"],
        )

        contribution = EvidenceContribution(
            target_scope="cell",
            source_id=mismatch.source_id,
            family=mismatch.family,
            polarity=mismatch.polarity,
            reason_code=mismatch.reason_code,
            strength=mismatch.strength,
            hard=mismatch.hard,
            feature_name="cell:pattern:pattern_mismatch",
            feature_weight=1.0,
            signed_feature_value=mismatch.strength,
            weighted_logit=1.0,
            posterior_without=0.2,
            posterior_contribution=0.4,
            metadata=mismatch.metadata,
        )
        text = main._explanation_reason_text(contribution)
        self.assertIn("pattern_consistency_0", text)
        self.assertIn("failed 1 of 1", text)

    def test_contextual_evidence_keeps_rule_details(self):
        df = pd.DataFrame(
            {
                "brewery_id": ["1", "1", "1", "1", "1", "2", "2", "2"],
                "city": [
                    "San Diego",
                    "San Diego",
                    "San Diego",
                    "San Diego",
                    "San Diego CA",
                    "Austin",
                    "Austin",
                    "Austin",
                ],
            }
        )
        observations = {}
        main._add_contextual_consensus_evidence(
            df,
            {"brewery_id": {}, "city": {}},
            observations,
        )
        evidence = next(
            item
            for item in observations["4::city"]
            if item.source_id == "contextual_disagreement"
        )
        self.assertEqual(evidence.metadata["determinant_column"], "brewery_id")
        self.assertEqual(evidence.metadata["dependent_column"], "city")
        self.assertEqual(evidence.metadata["expected_value"], "San Diego")
        self.assertEqual(evidence.metadata["observed_value"], "San Diego CA")
        self.assertEqual(evidence.metadata["reference_rows"], [0, 1, 2, 3])
        self.assertTrue(evidence.metadata["rule_violated"])
        self.assertEqual(evidence.metadata["validation_status"], "data_validated")

        contribution = EvidenceContribution(
            target_scope="cell",
            source_id=evidence.source_id,
            family=evidence.family,
            polarity=evidence.polarity,
            reason_code=evidence.reason_code,
            strength=evidence.strength,
            hard=evidence.hard,
            feature_name="cell:contextual:contextual_disagreement",
            feature_weight=1.0,
            signed_feature_value=evidence.strength,
            weighted_logit=1.0,
            posterior_without=0.2,
            posterior_contribution=0.4,
            metadata=evidence.metadata,
        )
        text = main._explanation_reason_text(contribution)
        self.assertIn("brewery_id", text)
        self.assertIn("city", text)
        self.assertIn("San Diego CA", text)
        self.assertIn("San Diego", text)
        self.assertIn("supporting rows: 1, 2, 3, 4", text)

    def test_contextual_evidence_abstains_on_weak_majority(self):
        df = pd.DataFrame(
            {
                "brewery_id": ["1", "1", "1"],
                "city": ["San Diego", "San Diego", "San Diego CA"],
            }
        )
        observations = {}
        main._add_contextual_consensus_evidence(
            df,
            {"brewery_id": {}, "city": {}},
            observations,
        )
        self.assertNotIn("2::city", observations)

    def test_profile_relationship_keeps_columns_and_rule(self):
        df = pd.DataFrame(
            {
                "State": ["CA", "TX", "NY"],
                "Stateavg": ["CA_metric", "CA_metric", "NY_metric"],
            }
        )
        metadata = {
            "State": {
                "type": "categorical",
                "relationship_profiles": [],
            },
            "Stateavg": {
                "type": "categorical",
                "relationship_profiles": [
                    {
                        "other_column": "State",
                        "type": "stateavg_format",
                        "description": "Stateavg should use <state>_<metric> format",
                        "violation_rate": 0.05,
                        "applicable_count": 20,
                    }
                ],
            },
        }
        value_registry = main._build_value_registry(df, metadata)
        cell_registry = main._build_cell_registry(
            df,
            metadata,
            value_registry,
        )
        _, cell_observations = main._extract_evidence(
            df,
            metadata,
            {},
            value_registry,
            cell_registry,
            enable_canonicalization_evidence=False,
            enable_evidence_gating=False,
        )
        evidence = next(
            item
            for item in cell_observations["1::Stateavg"]
            if item.source_id == "strong_relationship_violation"
        )
        self.assertEqual(evidence.metadata["determinant_column"], "State")
        self.assertEqual(evidence.metadata["dependent_column"], "Stateavg")
        self.assertEqual(evidence.metadata["determinant_value"], "TX")
        self.assertEqual(evidence.metadata["observed_value"], "CA_metric")
        self.assertIn("Stateavg should use", evidence.metadata["rule_text"])
        self.assertEqual(len(evidence.metadata["violated_relationships"]), 1)

    def test_tableeg_alignment_reads_shared_relation_fields(self):
        annotation = {
            "row_id": "2",
            "column": "city",
            "constraint": (
                "Constraint Violation: The brewery_id of Row 1 is equal to "
                "the brewery_id of Row 2, and the city values differ."
            ),
            "tuple_pairs": "(1, 2)",
            "right_value": "San Diego",
        }
        trace = {
            "decision": "suspicious",
            "evidence": [
                {
                    "source_id": "contextual_disagreement",
                    "family": "contextual",
                    "reason_code": "conditional_value_disagreement",
                    "posterior_contribution": 0.4,
                    "metadata": {
                        "determinant_column": "brewery_id",
                        "dependent_column": "city",
                        "related_columns": ["brewery_id", "city"],
                        "rule_text": (
                            "Rows with the same brewery_id should share city"
                        ),
                        "expected_value": "San Diego",
                        "reference_rows": [1, 2],
                    },
                }
            ],
        }
        alignment = tableeg_eval.rule_explanation_alignment(
            annotation,
            trace,
            {"1": 1, "2": 2},
        )
        self.assertTrue(alignment["type_aligned"])
        self.assertTrue(alignment["column_aligned"])
        self.assertTrue(alignment["detail_available"])
        self.assertTrue(alignment["reference_available"])
        self.assertTrue(alignment["expected_value_matches"])
        self.assertTrue(alignment["column_pair_matches"])
        self.assertTrue(alignment["reference_hit_at_1"])
        self.assertTrue(alignment["grounded_matches"])


if __name__ == "__main__":
    unittest.main()
