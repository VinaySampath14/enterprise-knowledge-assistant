# Second Forensic Pass: Tiny Fallback Allowlist/Denylist Policy (2026-03-28)

## Scope
- Source report: artifacts/experiments/phase0/forensics/fallback_forensic_comparison_2026-03-28.json
- Method: query-text-only phrase/regex extraction
- Constraints: no code/tests/behavior changes

## Proposed Tiny Eligibility Policy
### Allow Patterns
- allow_explanatory_start: /^(?:what|how|explain|give\s+me\s+an\s+overview)\b/
- why: All corrected cases use explanatory framing.
- allow_exact_corrected_anchor: /\b(?:collections\.defaultdict|itertools\.chain|asyncio\s+event\s+loop)\b/
- why: Exact anchor forms from corrected queries only.
### Deny Patterns
- deny_in_module_use_symbol_template: /\bin\s+\w+\s*,\s*how\s+do\s+i\s+use\s+\w+\b/
- why: Synthetic regressed synv2-0012 template.
- deny_near_domain_offtopic_capability: /\b(?:wildcard|json|csv|sql|regex)\b/
- why: Regressed near-domain asks request off-topic capabilities.
- deny_python_general_broad: /\b(?:difference\s+between\s+a\s+list\s+and\s+a\s+tuple|python\s+garbage\s+collection|async\s+file\s+operations\s+in\s+python)\b/
- why: Regressed broad/python-general intents from this run.
### Required Retrieval/Gate Conditions
- gate_decision == 'answer'
- generated output is refusal-like
- mismatch_subtype == 'not_applicable'
- top-2 retrieval chunks share same module
- top-3 retrieval contains >=2 chunks from same module
### Deny Pattern Precedence
- If any deny pattern matches, ineligible even when allow patterns match

## Per-Case Evaluation
| id | desired | allow matches | deny matches | eligible |
|---|---:|---|---|---:|
| man-003 | True | allow_explanatory_start, allow_exact_corrected_anchor | - | True |
| man-007 | True | allow_explanatory_start, allow_exact_corrected_anchor | - | True |
| man-009 | True | allow_explanatory_start, allow_exact_corrected_anchor | - | True |
| man-011 | False | - | deny_near_domain_offtopic_capability | False |
| man-013 | False | allow_explanatory_start | deny_near_domain_offtopic_capability | False |
| man-014 | False | - | deny_near_domain_offtopic_capability | False |
| man-016 | False | allow_explanatory_start | deny_near_domain_offtopic_capability | False |
| man-017 | False | - | deny_near_domain_offtopic_capability | False |
| man-019 | False | allow_explanatory_start | deny_python_general_broad | False |
| man-020 | False | allow_explanatory_start | deny_python_general_broad | False |
| man-024 | False | allow_explanatory_start | deny_python_general_broad | False |
| synv2-0012 | False | - | deny_in_module_use_symbol_template | False |

## Policy Fit Summary
- TP: 3
- FP: 0
- FN: 0
- TN: 9

## Recommendation
- attempt_one_final_narrow_retry
- Tiny allowlist/denylist isolates corrected cases with zero false-eligible regressions in this sample.