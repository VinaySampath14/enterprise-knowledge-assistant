# Failure Bucket Summary - phase2_conceptual_manual_candidate

1. bucket: **clarify vs refuse confusion**
   count: 5
   likely_fixability: high
   trigger_counts: {'mismatch_recoverable': 3, 'ambiguity': 2}
   example_query_ids: ['man-012', 'man-024', 'man-029', 'man-030', 'man-032']

2. bucket: **conceptual in-domain false refusals**
   count: 2
   likely_fixability: high
   trigger_counts: {'ambiguity': 2}
   example_query_ids: ['man-007', 'man-009']

3. bucket: **python-general leakage**
   count: 2
   likely_fixability: high
   trigger_counts: {'ambiguity': 2}
   example_query_ids: ['man-021', 'man-023']

4. bucket: **other repeated bucket (clarify->answer, ambiguity)**
   count: 1
   likely_fixability: high
   trigger_counts: {'ambiguity': 1}
   example_query_ids: ['man-027']

