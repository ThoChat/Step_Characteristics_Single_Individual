"""Validation of the method.

- ``previous_results``: comparison with the results and raw files of the previous analysis.
- ``previous_code``: reference copies of the previous functions. Import it explicitly
  (``from step_analysis.validation import previous_code``): it needs ``shapely``, a
  test-only dependency.
- ``synthetic``: synthetic trials with known answers.
"""

from . import previous_results, synthetic  # noqa: F401
