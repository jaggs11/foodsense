# results/

Empty of current numbers, on purpose.

## Where the numbers went

Everything this project had evaluated before the post-review repositioning is in
`archive/v1_multiage_counterfactual/`, byte-identical to how it was produced, with
its original `MANIFEST.sha256`. The code that produced it is reachable from the
annotated tag `v1-multiage-counterfactual`.

Verify it at any time:

    python scripts/verify_results.py --results-dir results/archive/v1_multiage_counterfactual

That is what `make verify-results` runs. The digests are the ones the artefacts
were written with -- archiving moved the files and rewrote the path prefixes in
the manifest, and changed nothing else.

## Why it is empty

Those results measure a system that no longer exists: three age groups, three
goals, a USDA-only corpus. The repositioning (see `docs/pivot.md`) makes toddlers
the only population and conditions the only objectives, so the 300-case
multi-age comparison, the lambda sweep, the validity decomposition and the
surrogate-boundary study are all measurements of a population the system no
longer serves.

They are archived rather than deleted because the re-run on the toddler and
condition harness in P9 needs something to be compared against.

## What lands here next

New evaluation artefacts, in P9, produced by the evaluation framework built in
that phase. Until then this directory holds no current numbers, and no number
appears anywhere in this repository before the script that produces it has run.
