# data/indian/

Indian food composition data, and the manifest that describes where each file
came from.

Under the repositioning (`docs/pivot.md`), Indian regional food data is the
**primary** corpus and USDA is supplementary. This directory is where the primary
corpus enters the project.

## The rule

**Nothing is ingested that is not listed in `manifest.yaml`, and nothing is
listed in `manifest.yaml` that is not on disk.** A manifest entry is written
after the bytes are local, never after finding a promising link. Bibliographic
fields are read out of the document itself -- its title page or its metadata --
not from a search result, a landing page, or memory. A field that cannot be read
from the file is `unknown`.

Raw downloads are gitignored. The manifest is committed: it holds metadata and
hashes, not content.

## manifest.yaml

```yaml
sources:
  - id: <short stable slug, referenced by every row derived from this file>
    name: <title, as printed in the document>
    organisation: <publisher, as printed in the document>
    reference: <citation, DOI or URL>
    license_note: <terms as stated by the document or its host; "unknown" is fine>
    file: <path relative to data/raw/indian/>
    basis: per_100g_edible_portion | per_serving
    source_type: indian_dataset | research_dataset | government | derived_recipe
    confidence: high | medium | low | unknown
```

`basis` is read from the document, never assumed. `foodsense/data/units.py`
treats it as authoritative when converting to the canonical per-100g storage, so
a wrong value here becomes wrong nutrition everywhere downstream.

## What may enter

In order of preference:

1. A national composition table for Indian foods, raw and cooked, per 100 g
   edible portion, with an organisational publisher. `source_type:
   indian_dataset`, `confidence: high`.
2. A peer-reviewed recipe-level Indian nutrient databank.
3. Standard recipes from a citable source, for composite dishes absent from the
   above. These feed the `derived_recipe` path: ingredients in grams, yield
   factor, method and the ingredient rows used are all stored in
   `recipe_derivations` so the computation can be audited. `confidence: medium`.

Unsourced community datasets (dataset-site scrapes, blog tables, app exports) may
enter only as `verification_status: unverified`, and are excluded from
recommendation by default. Prefer not entering them at all. A smaller verified
corpus beats a larger unverifiable one -- verification is the project's
contribution, so a corpus that cannot be verified subtracts from it.

## Also here

- `derived/<source_id>.csv` -- nutrient rows derived from a source, one row per
  food, every row carrying `source_id` and a page or table reference. Committed.
- `coverage.md` -- for each food named in requirements section 7, whether it is
  covered and by which source. "Not covered" is a legitimate outcome and is what
  tells the architect what can be promised.

## Status

`manifest.yaml` is empty. Populating it is P2.0; ingestion is P2. Both are
blocked until real files are here.
