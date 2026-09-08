# knowledge/sources/

Condition evidence documents, and the manifest that describes them.

Every citation the system emits resolves to an entry here. Ruling R9: chunks come
only from documents listed in `sources.yaml` that physically exist under
`knowledge/raw/`. A manifest entry is never authored for a document that has not
been read from disk, and no URL, title, author or year is written that does not
come from the document itself.

Raw documents are gitignored. The manifest is committed.

## sources.yaml

```yaml
sources:
  - id: <short stable slug, cited by conditions and by generated explanations>
    title: <as printed on the document title page>
    organisation: <publisher, as printed>
    year: <as printed; "unknown" if absent>
    url: <where it was retrieved from>
    doc_type: guideline | review | primary_study | government | composition_table
    population: toddler | children | infant | general
    conditions: [<condition ids this document bears on>]
    nutrients: [<nutrient keys this document gives figures for>]
    file: <path relative to knowledge/raw/>
    license_note: <terms as stated by the document or its host; "unknown" is fine>
```

## Why this gates P3

A condition in `configs/conditions/` is `enabled: true` only when **every rule it
scores cites a source_id present here**. A condition with no sources exists as a
skeleton, `enabled: false`, and is hidden from the UI. That is checked
mechanically rather than by review, so a condition cannot be switched on by
someone who forgot the evidence.

## Extracts

Each acquired document gets `knowledge/extracts/<source_id>.md`: one block per
claim, each carrying the value, a short verbatim quote containing that value, a
page number, and the population band as the document states it. A threshold that
reaches `configs/conditions/*.yaml` without an extract behind it does not get
written.

## Failed acquisitions

Recorded in `unavailable.md`, not silently replaced. A paywalled or missing
document is a normal outcome; substituting a different document that looks
similar is not.

## Status

`sources.yaml` is empty. Populating it is P2.0. P3 cannot enable any condition
until at least one source exists per condition to be enabled.
