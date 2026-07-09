# Internationalisation Policy

Conventions for adding translated documents to penta2himajin's repos. Adopted from `tympan-ladspa`'s ADR 0006.

## Premises

- English is the canonical language for code, comments, commit messages, design documents, and external communication.
- Japanese is the only currently tracked second language. Future languages follow [BCP 47](https://www.rfc-editor.org/info/bcp47) tags (`.zh-Hans.md`, `.de.md`, …).
- Translations must never block a PR.

## Layout: suffix files

Translated documents live next to their English originals with a `.<lang>.md` suffix:

```
README.md              # English, authoritative
README.ja.md           # Japanese

docs/overview.md
docs/overview.ja.md
```

The English file is the source of truth in every pair. No directories named after a language are added.

Why suffix files (vs. parallel directories `docs/en/`, `docs/ja/`):

- Zero migration: existing `CLAUDE.md` and PR-template references stay valid.
- `cargo publish` continues to use the root `README.md` without configuration.
- `ls docs/` reveals at a glance which files have translations and which do not.

## Scope: basic user-facing content only

**In scope** for translation:

- `README.md`
- The "user-facing introduction" tier of `docs/` (typically `docs/overview.md`). Expand by amending this document.
- Public-API rustdoc on crate-level items, when and if a stable public API ships.

**Out of scope** (English only):

- Engineering-internal docs: `docs/architecture.md`, `docs/handoff-protocol.md`, `docs/references.md`, every `docs/decisions/*.md` ADR.
- `CLAUDE.md` and other project-instruction files.
- Code comments, doc-comments inside non-crate-level items, commit messages, PR descriptions, issue text.

## Translations never block

A PR that updates an English document is not required to update its translated twin. CI does not enforce parity. Reviewers do not request `.ja.md` edits as a blocker. Translation drift is expected and visible by construction (see Source header below).

## Source header in every translated file

The first non-title line of each `.ja.md` file is:

```
> Source: <basename>.md @ <commit-sha-of-source-at-time-of-translation>
```

The header documents which revision of the English source the translation is derived from. A reader who notices drift compares the English source's current `git log` to the recorded SHA. Automated drift detection is not required; manual audit before a release is sufficient.

## Language switcher in the English source

Each English file that has a translation pair carries one line near the top, immediately under the H1:

```
[日本語](./<basename>.ja.md)
```

The translated file mirrors a back-link to its English source.

## Adding a translation pair

1. Add the entry to the "Tracked translations" table below in the same PR.
2. Create `<name>.<lang>.md` with the Source header pointing at the current HEAD SHA of `<name>.md`.
3. Insert the switcher link into `<name>.md`.

Removing a translation: reverse the steps above.

## Tracked translations

| English source | Japanese translation | Status |
|---|---|---|
| `README.md` | `README.ja.md` | Placeholder |

## Trigger for revisiting

Re-evaluate when any of the following holds:

- A third language is requested. At three or more languages the suffix scheme starts to clutter `ls`; reconsider a bucketed `translations/<lang>/` layout.
- A static-site generator is adopted (Docusaurus, mdBook with i18n, …). Their i18n conventions supersede the suffix scheme.
- A user reports confusion finding the translated entry; the switcher convention may need amplification.
