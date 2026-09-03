# Dependabot Triage — 2026-09 (CI enablement pass)

Triage record for the dependency state of `synthesisengineering/ragbot` at the
point the repository gained continuous integration. Unlike the
[2026-05 pass](dependabot-triage-2026-05.md), which worked from GitHub's alert
list, this one was driven by a CI run: the first workflow to execute on a pull
request failed on both jobs, and the two failures were more consequential than
the twenty-one open dependency pull requests they were meant to adjudicate.

This document is the source of truth for the classifications and fixes below.
`.github/dependabot.yml` references it.

## Summary

| Bucket | Count |
| --- | --- |
| Open dependency PRs at pass start | 21 |
| Blocking defects found by the first CI run | 2 |
| Transitive alerts fixed by override | 1 |
| PRs made obsolete by the tree having already moved past them | 2 |
| PRs superseded by a fix in this pass | 1 |

## The two blocking defects

### 1. `mcp` — a floor-only constraint let a breaking major in

**Classification: real-exposure, install-breaking.**

`requirements.txt` declared `mcp>=1.27.0` with no upper bound. `mcp` has since
released 2.x (2.1.1 at the time of this pass), and 2.x removed
`mcp.client.experimental` and moved `AnyUrl` out of `mcp.types`. A clean
`make install` therefore resolved to a version that breaks four modules at
import time:

- `src/synthesis_engine/mcp/tasks.py` — `mcp.client.experimental`
- `src/synthesis_engine/mcp/proxy.py` — `AnyUrl`
- `src/synthesis_engine/mcp/primitives/resources.py` — `AnyUrl`
- `src/synthesis_engine/mcp_server/server.py` — `AnyUrl`

The suite could not be collected: six errors before a single test ran. The
tests passed on developer machines only because those already had 1.27.1
installed, so the breakage was invisible to everyone who had ever worked on the
repository and total for anyone cloning it.

**Fix:** `mcp>=1.27.1,<2`. The floor rises to the version actually verified
working; the ceiling is the part that matters.

**Consequence for the open PR queue:** PR #29 proposed `>=1.27.0` → `>=1.27.1`.
That would not have fixed this. Raising a floor does not stop pip from taking a
newer major, and merging it would have looked like maintenance while changing
nothing. Closed as superseded.

**Not done here:** migrating to `mcp` 2.x. That is a real change across four
modules against a new SDK surface, and it belongs in its own pull request with
its own review, not smuggled into a CI enablement pass.

**Generalisation worth noting:** every Python dependency in this repository is
declared floor-only, and every open pip PR is a floor bump. `mcp` is the one
that has bitten so far; it is not structurally special. A future pass should
decide whether the `>=`-only convention is deliberate.

### 2. `web/package-lock.json` — out of sync with `package.json`

**Classification: build-breaking, no security component.**

`npm ci` refused to run: `npm ci` requires the lockfile and manifest to agree,
and the lockfile was missing seven entries — the wasm and native fallback
shims under `@tailwindcss/oxide-wasm32-wasi`, `@rolldown/binding-wasm32-wasi`,
and `@emnapi/*`.

**Fix:** regenerated with `npm install --package-lock-only`. Seven packages
added, none removed, and no dependency version changed apart from the override
below. The web suite passes unchanged (24 tests).

**Consequence for the open PR queue:** this is why all nine npm PRs reported
`CONFLICTING`. They were each built on a lockfile that did not match the
manifest, so none of them could merge, and rebasing them one at a time would
have re-conflicted each survivor against the last merge.

## Transitive alert fixed by override

### `nanoid` — high severity, `<3.3.18`

Surfaced by `npm audit` during the lockfile regeneration; not a direct
dependency. Handled per the standing policy — transitive alerts get an override
in `web/package.json` rather than an ignore in `.github/dependabot.yml`:

```json
"nanoid@<3.3.18": ">=3.3.18 <4"
```

Resolved 3.3.16 → 3.3.18. `npm audit` reports 0 vulnerabilities after the
change, and the web suite still passes.

## PRs already obsolete before this pass

The tree had moved past two of the open npm PRs, which is worth recording
because it is invisible from the PR list alone:

| PR | Proposes | Tree already at |
| --- | --- | --- |
| #42 | `next` 16.2.6 → 16.2.7 | 16.2.12 |
| #44 | `eslint` 9.39.4 → 10.4.1 | 10.8.0 |

## What CI now guarantees

`make test-fast` and the web lint plus suite run on every pull request and on
every push to `main`, secret-free so they also run on forked and Dependabot
pull requests. The queue that motivated this pass can now be judged on
evidence: each remaining dependency PR gets a pass or fail rather than a
reviewer's reading of a changelog.

## Unverified remainder

- Whether any Python dependency other than `mcp` is currently broken by its
  own floor-only constraint. CI answers this per-PR from here on, but no
  exhaustive sweep of latest-version resolution was performed in this pass.
- `mcp` 2.x compatibility, deliberately deferred as above.
