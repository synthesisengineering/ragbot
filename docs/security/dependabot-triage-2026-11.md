# Dependabot Triage — 2026-11 (queue clearance)

Triage record for the 20 Dependabot pull requests open on
`synthesisengineering/ragbot` on 2026-11-06, between 24 and 64 days old. Earlier
passes: [2026-05](dependabot-triage-2026-05.md) (alert-driven) and
[2026-09](dependabot-triage-2026-09.md) (CI enablement). This document is the
source of truth for the decisions below.

## Summary

| Bucket | Count |
| --- | --- |
| Open Dependabot PRs at pass start | 20 (10 pip, 10 npm) |
| Merged | 19 |
| Closed without merge | 1 (#62) |
| Web advisories (`npm audit`) on `main` before the pass | 22: 1 critical, 6 high, 15 moderate |
| Web advisories after this pass | 0 |
| Python advisories (`pip-audit`) after this pass | 13, all `filelock`, blocked upstream (see below) |

Both ecosystems were at their `open-pull-requests-limit` of 10 when the pass
began.

## CI freshness

A green check is evidence only for the tree it ran against.

| Group | CI ran | Against | Changed on `main` since | Treated as |
| --- | --- | --- | --- | --- |
| pip (#95–#106) | 2026-09-21; #101 2026-09-25 | `7b75de3`; #101 `a360a14` | `a360a14` changed source and `requirements.txt` | stale; verified locally |
| npm (#117, #118, #123–#129) | 2026-10-05 and 2026-10-13 | `e64c3e7` | `README.md` only | current |
| #62 | 2026-09-20 | `dab5d13` | — | failing |

## Python group — verified locally, then merged

Two fresh Python 3.12 environments, one from `main` and one from `main` with all
ten floor bumps applied, each installed with `make install`:

- `pip freeze` output identical between the two. Every new floor sits at or
  below what pip already resolves, so no floor dragged another package
  backwards, which is the failure mode recorded in the 2026-09 addendum.
- `pip check` clean in both.
- `make test-fast`: 1015 passed, 13 skipped in both.

#101 (`langchain`) changes the line adjacent to #98 (`langchain-anthropic`), so
it conflicted after #98 merged. Dependabot rebased it, raising its target from
`>=1.4.2` to `>=1.4.8`, and CI passed on the rebased branch before merge.

Merged: #95 `pytest-asyncio>=1.4.0`, #96 `pytest>=9.1.1`,
#98 `langchain-anthropic>=1.7.2`, #100 `boto3>=1.43.97`, #101 `langchain>=1.4.8`,
#102 `transformers>=5.17.0`, #103 `anthropic>=1.6.0`,
#104 `prometheus-client>=0.26.0`, #105 `uvicorn[standard]>=0.53.0`,
#106 `langchain-google-genai>=4.4.0`.

## Web group

Six PRs touched non-adjacent lines and merged without conflict. Their combined
tree, merged exactly as GitHub would, passed `npm ci`, lint, the suite and
`next build` before any of them merged: #128 `jsdom` 30.1.2, #126 `@types/node`
26.6.5, #127 `@next/eslint-plugin-next` 16.4.0, #129 `@vitejs/plugin-react`
6.1.2, #123 `eslint` 10.12.0, #125 `typescript-eslint` 8.71.1.

#117 and #118 conflicted with those six, so Dependabot rebased them. The rebase
also moved both from 5.0.3 to 5.1.0, which clears GHSA-p458-7748-mcp3; 5.0.3
would have left it open. Both merged on green CI.

### #124 — `next`: green in CI, but the build failed

As opened, #124 proposed `next` 16.4.0. CI was green because CI ran lint and
the unit suite but never `next build`. On 2026-11-06, `next build` on 16.4.0
exited 1: Next.js refuses to build a version with known security advisories,
and three published advisories cover `>=16.0.0 <=16.4.0`:

- GHSA-2464-fhr4-w9xw — unauthenticated remote code execution in the Image
  Optimization API with SVG files
- GHSA-673q-mjm6-r5xr — unauthenticated remote code execution in `use cache`
  functions returning typed arrays
- GHSA-p3m5-59mp-jv5r — denial of service in the App Router through Server
  Actions

`main` was on 16.3.5, which those three cover along with eight more, among them
GHSA-vcvr-r3jv-pc5j (remote code execution in `next/og` `ImageResponse`).

Dependabot's rebase moved #124 to 16.4.4, which carries none of these. It
passed `npm ci`, lint, the suite and `next build` together with #117 and #118
locally, and passed CI, before merging. This pass then moves `next` to 16.4.5,
the current release.

**CI change.** The web job now runs `npm run build`. A `next` version that
cannot build should fail CI, not pass it. Next.js cited the npm advisory
database for its refusal, so a version that builds today can stop building once
an advisory against it is published. That failure is the signal the step exists
to give.

### #62 — `typescript` 7: closed

`typescript-eslint` 8.72.1, the latest release, declares a peer range of
`typescript >=4.8.4 <6.1.0`, so `npm ci` fails with `ERESOLVE` against
TypeScript 7. Nothing in this repository can change that. Installing with
`--legacy-peer-deps` would only hide the conflict.

This pass instead moves TypeScript from 5.9.3 to 6.0.3, the newest release the
lint toolchain accepts. `tsc --noEmit`, lint and `next build` pass with no
`tsconfig.json` change. The range is `~6.0.3` rather than `^6.0.3`, so a future
6.1 cannot enter the lockfile ahead of `typescript-eslint`'s support for it. TypeScript 7 becomes
possible once a `typescript-eslint` release accepts it.

## Transitive web advisories

Fixed in the same branch, following the standing policy: overrides in
`web/package.json` for transitive packages, not ignores.

| Package | Before | After | How |
| --- | --- | --- | --- |
| `sharp` | 0.35.4 (`<0.35.5`, librsvg) | 0.35.5 | arrived with `next` 16.4.4 (#124); override retargeted from `0.35.3` to `0.35.5` so it cannot pin a vulnerable version |
| `brace-expansion` | 5.0.9 | 5.0.12 | override retargeted; the old target was itself vulnerable |
| `minimatch` | 10.2.6 | 11.0.5 | override `>=10.0.0 <11.0.1` → `>=11.0.1 <12`; GHSA-g3hv-75wg-m249 covers 7.0.0–11.0.0 with no backport |
| `postcss-selector-parser` | 6.0.10 | 6.1.5 | new override; `@tailwindcss/typography` pins 6.0.10 exactly |
| `source-map-js`, `vite`, `zod`, `undici` | various | in-range patches | lockfile refresh within existing ranges |

`minimatch` 11 keeps 10's export map and engine range. Its consumers are ESLint
and `typescript-estree`; lint reports the same 24 files and zero findings before
and after.

The lockfile drops `@rolldown/binding-wasm32-wasi` and its `@emnapi` and
`@napi-rs` dependencies. That is upstream: `rolldown` 1.2.16 no longer lists
the wasm32 binding. It is not the macOS lockfile omission recorded in 2026-09.

## Open: `filelock` (Python), blocked by LiteLLM's cap

`pip-audit` reports 13 CVEs against `filelock` 3.32.7, fixed only in 4.x
(4.0.9 for one, 4.2.1 for the rest). `filelock` is transitive, through `torch`,
`huggingface_hub` and `litellm`.

LiteLLM 1.108.1, the latest release, declares `filelock>=3.16.1,<4.0`. Adding
`filelock>=4.2.1` to `requirements.txt` does resolve, and that is the problem:
the resolver moves LiteLLM back from 1.108.1 to 1.103.10, the newest release
without the cap. It is the openai 3 case from 2026-09 again. The result would
pair LiteLLM with a `filelock` major its own later releases forbid, and would
fix the default gateway at an older version.

Two ways to resolve it, neither one a dependency bump:

1. A LiteLLM release that accepts `filelock` 4. The next pass checks this first.
2. Make the `direct` backend the default and drop LiteLLM from
   `requirements.txt`. The 2026-09 triage already named this as a decision to
   make on its own merits.

## Verification commands

```bash
make install && python -m pip check && make test-fast
cd web && npm ci && npm run lint && npm run test && npm run build && npm audit
```

## Second batch — same day

Clearing the queue freed both ecosystems' slots, and Dependabot opened 15 more
pull requests (#132–#146) at 05:32–05:34 UTC. They were triaged the same
morning.

CI on all 15 ran after #131 merged, against the `main` they would land on, and
included the new `next build` step, so it was treated as current. 13 were green.

**Python (6), verified locally, then merged.** Fresh Python 3.12 environments
from `main` and from `main` with all six floors gave identical `pip freeze`,
clean `pip check`, and 1015 passed / 13 skipped in both. Merged #133
`boto3>=1.43.123`, #135 `fastapi>=0.162.0`,
#137 `opentelemetry-exporter-otlp-proto-grpc>=1.45.2`, #139 `psycopg>=3.3.7`
and #141 `langchain-anthropic>=1.7.9`. #142 `uvicorn[standard]>=0.54.0` edits
the line after #135's, so it conflicted and was sent back to Dependabot for a
rebase.

**Web (7), verified as a combined tree, then merged.** The seven green npm PRs
touch no shared lines. Merged together on top of `main`, they passed `npm ci`,
lint (24 files, 0 findings), 27 tests and `next build`, and `npm audit` found 0
vulnerabilities. Merged #132 `@types/node` 26.6.6, #134 `eslint` 10.14.0,
#136 `jsdom` 30.1.3, #138 `@testing-library/user-event` 14.6.8, #143
`tailwindcss` 4.3.4, #145 `react-syntax-highlighter` 16.1.2 and #146
`@vitejs/plugin-react` 6.2.1.

### #140 and #144 — `react` and `react-dom` opened separately: closed, fixed together

Each bumped one package to 19.3.1 and left the other at 19.3.0. Every web test
file then failed to load with React's own check: `react` and `react-dom` must
have the exact same version. Neither pull request could pass alone, and the two
conflict on adjacent `package.json` lines, so they cannot be merged as a pair
either.

Both packages are now bumped to 19.3.1 in one change, verified with `npm ci`,
lint, tests, `next build` and `npm audit` (0). A Dependabot `groups` entry in
`.github/dependabot.yml` puts the two in one pull request from now on. The group
covers only these two packages, because only they must match exactly. #117 and
#118 showed `vitest` and `@vitest/ui` passing as separate pull requests.
