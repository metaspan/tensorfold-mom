# External reviews

Reviews of this project produced by other AI models (or humans). One file per
review; keep the reviewer's text verbatim, our triage appended at the bottom.

## Naming

`YYYY-MM-DD-<reviewer>-<topic>.md` — reviewer = model/product name, lowercase.

## Required header (top of every file)

```
Reviewer: <model name/version>
Date: YYYY-MM-DD
Reviewed commit: <sha or "unanchored">
Reviewed scope: <files/areas/docs given to the reviewer>
Channel: <how it arrived: pasted in chat / file / PR link>
```

## Triage protocol (Hermes, appended under `## Triage`)

Every substantive point gets exactly one verdict:

| verdict | meaning |
|---|---|
| ACT | accepted — work item opened (name it here) |
| BACKLOG | valid, deferred — named backlog entry + where |
| DISAGREE | contested — evidence cited (code path, measurement, doc) |
| FIXED | already addressed — cite commit |

Rules: verify claims against the code before accepting (a review is a
self-report, not a fact); never edit the reviewer's text; if a review is
superseded (code moved on), add a `## Superseded` note with the sha, don't
delete.
