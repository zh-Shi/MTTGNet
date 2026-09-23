#!/usr/bin/env python3
"""Download CMIP6 `tas` (near-surface air temperature) for the century-projection experiments.

Why this exists
---------------
Observations (ERA5 / HadCRUT5) stop in 2024-2026, but the SSP CO2 pathways run to
2100.  A model trained only on the observational record therefore never sees the
forced response it is asked to extrapolate, and the literature shows it fails:
DL emulators trained on history alone lose skill under future forcing, with
exaggerated or misattributed sensitivities (see 04_audit.md §10.7 and the
references there).  CMIP6 supplies the *same physical quantity* under historical
+ SSP forcing over 1850-2100, so a network trained on it sees the whole range and
the 2100 projection becomes interpolation rather than extrapolation.

This script only fetches the raw NetCDF.  Run `scripts/prepare_cmip6_tas.py`
afterwards to reduce each file to a global-mean monthly series.

Data
----
Global monthly (`Amon`) `tas`, one file per model/experiment (~40-80 MB each).
Default set = CNRM-ESM2-1 (already used for the CO2 pathways in this project,
so the CO2-temperature pairing is internally consistent) plus a few other ESMs
so leave-one-model-out is possible.

Usage
-----
    python scripts/download_cmip6_tas.py                     # default model list
    python scripts/download_cmip6_tas.py --models CNRM-ESM2-1
    python scripts/download_cmip6_tas.py --dry-run           # resolve URLs only
    python scripts/download_cmip6_tas.py --out data/cmip6_tas

Downloads are resumable (a partial file is re-fetched with a Range request) and
skipped when the target already exists with the expected size.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SEARCH_URL = "https://esgf-node.llnl.gov/esg-search/search"

# source_id -> (institution_id, preferred member_id or None to auto-pick)
MODELS: dict[str, tuple[str, str | None]] = {
    "CNRM-ESM2-1": ("CNRM-CERFACS", "r3i1p1f2"),
    "MPI-ESM1-2-LR": ("MPI-M", None),
    "CanESM5": ("CCCma", None),
    "IPSL-CM6A-LR": ("IPSL", None),
    "MIROC6": ("MIROC", None),
    "GFDL-ESM4": ("NOAA-GFDL", None),
}

EXPERIMENTS = ["historical", "ssp126", "ssp245", "ssp370", "ssp585"]

# Preference order for the transfer protocol; globus URLs need a client, so the
# plain HTTPS nodes come first.
URL_PREFERENCE = ("HTTPServer", "OPENDAP")


def _query(params: dict, retries: int = 4) -> dict:
    """One ESGF search-API call, with backoff.  Returns the parsed JSON."""
    q = urllib.parse.urlencode({**params, "format": "application/solr+json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(f"{SEARCH_URL}?{q}", timeout=90) as fh:
                return json.load(fh)
        except Exception as exc:      # noqa: BLE001 — the ESGF nodes throw a
            last = exc                # mixed bag (URLError, IncompleteRead,
            time.sleep(2 ** attempt)  # JSONDecodeError); all are worth retrying
    raise RuntimeError(f"ESGF query failed after {retries} tries: {last}")


def member_candidates(source_id: str, institution_id: str) -> list[str]:
    """Member ids that publish historical tas, most widely used first.

    These are only *candidates*: whether a member also has every SSP varies
    (IPSL-CM6A-LR's r10i1p1f1 has no ssp126 or ssp585), and the ESGF facet lists
    are not reliable enough to decide it up front — so the caller verifies by
    actually resolving the files.  `historical` is used to seed the list because
    it is the experiment every member publishes.
    """
    doc = _query({
        "type": "Dataset", "project": "CMIP6", "source_id": source_id,
        "institution_id": institution_id, "variable_id": "tas",
        "table_id": "Amon", "experiment_id": "historical", "limit": 60,
    })
    docs = doc.get("response", {}).get("docs", [])
    counts: dict[str, int] = {}
    for d in docs:
        for m in d.get("member_id", []):
            counts[m] = counts.get(m, 0) + 1
    if not counts:
        raise RuntimeError(f"no historical tas dataset for {source_id}")
    return sorted(counts, key=lambda m: (-counts[m], m))


def find_files(source_id: str, institution_id: str, member_id: str,
               experiment: str) -> list[dict]:
    """Every FILE of one (model, experiment), each with its mirror list.

    Two things this has to get right:

    * An experiment is not always one file.  MPI-ESM1-2-LR, IPSL-CM6A-LR,
      MIROC6 and others split `historical` and each SSP into calendar chunks
      (e.g. ``..._historical_r10i1p1f1_gn_187001-188912.nc``), so returning a
      single URL would silently train on 20 years of a 165-year record.
    * A search returns one document per replica *node*, and any single node can
      be stale (404 on an otherwise valid dataset), so keep every HTTPS mirror
      per file and let the caller fall through.
    """
    doc = _query({
        "type": "File", "project": "CMIP6", "source_id": source_id,
        "institution_id": institution_id, "variable_id": "tas",
        "table_id": "Amon", "experiment_id": experiment,
        "member_id": member_id, "limit": 500,
    })
    docs = doc.get("response", {}).get("docs", [])
    by_name: dict[str, dict] = {}
    for doc in docs:
        for entry in doc.get("url", []):
            parts = entry.split("|")
            if len(parts) < 3:
                continue
            url, _mime, proto = parts[0], parts[1], parts[2]
            if proto not in URL_PREFERENCE:
                continue
            filename = url.rsplit("/", 1)[-1]
            # The OPeNDAP nodes also advertise `<file>.nc.html` landing pages as
            # HTTPServer URLs; they are ~100 KB of HTML, not data.
            if not filename.endswith(".nc"):
                continue
            rec = by_name.setdefault(
                filename, {"filename": filename, "size": int(doc.get("size", 0)),
                           "urls": []})
            if url not in rec["urls"]:      # dedup by URL, not by filename
                rec["urls"].append(url)
    # Sort by the time range embedded in the name so the pieces come out ordered.
    return sorted(by_name.values(), key=lambda r: r["filename"])


def download(url: str, dest: Path, expected: int, resume: bool = True) -> None:
    """Stream to `dest`, resuming a partial download when the server allows it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    have = dest.stat().st_size if dest.exists() else 0
    if have and not resume:
        have = 0
    headers = {"Range": f"bytes={have}-"} if have else {}
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if have else "wb"
    with urllib.request.urlopen(req, timeout=180) as r, open(dest, mode) as fh:
        total = have + int(r.headers.get("Content-Length", 0))
        done = have
        last_log = 0.0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if time.time() - last_log > 5:
                pct = 100 * done / total if total else 0
                print(f"\r      {done/1e6:8.1f} / {total/1e6:.1f} MB  ({pct:5.1f}%)",
                      end="", flush=True)
                last_log = time.time()
    print(f"\r      {done/1e6:8.1f} MB  done" + " " * 20)
    if expected and dest.stat().st_size != expected:
        raise RuntimeError(f"{dest.name}: size {dest.stat().st_size} != expected {expected}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODELS),
                    help="CMIP6 source_id list (default: the built-in set)")
    ap.add_argument("--out", default=str(ROOT / "data/cmip6_tas"))
    ap.add_argument("--dry-run", action="store_true", help="resolve and print URLs only")
    args = ap.parse_args()

    out_root = Path(args.out)
    manifest: list[dict] = []
    failures: list[str] = []

    for source_id in args.models:
        if source_id not in MODELS:
            print(f"!! {source_id}: not in the built-in table (add institution_id first)")
            continue
        institution_id, member_pref = MODELS[source_id]
        print(f"\n=== {source_id} ({institution_id}) ===")
        try:
            candidates = [member_pref] if member_pref \
                else member_candidates(source_id, institution_id)
        except RuntimeError as exc:
            print(f"  !! {exc}")
            failures.append(f"{source_id}: {exc}")
            continue

        # Pick the first member that actually resolves EVERY experiment.
        resolved: dict[str, list[dict]] = {}
        member = None
        for cand in candidates[:6]:
            got = {e: find_files(source_id, institution_id, cand, e)
                   for e in EXPERIMENTS}
            n = sum(1 for v in got.values() if v)
            if n == len(EXPERIMENTS):
                member, resolved = cand, got
                break
            print(f"    member {cand}: {n}/{len(EXPERIMENTS)} experiments"
                  f"{'  (best so far)' if not resolved else ''}")
            if not resolved or n > sum(1 for v in resolved.values() if v):
                resolved = got
        if member is None:
            member = candidates[0]
            print(f"  !! no member covers all {len(EXPERIMENTS)} experiments; "
                  f"using {member} — missing "
                  f"{[e for e, v in resolved.items() if not v]}")
        else:
            print(f"  member_id: {member}  (all {len(EXPERIMENTS)} experiments)")

        for experiment in EXPERIMENTS:
            files = resolved.get(experiment) or find_files(
                source_id, institution_id, member, experiment)
            if not files:
                print(f"  !! {experiment}: no HTTPS file found — skipped")
                failures.append(f"{source_id}/{experiment}: no file")
                continue
            total = sum(f["size"] for f in files)
            print(f"  {experiment:11s} {len(files):2d} file(s)  {total/1e6:7.1f} MB"
                  f"  {files[0]['filename'] if len(files) == 1 else '(chunked)'}")
            if args.dry_run:
                for f in files:
                    print(f"      {f['filename']}  ({len(f['urls'])} mirror(s))")
                    for u in f["urls"][:1]:
                        print(f"          {u}")
                continue

            for f in files:
                dest = out_root / source_id / f["filename"]
                manifest.append({
                    "source_id": source_id, "institution_id": institution_id,
                    "member_id": member, "experiment_id": experiment,
                    "url": f["urls"][0], "size": f["size"], "path": str(dest),
                    "mirrors": len(f["urls"]),
                })
                if dest.exists() and f["size"] and dest.stat().st_size == f["size"]:
                    continue                       # already complete
                errs = []
                for i, url in enumerate(f["urls"]):
                    try:
                        download(url, dest, f["size"])
                        break
                    except Exception as exc:                   # noqa: BLE001
                        errs.append(f"{url.split('/')[2]}: {exc}")
                        # A stale mirror can leave a short file behind; drop it
                        # so the next attempt starts clean instead of "resuming"
                        # onto garbage.
                        if dest.exists() and dest.stat().st_size < f["size"]:
                            dest.unlink()
                else:
                    print(f"      !! {f['filename']}: all {len(f['urls'])} mirror(s) failed")
                    failures.append(f"{source_id}/{experiment}/{f['filename']}: "
                                    + "; ".join(errs))

    if not args.dry_run:
        out_root.mkdir(parents=True, exist_ok=True)
        (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"\nmanifest -> {out_root / 'manifest.json'}  ({len(manifest)} entries)")

    if failures:
        print(f"\n{len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("\nall requested files are present.")


if __name__ == "__main__":
    main()
