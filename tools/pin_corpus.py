"""Record the pins of the SynthChartNet sample shard in src/deplot_chart_pipeline/samples.py.

Uses the shard already in the cache when it is there (its size must equal the committed pin), otherwise
downloads `CORPUS_FILE["path"]` of `CORPUS_REPO` at the immutable `CORPUS_REVISION` (needs Hub access). It
computes the shard's SHA-256, counts its rows and chart types, rewrites the `sha256` and `rows` entries of
`CORPUS_FILE`, and prints the realised split counts of the default sample. Regenerate the notebook afterwards
(`python tools/build_notebook.py`), because the carried module changed. Run it once; re-running on an
already-pinned file only confirms the pins.

    python tools/pin_corpus.py [--cache weights/synthchartnet]
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from deplot_chart_pipeline import samples  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cache", default=str(ROOT / "weights" / "synthchartnet"))
    args = parser.parse_args()
    cache = Path(args.cache)
    path = cache / samples.CORPUS_FILE["path"]
    if not path.is_file():
        path = samples._hub_download(cache)
    size = path.stat().st_size
    if size != samples.CORPUS_FILE["bytes"]:
        pinned_size = samples.CORPUS_FILE["bytes"]
        raise SystemExit(f"{path}: {size} bytes, committed pin {pinned_size}; refusing to pin")
    digest = samples._sha256_file(path)
    rows = samples.read_corpus(path)
    pinned = samples.CORPUS_FILE.get("sha256")
    if pinned is not None and pinned != digest:
        raise SystemExit(f"{path}: sha256 {digest} differs from the recorded pin {pinned}; investigate")
    module = ROOT / "src" / "deplot_chart_pipeline" / "samples.py"
    text = module.read_text(encoding="utf-8")
    new, count = re.subn(
        r'("sha256": )(None|"[0-9a-f]{64}"),\n(\s*"rows": )(None|\d+),',
        lambda m: f'{m.group(1)}"{digest}",\n{m.group(3)}{len(rows)},',
        text,
        count=1,
    )
    if count != 1:
        raise SystemExit("could not find the CORPUS_FILE pin entries in samples.py")
    module.write_text(new, encoding="utf-8")
    samples.CORPUS_FILE.update({"sha256": digest, "rows": len(rows)})
    types = Counter(str(r["chart_type"]) for r in rows)
    print({"sha256": digest, "bytes": size, "rows": len(rows), "chart_types": dict(sorted(types.items()))})
    splits = samples.build_sample_dataset(rows, path, image_dir=cache / "images")
    for name, part in splits.items():
        by_type = dict(sorted(Counter(r["chart_type"] for r in part).items()))
        print({name: {"charts": len(part), "by_type": by_type, "digest": samples.dataset_digest(part)[:16]}})
    longest = max(len(r["target_text"]) for part in splits.values() for r in part)
    print({"longest_target_chars": longest})
    print("pins written to", module.relative_to(ROOT), "- now run: python tools/build_notebook.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
