"""Chart-to-table scoring: relaxed position-wise cell accuracy, RNSS, exact-table match and a per-chart-type
breakdown, plus three non-neural baselines that never look at the chart image.

**Cell accuracy** (`pipeline.cell_accuracy`) matches every expected cell only against the predicted cell at
the same (row, column) position; text cells compare case- and whitespace-insensitively, numeric cells
within 5 % relative tolerance (the ChartQA "relaxed accuracy" convention). A missing row or column is a miss;
extra predicted rows or columns are not penalised (they show in the shapes). The ``TITLE`` row is not a cell.

**RNSS** — the relative number set similarity of Masry et al. (ChartQA, 2022, arXiv:2203.10244) as used for
plot-to-table evaluation by Liu et al. (DePlot, 2022, arXiv:2212.10505, §5.1): the numbers of the predicted
and the target table are matched one-to-one at minimal total cost, with the cost of a pair
``D(p, t) = min(1, |p - t| / |t|)``, and ``RNSS = 1 - cost / max(N, M)``. It ignores text cells and table
layout. Our reading of the unmatched case, stated because the formula leaves it implicit: a number left over
on the longer side costs the maximal distance 1, so emitting fewer or more numbers than the target lowers the
score; two tables without numbers score 1. The paper's RMS (relative mapping similarity, which also matches
row and column headers) is not implemented.

**Exact-table match** is the fraction of charts whose parsed rows equal the target's after lower-casing and
whitespace collapsing.

The baselines see only the training split and the test chart's type — never its image or its target. The
**empty baseline** emits no table (the floor: cell accuracy 0). The **header-only baseline** emits, per chart
type, the most frequent header row of the training targets (a header being a first row with an empty corner
cell) and no data rows. The **medoid baseline** emits, per chart type, the training target with the highest
mean cell accuracy against the other training targets of that type — the single most "typical" table. A
system that does not beat them has not shown that it reads the chart.
"""

from __future__ import annotations

import statistics
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

from .pipeline import CELL_SEPARATOR, ROW_SEPARATOR, _as_number, cell_accuracy, parse_table

METRIC_DEFINITIONS: dict[str, str] = {
    "cell_accuracy": (
        "mean over charts of the relaxed position-wise cell accuracy (text case/whitespace-insensitive, "
        "numbers within 5 % relative tolerance; missing cells are misses, extra cells are not penalised)"
    ),
    "rnss": (
        "mean over charts of the relative number set similarity (ChartQA / DePlot): minimal-cost one-to-one "
        "matching of the tables' numbers with cost min(1, |p - t| / |t|), unmatched numbers at cost 1"
    ),
    "exact_table_match": "fraction of charts whose normalised parsed rows equal the target's",
    "by_chart_type": "the same means per chart type",
}


def _normalise_rows(rows: Sequence[Sequence[str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(tuple(" ".join(cell.lower().split()) for cell in row) for row in rows)


def _numbers(rows: Sequence[Sequence[str]]) -> list[float]:
    return [value for row in rows for cell in row if (value := _as_number(cell)) is not None]


def _min_cost_assignment(cost: Sequence[Sequence[float]]) -> float:
    """Total cost of the minimal-cost perfect matching of a square cost matrix (Hungarian algorithm,
    O(n^3), potentials form)."""
    n = len(cost)
    if n == 0:
        return 0.0
    inf = float("inf")
    u, v = [0.0] * (n + 1), [0.0] * (n + 1)
    match, way = [0] * (n + 1), [0] * (n + 1)
    for i in range(1, n + 1):
        match[0], j0 = i, 0
        minv, used = [inf] * (n + 1), [False] * (n + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = match[j0], inf, 0
            for j in range(1, n + 1):
                if not used[j]:
                    current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minv[j]:
                        minv[j], way[j] = current, j0
                    if minv[j] < delta:
                        delta, j1 = minv[j], j
            for j in range(n + 1):
                if used[j]:
                    u[match[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if match[j0] == 0:
                break
        while True:
            j1 = way[j0]
            match[j0] = match[j1]
            j0 = j1
            if j0 == 0:
                break
    return sum(cost[match[j] - 1][j - 1] for j in range(1, n + 1))


def rnss(predicted_rows: Sequence[Sequence[str]], expected_rows: Sequence[Sequence[str]]) -> float:
    """Relative number set similarity of two tables (see the module docstring for the definition)."""
    predicted, expected = _numbers(predicted_rows), _numbers(expected_rows)
    size = max(len(predicted), len(expected))
    if size == 0:
        return 1.0

    def distance(p: float, t: float) -> float:
        if t == 0:
            return 0.0 if p == 0 else 1.0
        return min(1.0, abs(p - t) / abs(t))

    cost = [
        [
            distance(predicted[i], expected[j]) if i < len(predicted) and j < len(expected) else 1.0
            for j in range(size)
        ]
        for i in range(size)
    ]
    return 1.0 - _min_cost_assignment(cost) / size


def score_table(prediction: str, expected: str | Mapping[str, Any]) -> dict[str, Any]:
    """Cell accuracy, RNSS and exact match of one predicted linearised table against its target."""
    target = expected if isinstance(expected, Mapping) else parse_table(str(expected))
    predicted = parse_table(str(prediction))
    accuracy = cell_accuracy(predicted["rows"], target["rows"])
    return {
        "cell_accuracy": accuracy["value"],
        "rnss": rnss(predicted["rows"], target["rows"]),
        "exact_table_match": _normalise_rows(predicted["rows"]) == _normalise_rows(target["rows"]),
        "predicted_shape": accuracy["predicted_shape"],
        "expected_shape": accuracy["expected_shape"],
    }


def chart_metrics(predictions: Sequence[str], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aligned predictions scored against the records' targets: means overall and per chart type, plus one
    row per chart."""
    if len(predictions) != len(records):
        raise ValueError(f"{len(predictions)} predictions for {len(records)} records")
    if not records:
        raise ValueError("records must not be empty")
    rows = []
    for prediction, record in zip(predictions, records, strict=True):
        scored = score_table(prediction, record.get("table") or str(record["target_text"]))
        rows.append(
            {
                "id": record["id"],
                "chart_type": str(record.get("chart_type", "other")),
                "prediction": str(prediction),
                "target_text": str(record["target_text"]),
                **scored,
            }
        )
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[row["chart_type"]].append(row)

    def means(part: Sequence[Mapping[str, Any]]) -> dict[str, float]:
        return {
            "cell_accuracy": round(statistics.fmean(r["cell_accuracy"] for r in part), 4),
            "rnss": round(statistics.fmean(r["rnss"] for r in part), 4),
            "exact_table_match": round(statistics.fmean(float(r["exact_table_match"]) for r in part), 4),
        }

    return {
        "n": len(rows),
        **means(rows),
        "by_chart_type": {name: {"n": len(part), **means(part)} for name, part in sorted(groups.items())},
        "rows": rows,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def _format(rows: Sequence[Sequence[str]]) -> str:
    """DePlot's linearisation with an empty title (the same as `samples.format_table`)."""
    lines = [f"TITLE {CELL_SEPARATOR} ", *(f" {CELL_SEPARATOR} ".join(row) for row in rows)]
    return f" {ROW_SEPARATOR} ".join(lines)


def _by_type(records: Sequence[Mapping[str, Any]]) -> dict[str, list[Mapping[str, Any]]]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record.get("chart_type", "other"))].append(record)
    return groups


def _score(name: str, note: str, predictions: list[str], test: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {**chart_metrics(predictions, test), "baseline": name, "note": note}


def empty_baseline(test: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Emit no table for every test chart."""
    if not test:
        raise ValueError("test must be non-empty")
    return _score("empty", "no table at all (the floor)", [""] * len(test), test)


def header_only_baseline(
    train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Emit, per chart type, the most frequent training header row and no data rows."""
    if not train or not test:
        raise ValueError("train and test must be non-empty")
    headers: dict[str, str] = {}
    for chart_type, part in _by_type(train).items():
        counts = Counter(
            tuple(rows[0])
            for rows in (parse_table(str(r["target_text"]))["rows"] for r in part)
            if rows and rows[0] and rows[0][0] == ""
        )
        headers[chart_type] = (
            _format([list(min(counts, key=lambda k: (-counts[k], k)))]) if counts else _format([])
        )
    predictions = [headers.get(str(r.get("chart_type", "other")), _format([])) for r in test]
    note = "the most frequent training header row of the chart's type, no data rows"
    return {**_score("header-only", note, predictions, test), "headers": headers}


def medoid_baseline(train: Sequence[Mapping[str, Any]], test: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Emit, per chart type, the training target with the highest mean cell accuracy against the other
    training targets of that type (the overall medoid for a type absent from training)."""
    if not train or not test:
        raise ValueError("train and test must be non-empty")

    def medoid(part: Sequence[Mapping[str, Any]]) -> str:
        tables = [parse_table(str(r["target_text"]))["rows"] for r in part]
        if len(tables) == 1:
            return str(part[0]["target_text"])
        best, best_score = 0, -1.0
        for i, candidate in enumerate(tables):
            score = statistics.fmean(
                cell_accuracy(candidate, other)["value"] for j, other in enumerate(tables) if j != i and other
            )
            if score > best_score:
                best, best_score = i, score
        return str(part[best]["target_text"])

    tables = {chart_type: medoid(part) for chart_type, part in sorted(_by_type(train).items())}
    fallback = medoid(train)
    predictions = [tables.get(str(r.get("chart_type", "other")), fallback) for r in test]
    note = "the training medoid table of the chart's type (highest mean cell accuracy against its type)"
    return {**_score("medoid", note, predictions, test), "tables": tables}
