# GraphLM

**[한국어](docs/ko/README.md)** | English (TBD)

> Research codebase treating the **internal structure of Transformers as a graph** — each FFN expert, attention head, or layer block is a node; tokens are routed and aggregated across them. (computation-as-graph, **not** data-as-graph.)

Python 3.11+, Jupyter notebooks for experiments, package code in `src/graphlm/`.

See [CLAUDE.md](CLAUDE.md#핵심-패러다임--computation-as-graph) for the full paradigm definition.

## Quick start

```bash
pip install -e ".[dev]"
make test
make lint
```

See [`docs/ko/README.md`](docs/ko/README.md) for the full guide, and
[`.claude/rules/`](.claude/rules/) for development conventions.

## Project layout

```
src/graphlm/    Python package (models, graph utils, training, eval)
notebooks/      Jupyter experiment notebooks
tests/          pytest tests (mirrors src/graphlm/)
data/           experiment data (gitignored)
docs/           CI docs, Korean documentation, paper summaries (docs/papers/)
.claude/        AI collaboration rules and infra
```

## License

TBD.
