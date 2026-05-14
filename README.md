# GraphLM

**[한국어](docs/ko/README.md)** | English (TBD)

> Research codebase combining graph structures with language models.

Python 3.11+, Jupyter notebooks for experiments, package code in `src/graphlm/`.

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
docs/           CI docs + Korean documentation
.claude/        AI collaboration rules and infra
```

## License

TBD.
