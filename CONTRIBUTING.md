# Contributing to agentdx

Thank you for your interest in contributing to agentdx.

## How to Contribute

### Reporting Issues

- Use GitHub Issues to report bugs or request features
- Include your Python version, agentdx version, and a minimal reproduction trace
- For pathology detection issues, include the trace (or an anonymised version) and the expected vs. actual diagnostic output

### Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Write tests for any new detectors or parsers
4. Ensure all existing tests pass: `pytest tests/`
5. Submit a pull request with a clear description of the change

### Adding a New Parser

Parsers convert framework-specific trace formats into agentdx's normalised trace schema. To add a parser:

1. Create `agentdx/parsers/your_framework.py`
2. Implement the `BaseParser` interface
3. Add test traces in `tests/fixtures/your_framework/`
4. Add tests in `tests/test_your_framework.py`
5. Export from `agentdx/parsers/__init__.py` and `agentdx/__init__.py`

### Adding a New Detector

Detectors identify specific failure pathologies in normalised traces. To add a detector:

1. Create `agentdx/detectors/your_pathology.py`
2. Implement the `BaseDetector` interface
3. Map the pathology to the taxonomy in `agentdx/taxonomy.py`
4. Add test cases covering true positives, true negatives, and edge cases in `tests/test_your_pathology.py`
5. Export from `agentdx/detectors/__init__.py` and `agentdx/__init__.py`
6. Document the detection logic and known limitations

### Taxonomy Contributions

If you believe the failure pathology taxonomy should be extended or refined, please open an Issue first to discuss the proposed change before submitting a PR. Taxonomy changes affect the entire detection pipeline and require careful consideration.

## Code Standards

- Python 3.10+
- Type hints on all public functions
- Docstrings on all public classes and methods
- `ruff` for linting and formatting (`ruff check` + `ruff format`)
- Test coverage target: 80%+

## Code of Conduct

Be respectful, constructive, and collaborative. We are building safety infrastructure for AI systems — the work matters, and so does how we treat each other.

## Questions?

Open an Issue or reach out via [mldeep.io](https://mldeep.io).
