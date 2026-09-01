# Contributing to RetainIQ

Thank you for your interest in contributing to RetainIQ Predictive Customer Retention Intelligence!

## How to Contribute

### Reporting Issues

- Use the GitHub Issues tab to report bugs
- Include steps to reproduce the issue
- Mention your Python version and OS

### Suggesting Features

- Open a GitHub Issue with the `enhancement` label
- Describe the feature and its business value

### Submitting Code

1. **Fork** the repository
2. **Create a branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes** and add tests if applicable
4. **Run the test suite**:
   ```bash
   python -m pytest tests/
   ```
5. **Commit** with a clear message:
   ```bash
   git commit -m "Add: description of your change"
   ```
6. **Push** to your fork and open a **Pull Request**

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/RetainIQ_Predictive_Customer_Retention_Intelligence-.git
cd RetainIQ_Predictive_Customer_Retention_Intelligence-

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py

# Run tests
python -m pytest tests/
```

## Code Style

- Follow PEP 8 guidelines
- Add docstrings to new functions and classes
- Keep functions focused and small

## Pull Request Guidelines

- Reference any related issues in the PR description
- Keep PRs focused on a single change
- Ensure all tests pass before submitting
- Update documentation if needed

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE).
