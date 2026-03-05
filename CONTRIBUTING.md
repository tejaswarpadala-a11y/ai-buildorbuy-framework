# Contributing to Bank Rage Classifier

Thank you for your interest in contributing! This project demonstrates PM thinking + technical execution, and we welcome improvements.

---

## How to Contribute

### 1. Report Issues
- Use GitHub Issues to report bugs
- Include: Python version, error message, steps to reproduce
- Check existing issues first to avoid duplicates

### 2. Suggest Features
- Open an issue with "Feature Request" label
- Explain: Problem → Proposed Solution → Expected Impact
- Bonus: Include cost-benefit analysis (PM thinking!)

### 3. Submit Code
- Fork the repo
- Create a feature branch: `git checkout -b feature/your-feature`
- Make changes with clear commit messages
- Submit a Pull Request

---

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/bank-rage-classifier.git
cd bank-rage-classifier

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies + dev tools
pip install -r requirements.txt
pip install black isort pytest  # Code formatting + testing
```

---

## Code Style

### Python
- **Formatter**: Black (line length 100)
- **Imports**: isort
- **Docstrings**: Google style

```python
def classify_complaint(text: str, model: str = "specialist") -> dict:
    """
    Classify a consumer complaint into one of 7 categories.
    
    Args:
        text: Complaint text (max 512 characters)
        model: "specialist" or "genai" (default: specialist)
        
    Returns:
        Dictionary with keys: label, confidence, category_id
        
    Example:
        >>> result = classify_complaint("They charged a hidden fee!")
        >>> result['label']
        'Hidden Fees'
    """
    pass
```

### Notebooks
- Clear markdown headers (## Executive Summary, ## Methodology, etc.)
- Explain WHY before code cells (not just WHAT)
- Include visualizations for key results
- Add "Key Takeaways" at end of each section

---

## Areas for Contribution

### High Priority
1. **Spanish Language Support** (40% of CFPB complaints)
   - Translate label codebook
   - Retrain on multilingual RoBERTa
   - Benchmark against Spanish GenAI models

2. **Active Learning Pipeline**
   - Implement uncertainty sampling
   - Reduce labeling cost 60% ($180 → $72 per 15k)
   - See Future Work in README for spec

3. **Vector DB Integration**
   - ChromaDB for error clustering
   - Dynamic few-shot example retrieval
   - Expected +2-5% F1 improvement

### Medium Priority
4. **Additional GenAI Benchmarks**
   - Llama 3.1 70B
   - Mistral Large
   - DeepSeek v3

5. **Production Deployment**
   - FastAPI REST API
   - Docker container
   - Load testing results

6. **Explainability**
   - LIME/SHAP integration
   - Attention visualization
   - Generate human-readable explanations

### Documentation
7. **Blog Post** on approach (Medium, Substack)
8. **Video Walkthrough** (YouTube)
9. **Case Study** for specific industry vertical

---

## Pull Request Guidelines

### Before Submitting
- [ ] Code follows style guide (Black + isort)
- [ ] All notebooks run end-to-end without errors
- [ ] Added tests for new functionality (if applicable)
- [ ] Updated README if adding features
- [ ] No API keys or sensitive data in commits

### PR Description Template
```markdown
## Problem
[What issue does this solve?]

## Solution
[How does this PR address it?]

## Testing
[How was this tested?]

## Impact
[Expected performance/cost/usability improvement]
```

### Review Process
1. Automated checks (code style, notebook execution)
2. Maintainer review (1-3 days)
3. Feedback & iteration
4. Merge!

---

## Testing

### Run Tests
```bash
pytest tests/
```

### Test Coverage
```bash
pytest --cov=src tests/
```

### Notebook Smoke Test
```bash
# Test all notebooks execute without errors
jupyter nbconvert --to notebook --execute notebooks/*.ipynb
```

---

## Release Process

1. Update version in `setup.py` (if applicable)
2. Update `CHANGELOG.md`
3. Create release tag: `git tag v1.1.0`
4. Push: `git push origin v1.1.0`
5. GitHub Actions builds and deploys (TBD)

---

## Questions?

- **Email**: tejaswar.padala@gmail.com
- **LinkedIn**: https://www.linkedin.com/in/teja-padala/
- **GitHub Discussions**: Coming soon

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

Thank you for helping make this project better! 🚀
