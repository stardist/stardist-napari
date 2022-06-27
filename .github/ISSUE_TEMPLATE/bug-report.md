---
name: "\U0001F41B Bug Report"
about: Submit a bug report
title: ''
labels: bug
assignees: ''

---

**Describe the bug**
A clear and concise description of what the bug is.

**To reproduce**
Steps to reproduce the behavior, ideally by providing a runnable code snippet, Python script, etc.

**Expected behavior**
A clear and concise description of what you expected to happen.

**Data and screenshots**
If applicable, add data (e.g. images) or screenshots to help explain your problem.

**Environment:**

1. Please copy and paste the information at napari info option in the help menubar.

2. Please run this code and paste the output:
```python
import importlib, platform

print(f'os: {platform.platform()}')
for m in ('stardist_napari','stardist','csbdeep','napari','magicgui','tensorflow'):
    try:
        print(f'{m}: {importlib.import_module(m).__version__}')
    except ModuleNotFoundError:
        print(f'{m}: not installed')
```
