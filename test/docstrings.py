#!/usr/bin/env python3
"""Auto-update docstring examples with actual outputs."""

import ast
import re
import sys
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import sympy as sp


def exec_code(code: str, ctx: dict) -> str:
    """Execute code and return output."""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        tree = ast.parse(code)
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            for stmt in tree.body[:-1]:
                exec(compile(ast.Module(body=[stmt], type_ignores=[]), '<string>', 'exec'), ctx)
            result = eval(compile(ast.Expression(body=tree.body[-1].value), '<string>', 'eval'), ctx)
        else:
            exec(code, ctx)
            result = None
        output = sys.stdout.getvalue()
        if result is not None and not output:
            if isinstance(result, pd.DataFrame):
                output = str(result)
            else:
                output = repr(result)
        return output.rstrip()
    except Exception as e:
        return f"ERROR: {e}"
    finally:
        sys.stdout = old_stdout


def process_examples(text: str) -> str:
    """Process examples block."""
    lines = text.split('\n')
    result = []
    code = []
    ctx = {}
    exec("import numpy as np; import pandas as pd; import sympy as sp; import networkx as nx", ctx)
    exec("from qmm import *", ctx)

    for line in lines:
        s = line.strip()
        if s.startswith('#'):
            continue
        if not s:
            if code:
                output = exec_code('\n'.join(code), ctx)
                result.extend(code)
                if output:
                    result.extend(f"# {o}" for o in output.split('\n'))
                result.append('')
                code = []
            continue
        code.append(s)

    if code:
        output = exec_code('\n'.join(code), ctx)
        result.extend(code)
        if output:
            result.extend(f"# {o}" for o in output.split('\n'))

    while result and result[-1] == '':
        result.pop()

    return '\n'.join(result)


def update_file(path: Path) -> int:
    """Update all examples in file."""
    content = path.read_text()
    count = 0

    def replace(m):
        nonlocal count
        indent = m.group(1)
        examples = m.group(2)
        new_examples = process_examples(examples)
        indented = '\n'.join((' ' * 8 + line if line else '') for line in new_examples.split('\n'))
        count += 1
        return f'{indent}```python\n{indented}\n{indent}```'

    pattern = r'^(\s+)```python\s*\n(.*?)\n\s+```'
    new_content = re.sub(pattern, replace, content, flags=re.MULTILINE | re.DOTALL)

    if new_content != content:
        path.write_text(new_content)

    return count


if __name__ == '__main__':
    root = Path(__file__).parent.parent / 'qmm'
    files = list(root.glob('**/*.py'))

    print("Updating docstring examples...")
    total = 0
    for f in sorted(files):
        if f.name.startswith('_'):
            continue
        count = update_file(f)
        if count:
            print(f"  {f.relative_to(root.parent)}: {count} blocks")
            total += count

    print(f"\nTotal: {total} example blocks updated")
