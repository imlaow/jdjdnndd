#!/usr/bin/env python
"""Test if app.py has any syntax or basic import errors"""

import sys
import ast

# Check 1: Validate Python syntax
print("=" * 60)
print("CHECK 1: Python Syntax Validation")
print("=" * 60)
try:
    with open('app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("✅ PASSED: Python syntax is valid")
except SyntaxError as e:
    print(f"❌ FAILED: Syntax Error at line {e.lineno}: {e.msg}")
    sys.exit(1)

# Check 2: Verify key functions exist
print("\n" + "=" * 60)
print("CHECK 2: Function Definitions")
print("=" * 60)
required_functions = [
    'initialize_model',
    'enhance_prompt',
    'build_negative_prompt',
    'generate_image',
    'create_interface',
]
tree = ast.parse(code)
function_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

for func in required_functions:
    if func in function_names:
        print(f"✅ FOUND: {func}")
    else:
        print(f"❌ MISSING: {func}")
        sys.exit(1)

# Check 3: Verify imports
print("\n" + "=" * 60)
print("CHECK 3: Critical Imports")
print("=" * 60)
required_imports = [
    'os',
    'torch',
    'gradio',
    'diffusers',
    'PIL',
    'compel'
]

for module in required_imports:
    try:
        __import__(module)
        print(f"✅ FOUND: {module}")
    except ImportError:
        print(f"⚠️  NOT INSTALLED: {module} (this may be required)")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print("✅ app.py is syntactically correct")
print("✅ All required functions are defined")
print("✅ No errors detected in code structure")
print("\n📝 NOTE: When running app.py, it will:")
print("   1. Try to download the model from HuggingFace Hub")
print("   2. Initialize the Stable Diffusion XL pipeline")
print("   3. Start a Gradio web server on http://0.0.0.0:7860")
print("\n⏱️  First run may take several minutes due to model loading")
