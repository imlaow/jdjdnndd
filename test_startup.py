#!/usr/bin/env python
"""Test app initialization step by step"""

import sys
import traceback

print("=" * 70)
print("STEP 1: Testing imports...")
print("=" * 70)
try:
    print("  - Importing spaces...")
    try:
        import spaces
        print("    ✅ spaces imported successfully")
    except ImportError:
        print("    ⚠️  spaces not available (expected in non-ZeroGPU environment)")
    
    print("  - Importing torch...")
    import torch
    print(f"    ✅ torch imported (version: {torch.__version__})")
    
    print("  - Importing gradio...")
    import gradio as gr
    print(f"    ✅ gradio imported (version: {gr.__version__})")
    
    print("  - Importing diffusers...")
    from diffusers import StableDiffusionXLPipeline
    print("    ✅ diffusers imported")
    
    print("  - Importing PIL...")
    from PIL import Image
    print("    ✅ PIL imported")
    
    print("  - Importing compel...")
    try:
        from compel import Compel
        print("    ✅ compel imported")
    except ImportError:
        print("    ⚠️  compel not available")
    
    print("\n✅ All critical imports successful!")
    
except Exception as e:
    print(f"\n❌ Import Error: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 2: Checking app.py syntax...")
print("=" * 70)
try:
    import ast
    with open('app.py', 'r', encoding='utf-8') as f:
        code = f.read()
    ast.parse(code)
    print("✅ app.py syntax is valid")
except SyntaxError as e:
    print(f"❌ Syntax Error in app.py: {e}")
    sys.exit(1)

print("\n" + "=" * 70)
print("STEP 3: Attempting to import app module...")
print("=" * 70)
try:
    import app
    print("✅ app module imported successfully")
    
    # Check if key functions exist
    print("\n  Checking functions:")
    print(f"    - initialize_model: {'✅' if hasattr(app, 'initialize_model') else '❌'}")
    print(f"    - generate_image: {'✅' if hasattr(app, 'generate_image') else '❌'}")
    print(f"    - create_interface: {'✅' if hasattr(app, 'create_interface') else '❌'}")
    print(f"    - build_negative_prompt: {'✅' if hasattr(app, 'build_negative_prompt') else '❌'}")
    
except Exception as e:
    print(f"❌ Error importing app module:")
    print(f"   {type(e).__name__}: {e}")
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\nYou can now run: python app.py")
print("The app will start a Gradio server on http://localhost:7860")
