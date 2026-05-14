#!/usr/bin/env python
import sys
import traceback

try:
    print("Starting import test...")
    import app
    print("✅ Import successful - no errors detected")
    sys.exit(0)
except Exception as e:
    print(f"❌ Error during import: {e}")
    traceback.print_exc()
    sys.exit(1)
