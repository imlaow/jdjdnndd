import sys
print("Starting basic import test...")
try:
    print("Importing torch...")
    import torch
    print("torch OK")
except Exception as e:
    print(f"torch ERROR: {e}")
    sys.exit(1)
print("Done")
