#!/usr/bin/env python
"""Minimal test to check if Gradio works"""

import sys
print("Starting minimal Gradio test...", file=sys.stderr, flush=True)

try:
    print("Importing gradio...", file=sys.stderr, flush=True)
    import gradio as gr
    print("✅ Gradio imported", file=sys.stderr, flush=True)
    
    print("Creating Gradio interface...", file=sys.stderr, flush=True)
    with gr.Blocks(title="Test") as demo:
        gr.Textbox(label="Test Input")
        gr.Button("Test Button")
    
    print("✅ Gradio interface created successfully!", file=sys.stderr, flush=True)
    print("Demo object created, would launch now", file=sys.stderr, flush=True)
    
except Exception as e:
    print(f"❌ Error: {e}", file=sys.stderr, flush=True)
    import traceback
    traceback.print_exc(file=sys.stderr)
    sys.exit(1)

print("✅ Test passed!", file=sys.stderr, flush=True)
