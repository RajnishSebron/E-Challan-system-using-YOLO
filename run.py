"""
run.py - Root entry point to launch the AI ANPR & E-Challan System
"""
import os
import sys

# Ensure backend directory is in python path
current_dir = os.path.dirname(os.path.abspath(__file__))
backend_dir = os.path.join(current_dir, "anpr_echallan_system", "backend")

if not os.path.exists(backend_dir):
    backend_dir = os.path.join(current_dir, "backend")

sys.path.insert(0, backend_dir)

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print(" Starting AI ANPR & E-Challan Traffic Enforcement Command Center ")
    print(" Dashboard: http://localhost:8000 ")
    print("=" * 60)
    os.chdir(backend_dir)
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

