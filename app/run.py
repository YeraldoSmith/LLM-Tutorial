#!/usr/bin/env python3
"""Start the tutorial app.

    pip install -r requirements.txt
    python run.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

if __name__ == "__main__":
    from tutor.app import main
    sys.exit(main())
