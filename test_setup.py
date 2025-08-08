#!/usr/bin/env python3
"""Quick setup test for the SEM keyword research platform."""

import sys
import subprocess
import importlib

def check_python_version():
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 10):
        print(f"❌ Python {version.major}.{version.minor} - Need Python 3.10+")
        return False
    print(f"✅ Python {version.major}.{version.minor} - Compatible")
    return True

def check_required_packages():
    packages = ['spacy', 'sentence_transformers', 'keybert', 'sklearn', 'requests', 'bs4', 'google.generativeai']
    missing = []
    for pkg in packages:
        try:
            if pkg == 'sklearn': importlib.import_module('sklearn')
            elif pkg == 'bs4': importlib.import_module('bs4') 
            elif pkg == 'google.generativeai': import google.generativeai
            else: importlib.import_module(pkg)
            print(f"✅ {pkg}")
        except ImportError:
            print(f"❌ {pkg}")
            missing.append(pkg)
    return len(missing) == 0

def test_basic_run():
    try:
        result = subprocess.run([sys.executable, "run.py", "--config", "config.yaml", "--dry-run"], 
                              capture_output=True, text=True, timeout=30)
        return result.returncode == 0 and ("completed successfully" in result.stdout or "SUCCESS" in result.stdout)
    except:
        return False

def main():
    print("🚀 SEM Platform Setup Test\n")
    tests = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages), 
        ("Basic Functionality", test_basic_run)
    ]
    
    passed = sum(1 for name, test in tests if (print(f"Testing {name}...") or test()))
    
    if passed == len(tests):
        print("\n🎉 Setup successful! Run: python run.py --config config.yaml")
    else:
        print(f"\n⚠️ {len(tests)-passed} issues found. Check README.md for troubleshooting.")
    
    return passed == len(tests)

if __name__ == "__main__":
    sys.exit(0 if main() else 1)