import sys, os, openpi

print(">>> sys.executable =", sys.executable)
print(">>> openpi.__file__ =", openpi.__file__)
print(">>> openpi.__path__ =", list(openpi.__path__))

print("\\n>>> sys.path:")
for p in sys.path:
    print("   ", p)

print("\\n>>> PYTHONPATH =", os.environ.get("PYTHONPATH"))