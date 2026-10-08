import subprocess, sys
steps = ["data.py", "make_splits.py", "features_mfcc.py", "features_mfmc.py",
         "sanity.py", "plots.py", "verify.py"]
for s in steps:
    print("\n===== running", s, "=====")
    r = subprocess.run([sys.executable, "src/" + s])
    if r.returncode != 0:
        print("FAILED at", s)
        sys.exit(1)
print("\nPIPELINE COMPLETE")
