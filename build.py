# ----------------------------------------------------------------------
# Author:   yury.matveev@desy.de
# ----------------------------------------------------------------------

import os
import sys

# ----------------------------------------------------------------------
in_dirs = ["src/ui"]
out_dirs = ["src/gui"]

ui_compilers = {"linux2": "python -m PyQt5.uic.pyuic",
                "linux": "python -m PyQt5.uic.pyuic",
                "win32": "C://Users//matveyev//AppData//Local//Programs//Python//Python37-32//Scripts//pyuic5.exe"}

rc_compilers = {"linux2": "pyrcc5",
                "linux": "pyrcc5",
                "win32":  "C://Users//matveyev//AppData//Local//Programs//Python//Python37-32//Scripts//pyrcc5.exe"}

# ----------------------------------------------------------------------
def compile_uis(ui_compiler, rc_compiler, in_dirs, out_dirs):
    """
    """
    for in_dir, out_dir in zip(in_dirs, out_dirs):
        for f in [f for f in os.listdir(in_dir) if os.path.isfile(os.path.join(in_dir, f))
                  and os.path.splitext(f)[-1] in [".ui", ".qrc"]]:

            base, ext = os.path.splitext(f)
            if ext == ".ui":
                cmd = "{} {}/{} -o {}/{}{}.py".format(ui_compiler, in_dir, f, out_dir, base, "_ui")
            else:
                cmd = "{} {}/{} -o {}{}.py".format(rc_compiler, in_dir, f, base, "_rc")

            print(cmd)
            os.system(cmd)


# ----------------------------------------------------------------------
if __name__ == "__main__":

    print("Removing pyc files...")

    for out_dir in out_dirs:
        for root, dirs, files in os.walk(out_dir):
            for f in [f for f in files if f.endswith(".pyc")]:
                if sys.platform == "linux" or sys.platform == "linux2":
                    os.system("rm {}".format(os.path.join(root, f)))
                elif sys.platform == "win32":
                    os.remove(os.path.join(root, f))

    print("Removing uis and rcs...")
    for out_dir in out_dirs:
        for root, dirs, files in os.walk(out_dir):
            for f in [f for f in files if (f.endswith(".pyc") or f.endswith(".py"))
                                          and f != "__init__.py"]:
                if sys.platform == "linux" or sys.platform == "linux2":
                    os.system("rm {}".format(os.path.join(root, f)))
                elif sys.platform == "win32":
                    os.remove(os.path.join(root, f))

    print("All removed!")

    compile_uis(ui_compilers[sys.platform],
                rc_compilers[sys.platform], in_dirs, out_dirs)

    print("All OK!")