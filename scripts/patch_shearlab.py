import re, pathlib, pyshearlab

pkg = pathlib.Path(pyshearlab.__file__).parent
pattern = re.compile(
    r"^(\s*)(h0, ?h1) = dfilters\(([^)]*)\)/np\.sqrt\(2\)", re.M)
repl = (r"\1\2 = dfilters(\3)"
        "\n" r"\1h0 = h0/np.sqrt(2)"
        "\n" r"\1h1 = h1/np.sqrt(2)")

for f in pkg.glob("*.py"):
    s = f.read_text()
    s2 = pattern.sub(repl, s)
    if s2 != s:
        f.write_text(s2)
        print("patched", f.name, "-", len(pattern.findall(s)), "spot(s)")
print("done")