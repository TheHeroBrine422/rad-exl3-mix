import sys
from PIL import Image
a, b, out = sys.argv[1:4]
A = Image.open(a).convert("RGB"); B = Image.open(b).convert("RGB")
w = max(A.width, B.width)
C = Image.new("RGB", (w, A.height + B.height), "white"); C.paste(A, (0, 0)); C.paste(B, (0, A.height)); C.save(out)
