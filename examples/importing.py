import time

from dxtb.param import GFN1_XTB

print(GFN1_XTB.meta)
# print(GFN1_XTB.meta)

print("before import")
start = time.perf_counter()
import scipy

end = time.perf_counter()
print("after import")
print(end - start)
