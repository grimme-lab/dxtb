import sys
import torch
from json import load as json_load

path = "samples.json"
JSON = "gfn1.json"
JSON_GFN2 = "gfn2.json"
JSON_FIT = "gfn1.fit.json"

l = []
l_fit = []
l_gfn2 = []
loss_fn = torch.nn.L1Loss(reduction="sum")

with open(path, "rb") as f:
	data = json_load(f)
	for uid, features in data.items():
		suid = uid.split(":")[1]
		gref = torch.tensor(features["gref"])
		#print(suid, gref)

		with open(f"benchmark/{suid}/{JSON}") as j:
			js = json_load(j)
			g = torch.tensor(js["gradient"]).reshape(-1, 3)

		with open(f"benchmark/{suid}/{JSON_FIT}") as j_fit:
			js_fit = json_load(j_fit)
			g_fit = torch.tensor(js_fit["gradient"]).reshape(-1, 3)

		with open(f"benchmark/{suid}/{JSON_GFN2}") as j:
			js = json_load(j)
			g_gfn2 = torch.tensor(js["gradient"]).reshape(-1, 3)


		loss = loss_fn(g, gref)
		loss_fit = loss_fn(g_fit, gref)
		loss_gfn2 = loss_fn(g_gfn2, gref)
		l.append(loss.item())
		l_fit.append(loss_fit.item())
		l_gfn2.append(loss_gfn2.item())
		#print(suid, loss.item(), loss_fit.item())

mean = torch.tensor(l).mean().item()
mean_fit = torch.tensor(l_fit).mean().item()
mean_gfn2 = torch.tensor(l_gfn2).mean().item()

with open("../grad.txt", "a") as f:
	f.write(f"{sys.argv[1]},{mean},{mean_fit},{mean_gfn2}\n")

print(f"MEAN orig {torch.tensor(l).mean().item():.3E}")
print(f"MEAN fit  {torch.tensor(l_fit).mean().item():.3E}")
print(f"MEAN gfn2 {torch.tensor(l_gfn2).mean().item():.3E}")
