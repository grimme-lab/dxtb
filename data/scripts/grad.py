import torch
from json import load as json_load

path = "samples.json"
JSON = "gfn1.json"
JSON_FIT = "gfn1.fit.json"

l = []
l_fit = []
loss_fn = torch.nn.MSELoss(reduction="sum")

with open(path, "rb") as f:
	data = json_load(f)
	for uid, features in data.items():
		suid = uid.split(":")[1]
		gref = torch.tensor(features["gref"])

		with open(f"benchmark/{suid}/{JSON}") as j:
			js = json_load(j)
			g = torch.tensor(js["gradient"]).reshape(-1, 3)

		with open(f"benchmark/{suid}/{JSON_FIT}") as j_fit:
			js_fit = json_load(j_fit)
			g_fit = torch.tensor(js_fit["gradient"]).reshape(-1, 3)


		loss = loss_fn(g, gref)
		loss_fit = loss_fn(g_fit, gref)
		l.append(loss.item())
		l_fit.append(loss_fit.item())
		#print(suid, loss.item(), loss_fit.item())

print(f"MEAN orig {torch.tensor(l).mean().item():.3E}")
print(f"MEAN fit  {torch.tensor(l_fit).mean().item():.3E}")
