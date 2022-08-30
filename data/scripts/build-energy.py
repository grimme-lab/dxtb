import json
import os
import sys



def walklevel(some_dir: str, level=1):
    """Identical to os.walk() but allowing for limited depth."""
    some_dir = some_dir.rstrip(os.path.sep)
    assert os.path.isdir(some_dir), f"Not a directory: {some_dir}"
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]

def main():
  FUNC=sys.argv[1]
  
  FILE = f"{FUNC}.json"
  ENERGY_FILE = "energy"
  root = "./"
  for (dirpath, dirnames, filenames) in walklevel(root, level=2):
    if FILE not in filenames:
      continue
    
    if not os.path.exists("/".join([dirpath, FUNC])):
        os.makedirs("/".join([dirpath, FUNC]))
        
    with open("/".join([dirpath, FILE]), "r") as f:
      data = json.loads(f.read())
      
      #print(dirpath)
      with open("/".join([dirpath, FUNC, ENERGY_FILE]), "w") as f2:
        energy = data["energy"] 
        f2.write(f"$energy\n     1   {energy}   {energy}   {energy}\n$end")
      

if __name__ == '__main__':
  main()
