with open('infer_all_steps_nodes.py', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace('ad = ckpt_mgn["args"]', 'ad = ckpt_mgn.get("args", {})')
c = c.replace('e_mean, e_std = ckpt_mgn["e_mean"].to(device), ckpt_mgn["e_std"].to(device)', '''e_mean = ckpt_mgn.get("e_mean", torch.tensor([[0.0, 0.0, 0.0]])).to(device)
    e_std = ckpt_mgn.get("e_std", torch.tensor([[1.0, 1.0, 1.0]])).to(device)''')
c = c.replace('processor_size=ad.get("processor_size", 15)', 'processor_size=ad.get("processor_size", 10)')

with open('infer_all_steps_nodes.py', 'w', encoding='utf-8') as f:
    f.write(c)