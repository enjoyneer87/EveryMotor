import os

f1 = 'D:/KDH/gitEmach/eMach/tools/motorCAD/pyMCAD/magnetic.py'
with open(f1, 'r', encoding='utf-8') as f:
    c1 = f.read()

if 'vmin=None' not in c1:
    c1 = c1.replace('mesh_kwargs=None,\n    ):', 'mesh_kwargs=None,\n        vmin=None,\n        vmax=None,\n    ):')
    c1 = c1.replace('sc = ax.scatter(xs, ys, c=cs, s=s, cmap=cmap, marker=".")', 'sc = ax.scatter(xs, ys, c=cs, s=s, cmap=cmap, marker=".", vmin=vmin, vmax=vmax)')
    with open(f1, 'w', encoding='utf-8') as f:
        f.write(c1)
    print("Patched magnetic.py")

f2 = 'D:/KDH/gitEmach/eMach/tools/motorCAD/pyMCAD/__init__.py'
with open(f2, 'r', encoding='utf-8') as f:
    c2 = f.read()

if 'vmin=None' not in c2:
    c2 = c2.replace('quantity="b", reg_code=None, s=2, cmap="jet"):', 'quantity="b", reg_code=None, s=2, cmap=None, vmin=None, vmax=None):')
    
    inject_str = '''qty = str(qty_dd.value).lower()
            
            _vmin, _vmax = vmin, vmax
            _cmap = cmap if cmap else ("jet" if qty == "b" else "twilight_shifted" if qty == "a" else "RdBu_r")
            
            if _vmin is None and _vmax is None:
                if qty == "b":
                    _vmin, _vmax = 0.0, 2.0
                elif qty == "a":
                    _vmin, _vmax = -0.01, 0.01
                elif qty == "j":
                    _vmin, _vmax = -60.0, 60.0
'''
    c2 = c2.replace('qty = str(qty_dd.value).lower()', inject_str)
    c2 = c2.replace('s=size_slider.value,\n                cmap=cmap,', 's=size_slider.value,\n                cmap=_cmap,\n                vmin=_vmin,\n                vmax=_vmax,')
    
    with open(f2, 'w', encoding='utf-8') as f:
        f.write(c2)
    print("Patched __init__.py")
