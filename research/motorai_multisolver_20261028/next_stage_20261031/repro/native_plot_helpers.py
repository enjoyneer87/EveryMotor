"""Exact reused flat-native polygon/crop/clip helpers; no array loading."""
import numpy as np
from matplotlib.collections import PolyCollection
from matplotlib.colors import Normalize

def clip_record(value,area,limits):
    lo,hi=value<limits[0],value>limits[1]
    return dict(below_elements=int(lo.sum()),above_elements=int(hi.sum()),
                count_scope='full retained native support; crop changes viewport only',
                below_area_fraction=float(area[lo].sum()/area.sum()),
                above_area_fraction=float(area[hi].sum()/area.sum()),
                minimum_T=float(value.min()),maximum_T=float(value.max()))

def paint(ax,vertices,value,limits,cmap,title,bounds,area):
    c=clip_record(value,area,limits)
    artist=PolyCollection(vertices,array=value,cmap=cmap,norm=Normalize(*limits),
                          edgecolors='none',antialiaseds=False)
    ax.set_facecolor('#d9dde2');ax.add_collection(artist)
    ax.set(xlim=bounds[:2],ylim=bounds[2:],aspect='equal',xlabel='x [mm]',ylabel='y [mm]')
    ax.set_title(title+f"\n전체 요소 범위 밖 ↓{c['below_elements']} / ↑{c['above_elements']}",fontsize=8)
    ax.tick_params(labelsize=7)
    return artist,c

def a1_crop(vertices,mask):
    points=vertices[mask].reshape(-1,2)
    theta=np.arctan2(points[:,1],points[:,0]);radius=np.linalg.norm(points,axis=1)
    assert theta.max()-theta.min()<np.pi, 'no inferred angle wrapping allowed'
    middle=(theta.min()+theta.max())/2
    low=max(theta.min(),middle-np.deg2rad(5));high=min(theta.max(),middle+np.deg2rad(5))
    corners=np.asarray([(r*np.cos(t),r*np.sin(t)) for r in (radius.min(),radius.max()) for t in (low,high)])
    return (float(corners[:,0].min()-.5),float(corners[:,0].max()+.5),
            float(corners[:,1].min()-.5),float(corners[:,1].max()+.5))
