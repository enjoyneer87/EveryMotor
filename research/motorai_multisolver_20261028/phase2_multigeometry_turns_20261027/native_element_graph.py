"""Motor-CAD element-support graph with explicit native ID adjacency.

Graph vertices are native FE triangles, not conductor centers or FE nodes.
No coordinate-nearest ID reconstruction, node averaging, J interpretation or A curl.
The saved sliding a2 band is excluded explicitly; this is not a whole-mesh model.
"""
from pathlib import Path
import hashlib
import numpy as np

NODE_FEATURES=('centroid_x_0.1m','centroid_y_0.1m','area_mm2','stator','rotor','magnet','copper','air','wedge','shaft','impreg','other','turn_layer_over8','slot_over6')
EDGE_FEATURES=('dx_0.1m','dy_0.1m','distance_0.1m','different_native_region')
DESIGN_FEATURES=('I_rms_over460','sin_phase','cos_phase','bore_over198','slot_depth_over20','Cu_width_over5','Cu_height_over3','parallel_over2','conductors_over8','rpm_over16000','sin_mechanical_angle','cos_mechanical_angle')

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()

def region_features(reg,names):
 out=np.zeros((len(reg),11),np.float32)
 for i,code in enumerate(reg):
  name=names[int(code)];low=name.lower()
  category=3 if low.startswith('turn_') else (2 if 'magnet' in low else (5 if 'wedge' in low else (7 if 'impreg' in low else (6 if 'shaft' in low else (4 if 'air' in low or 'pocket' in low or low in ['a1','a2','a3','a4'] else (1 if low=='rotor' else (0 if low=='stator' else 8)))))))
  out[i,category]=1
  if low.startswith('turn_'):
   _,layer,slot=low.split('_');out[i,9:]=[int(layer)/8,int(slot)/6]
 return out

def native_dual_edges(tri):
 """Shared pairs of explicit native node IDs define neighboring FE triangles."""
 pairs=np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]],axis=0),axis=1)
 owners=np.tile(np.arange(len(tri)),3);order=np.lexsort((pairs[:,1],pairs[:,0]));pairs=pairs[order];owners=owners[order]
 same=np.all(pairs[1:]==pairs[:-1],axis=1);where=np.flatnonzero(same)
 if np.any(same[1:]&same[:-1]):raise ValueError('Non-manifold native edge')
 a,b=owners[where],owners[where+1]
 return np.vstack([np.r_[a,b],np.r_[b,a]]).astype(np.int64)

def read_case(path,native_input,selected_steps):
 import h5py
 with h5py.File(path,'r') as h:
  node_id=h['mesh/node_id'][:].astype(np.int64);element_id=h['mesh/tri_index'][:].astype(np.int64)
  assert len(set(node_id))==len(node_id) and len(set(element_id))==len(element_id)
  native_tri=np.column_stack([h[f'mesh/node_{k}'][:] for k in [1,2,3]]).astype(np.int64)
  order=np.argsort(node_id);positions=np.searchsorted(node_id[order],native_tri)
  assert positions.max()<len(node_id) and np.array_equal(node_id[order][positions],native_tri)
  tri=order[positions];reg=h['mesh/reg_code'][:].astype(np.int64)
  names=dict(zip(h['regions/reg_code'][:].tolist(),h['regions/name'].asstr()[:].tolist()))
  assert set(reg).issubset(names)
  base=np.column_stack([h['mesh/node_x_mm'][:],h['mesh/node_y_mm'][:]]).astype(np.float64)
  moving=h['mesh/moving_node_indices'][:].astype(np.int64)
  assert np.array_equal(node_id[moving],h['mesh/moving_node_id'][:]) and len(set(moving))==len(moving)
  nsteps=len(h['steps']);assert nsteps==45 and h['fields/bx'].shape==(nsteps,len(tri)) and h['fields/by'].shape==(nsteps,len(tri))
  # RotateStep is an incremental mechanical angle in this native export.
  angles=np.cumsum(h['meta/rotate_step'][:].astype(float));assert np.isfinite(angles).all()
  keep=np.array([names[int(code)].lower()!='a2' for code in reg]);assert keep.any()
  edges=native_dual_edges(native_tri[keep]);region=region_features(reg[keep],names);graphs=[];diagnostics=[];reference_sign=None
  for step in range(nsteps):
   xy=base.copy();xy[moving,0]=h['mesh/node_x_mm_by_step_moving'][step];xy[moving,1]=h['mesh/node_y_mm_by_step_moving'][step]
   vertices=xy[tri];ab=vertices[:,1]-vertices[:,0];ac=vertices[:,2]-vertices[:,0];signed=ab[:,0]*ac[:,1]-ab[:,1]*ac[:,0]
   if reference_sign is None:reference_sign=np.sign(signed)
   invalid=(np.abs(signed)<1e-10)|(np.sign(signed)!=reference_sign)|~np.isfinite(signed)
   if invalid[keep].any():raise ValueError(f'Invalid geometry outside explicitly excluded a2 at step{step}')
   b=np.column_stack([h['fields/bx'][step],h['fields/by'][step]]).astype(np.float64)
   assert np.isfinite(b[keep]).all()
   magnitude=h['fields/b'][step].astype(float);error=float(np.max(np.abs(np.linalg.norm(b[keep],axis=1)-magnitude[keep])))
   tol=float(16*np.finfo(np.float32).eps*max(1,float(np.nanmax(magnitude[keep]))))
   if not np.isfinite(error) or error>tol:raise ValueError('Stored B magnitude/vector arithmetic differs beyond float32 bound')
   diagnostics.append(dict(step=step,invalid_elements_total=int(invalid.sum()),invalid_elements_outside_a2=int(invalid[keep].sum()),B_magnitude_roundtrip_max_T=error,float32_arithmetic_tolerance_T=tol))
   if step not in selected_steps:continue
   area=np.abs(signed[keep])/2*1e-6;centroids=vertices[keep].mean(axis=1)*1e-3
   node=np.column_stack([centroids/.1,area*1e6,region]).astype(np.float32)
   src,dst=edges;delta=(centroids[dst]-centroids[src])/.1
   edge=np.column_stack([delta,np.linalg.norm(delta,axis=1),(reg[keep][src]!=reg[keep][dst]).astype(float)]).astype(np.float32)
   v=native_input;phase=np.deg2rad(v['PhaseAdvance']);angle=np.deg2rad(angles[step])
   design=np.array([v['RMSCurrent']/460,np.sin(phase),np.cos(phase),v['Stator_Bore']/198,v['Slot_Depth']/20,v['Copper_Width']/5,v['Copper_Height']/3,v['ParallelPaths']/2,v['ConductorsPerSlot']/8,v['ShaftSpeed']/16000,np.sin(angle),np.cos(angle)],np.float32)
   graphs.append(dict(step=np.array(step),native_element_id=element_id[keep],native_tri_node_ids=native_tri[keep],native_reg_code=reg[keep],node=node,edge_index=edges,edge_attr=edge,weights=area.astype(np.float32),target_B_T=b[keep].astype(np.float32),design_op=design,mechanical_angle_deg=np.array(angles[step])))
  return graphs,dict(native_elements=len(tri),native_nodes=len(node_id),excluded_a2_elements=int((~keep).sum()),kept_elements=int(keep.sum()),all_45_steps_checked=True,selected_steps=selected_steps,
    scope='native element-reported B supervision, a2 excluded; no nodal A curl, no AC-loss reconstruction',field_units='T',coordinate_unit='m after explicit mm conversion',target_location='native_element_reported',
    time_first_nan=bool(np.isnan(h['meta/time_s'][0])),time_not_used_as_model_input=True,steps=diagnostics)
