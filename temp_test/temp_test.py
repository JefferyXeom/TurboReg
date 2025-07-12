import numpy as np
import torch
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
import turboreg_gpu  # NOTE: torch must be imported before turboreg_gpu


def preprocess(pcd, voxel_size):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*3, max_nn=30))
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size*5, max_nn=100)
    )
    return pcd_down, fpfh


# Load input correspondences
# kpts_src = torch.from_numpy(np.loadtxt('demo_data/000_fpfh_kpts_src.txt')).cuda().float()
# kpts_dst = torch.from_numpy(np.loadtxt('demo_data/000_fpfh_kpts_dst.txt')).cuda().float()

src = o3d.io.read_point_cloud("1_100.pcd")
dst = o3d.io.read_point_cloud("2501_2600.pcd")

voxel_size = 1
src_down, src_fpfh = preprocess(src, voxel_size)
dst_down, dst_fpfh = preprocess(dst, voxel_size)

src_feats = np.array(src_fpfh.data).T
dst_feats = np.array(dst_fpfh.data).T

nn = NearestNeighbors(n_neighbors=1).fit(dst_feats)
_, indices = nn.kneighbors(src_feats)

kpts_src = np.asarray(src_down.points)
kpts_dst = np.asarray(dst_down.points)[indices[:, 0]]

np.savetxt("1_100_kpts.pcd", kpts_src)
np.savetxt("2501_2600_kpts.pcd", kpts_dst)

def numpy_to_torch32(device, *arrays):
    return [torch.tensor(array, device=device, dtype=torch.float32) for array in arrays]

kpts_src_torch, kpts_dst_torch = numpy_to_torch32("cuda:0", kpts_src, kpts_dst)
        
# Initialize TurboReg with specific parameters:
reger = turboreg_gpu.TurboRegGPU(
    6000,      # max_N: Maximum number of correspondences
    0.05,     # tau_length_consis: \tau (consistency threshold for feature length/distance)
    500,      # num_pivot: Number of pivot points, K_1
    0.3,      # radiu_nms: Radius for avoiding the instability of the solution
    0.2,       # tau_inlier: Threshold for inlier points. NOTE: just for post-refinement (REF@PointDSC/SC2PCR/MAC)
    "MAE"       # eval_metric: MetricType (e.g., "IN" for Inlier Number, or "MAE" / "MSE")
)

# Run registration
trans = reger.run_reg(kpts_src_torch, kpts_dst_torch).cpu().numpy()
print(trans, '\n')