import numpy as np
import torch
from torch.autograd.functional import jacobian

from torch_robotics.environments.primitives import ObjectField, MultiSphereField
from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robots import DifferentiableFrankaPanda
from torch_robotics.torch_planning_objectives.fields.distance_fields import CollisionObjectDistanceField
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import DEFAULT_TENSOR_ARGS


tensor_args = DEFAULT_TENSOR_ARGS


########################################################################################################################
# Object field composed of a sphere
def df_obj_list_fn():
    sphere = MultiSphereField(
        torch.zeros(3, **tensor_args).view(1, -1),
        torch.ones(1, **tensor_args).view(1, -1) * 0.1,
        tensor_args=tensor_args,
    )
    obj_field_1 = ObjectField([sphere])
    theta = np.deg2rad(45)
    obj_field_1.set_position_orientation(pos=[-1.5, 0.0, 1.0], ori=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)])

    obj_field_2 = ObjectField([sphere])
    theta = np.deg2rad(30)
    obj_field_2.set_position_orientation(pos=[1.0, 0.0, 1.0], ori=[np.cos(theta / 2), 0, 0, np.sin(theta / 2)])

    return [obj_field_1, obj_field_2]


# Collision field
collision_df = CollisionObjectDistanceField(
    df_obj_list_fn=df_obj_list_fn, num_interpolated_points=50, tensor_args=tensor_args
)

# Robot
diff_panda = DifferentiableFrankaPanda(device=tensor_args["device"])

batch_size = 20


def compute_distances_objects(q_):
    # Compute the shortest distance to all objects
    link_tensor = diff_panda.compute_forward_kinematics_all_links(q_)
    link_pos = link_pos_from_link_tensor(link_tensor)
    distances = collision_df.compute_distance(link_pos)
    return distances


q = torch.rand(batch_size, diff_panda._n_dofs).to(**tensor_args).requires_grad_(True)
distances = compute_distances_objects(q)
print(distances)
print(f"distances.shape: {distances.shape}")


# Compute the distances gradient wrt to the configurations
def surrogate_compute_distances_objects(q_):
    return compute_distances_objects(q_).sum(0)


with TimerCUDA() as t:
    q_new = torch.rand(batch_size, diff_panda._n_dofs).to(**tensor_args)
    grad_distance_wrt_q = jacobian(surrogate_compute_distances_objects, q_new).movedim(-2, 0)
    print(grad_distance_wrt_q)
print(f"grad_distance_wrt_q.shape: {grad_distance_wrt_q.shape}")
print(f"jacobian_distance time: {t.elapsed}")
