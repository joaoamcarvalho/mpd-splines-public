import torch

from torch_robotics.torch_kinematics_tree.geometrics.utils import link_pos_from_link_tensor, link_quat_from_link_tensor
from torch_robotics.torch_kinematics_tree.models.robots import Differentiable2LinkPlanar, DifferentiableFrankaPanda
from torch_robotics.torch_utils.torch_timer import TimerCUDA

if __name__ == "__main__":

    batch_size = 100
    # device = "cpu"
    device = "cuda"

    print("\n===========================2 Link Planar Model===============================")
    diff_planar = Differentiable2LinkPlanar(device=device)
    diff_planar.print_link_names()
    print(f"joint limits: {diff_planar.get_joint_limits()}")
    print(f"n dofs: {diff_planar._n_dofs}")

    print()
    # Compute forward kinematics all task space links
    with TimerCUDA() as t:
        q = torch.rand(batch_size, diff_planar._n_dofs).to(device).requires_grad_(True)
        data_links_fk = diff_planar.compute_forward_kinematics_all_links(q)
        print(f"data_links_fk.shape: {data_links_fk.shape}")
    print(f"Computational Time {t.elapsed:.4f}")

    print()
    # Compute analytical jacobian for all task space links
    with TimerCUDA() as t:
        link_analytical_jac = diff_planar.compute_analytical_jacobian_all_links(q)
        print(f"link_analytical_jac.shape: {link_analytical_jac.shape}")
    print(f"Computational Time {t.elapsed:.4f}")

    print()
    # Compute linear (position) and angular geometrical jacobians for the end-effector
    with TimerCUDA() as t:
        ee_pos, ee_rot, lin_jac, ang_jac = diff_planar.compute_forward_kinematics_and_geometric_jacobian(
            q, torch.zeros_like(q), "link_ee"
        )
        print(f"lin_jac.shape: {lin_jac.shape}")
        print(f"ang_jac.shape: {ang_jac.shape}")
    print(f"Computational Time {t.elapsed:.4f}")

    print()
    # Compute linear (position) and angular geometrical jacobians for all links
    with TimerCUDA() as t:
        for link_name in diff_planar.get_link_names():
            print(link_name)
            ee_pos, ee_rot, lin_jac, ang_jac = diff_planar.compute_forward_kinematics_and_geometric_jacobian(
                q, torch.zeros_like(q), link_name
            )
            print(f"lin_jac.shape: {lin_jac.shape}")
            print(f"ang_jac.shape: {ang_jac.shape}")
    print(f"Computational Time {t.elapsed:.4f}")

    print("\n===========================Panda Model===============================")
    diff_panda = DifferentiableFrankaPanda(gripper=False, device=device)
    diff_panda.print_link_names()
    print(f"joint limits: {diff_panda.get_joint_limits()}")
    print(f"n dofs: {diff_panda._n_dofs}")

    print()
    # Compute forward kinematics all task space links
    with TimerCUDA() as t:
        q = torch.rand(batch_size, diff_panda._n_dofs).to(device).requires_grad_(True)
        data_links_fk = diff_panda.compute_forward_kinematics_all_links(q)
        print(f"data_links_fk.shape: {data_links_fk.shape}")
    print(f"Computational Time {t.elapsed:.4f}")

    print()
    # Compute analytical jacobian for all task space links
    with TimerCUDA() as t:
        link_analytical_jac = diff_panda.compute_analytical_jacobian_all_links(q)
        print(f"link_analytical_jac.shape: {link_analytical_jac.shape}")
    print(f"Computational Time {t.elapsed:.4f}")

    print()

    print()
    # Compute linear (position) and angular geometrical jacobians for all links
    with TimerCUDA() as t:
        for link_name in diff_panda.get_link_names():
            print(link_name)
            ee_pos, ee_rot, lin_jac, ang_jac = diff_panda.compute_forward_kinematics_and_geometric_jacobian(
                q, torch.zeros_like(q), link_name
            )
            print(f"lin_jac.shape: {lin_jac.shape}")
            print(f"ang_jac.shape: {ang_jac.shape}")
    print(f"Computational Time {t.elapsed:.4f}")

    print()
    # Compute linear (position) and angular geometrical jacobians for end-effector
    with TimerCUDA() as t:
        ee_pos, ee_rot, lin_jac, ang_jac = diff_panda.compute_forward_kinematics_and_geometric_jacobian(
            q, torch.zeros_like(q), "ee_link"
        )
        print(f"lin_jac.shape: {lin_jac.shape}")
        print(f"ang_jac.shape: {ang_jac.shape}")
    print(f"Computational Time {t.elapsed:.4f}")

    # Compute linear (position) and angular geometrical jacobians for the end-effector
    # with Timer() as t:
    #     ee_pos, ee_rot, lin_jac, ang_jac = diff_panda.compute_forward_kinematics_and_geometric_jacobian(q, torch.zeros_like(q), 'ee_link')
    #     print(f'lin_jac.shape: {lin_jac.shape}')
    #     print(f'ang_jac.shape: {ang_jac.shape}')
    # print(f"Computational Time {t.elapsed:.4f}")

    link_tensor_v1 = data_links_fk[:, -1, ...]
    ee_pos_v1 = link_pos_from_link_tensor(link_tensor_v1).squeeze()
    ee_rot_v1 = link_quat_from_link_tensor(link_tensor_v1).squeeze()

    # ee_pos_v2, ee_rot_v2 = ee_pos, ee_rot
    # ee_pos_v2 = ee_pos_v2.squeeze()
    # ee_rot_v2 = ee_rot_v2.squeeze()

    # print('\n----- TEST ')
    # print(f'torch.allclose(ee_pos_v1, ee_pos_v2): {torch.allclose(ee_pos_v1, ee_pos_v2)}')
    # print(f'torch.allclose(ee_rot_v1, ee_rot_v2): {torch.allclose(ee_rot_v1, ee_rot_v2)}')
    # print()
