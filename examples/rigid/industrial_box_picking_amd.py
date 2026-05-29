import argparse

import numpy as np

import genesis as gs


BOX_SIZE = (0.055, 0.055, 0.045)
PICK_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def parse_args():
    parser = argparse.ArgumentParser(description="Industrial robot box-picking cell for AMD ROCm/HIP.")
    parser.add_argument(
        "--backend",
        choices=("amdgpu", "cpu", "gpu", "cuda"),
        default="amdgpu",
        help="Genesis backend. Use cpu for smoke tests outside a ROCm container.",
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer.")
    parser.add_argument("--cycles", type=int, default=3, help="Number of boxes to move from infeed to pallet.")
    parser.add_argument("--fast", action="store_true", help="Run fewer simulation steps for quick validation.")
    return parser.parse_args()


def wait(scene, steps):
    for _ in range(steps):
        scene.step()


def move_ee(scene, robot, end_effector, pos, steps, motors_dof, fingers_dof, gripper_width=0.04):
    qpos = robot.inverse_kinematics(link=end_effector, pos=np.asarray(pos), quat=PICK_QUAT)
    robot.control_dofs_position(qpos[:-2], motors_dof)
    robot.control_dofs_position(np.array([gripper_width, gripper_width]), fingers_dof)
    wait(scene, steps)
    return qpos


def add_static_box(scene, pos, size, color):
    return scene.add_entity(
        gs.morphs.Box(pos=pos, size=size, fixed=True),
        surface=gs.surfaces.Plastic(color=color),
    )


def main():
    args = parse_args()
    backend = getattr(gs, args.backend)
    gs.init(backend=backend, precision="32")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.8, -1.9, 1.25),
            camera_lookat=(0.52, 0.02, 0.18),
            camera_fov=38,
            res=(1280, 720),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(box_box_detection=True),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    # Industrial fixtures: an infeed table, a pallet table, and low retaining rails.
    add_static_box(scene, pos=(0.62, -0.24, 0.015), size=(0.44, 0.28, 0.03), color=(0.35, 0.38, 0.40))
    add_static_box(scene, pos=(0.42, 0.34, 0.015), size=(0.38, 0.30, 0.03), color=(0.26, 0.30, 0.33))
    add_static_box(scene, pos=(0.62, -0.39, 0.08), size=(0.44, 0.018, 0.10), color=(0.08, 0.11, 0.13))
    add_static_box(scene, pos=(0.62, -0.09, 0.08), size=(0.44, 0.018, 0.10), color=(0.08, 0.11, 0.13))
    add_static_box(scene, pos=(0.22, 0.34, 0.08), size=(0.018, 0.30, 0.10), color=(0.08, 0.11, 0.13))
    add_static_box(scene, pos=(0.62, 0.34, 0.08), size=(0.018, 0.30, 0.10), color=(0.08, 0.11, 0.13))

    source_positions = [(0.53, -0.25, 0.055), (0.62, -0.25, 0.055), (0.71, -0.25, 0.055)]
    place_positions = [(0.36, 0.30, 0.055), (0.45, 0.30, 0.055), (0.54, 0.30, 0.055)]
    boxes = []
    for i, pos in enumerate(source_positions):
        boxes.append(
            scene.add_entity(
                gs.morphs.Box(size=BOX_SIZE, pos=pos),
                surface=gs.surfaces.Plastic(color=(0.88, 0.50 + 0.08 * i, 0.16)),
            )
        )

    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"), vis_mode="collision")
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = franka.get_link("hand")

    franka.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 120, 120]))
    franka.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 12, 12]))
    franka.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )

    arm_ready = np.array([-0.7, 0.9, 0.7, -1.7, -0.8, 1.5, 0.6, 0.04, 0.04])
    franka.set_dofs_position(arm_ready)
    wait(scene, 80 if not args.fast else 8)

    rigid = scene.sim.rigid_solver
    hand_link = franka.get_link("hand").idx
    travel_steps = 90 if not args.fast else 8
    settle_steps = 35 if not args.fast else 4

    for i, box in enumerate(boxes[: args.cycles]):
        pick = np.array(source_positions[i])
        place = np.array(place_positions[i])
        hover_pick = pick + np.array([0.0, 0.0, 0.23])
        touch_pick = pick + np.array([0.0, 0.0, 0.095])
        hover_place = place + np.array([0.0, 0.0, 0.24])
        release_place = place + np.array([0.0, 0.0, 0.11])

        move_ee(scene, franka, end_effector, hover_pick, travel_steps, motors_dof, fingers_dof, 0.04)
        move_ee(scene, franka, end_effector, touch_pick, settle_steps, motors_dof, fingers_dof, 0.012)

        box_link = box.get_link("box_baselink").idx
        rigid.add_weld_constraint(box_link, hand_link)
        wait(scene, settle_steps)

        move_ee(scene, franka, end_effector, hover_pick, travel_steps, motors_dof, fingers_dof, 0.012)
        move_ee(scene, franka, end_effector, hover_place, travel_steps, motors_dof, fingers_dof, 0.012)
        move_ee(scene, franka, end_effector, release_place, settle_steps, motors_dof, fingers_dof, 0.012)

        rigid.delete_weld_constraint(box_link, hand_link)
        franka.control_dofs_position(np.array([0.04, 0.04]), fingers_dof)
        wait(scene, settle_steps)
        move_ee(scene, franka, end_effector, hover_place, settle_steps, motors_dof, fingers_dof, 0.04)

    print(f"Completed {min(args.cycles, len(boxes))} AMD box-picking cycle(s).")


if __name__ == "__main__":
    main()
