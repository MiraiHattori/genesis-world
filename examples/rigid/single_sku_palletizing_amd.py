import argparse

import numpy as np

import genesis as gs


CARTON_SIZE = np.array([0.105, 0.075, 0.055])
PALLET_TOP_Z = 0.055
PICK_QUAT = np.array([0.0, 1.0, 0.0, 0.0])


def parse_args():
    parser = argparse.ArgumentParser(description="Single-SKU palletizing cell for AMD ROCm/HIP.")
    parser.add_argument(
        "--backend",
        choices=("amdgpu", "cpu", "gpu", "cuda"),
        default="amdgpu",
        help="Genesis backend. Use cpu for smoke tests outside a ROCm container.",
    )
    parser.add_argument("-v", "--vis", action="store_true", help="Show the interactive viewer.")
    parser.add_argument("--fast", action="store_true", help="Run fewer simulation steps for quick validation.")
    parser.add_argument("--rows", type=int, default=2, help="Carton rows per layer.")
    parser.add_argument("--cols", type=int, default=3, help="Carton columns per layer.")
    parser.add_argument("--layers", type=int, default=2, help="Number of pallet layers.")
    parser.add_argument("--cartons", type=int, default=None, help="Override total cartons to palletize.")
    return parser.parse_args()


def wait(scene, steps):
    for _ in range(steps):
        scene.step()


def add_static_box(scene, pos, size, color):
    return scene.add_entity(
        gs.morphs.Box(pos=pos, size=size, fixed=True),
        surface=gs.surfaces.Plastic(color=color),
    )


def move_ee(scene, robot, end_effector, pos, steps, motors_dof):
    qpos = robot.inverse_kinematics(link=end_effector, pos=np.asarray(pos), quat=PICK_QUAT)
    robot.control_dofs_position(qpos[motors_dof], motors_dof)
    wait(scene, steps)


def pallet_pattern(rows, cols, layers):
    center = np.array([0.43, 0.30, PALLET_TOP_Z])
    spacing = CARTON_SIZE[:2] + np.array([0.008, 0.008])
    pattern = []
    for layer in range(layers):
        layer_xy = []
        for row in range(rows):
            for col in range(cols):
                x = center[0] + (col - (cols - 1) * 0.5) * spacing[0]
                y = center[1] + (row - (rows - 1) * 0.5) * spacing[1]
                if layer % 2:
                    x, y = center[0] + (y - center[1]), center[1] + (x - center[0])
                z = PALLET_TOP_Z + CARTON_SIZE[2] * (layer + 0.5)
                layer_xy.append(np.array([x, y, z]))
        pattern.extend(layer_xy)
    return pattern


def main():
    args = parse_args()
    backend = getattr(gs, args.backend)
    gs.init(backend=backend, precision="32")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.65, -1.7, 1.18),
            camera_lookat=(0.50, 0.06, 0.18),
            camera_fov=42,
            res=(1280, 720),
            max_FPS=60,
        ),
        sim_options=gs.options.SimOptions(dt=0.01),
        rigid_options=gs.options.RigidOptions(box_box_detection=True),
        show_viewer=args.vis,
    )

    scene.add_entity(gs.morphs.Plane())

    add_static_box(scene, pos=(0.66, -0.34, 0.025), size=(0.54, 0.16, 0.05), color=(0.18, 0.21, 0.23))
    add_static_box(scene, pos=(0.66, -0.45, 0.095), size=(0.54, 0.018, 0.11), color=(0.06, 0.08, 0.09))
    add_static_box(scene, pos=(0.66, -0.23, 0.095), size=(0.54, 0.018, 0.11), color=(0.06, 0.08, 0.09))
    add_static_box(scene, pos=(0.43, 0.30, 0.025), size=(0.42, 0.30, 0.05), color=(0.42, 0.27, 0.12))
    add_static_box(scene, pos=(0.43, 0.30, PALLET_TOP_Z + 0.003), size=(0.40, 0.28, 0.006), color=(0.55, 0.36, 0.16))

    requested_cartons = args.rows * args.cols * args.layers if args.cartons is None else args.cartons
    requested_cartons = max(1, min(requested_cartons, args.rows * args.cols * args.layers))
    source_positions = [
        np.array([0.54 + 0.075 * (i % 4), -0.34, 0.05 + CARTON_SIZE[2] * 0.5])
        for i in range(requested_cartons)
    ]
    place_positions = pallet_pattern(args.rows, args.cols, args.layers)[:requested_cartons]

    cartons = []
    for pos in source_positions:
        cartons.append(
            scene.add_entity(
                gs.morphs.Box(size=CARTON_SIZE, pos=pos),
                surface=gs.surfaces.Plastic(color=(0.78, 0.56, 0.30)),
            )
        )

    robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_sim/franka_panda_no_finger.xml"), vis_mode="collision")
    scene.build()

    motors_dof = np.arange(7)
    end_effector = robot.get_link("panda0_gripper")
    robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000]))
    robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200]))
    robot.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12]),
        np.array([87, 87, 87, 87, 12, 12, 12]),
    )
    robot.set_dofs_position(np.array([-0.65, 0.9, 0.65, -1.65, -0.75, 1.45, 0.55]))
    wait(scene, 80 if not args.fast else 8)

    rigid = scene.sim.rigid_solver
    suction_link = end_effector.idx
    travel_steps = 90 if not args.fast else 8
    settle_steps = 35 if not args.fast else 4

    for carton, pick, place in zip(cartons, source_positions, place_positions):
        hover_pick = pick + np.array([0.0, 0.0, 0.22])
        touch_pick = pick + np.array([0.0, 0.0, 0.085])
        hover_place = place + np.array([0.0, 0.0, 0.24])
        release_place = place + np.array([0.0, 0.0, 0.095])

        move_ee(scene, robot, end_effector, hover_pick, travel_steps, motors_dof)
        move_ee(scene, robot, end_effector, touch_pick, settle_steps, motors_dof)
        carton_link = carton.get_link("box_baselink").idx
        rigid.add_weld_constraint(carton_link, suction_link)
        wait(scene, settle_steps)

        move_ee(scene, robot, end_effector, hover_pick, travel_steps, motors_dof)
        move_ee(scene, robot, end_effector, hover_place, travel_steps, motors_dof)
        move_ee(scene, robot, end_effector, release_place, settle_steps, motors_dof)
        rigid.delete_weld_constraint(carton_link, suction_link)
        wait(scene, settle_steps)
        move_ee(scene, robot, end_effector, hover_place, settle_steps, motors_dof)

    print(f"Completed single-SKU palletizing for {requested_cartons} carton(s).")


if __name__ == "__main__":
    main()
