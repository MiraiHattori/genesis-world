import argparse

import numpy as np

import genesis as gs


CARTON_SIZE = np.array([0.105, 0.075, 0.055])
PALLET_TOP_Z = 0.055
INFEED_Y = -0.08
PICK_QUAT = np.array([0.0, 1.0, 0.0, 0.0])
SUCTION_CUP_HEIGHT = 0.025
SUCTION_CUP_RADIUS = 0.035
ROBOT_TOOL_OFFSET = np.array([0.0, 0.0, 0.14])
TOOL_STEM_HEIGHT = ROBOT_TOOL_OFFSET[2] - SUCTION_CUP_HEIGHT
FINGER_OPEN = np.array([0.04, 0.04])


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


def set_entity_pos(entity, pos):
    entity.set_pos(np.asarray(pos), zero_velocity=True)


def update_tool_visuals(tool, cup_pos):
    cup, stem, flange = tool
    set_entity_pos(cup, cup_pos)
    set_entity_pos(stem, cup_pos + np.array([0.0, 0.0, SUCTION_CUP_HEIGHT * 0.5 + TOOL_STEM_HEIGHT * 0.5]))
    set_entity_pos(flange, cup_pos + ROBOT_TOOL_OFFSET)


def move_vacuum_cup(
    scene,
    tool,
    robot,
    end_effector,
    target,
    steps,
    motors_dof,
    fingers_dof=None,
    attached_carton=None,
):
    cup = tool[0]
    start = entity_pos(cup)
    target = np.asarray(target)
    for alpha in np.linspace(0.0, 1.0, max(2, steps)):
        pos = start + alpha * (target - start)
        update_tool_visuals(tool, pos)
        if attached_carton is not None:
            carton_center = pos - np.array([0.0, 0.0, SUCTION_CUP_HEIGHT * 0.5 + CARTON_SIZE[2] * 0.5])
            set_entity_pos(attached_carton, carton_center)

        qpos = robot.inverse_kinematics(link=end_effector, pos=pos + ROBOT_TOOL_OFFSET, quat=PICK_QUAT)
        robot.control_dofs_position(qpos[motors_dof], motors_dof)
        if fingers_dof is not None:
            robot.control_dofs_position(FINGER_OPEN, fingers_dof)
        scene.step()


def tensor_to_numpy(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    return np.asarray(value, dtype=float)


def entity_pos(entity):
    return tensor_to_numpy(entity.get_pos())


def carton_top_center(carton):
    return entity_pos(carton) + np.array([0.0, 0.0, CARTON_SIZE[2] * 0.5])


def suction_errors(cup, carton):
    tool_contact = entity_pos(cup) - np.array([0.0, 0.0, SUCTION_CUP_HEIGHT * 0.5])
    top = carton_top_center(carton)
    xy_error = np.linalg.norm(tool_contact[:2] - top[:2])
    z_error = abs(tool_contact[2] - top[2])
    return tool_contact, top, xy_error, z_error


def suction_sealed(cup, carton, xy_tolerance=0.018, z_tolerance=0.018):
    _, _, xy_error, z_error = suction_errors(cup, carton)
    return xy_error <= xy_tolerance and z_error <= z_tolerance


def attach_with_suction(scene, tool, robot, end_effector, carton, motors_dof, fingers_dof, travel_steps, settle_steps):
    cup = tool[0]
    top = carton_top_center(carton)
    hover = top + np.array([0.0, 0.0, 0.16 + SUCTION_CUP_HEIGHT * 0.5])
    move_vacuum_cup(scene, tool, robot, end_effector, hover, travel_steps, motors_dof, fingers_dof)

    for _ in range(4):
        top = carton_top_center(carton)
        seal = top + np.array([0.0, 0.0, SUCTION_CUP_HEIGHT * 0.5])
        move_vacuum_cup(scene, tool, robot, end_effector, seal, settle_steps, motors_dof, fingers_dof)
        if suction_sealed(cup, carton):
            wait(scene, settle_steps)
            return

    tool_contact, top, xy_error, z_error = suction_errors(cup, carton)
    raise RuntimeError(
        "Vacuum seal failed: tool contact point did not reach carton top. "
        f"tool_contact={tool_contact.round(4).tolist()} carton_top={top.round(4).tolist()} "
        f"xy_error={xy_error:.4f} z_error={z_error:.4f}"
    )


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

    add_static_box(scene, pos=(0.66, INFEED_Y, 0.025), size=(0.54, 0.16, 0.05), color=(0.18, 0.21, 0.23))
    add_static_box(scene, pos=(0.66, INFEED_Y - 0.11, 0.095), size=(0.54, 0.018, 0.11), color=(0.06, 0.08, 0.09))
    add_static_box(scene, pos=(0.66, INFEED_Y + 0.11, 0.095), size=(0.54, 0.018, 0.11), color=(0.06, 0.08, 0.09))
    add_static_box(scene, pos=(0.43, 0.30, 0.025), size=(0.42, 0.30, 0.05), color=(0.42, 0.27, 0.12))
    add_static_box(scene, pos=(0.43, 0.30, PALLET_TOP_Z + 0.003), size=(0.40, 0.28, 0.006), color=(0.55, 0.36, 0.16))

    requested_cartons = args.rows * args.cols * args.layers if args.cartons is None else args.cartons
    requested_cartons = max(1, min(requested_cartons, args.rows * args.cols * args.layers))
    source_positions = [
        np.array([0.54 + 0.075 * (i % 4), INFEED_Y, 0.05 + CARTON_SIZE[2] * 0.5])
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

    vacuum_cup_start = np.array([source_positions[0][0], INFEED_Y, 0.32])
    vacuum_cup = scene.add_entity(
        gs.morphs.Cylinder(
            radius=SUCTION_CUP_RADIUS,
            height=SUCTION_CUP_HEIGHT,
            pos=vacuum_cup_start,
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Plastic(color=(0.02, 0.08, 0.10)),
    )
    vacuum_stem = scene.add_entity(
        gs.morphs.Cylinder(
            radius=0.012,
            height=TOOL_STEM_HEIGHT,
            pos=vacuum_cup_start + np.array([0.0, 0.0, SUCTION_CUP_HEIGHT * 0.5 + TOOL_STEM_HEIGHT * 0.5]),
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Plastic(color=(0.62, 0.65, 0.67)),
    )
    vacuum_flange = scene.add_entity(
        gs.morphs.Box(
            size=(0.07, 0.07, 0.018),
            pos=vacuum_cup_start + ROBOT_TOOL_OFFSET,
            fixed=True,
            collision=False,
        ),
        surface=gs.surfaces.Plastic(color=(0.50, 0.53, 0.55)),
    )

    robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    motors_dof = np.arange(7)
    fingers_dof = np.arange(7, 9)
    end_effector = robot.get_link("hand")
    robot.set_dofs_kp(np.array([4500, 4500, 3500, 3500, 2000, 2000, 2000, 120, 120]))
    robot.set_dofs_kv(np.array([450, 450, 350, 350, 200, 200, 200, 12, 12]))
    robot.set_dofs_force_range(
        np.array([-87, -87, -87, -87, -12, -12, -12, -100, -100]),
        np.array([87, 87, 87, 87, 12, 12, 12, 100, 100]),
    )
    robot.set_dofs_position(np.array([-0.65, 0.9, 0.65, -1.65, -0.75, 1.45, 0.55, *FINGER_OPEN]))
    wait(scene, 80 if not args.fast else 8)

    travel_steps = 90 if not args.fast else 35
    settle_steps = 35 if not args.fast else 12

    for carton, pick, place in zip(cartons, source_positions, place_positions):
        hover_pick = pick + np.array([0.0, 0.0, 0.22])
        hover_place = place + np.array([0.0, 0.0, 0.24])
        release_cup = place + np.array([0.0, 0.0, CARTON_SIZE[2] * 0.5 + SUCTION_CUP_HEIGHT * 0.5])
        hover_place_cup = release_cup + np.array([0.0, 0.0, 0.16])

        attach_with_suction(
            scene,
            (vacuum_cup, vacuum_stem, vacuum_flange),
            robot,
            end_effector,
            carton,
            motors_dof,
            fingers_dof,
            travel_steps,
            settle_steps,
        )

        tool = (vacuum_cup, vacuum_stem, vacuum_flange)
        move_vacuum_cup(scene, tool, robot, end_effector, hover_pick, travel_steps, motors_dof, fingers_dof, carton)
        move_vacuum_cup(scene, tool, robot, end_effector, hover_place_cup, travel_steps, motors_dof, fingers_dof, carton)
        move_vacuum_cup(scene, tool, robot, end_effector, release_cup, settle_steps, motors_dof, fingers_dof, carton)
        wait(scene, settle_steps)
        wait(scene, settle_steps)
        move_vacuum_cup(scene, tool, robot, end_effector, hover_place, settle_steps, motors_dof, fingers_dof)

    print(f"Completed single-SKU palletizing for {requested_cartons} carton(s).")


if __name__ == "__main__":
    main()
