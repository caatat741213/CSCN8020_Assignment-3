from __future__ import annotations

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer


def get_name(
    model: mujoco.MjModel,
    object_type: mujoco.mjtObj,
    object_id: int,
) -> str:
    """Return a MuJoCo object's name or a readable fallback."""
    name = mujoco.mj_id2name(model, object_type, object_id)
    return name if name is not None else f"<unnamed-{object_id}>"


def print_model_summary(model: mujoco.MjModel) -> None:
    """Print the main dimensions of the loaded MuJoCo model."""
    print("\n=== G1 MODEL SUMMARY ===")
    print(f"Model name:          {model.names.decode(errors='ignore').split(chr(0))[0]}")
    print(f"Bodies:              {model.nbody}")
    print(f"Joints:              {model.njnt}")
    print(f"Degrees of freedom:  {model.nv}")
    print(f"Position variables:  {model.nq}")
    print(f"Actuators:           {model.nu}")
    print(f"Sensors:             {model.nsensor}")
    print(f"Simulation timestep: {model.opt.timestep:.6f} seconds")


def print_joint_information(model: mujoco.MjModel) -> None:
    """Print joint names, types, and model indices."""
    joint_type_names = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }

    print("\n=== JOINTS ===")

    for joint_id in range(model.njnt):
        name = get_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        joint_type = int(model.jnt_type[joint_id])

        print(
            f"{joint_id:2d} | "
            f"{name:35s} | "
            f"type={joint_type_names.get(joint_type, str(joint_type)):6s} | "
            f"qpos_address={model.jnt_qposadr[joint_id]:2d} | "
            f"dof_address={model.jnt_dofadr[joint_id]:2d}"
        )


def print_actuator_information(model: mujoco.MjModel) -> None:
    """Print actuator names and control ranges."""
    print("\n=== ACTUATORS ===")

    for actuator_id in range(model.nu):
        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_id,
        )

        limited = bool(model.actuator_ctrllimited[actuator_id])
        low, high = model.actuator_ctrlrange[actuator_id]

        print(
            f"{actuator_id:2d} | "
            f"{name:35s} | "
            f"limited={limited!s:5s} | "
            f"control range=[{low:9.3f}, {high:9.3f}]"
        )


def print_sensor_information(model: mujoco.MjModel) -> None:
    """Print available MuJoCo sensor definitions."""
    print("\n=== SENSORS ===")

    if model.nsensor == 0:
        print("No MuJoCo sensors are defined in this scene.")
        return

    for sensor_id in range(model.nsensor):
        name = get_name(
            model,
            mujoco.mjtObj.mjOBJ_SENSOR,
            sensor_id,
        )

        print(
            f"{sensor_id:2d} | "
            f"{name:35s} | "
            f"dimension={model.sensor_dim[sensor_id]:2d} | "
            f"data_address={model.sensor_adr[sensor_id]:3d}"
        )


def run_viewer(model: mujoco.MjModel) -> None:
    """Open the passive MuJoCo viewer and run the model."""
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    print("\nOpening MuJoCo viewer.")
    print("Close the viewer window to end the program.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.perf_counter()

            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.perf_counter() - step_start
            delay = model.opt.timestep - elapsed

            if delay > 0:
                time.sleep(delay)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and inspect a Unitree G1 MuJoCo scene."
    )

    parser.add_argument(
        "scene",
        type=Path,
        help="Path to a G1 MuJoCo scene XML file.",
    )

    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Print the model information without opening the viewer.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    scene_path = args.scene.expanduser().resolve()

    if not scene_path.is_file():
        raise FileNotFoundError(
            f"G1 scene file was not found: {scene_path}"
        )

    print(f"Loading scene: {scene_path}")

    model = mujoco.MjModel.from_xml_path(str(scene_path))

    print_model_summary(model)
    print_joint_information(model)
    print_actuator_information(model)
    print_sensor_information(model)

    if not args.no_viewer:
        run_viewer(model)


if __name__ == "__main__":
    main()