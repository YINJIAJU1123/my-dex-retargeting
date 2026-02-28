# import multiprocessing
# import time
# from pathlib import Path
# from queue import Empty
# from typing import Optional

# import cv2
# import numpy as np
# import sapien
# import tyro
# from loguru import logger
# from sapien.asset import create_dome_envmap
# from sapien.utils import Viewer

# from dex_retargeting.constants import (
#     RobotName,
#     RetargetingType,
#     HandType,
#     get_default_config_path,
# )
# from dex_retargeting.retargeting_config import RetargetingConfig
# from single_hand_detector import SingleHandDetector


# def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
#     RetargetingConfig.set_default_urdf_dir(str(robot_dir))
#     logger.info(f"Start retargeting with config {config_path}")
#     retargeting = RetargetingConfig.load_from_file(config_path).build()

#     hand_type = "Right" if "right" in config_path.lower() else "Left"
#     detector = SingleHandDetector(hand_type=hand_type, selfie=False)

#     sapien.render.set_viewer_shader_dir("default")
#     sapien.render.set_camera_shader_dir("default")

#     config = RetargetingConfig.load_from_file(config_path)

#     # Setup
#     scene = sapien.Scene()
#     render_mat = sapien.render.RenderMaterial()
#     render_mat.base_color = [0.06, 0.08, 0.12, 1]
#     render_mat.metallic = 0.0
#     render_mat.roughness = 0.9
#     render_mat.specular = 0.8
#     scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

#     # Lighting
#     scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
#     scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
#     scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
#     scene.set_environment_map(
#         create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
#     )
#     scene.add_area_light_for_ray_tracing(
#         sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
#     )

#     # Camera
#     cam = scene.add_camera(
#         name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
#     )
#     cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

#     viewer = Viewer()
#     viewer.set_scene(scene)
#     viewer.control_window.show_origin_frame = False
#     viewer.control_window.move_speed = 0.01
#     viewer.control_window.toggle_camera_lines(False)
#     viewer.set_camera_pose(cam.get_local_pose())

#     # Load robot and set it to a good pose to take picture
#     loader = scene.create_urdf_loader()
#     filepath = Path(config.urdf_path)
#     robot_name = filepath.stem
#     loader.load_multiple_collisions_from_file = True
#     if "ability" in robot_name:
#         loader.scale = 1.5
#     elif "dclaw" in robot_name:
#         loader.scale = 1.25
#     elif "allegro" in robot_name:
#         loader.scale = 1.4
#     elif "shadow" in robot_name:
#         loader.scale = 0.9
#     elif "bhand" in robot_name:
#         loader.scale = 1.5
#     elif "leap" in robot_name:
#         loader.scale = 1.4
#     elif "svh" in robot_name:
#         loader.scale = 1.5

#     if "glb" not in robot_name:
#         filepath = str(filepath).replace(".urdf", "_glb.urdf")
#     else:
#         filepath = str(filepath)

#     robot = loader.load(filepath)

#     if "ability" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.15]))
#     elif "shadow" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.2]))
#     elif "dclaw" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.15]))
#     elif "allegro" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.05]))
#     elif "bhand" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.2]))
#     elif "leap" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.15]))
#     elif "svh" in robot_name:
#         robot.set_pose(sapien.Pose([0, 0, -0.13]))

#     # Different robot loader may have different orders for joints
#     sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
#     retargeting_joint_names = retargeting.joint_names
#     retargeting_to_sapien = np.array(
#         [retargeting_joint_names.index(name) for name in sapien_joint_names]
#     ).astype(int)

#     while True:
#         try:
#             bgr = queue.get(timeout=5)
#             rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
#         except Empty:
#             logger.error(
#                 "Fail to fetch image from camera in 5 secs. Please check your web camera device."
#             )
#             return

#         _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
#         bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
#         cv2.imshow("realtime_retargeting_demo", bgr)
#         if cv2.waitKey(1) & 0xFF == ord("q"):
#             break

#         if joint_pos is None:
#             logger.warning(f"{hand_type} hand is not detected.")
#         else:
#             retargeting_type = retargeting.optimizer.retargeting_type
#             indices = retargeting.optimizer.target_link_human_indices
#             if retargeting_type == "POSITION":
#                 indices = indices
#                 ref_value = joint_pos[indices, :]
#             else:
#                 origin_indices = indices[0, :]
#                 task_indices = indices[1, :]
#                 ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
#             qpos = retargeting.retarget(ref_value)
#             robot.set_qpos(qpos[retargeting_to_sapien])

#         for _ in range(2):
#             viewer.render()


import multiprocessing
import time
from pathlib import Path
from queue import Empty
from typing import Optional

import cv2
import numpy as np
import sapien
import tyro
from loguru import logger
from sapien.asset import create_dome_envmap
from sapien.utils import Viewer

from dex_retargeting.constants import (
    RobotName,
    RetargetingType,
    HandType,
    get_default_config_path,
)
from dex_retargeting.retargeting_config import RetargetingConfig
from single_hand_detector import SingleHandDetector


def start_retargeting(queue: multiprocessing.Queue, robot_dir: str, config_path: str):
    RetargetingConfig.set_default_urdf_dir(str(robot_dir))
    logger.info(f"Start retargeting with config {config_path}")
    retargeting = RetargetingConfig.load_from_file(config_path).build()

    hand_type = "Right" if "right" in config_path.lower() else "Left"
    detector = SingleHandDetector(hand_type=hand_type, selfie=False)

    sapien.render.set_viewer_shader_dir("default")
    sapien.render.set_camera_shader_dir("default")

    config = RetargetingConfig.load_from_file(config_path)

    # Setup
    scene = sapien.Scene()
    render_mat = sapien.render.RenderMaterial()
    render_mat.base_color = [0.06, 0.08, 0.12, 1]
    render_mat.metallic = 0.0
    render_mat.roughness = 0.9
    render_mat.specular = 0.8
    scene.add_ground(-0.2, render_material=render_mat, render_half_size=[1000, 1000])

    # Lighting
    scene.add_directional_light(np.array([1, 1, -1]), np.array([3, 3, 3]))
    scene.add_point_light(np.array([2, 2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.add_point_light(np.array([2, -2, 2]), np.array([2, 2, 2]), shadow=False)
    scene.set_environment_map(
        create_dome_envmap(sky_color=[0.2, 0.2, 0.2], ground_color=[0.2, 0.2, 0.2])
    )
    scene.add_area_light_for_ray_tracing(
        sapien.Pose([2, 1, 2], [0.707, 0, 0.707, 0]), np.array([1, 1, 1]), 5, 5
    )

    # Camera
    cam = scene.add_camera(
        name="Cheese!", width=600, height=600, fovy=1, near=0.1, far=10
    )
    cam.set_local_pose(sapien.Pose([0.50, 0, 0.0], [0, 0, 0, -1]))

    viewer = Viewer()
    viewer.set_scene(scene)
    viewer.control_window.show_origin_frame = False
    viewer.control_window.move_speed = 0.01
    viewer.control_window.toggle_camera_lines(False)
    viewer.set_camera_pose(cam.get_local_pose())

    # Load robot
    loader = scene.create_urdf_loader()
    filepath = Path(config.urdf_path)
    robot_name = filepath.stem
    loader.load_multiple_collisions_from_file = True
    
    # Scale setting
    if "ability" in robot_name: loader.scale = 1.5
    elif "dclaw" in robot_name: loader.scale = 1.25
    elif "allegro" in robot_name: loader.scale = 1.4
    elif "shadow" in robot_name: loader.scale = 0.9
    elif "bhand" in robot_name: loader.scale = 1.5
    elif "leap" in robot_name: loader.scale = 1.4
    elif "svh" in robot_name: loader.scale = 1.5
    

    if "glb" not in robot_name:
        glb_path = str(filepath).replace(".urdf", "_glb.urdf")
        if Path(glb_path).exists():
            filepath = glb_path
        else:
            filepath = str(filepath) # Keep original if glb missing
    else:
        filepath = str(filepath)

    try:
        robot = loader.load(filepath)
    except Exception as e:
        logger.error(f"Failed to load robot from {filepath}: {e}")
        return

    # Set Pose
    if "ability" in robot_name: robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "shadow" in robot_name: robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "dclaw" in robot_name: robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "allegro" in robot_name: robot.set_pose(sapien.Pose([0, 0, -0.05]))
    elif "bhand" in robot_name: robot.set_pose(sapien.Pose([0, 0, -0.2]))
    elif "leap" in robot_name: robot.set_pose(sapien.Pose([0, 0, -0.15]))
    elif "svh" in robot_name: robot.set_pose(sapien.Pose([0, 0, -0.13]))

    # Joint Mapping
    sapien_joint_names = [joint.get_name() for joint in robot.get_active_joints()]
    retargeting_joint_names = retargeting.joint_names
    retargeting_to_sapien = np.array(
        [retargeting_joint_names.index(name) for name in sapien_joint_names]
    ).astype(int)

    while True:
        try:
            bgr = queue.get(timeout=5)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Empty:
            logger.error("Fail to fetch image from camera.")
            return

        _, joint_pos, keypoint_2d, _ = detector.detect(rgb)
        bgr = detector.draw_skeleton_on_image(bgr, keypoint_2d, style="default")
        cv2.imshow("realtime_retargeting_demo", bgr)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        if joint_pos is None:
            logger.warning(f"{hand_type} hand is not detected.")
        else:
            retargeting_type = retargeting.optimizer.retargeting_type
            indices = retargeting.optimizer.target_link_human_indices
            
            # Prepare Input
            if retargeting_type == "POSITION":
                indices = indices
                ref_value = joint_pos[indices, :]
            else:
                origin_indices = indices[0, :]
                task_indices = indices[1, :]
                ref_value = joint_pos[task_indices, :] - joint_pos[origin_indices, :]
            
            # 1. Run Optimizer
            qpos = retargeting.retarget(ref_value)

            

##################################################################################
            try:
                joint_names = retargeting.joint_names
                
                # ------------------------------------------------------
                # 1. 普通四指的回零 (阈值 0.15, 约 8 度)
                # ------------------------------------------------------
                for i, name in enumerate(joint_names):
                    # 跳过拇指，因为我们要下面单独处理它
                    if "thumb" in name: 
                        continue
                        
                    if "MCP" in name or "PIP" in name or "DIP" in name:
                        if qpos[i] < 0.15:
                            qpos[i] = 0.0
                        else:
                            qpos[i] -= 0.15 # 平滑减去阈值

                # ------------------------------------------------------
                # 2. 拇指的特殊处理 (逻辑核心)
                # ------------------------------------------------------
                thumb_pip_name = "right_thumb_PIP_joint"
                thumb_dip_name = "right_thumb_DIP_joint"
                
                if thumb_pip_name in joint_names and thumb_dip_name in joint_names:
                    p_idx = joint_names.index(thumb_pip_name)
                    d_idx = joint_names.index(thumb_dip_name)
                    
                    pip_val = qpos[p_idx]
                    
                    # 【关键点】拇指的“超强死区”
                    # 设置为 0.4 (约 23 度)。
                    # 意思是：只要优化器算出来小于 23 度，我都认为它是直的。
                    thumb_deadzone = 0.4 
                    
                    if pip_val < thumb_deadzone:
                        # 强制拉直 PIP 和 DIP
                        qpos[p_idx] = 0.0
                        qpos[d_idx] = 0.0
                    else:
                        # 超过阈值后，恢复“软件耦合”
                        # 减去阈值，保证从 0 开始平滑弯曲
                        corrected_pip = pip_val - thumb_deadzone
                        qpos[p_idx] = corrected_pip
                        
                        # 强制 DIP 跟随 (系数 1.3 让指尖扣得紧)
                        qpos[d_idx] = corrected_pip * 1.3
                        
            except Exception as e:
                print(f"Error in custom optimization: {e}")
            # ==========================================================

            # 2. Set Robot
            robot.set_qpos(qpos[retargeting_to_sapien])

        for _ in range(2):
            viewer.render()


def produce_frame(queue: multiprocessing.Queue, camera_path: Optional[str] = None):
    if camera_path is None:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(camera_path)

    while cap.isOpened():
        success, image = cap.read()
        time.sleep(1 / 30.0)
        if not success:
            continue
        queue.put(image)


def main(
    robot_name: RobotName,
    retargeting_type: RetargetingType,
    hand_type: HandType,
    camera_path: Optional[str] = None,
):
    """
    Detects the human hand pose from a video and translates the human pose trajectory into a robot pose trajectory.

    Args:
        robot_name: The identifier for the robot. This should match one of the default supported robots.
        retargeting_type: The type of retargeting, each type corresponds to a different retargeting algorithm.
        hand_type: Specifies which hand is being tracked, either left or right.
            Please note that retargeting is specific to the same type of hand: a left robot hand can only be retargeted
            to another left robot hand, and the same applies for the right hand.
        camera_path: the device path to feed to opencv to open the web camera. It will use 0 by default.
    """
    config_path = get_default_config_path(robot_name, retargeting_type, hand_type)
    robot_dir = (
        Path(__file__).absolute().parent.parent.parent / "assets" / "robots" / "hands"
    )

    queue = multiprocessing.Queue(maxsize=1000)
    producer_process = multiprocessing.Process(
        target=produce_frame, args=(queue, camera_path)
    )
    consumer_process = multiprocessing.Process(
        target=start_retargeting, args=(queue, str(robot_dir), str(config_path))
    )

    producer_process.start()
    consumer_process.start()

    producer_process.join()
    consumer_process.join()
    time.sleep(5)

    print("done")


if __name__ == "__main__":
    tyro.cli(main)
