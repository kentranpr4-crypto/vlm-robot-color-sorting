#!/usr/bin/env python3
"""Physical Panda pick-and-place demo; use --headless for a DIRECT smoke test.

The vision simulation reuses this controller without importing CLIP here.
Motion targets are world coordinates of panda_grasptarget.
"""
import argparse
import time
import numpy as np
import pybullet as p
import pybullet_data


class PandaRobotSimulation:
    """Panda control with physical grasp and placement verification."""

    def __init__(self, gui_mode=True):
        self.gui_mode = gui_mode
        self.physics_client = p.connect(p.GUI if gui_mode else p.DIRECT)
        if self.physics_client < 0:
            raise RuntimeError("Could not connect to PyBullet")
        p.setAdditionalSearchPath(pybullet_data.getDataPath(),
                                  physicsClientId=self.physics_client)
        self.time_step = 1.0 / 240.0
        self.robot_id = self.table_id = self.cube_id = None
        self.gripper_open_position = 0.04
        self.gripper_close_position = 0.0
        self.pick_position = [0.5, 0.0, 0.65]
        self.place_position = [0.3, 0.3, 0.65]
        self.home_position = [0.3, 0.0, 0.95]
        self.held_object_id = None
        self.last_error = ""
        self._arm_target = None

    def _step(self, seconds):
        for _ in range(max(1, int(np.ceil(seconds / self.time_step)))):
            p.stepSimulation(physicsClientId=self.physics_client)
            if self.gui_mode:
                time.sleep(self.time_step)

    def _fail(self, reason):
        self.last_error = reason
        print(f"Robot control failed: {reason}")
        return False

    def _setup_world(self):
        client = self.physics_client
        p.setRealTimeSimulation(0, physicsClientId=client)
        p.setTimeStep(self.time_step, physicsClientId=client)
        p.setPhysicsEngineParameter(numSolverIterations=150, physicsClientId=client)
        p.setGravity(0, 0, -9.81, physicsClientId=client)
        p.loadURDF("plane.urdf", physicsClientId=client)
        self.table_id = p.loadURDF("table/table.urdf", [0.5, 0, 0],
                                   useFixedBase=True, physicsClientId=client)
        self.table_height = p.getAABB(self.table_id, physicsClientId=client)[1][2]
        self.robot_id = p.loadURDF("franka_panda/panda.urdf",
                                   [0, 0, self.table_height + 0.001],
                                   useFixedBase=True, physicsClientId=client)
        joints, links = {}, {}
        for index in range(p.getNumJoints(self.robot_id, physicsClientId=client)):
            info = p.getJointInfo(self.robot_id, index, physicsClientId=client)
            joints[info[1].decode()] = index
            links[info[12].decode()] = index
        self.arm_joints = [joints[f"panda_joint{i}"] for i in range(1, 8)]
        self.finger_joints = [joints["panda_finger_joint1"], joints["panda_finger_joint2"]]
        self.end_effector_link = links["panda_grasptarget"]
        self.arm_limits = [p.getJointInfo(self.robot_id, j, physicsClientId=client)[8:10]
                           for j in self.arm_joints]
        movable = self.arm_joints + self.finger_joints
        limits = [p.getJointInfo(self.robot_id, j, physicsClientId=client)[8:10]
                  for j in movable]
        self._ik_lower = [limit[0] for limit in limits]
        self._ik_upper = [limit[1] for limit in limits]
        self._ik_ranges = [hi - lo for lo, hi in limits]
        # Reset only the robot's startup pose, before stepping physics.
        # Manipulated objects are never reset, attached, or teleported.
        self.initial_joint_positions = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
        for joint, value in zip(self.arm_joints, self.initial_joint_positions):
            p.resetJointState(self.robot_id, joint, value, physicsClientId=client)
        for joint in self.finger_joints:
            p.resetJointState(self.robot_id, joint, 0.04, physicsClientId=client)
            p.changeDynamics(self.robot_id, joint, lateralFriction=1.5,
                             spinningFriction=0.01, frictionAnchor=1,
                             physicsClientId=client)
        # Couple only the robot's fingers. No constraint connects to an object.
        self.finger_gear = p.createConstraint(
            self.robot_id, self.finger_joints[0], self.robot_id, self.finger_joints[1],
            p.JOINT_GEAR, [1, 0, 0], [0, 0, 0], [0, 0, 0], physicsClientId=client)
        p.changeConstraint(self.finger_gear, gearRatio=-1, erp=0.1,
                           maxForce=50, physicsClientId=client)
        self._command_arm(self.initial_joint_positions)
        self.control_gripper(self.gripper_open_position)

    def setup_simulation(self):
        self._setup_world()
        self.create_cube()
        self.setup_camera()
        self._step(0.5)

    def create_cube(self):
        half_size = 0.025
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[half_size] * 3,
                                           physicsClientId=self.physics_client)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[half_size] * 3,
                                     rgbaColor=[1, 0, 0, 1],
                                     physicsClientId=self.physics_client)
        self.cube_id = p.createMultiBody(
            baseMass=0.1, baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
            basePosition=[*self.pick_position[:2], self.table_height + half_size + 0.002],
            physicsClientId=self.physics_client)
        self._configure_object(self.cube_id)

    def _configure_object(self, body):
        p.changeDynamics(body, -1, lateralFriction=1.0, spinningFriction=0.01,
                         rollingFriction=0.001, restitution=0,
                         physicsClientId=self.physics_client)

    def setup_camera(self):
        p.resetDebugVisualizerCamera(1.5, 45, -30, [0.4, 0, 0.65],
                                     physicsClientId=self.physics_client)

    def get_joint_info(self):
        for joint in range(p.getNumJoints(self.robot_id, physicsClientId=self.physics_client)):
            info = p.getJointInfo(self.robot_id, joint, physicsClientId=self.physics_client)
            print(f"Joint {joint}: {info[1].decode()} -> {info[12].decode()}")

    def _command_arm(self, positions):
        self._arm_target = list(positions[:7])
        for joint, target in zip(self.arm_joints, self._arm_target):
            p.setJointMotorControl2(self.robot_id, joint, p.POSITION_CONTROL,
                                    targetPosition=target, force=87, maxVelocity=1.2,
                                    positionGain=0.2, physicsClientId=self.physics_client)

    def set_initial_robot_pose(self):
        self._command_arm(self.initial_joint_positions)
        return self.wait_for_movement()

    def _ee_pose(self):
        # IK targets the link frame rather than its inertial COM frame.
        state = p.getLinkState(self.robot_id, self.end_effector_link,
                               computeForwardKinematics=True, physicsClientId=self.physics_client)
        return np.array(state[4]), state[5]

    def move_to_position(self, target_position, target_orientation=None):
        """Follow short Cartesian segments; return False on tracking failure."""
        target = np.asarray(target_position, dtype=float)
        if target.shape != (3,) or not np.all(np.isfinite(target)):
            return self._fail("Target must contain three finite world coordinates")
        orientation = (p.getQuaternionFromEuler([np.pi, 0, 0])
                       if target_orientation is None else target_orientation)
        start, _ = self._ee_pose()
        segments = max(1, int(np.ceil(np.linalg.norm(target - start) / 0.008)))
        for waypoint in np.linspace(start, target, segments + 1)[1:]:
            solution = p.calculateInverseKinematics(
                self.robot_id, self.end_effector_link, waypoint.tolist(), orientation,
                lowerLimits=self._ik_lower, upperLimits=self._ik_upper,
                jointRanges=self._ik_ranges,
                restPoses=self.initial_joint_positions + [0.04, 0.04],
                maxNumIterations=200, residualThreshold=1e-5, physicsClientId=self.physics_client)
            arm = solution[:7]
            if any(not lo - 0.01 <= value <= hi + 0.01
                   for value, (lo, hi) in zip(arm, self.arm_limits)):
                return self._fail(f"IK solution exceeds Panda joint limits at {waypoint.tolist()}: {list(arm)}")
            self._command_arm([np.clip(value, lo, hi)
                               for value, (lo, hi) in zip(arm, self.arm_limits)])
            self._step(0.05)
        if not self.wait_for_movement():
            return self._fail("Arm did not settle before the motion timeout")
        actual, actual_orientation = self._ee_pose()
        error = np.linalg.norm(actual - target)
        angular_error = 2 * np.arccos(np.clip(abs(np.dot(actual_orientation, orientation)), 0, 1))
        if error > 0.01 or angular_error > 0.12:
            return self._fail(f"Unreachable or blocked pose (position error {error:.3f} m)")
        return True

    def safe_move_to_position(self, target_position, target_orientation=None, min_height=0.9):
        """Lift vertically, traverse at clearance, then descend vertically."""
        current, _ = self._ee_pose()
        height = max(min_height, current[2], target_position[2])
        for waypoint in ([current[0], current[1], height],
                         [target_position[0], target_position[1], height], target_position):
            if not self.move_to_position(waypoint, target_orientation):
                return False
        return True

    def control_gripper(self, position):
        """Drive both fingers while stepping physics; position is per finger."""
        position = float(np.clip(position, 0, 0.04))
        for joint in self.finger_joints:
            p.setJointMotorControl2(self.robot_id, joint, p.POSITION_CONTROL,
                                    targetPosition=position, force=20, maxVelocity=0.04,
                                    positionGain=0.2, physicsClientId=self.physics_client)
        self._step(1.2)
        states = p.getJointStates(self.robot_id, self.finger_joints, physicsClientId=self.physics_client)
        if position >= 0.039:
            self.held_object_id = None
            return all(state[0] >= 0.038 for state in states)
        # Fingers stop on an object; tracking the command cannot verify a grasp.
        return True

    def wait_for_movement(self, timeout=5.0):
        settled = 0
        for _ in range(max(1, int(timeout / self.time_step))):
            self._step(self.time_step)
            states = p.getJointStates(self.robot_id, self.arm_joints, physicsClientId=self.physics_client)
            if (max(abs(s[0] - target) for s, target in zip(states, self._arm_target)) < 0.008
                    and max(abs(s[1]) for s in states) < 0.03):
                settled += 1
                if settled >= 12:
                    return True
            else:
                settled = 0
        return False

    def _object_position(self, body):
        return np.array(p.getBasePositionAndOrientation(body, physicsClientId=self.physics_client)[0])

    def _is_grasping(self, body):
        contacts = p.getContactPoints(self.robot_id, body, physicsClientId=self.physics_client)
        touching = {contact[3] for contact in contacts if contact[9] > 0.02}
        states = p.getJointStates(self.robot_id, self.finger_joints, physicsClientId=self.physics_client)
        gap = sum(state[0] for state in states)
        ee, _ = self._ee_pose()
        return (all(joint in touching for joint in self.finger_joints)
                and 0.005 < gap < 0.079 and np.linalg.norm(self._object_position(body) - ee) < 0.07)

    def _pick_body(self, body):
        if self.held_object_id is not None:
            return self._fail("Release the currently held object before another pick")
        self.last_error = ""
        if not self.safe_move_to_position(self.home_position):
            return False
        if not self.control_gripper(self.gripper_open_position):
            return self._fail("Gripper could not open")
        position = self._object_position(body)
        if not self.safe_move_to_position(position + [0, 0, 0.20]):
            return False
        position = self._object_position(body)
        if not self.move_to_position(position + [0, 0, 0.005]):
            return False
        self.control_gripper(self.gripper_close_position)
        if not self._is_grasping(body):
            print(f"Grasp miss: object {self._object_position(body).tolist()}, "
                  f"tool {self._ee_pose()[0].tolist()}")
            self.control_gripper(self.gripper_open_position)
            return self._fail("No opposing finger contacts on the target object")
        if not self.move_to_position(position + [0, 0, 0.20]):
            return False
        self._step(0.4)
        lifted = self._object_position(body)
        if lifted[2] < position[2] + 0.08 or not self._is_grasping(body):
            self.control_gripper(self.gripper_open_position)
            return self._fail("Target did not remain grasped after the lift")
        self.held_object_id = body
        self._grasp_offset = self._ee_pose()[0] - lifted
        print(f"Verified grasp: body {body}, lift {lifted[2] - position[2]:.3f} m")
        return True

    def _place_body(self, xy, support_body, xy_tolerance=0.04):
        body = self.held_object_id
        if body is None or not self._is_grasping(body):
            return self._fail("No physically verified held object to place")
        support_z = p.getAABB(support_body, physicsClientId=self.physics_client)[1][2]
        position = self._object_position(body)
        bottom = p.getAABB(body, physicsClientId=self.physics_client)[0][2]
        target = np.array([xy[0], xy[1], support_z + position[2] - bottom + 0.006])
        target += self._grasp_offset
        approach = target + [0, 0, 0.18]
        if not self.safe_move_to_position(approach):
            return False
        if not self._is_grasping(body):
            return self._fail("Object slipped during transport")
        if not self.move_to_position(target):
            return False
        if not self.control_gripper(self.gripper_open_position):
            return self._fail("Gripper could not release the object")
        if not self.move_to_position(approach):
            return False
        if not self.safe_move_to_position(self.home_position):
            return False
        self._step(1.0)
        final = self._object_position(body)
        bottom = p.getAABB(body, physicsClientId=self.physics_client)[0][2]
        linear, angular = p.getBaseVelocity(body, physicsClientId=self.physics_client)
        supported = p.getContactPoints(body, support_body, physicsClientId=self.physics_client)
        if (np.linalg.norm(final[:2] - np.asarray(xy)) > xy_tolerance
                or abs(bottom - support_z) > 0.012 or not supported
                or np.linalg.norm(linear) > 0.03 or np.linalg.norm(angular) > 0.5
                or self._is_grasping(body)):
            return self._fail(f"Object did not settle at the placement target: {final.tolist()}")
        print(f"Verified placement: body {body} at {final.tolist()}")
        return True

    def debug_positions(self):
        print(f"Table surface: {self.table_height:.3f} m")
        print(f"Cube: {self._object_position(self.cube_id).tolist()}")

    def pick_and_place_sequence(self):
        return self._pick_body(self.cube_id) and self._place_body(self.place_position[:2], self.table_id)

    def run_simulation(self, duration=30):
        self.setup_simulation()
        if not self.set_initial_robot_pose():
            return self._fail("Could not reach the initial configuration")
        success = self.pick_and_place_sequence()
        print("Pick and place verified." if success else "Pick and place failed.")
        if duration > 0:
            self._step(duration)
        return success

    def cleanup(self):
        if p.isConnected(self.physics_client):
            p.disconnect(physicsClientId=self.physics_client)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true", help="Use PyBullet DIRECT")
    parser.add_argument("--duration", type=float, default=10,
                        help="Simulated seconds to observe after the sequence")
    args = parser.parse_args()
    sim = PandaRobotSimulation(gui_mode=not args.headless)
    try:
        return 0 if sim.run_simulation(duration=args.duration) else 1
    except KeyboardInterrupt:
        return 130
    finally:
        sim.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
