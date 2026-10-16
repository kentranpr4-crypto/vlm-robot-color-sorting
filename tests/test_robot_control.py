"""Run with: python -m unittest discover -s tests -v.

All robot and object behavior uses real PyBullet DIRECT physics. Only CLIP
loading is bypassed in robot tests, so these tests need no model download.
"""
import unittest
from unittest.mock import patch

import numpy as np
import pybullet as p

from simple_pick_place_demo import PandaRobotSimulation


class RobotControlTests(unittest.TestCase):
    def setUp(self):
        self.sim = PandaRobotSimulation(gui_mode=False)
        self.sim.setup_simulation()

    def tearDown(self):
        self.sim.cleanup()

    def test_physical_pick_and_place(self):
        sim = self.sim
        before = sim._object_position(sim.cube_id)
        self.assertTrue(sim._pick_body(sim.cube_id), sim.last_error)
        self.assertGreater(sim._object_position(sim.cube_id)[2], before[2] + 0.08)
        self.assertTrue(sim._is_grasping(sim.cube_id))
        for index in range(p.getNumConstraints(physicsClientId=sim.physics_client)):
            constraint = p.getConstraintUniqueId(index, physicsClientId=sim.physics_client)
            info = p.getConstraintInfo(constraint, physicsClientId=sim.physics_client)
            self.assertEqual((info[0], info[2]), (sim.robot_id, sim.robot_id))
        self.assertTrue(sim._place_body(sim.place_position[:2], sim.table_id), sim.last_error)
        self.assertLess(np.linalg.norm(sim._object_position(sim.cube_id)[:2]
                                       - np.array(sim.place_position[:2])), 0.04)
        self.assertIsNone(sim.held_object_id)

    def test_open_fingers_cannot_report_a_successful_pick(self):
        sim = self.sim
        original = sim.control_gripper
        # Simulate a gripper that fails to close; arm and object still use physics.
        with patch.object(sim, 'control_gripper',
                          side_effect=lambda _: original(sim.gripper_open_position)):
            self.assertFalse(sim._pick_body(sim.cube_id))
        self.assertIsNone(sim.held_object_id)
        self.assertLess(sim._object_position(sim.cube_id)[2], sim.table_height + 0.04)

    def test_unreachable_pose_and_empty_placement_fail(self):
        self.assertFalse(self.sim.move_to_position([3, 0, 2]))
        self.assertFalse(self.sim._place_body([0.3, 0.3], self.sim.table_id))

    def test_contacts_without_a_physical_lift_are_not_success(self):
        sim = self.sim
        original_position = sim._object_position(sim.cube_id)
        # An anchored cube can touch both fingers but cannot be lifted.
        p.changeDynamics(sim.cube_id, -1, mass=0, physicsClientId=sim.physics_client)
        self.assertFalse(sim._pick_body(sim.cube_id))
        np.testing.assert_allclose(sim._object_position(sim.cube_id), original_position, atol=0.001)
        self.assertIsNone(sim.held_object_id)

    def test_gripper_stepping_and_joint_mapping(self):
        sim = self.sim
        self.assertEqual(sim.arm_joints, list(range(7)))
        self.assertEqual(sim.finger_joints, [9, 10])
        self.assertEqual(sim.end_effector_link, 11)
        sim.control_gripper(0)
        closed = p.getJointStates(sim.robot_id, sim.finger_joints,
                                   physicsClientId=sim.physics_client)
        self.assertLess(sum(state[0] for state in closed), 0.003)
        self.assertTrue(sim.control_gripper(0.04))
        self.assertEqual(p.getPhysicsEngineParameters(
            physicsClientId=sim.physics_client)['fixedTimeStep'], 1 / 240)


class VisionRobotControlTests(unittest.TestCase):
    def setUp(self):
        from panda_vision_simulation import VisionLanguagePandaSimulation
        with patch.object(VisionLanguagePandaSimulation, '_initialize_clip'):
            self.sim = VisionLanguagePandaSimulation(gui_mode=False)
        self.sim.setup_simulation()
        self.sim._step(0.5)

    def tearDown(self):
        self.sim.cleanup()

    def test_all_objects_physically_sorted(self):
        sim = self.sim
        results = sim.run_color_sorting_demo(visualize=False)
        self.assertEqual(len(results), len(sim.objects))
        self.assertTrue(all(result['success'] for result in results), results)
        # Check the final scene again, including earlier objects after later picks.
        for name, info in sim.objects.items():
            zone_name = sim._get_target_zone_for_object(name)
            zone = sim.sorting_zones[zone_name]
            position = sim._object_position(info['id'])
            self.assertTrue(np.all(np.abs(position[:2] - zone['position'][:2])
                                   < np.array(zone['size'][:2]) / 2), name)
            self.assertTrue(p.getContactPoints(info['id'], sim.zone_ids[zone_name],
                                               physicsClientId=sim.physics_client), name)

    def test_unknown_object_and_unheld_object_fail(self):
        self.assertFalse(self.sim.pick_object('missing_object'))
        self.assertFalse(self.sim.place_object_in_sorting_zone('red_cube'))
        self.assertFalse(self.sim.place_object())

    def test_default_placement(self):
        self.assertTrue(self.sim.pick_object('blue_cube'), self.sim.last_error)
        self.assertTrue(self.sim.place_object(), self.sim.last_error)


if __name__ == '__main__':
    unittest.main()
