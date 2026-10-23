"""Focused migration regressions without starting Isaac Sim.

Run: python -m unittest discover -s tests -v
Requires NumPy. Production class methods are loaded via AST so Omniverse imports
and extension startup side effects do not require a simulator for these checks.
Physics calls are mocked; these tests do not establish actual flight behavior.
"""

import __future__
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOGIC = ROOT / "extensions/pegasus.simulator/pegasus/simulator/logic"


def load_class(relative_path, name, namespace, methods=None):
    path = LOGIC / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    cls.bases = []
    if methods is not None:
        cls.body = [node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    module = ast.fix_missing_locations(ast.Module(body=[cls], type_ignores=[]))
    exec(compile(module, str(path), "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


class FakeRigidPrim:
    instances = []
    ready = True

    def __init__(self, paths):
        self.paths = paths
        self.physics_valid = self.ready
        self.validity_checks = 0
        self.calls = []
        self.instances.append(self)

    def is_physics_tensor_entity_valid(self):
        self.validity_checks += 1
        return self.physics_valid

    def apply_forces_and_torques_at_pos(self, **kwargs):
        # Capture values at call time and identities to check buffer reuse separately.
        self.calls.append({key: value.copy() if isinstance(value, np.ndarray) else value
                           for key, value in kwargs.items()})
        self.buffer_ids = {key: id(value) for key, value in kwargs.items() if isinstance(value, np.ndarray)}


class VehicleTests(unittest.TestCase):
    def setUp(self):
        FakeRigidPrim.instances = []
        FakeRigidPrim.ready = True
        self.sim = Mock()
        self.render = Mock()
        self.manager = Mock()
        self.stage = Mock()
        self.prim = Mock()
        self.stage.GetPrimAtPath.return_value = self.prim
        self.xformable = Mock()
        omni = Mock()
        omni.usd.get_context.return_value.get_stage.return_value = self.stage
        self.namespace = dict(np=np, RigidPrim=FakeRigidPrim, SimulationManager=self.sim,
                              RenderingManager=self.render, VehicleManager=self.manager, omni=omni,
                              get_stage_next_free_path=lambda stage, prefix, _: prefix,
                              stage_utils=SimpleNamespace(define_prim=Mock()),
                              UsdGeom=SimpleNamespace(Xformable=Mock(return_value=self.xformable)),
                              Gf=SimpleNamespace(Vec3d=lambda *values: values,
                                                 Quatf=lambda *values: values),
                              State=Mock,
                              SimulationEvent=SimpleNamespace(
                                  PHYSICS_POST_STEP="physics",
                                  SIMULATION_STARTED="started",
                                  SIMULATION_STOPPED="stopped",
                              ),
                              RenderingEvent=SimpleNamespace(NEW_FRAME="frame"),
                              PegasusInterface=Mock)
        self.Vehicle = load_class("vehicles/vehicle.py", "Vehicle", self.namespace)

    def constructed_vehicle(self, graphical_sensors):
        return self.Vehicle(
            "/World/drone",
            usd_path="iris.usd",
            sensors=[],
            graphical_sensors=graphical_sensors,
            graphs=[],
            backends=[],
        )

    def vehicle(self, prefix="/World/drone"):
        vehicle = self.Vehicle.__new__(self.Vehicle)
        vehicle._stage_prefix = prefix
        vehicle._rigid_prims = {}
        vehicle._body_rigid_prim = None
        vehicle._force_buffer = np.empty((1, 3), dtype=np.float32)
        vehicle._torque_buffer = np.empty((1, 3), dtype=np.float32)
        vehicle._force_position_buffer = np.empty((1, 3), dtype=np.float32)
        vehicle._sensors = vehicle._graphical_sensors = vehicle._backends = []
        vehicle._sim_running = False
        vehicle._on_sim_started()
        return vehicle

    def test_multi_drone_wrappers_are_constant_after_warmup(self):
        vehicles = [self.vehicle(f"/World/drone{i}") for i in range(12)]
        for _ in range(100):
            for vehicle in vehicles:
                for rotor in range(4):
                    vehicle.apply_force([0, 0, 3], body_part=f"/rotor{rotor}")
                vehicle.apply_force([1, 0, 0])
                vehicle.apply_torque([0, 0, 0.2])
        self.assertEqual(len(FakeRigidPrim.instances), 12 * 5)
        self.assertEqual(len({id(v._force_buffer) for v in vehicles}), 12)
        self.assertEqual(len({id(v._rigid_prims[v._stage_prefix + '/rotor0']) for v in vehicles}), 12)

    def test_force_and_torque_share_wrapper_and_reuse_correct_buffers(self):
        vehicle = self.vehicle()
        vehicle.apply_force([1, 2, 3], [0.2, -0.1, 0.3], "/rotor0")
        rotor = vehicle._rigid_prims[vehicle._stage_prefix + "/rotor0"]
        first_ids = rotor.buffer_ids.copy()
        vehicle.apply_force([-1, -2, -3], body_part="/rotor0")
        self.assertEqual(first_ids, rotor.buffer_ids)
        np.testing.assert_allclose(rotor.calls[0]["positions"], [[0.2, -0.1, 0.3]])
        np.testing.assert_array_equal(rotor.calls[1]["positions"], [[0, 0, 0]])
        np.testing.assert_array_equal(rotor.calls[1]["forces"], [[-1, -2, -3]])
        self.assertNotIn("torques", rotor.calls[1])
        vehicle.apply_torque([4, 5, 6], "/rotor0")
        self.assertEqual(len(FakeRigidPrim.instances), 2)
        np.testing.assert_array_equal(rotor.calls[2]["torques"], [[4, 5, 6]])
        self.assertNotIn("forces", rotor.calls[2])
        self.assertTrue(all(call["local_frame"] for call in rotor.calls))
        self.assertEqual(vehicle._force_buffer.dtype, np.float32)

    def test_rotor_wrapper_is_validated_once_per_play_lifecycle(self):
        vehicle = self.vehicle()
        vehicle.apply_force([0, 0, 1], body_part="/rotor0")
        rotor = vehicle._rigid_prims[vehicle._stage_prefix + "/rotor0"]
        for _ in range(100):
            vehicle.apply_torque([0, 0, 1], "/rotor0")
            vehicle.apply_force([0, 0, 2], body_part="/rotor0")
        self.assertEqual(len(FakeRigidPrim.instances), 2)
        self.assertEqual(rotor.validity_checks, 1)

    def test_body_wrapper_is_validated_once_per_play_lifecycle(self):
        vehicle = self.vehicle()
        body = vehicle._body_rigid_prim
        for _ in range(100):
            vehicle.apply_force([0, 0, 1])
            vehicle.apply_torque([0, 0, 0.1])
        self.assertEqual(body.validity_checks, 1)
        self.assertEqual(len(FakeRigidPrim.instances), 1)
        self.assertTrue(vehicle._body_rigid_prim_validated)

    def test_body_wrapper_retries_until_replacement_is_ready(self):
        vehicle = self.vehicle()
        old_body = vehicle._body_rigid_prim
        old_body.physics_valid = False
        FakeRigidPrim.ready = False
        vehicle.apply_force([0, 0, 1])
        self.assertFalse(vehicle._body_rigid_prim_validated)
        self.assertNotIn(
            vehicle._stage_prefix + "/body",
            vehicle._rigid_prims,
        )

        FakeRigidPrim.ready = True
        vehicle.apply_force([0, 0, 2])
        self.assertIsNot(vehicle._body_rigid_prim, old_body)
        self.assertTrue(vehicle._body_rigid_prim_validated)
        self.assertEqual(len(vehicle._body_rigid_prim.calls), 1)

    def test_state_update_uses_tensor_attitude_and_one_rotation(self):
        rotation_api = Mock()
        rotation_api.from_quat.side_effect = (
            __import__("scipy.spatial.transform", fromlist=["Rotation"])
            .Rotation.from_quat
        )
        world_transform_query = Mock(
            side_effect=AssertionError("USD attitude query must not run")
        )
        self.namespace["Rotation"] = rotation_api
        self.namespace["get_world_transform_xform"] = world_transform_query

        array = lambda values: SimpleNamespace(
            numpy=lambda: np.asarray([values], dtype=np.float32)
        )
        body = Mock()
        body.get_world_poses.return_value = (
            array([1.0, 2.0, 3.0]),
            array([1.0, 0.0, 0.0, 0.0]),
        )
        body.get_velocities.return_value = (
            array([4.0, 5.0, 6.0]),
            array([0.1, 0.2, 0.3]),
        )

        vehicle = self.Vehicle.__new__(self.Vehicle)
        vehicle._body_rigid_prim = body
        vehicle._current_stage = Mock()
        vehicle._stage_prefix = "/World/drone"
        vehicle._state = SimpleNamespace(linear_velocity=np.zeros(3))
        vehicle.update_state(0.004)

        world_transform_query.assert_not_called()
        rotation_api.from_quat.assert_called_once()
        np.testing.assert_array_equal(
            vehicle._state.attitude,
            [0.0, 0.0, 0.0, 1.0],
        )
        np.testing.assert_allclose(
            vehicle._state.linear_body_velocity,
            [4.0, 5.0, 6.0],
        )
        np.testing.assert_allclose(
            vehicle._state.angular_velocity,
            [0.1, 0.2, 0.3],
        )

    def test_physics_dispatcher_preserves_callback_order_and_arguments(self):
        vehicle = self.Vehicle.__new__(self.Vehicle)
        calls = []
        vehicle.update_state = lambda dt, context: calls.append(("state", dt, context))
        vehicle.update = lambda dt, context: calls.append(("dynamics", dt, context))
        vehicle.update_sensors = lambda dt, context: calls.append(("sensors", dt, context))
        vehicle.update_sim_state = lambda dt, context: calls.append(("backend_state", dt, context))

        context = object()
        vehicle._physics_step(0.004, context)

        self.assertEqual(
            calls,
            [
                ("state", 0.004, context),
                ("dynamics", 0.004, context),
                ("sensors", 0.004, context),
                ("backend_state", 0.004, context),
            ],
        )

    def test_unready_replacement_is_not_used_or_cached(self):
        vehicle = self.vehicle()
        FakeRigidPrim.ready = False
        vehicle.apply_force([0, 0, 1], body_part="/rotor0")
        self.assertNotIn(vehicle._stage_prefix + "/rotor0", vehicle._rigid_prims)
        self.assertEqual(FakeRigidPrim.instances[-1].calls, [])

    def test_stop_start_rebuilds_handles_and_stopped_calls_do_nothing(self):
        vehicle = self.vehicle()
        vehicle.apply_force([0, 0, 1], body_part="/rotor0")
        old_body = vehicle._body_rigid_prim
        vehicle._on_sim_stopped()
        self.assertEqual(vehicle._rigid_prims, {})
        self.assertIsNone(vehicle._body_rigid_prim)
        vehicle.apply_force([0, 0, 1], body_part="/rotor0")
        vehicle.apply_torque([0, 0, 1])
        self.assertEqual(len(FakeRigidPrim.instances), 2)
        vehicle._on_sim_started()
        vehicle.apply_force([0, 0, 1], body_part="/rotor0")
        self.assertIsNot(vehicle._body_rigid_prim, old_body)
        self.assertEqual(len(FakeRigidPrim.instances), 4)

    def test_cleanup_handles_missing_fields_and_is_idempotent(self):
        vehicle = self.Vehicle.__new__(self.Vehicle)
        vehicle._cb_state = 0
        vehicle._cb_render = 7
        vehicle.__del__()
        vehicle.__del__()
        self.sim.deregister_callback.assert_called_once_with(0)
        self.render.deregister_callback.assert_called_once_with(7)
        self.manager.get_vehicle_manager.assert_not_called()

    def test_constructor_stage_failure_can_be_cleaned_up(self):
        vehicle = self.Vehicle.__new__(self.Vehicle)
        self.namespace["omni"].usd.get_context.side_effect = RuntimeError("stage unavailable")
        with self.assertRaisesRegex(RuntimeError, "stage unavailable"):
            vehicle.__init__("/World/drone")
        vehicle.__del__()
        self.manager.get_vehicle_manager.assert_not_called()

    def test_no_graphical_sensors_skip_render_callback(self):
        vehicle = self.constructed_vehicle([])
        self.assertIsNone(vehicle._cb_render)
        self.render.register_callback.assert_not_called()
        vehicle.__del__()

    def test_constructor_registers_one_physics_callback(self):
        vehicle = self.constructed_vehicle([])
        physics_callbacks = [
            call
            for call in self.sim.register_callback.call_args_list
            if call.args[1] == "physics"
        ]
        self.assertEqual(len(physics_callbacks), 1)
        self.assertEqual(physics_callbacks[0].args[0], vehicle._physics_step)
        self.assertIsNone(vehicle._cb_state)
        self.assertIsNone(vehicle._cb_sensors)
        self.assertIsNone(vehicle._cb_sim_state)
        vehicle.__del__()

    def test_graphical_sensor_still_registers_render_callback(self):
        self.render.register_callback.return_value = 17
        sensor = Mock()
        vehicle = self.constructed_vehicle([sensor])
        sensor.initialize.assert_called_once_with(vehicle)
        self.render.register_callback.assert_called_once_with(
            "frame",
            callback=vehicle.update_graphical_sensors,
        )
        self.assertEqual(vehicle._cb_render, 17)
        vehicle.__del__()


class FakeArticulation:
    instances = []

    def __init__(self, paths):
        self.paths = paths
        self.valid = False
        self.validity_checks = 0
        self.index_lookups = 0
        self.velocity_calls = []
        self.instances.append(self)

    def is_physics_tensor_entity_valid(self):
        self.validity_checks += 1
        return self.valid

    def get_dof_indices(self, joint_name):
        self.index_lookups += 1
        index = int(joint_name.removeprefix("joint"))
        return SimpleNamespace(numpy=lambda: np.array([index]))

    def set_dof_velocities(self, velocities, dof_indices):
        self.velocity_calls.append((velocities, dof_indices))


class MultirotorVisualTests(unittest.TestCase):
    def setUp(self):
        FakeArticulation.instances = []
        namespace = dict(Articulation=FakeArticulation)
        self.Multirotor = load_class(
            "vehicles/multirotor.py",
            "Multirotor",
            namespace,
            methods={
                "start",
                "stop",
                "update",
                "_get_rotor_dof_index",
                "_set_propeller_visuals",
                "handle_propeller_visual",
            },
        )
        self.Multirotor._PROPELLER_VISUAL_UPDATE_PERIOD = 1.0 / 60.0

    def test_articulation_retries_until_ready_then_caches_validity(self):
        drone = self.Multirotor()
        drone._stage_prefix = "/World/drone"
        drone._thrusters = SimpleNamespace(rot_dir=[1, -1, 1, -1])
        drone.start()
        articulation = FakeArticulation.instances[0]

        drone.handle_propeller_visual(0, 1.0)
        self.assertEqual(articulation.validity_checks, 1)
        self.assertEqual(articulation.velocity_calls, [])

        articulation.valid = True
        drone.handle_propeller_visual(0, 1.0)
        drone.handle_propeller_visual(0, 1.0)
        self.assertEqual(articulation.validity_checks, 2)
        self.assertEqual(articulation.index_lookups, 1)
        self.assertEqual(len(articulation.velocity_calls), 2)
        self.assertTrue(drone._articulation_validated)

        drone.stop()
        self.assertIsNone(drone._articulation)
        self.assertFalse(drone._articulation_validated)
        self.assertEqual(drone._rotor_dof_indices, {})

    def test_physics_update_batches_visual_targets_at_sixty_hz(self):
        drone = self.Multirotor()
        drone._stage_prefix = "/World/drone"
        drone._backends = []
        drone._thrusters = Mock()
        drone._thrusters._num_rotors = 4
        drone._thrusters.rot_dir = [1, -1, 1, -1]
        drone._thrusters.update.return_value = (
            [0.2, 0.2, 0.2, 0.2],
            None,
            0.05,
        )
        drone._drag = Mock()
        drone._drag.update.return_value = [0.0, 0.0, 0.0]
        drone._state = Mock()
        drone.apply_force = Mock()
        drone.apply_torque = Mock()
        drone._propeller_visual_elapsed = 0.0
        drone.start()
        articulation = FakeArticulation.instances[0]
        articulation.valid = True

        for _ in range(250):
            drone.update(0.004)

        self.assertEqual(drone.apply_force.call_count, 250 * 5)
        self.assertEqual(drone.apply_torque.call_count, 250)
        self.assertEqual(len(articulation.velocity_calls), 60)
        self.assertTrue(
            all(call[0] == [[100.0, -100.0, 100.0, -100.0]]
                for call in articulation.velocity_calls)
        )
        self.assertTrue(
            all(call[1] == [0, 1, 2, 3]
                for call in articulation.velocity_calls)
        )
        self.assertEqual(articulation.index_lookups, 4)


class PersonTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.client.Result.OK = "OK"
        self.client.ItemFlags.CAN_HAVE_CHILDREN = 1
        self.root = Mock(return_value="https://assets/Isaac/6.0")
        self.sim = Mock()
        self.log = Mock()
        self.namespace = dict(get_assets_root_path=self.root, carb=self.log,
                              omni=SimpleNamespace(client=self.client, usd=Mock()), SimulationManager=self.sim)
        self.Person = load_class("people/person.py", "Person", self.namespace)

    def test_missing_root_lists_nothing_without_client_call(self):
        self.root.return_value = None
        self.assertEqual(self.Person.get_character_asset_list(), [])
        self.client.list.assert_not_called()
        self.log.log_error.assert_called_once()

    def test_root_exception_is_reported_for_listing_and_chained_for_spawning(self):
        self.root.side_effect = OSError("server offline")
        self.assertEqual(self.Person.get_character_asset_list(), [])
        with self.assertRaises(RuntimeError) as error:
            self.Person.get_path_for_character_prim("worker")
        self.assertIsInstance(error.exception.__cause__, OSError)
        self.client.stat.assert_not_called()

    def test_successful_root_is_resolved_only_once(self):
        folder = lambda name, flags: SimpleNamespace(relative_path=name, flags=flags)
        self.client.list.return_value = ("OK", [folder("worker", 1), folder(".hidden", 1), folder("readme", 0)])
        self.assertEqual(self.Person.get_character_asset_list(), ["worker"])
        self.assertEqual(self.Person.get_character_asset_list(), ["worker"])
        self.root.assert_called_once()

    def test_configured_root_avoids_asset_discovery(self):
        self.Person.people_asset_folder = "https://custom/characters"
        self.Person._ensure_assets_root_path()
        self.assertEqual(self.Person.assets_root_path, "https://custom/characters")
        self.root.assert_not_called()

    def test_listing_failure_returns_an_empty_list(self):
        self.client.list.return_value = ("OFFLINE", [])
        self.assertEqual(self.Person.get_character_asset_list(), [])

    def test_missing_character_and_missing_usd_raise_clear_errors(self):
        self.client.stat.return_value = ("NOT_FOUND", None)
        with self.assertRaisesRegex(FileNotFoundError, "worker"):
            self.Person.get_path_for_character_prim("worker")
        self.client.stat.return_value = ("OK", None)
        self.client.list.return_value = ("OK", [SimpleNamespace(relative_path="readme.txt")])
        with self.assertRaisesRegex(FileNotFoundError, "No .usd"):
            self.Person.get_path_for_character_prim("worker")
        self.client.list.return_value = ("OFFLINE", [])
        with self.assertRaisesRegex(FileNotFoundError, "Unable to read"):
            self.Person.get_path_for_character_prim("worker")

    def test_valid_character_path_and_missing_name(self):
        self.client.stat.return_value = ("OK", None)
        self.client.list.return_value = ("OK", [SimpleNamespace(relative_path="worker.usd")])
        self.assertEqual(self.Person.get_path_for_character_prim("worker"),
                         "https://assets/Isaac/6.0/Isaac/People/Characters/worker/worker.usd")
        with self.assertRaises(ValueError):
            self.Person.get_path_for_character_prim(None)

    def test_cleanup_handles_partial_registration_and_continues_on_failure(self):
        person = self.Person.__new__(self.Person)
        person._cb_state = 0
        person._cb_update = 1
        self.sim.deregister_callback.side_effect = [RuntimeError("already removed"), None]
        person.__del__()
        person.__del__()
        self.assertEqual([call.args[0] for call in self.sim.deregister_callback.call_args_list], [0, 1])

    def test_constructor_failure_initializes_cleanup_fields(self):
        person = self.Person.__new__(self.Person)
        self.root.return_value = None
        with self.assertRaises(RuntimeError):
            person.__init__("worker", "worker")
        self.assertIsNone(person._cb_state)
        self.assertIsNone(person._cb_sim_stop)
        person.__del__()
        self.sim.deregister_callback.assert_not_called()


class RowMatrix:
    """Minimal row-vector transform adapter for exercising the production allocator."""
    def __init__(self, array):
        self.array = array

    def __mul__(self, other):
        return RowMatrix(self.array @ other.array)

    def GetInverse(self):
        return RowMatrix(np.linalg.inv(self.array))

    def ExtractTranslation(self):
        return self.array[3, :3]


class AllocationTests(unittest.TestCase):
    def test_rotor_allocation_is_invariant_under_world_translation_and_rotation(self):
        positions = [(0.2, 0.2), (-0.2, 0.2), (-0.2, -0.2), (0.2, -0.2)]
        matrices = {}
        stage = SimpleNamespace(GetPrimAtPath=lambda path: path)
        cache = SimpleNamespace(GetLocalToWorldTransform=lambda prim: matrices[prim])
        namespace = dict(np=np, omni=SimpleNamespace(usd=SimpleNamespace(
            get_context=lambda: SimpleNamespace(get_stage=lambda: stage))),
            UsdGeom=SimpleNamespace(XformCache=lambda: cache))
        Multirotor = load_class("vehicles/multirotor.py", "Multirotor", namespace,
                               methods={"force_and_torques_to_velocities"})
        drone = Multirotor()
        drone._stage_prefix = "/World/drone"
        drone._thrusters = SimpleNamespace(_num_rotors=4, _rotor_constant=[1.0] * 4,
            _rolling_moment_coefficient=[0.1] * 4, _rot_dir=[1, -1, 1, -1], max_rotor_velocity=[100.0])
        results = []
        for yaw, translation in [(0.0, [0, 0, 0]), (0.9, [12, -8, 3]), (-1.2, [-4, 7, 2])]:
            c, s = np.cos(yaw), np.sin(yaw)
            body = np.array([[c, s, 0, 0], [-s, c, 0, 0], [0, 0, 1, 0], [*translation, 1.0]])
            matrices["/World/drone/body"] = RowMatrix(body)
            for i, (x, y) in enumerate(positions):
                local = np.eye(4)
                local[3, :3] = [x, y, 0]
                matrices[f"/World/drone/rotor{i}"] = RowMatrix(local @ body)
            results.append(drone.force_and_torques_to_velocities(20.0, np.array([0.3, -0.2, 0.1])))
        allocation = np.array([[1, 1, 1, 1], [y for x, y in positions],
                               [-x for x, y in positions], [0.1, -0.1, 0.1, -0.1]])
        for result in results:
            np.testing.assert_allclose(allocation @ result ** 2, [20, 0.3, -0.2, 0.1], atol=1e-10)


if __name__ == "__main__":
    unittest.main()
