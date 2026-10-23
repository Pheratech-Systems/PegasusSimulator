"""
| File: vehicle.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Definition of the Vehicle class which is used as the base for all the vehicles.
"""

# Numerical computations
import numpy as np
from scipy.spatial.transform import Rotation

# Low level APIs
import carb
from pxr import Usd, Gf, UsdGeom

# High level Isaac sim APIs
import omni.usd
from omni.usd import get_stage_next_free_path
import isaacsim.core.experimental.utils.app as app_utils
import isaacsim.core.experimental.utils.stage as stage_utils
from isaacsim.core.simulation_manager import SimulationManager, SimulationEvent
from isaacsim.core.rendering_manager import RenderingManager, RenderingEvent
from isaacsim.core.experimental.prims import RigidPrim

# Extension APIs
from pegasus.simulator.logic.state import State
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface
from pegasus.simulator.logic.vehicle_manager import VehicleManager


def get_world_transform_xform(prim: Usd.Prim):
    """
    Get the local transformation of a prim using omni.usd.get_world_transform_matrix().
    See https://docs.omniverse.nvidia.com/kit/docs/omni.usd/latest/omni.usd/omni.usd.get_world_transform_matrix.html
    Args:
        prim (Usd.Prim): The prim to calculate the world transformation.
    Returns:
        A tuple of:
        - Translation vector.
        - Rotation quaternion, i.e. 3d vector plus angle.
        - Scale vector.
    """
    world_transform: Gf.Matrix4d = omni.usd.get_world_transform_matrix(prim)
    rotation: Gf.Rotation = world_transform.ExtractRotation()
    return rotation


class Vehicle:

    def __init__(
        self,
        stage_prefix: str,
        usd_path: str = None,
        init_pos=[0.0, 0.0, 0.0],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        sensors=[],
        graphical_sensors=[],
        graphs=[],
        backends=[]
    ):
        """
        Class that initializes a vehicle in the isaac sim's curent stage

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator when spawned. Defaults to "quadrotor".
            usd_path (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.0].
            init_orientation (list): The initial orientation of the vehicle in quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
        """

        # Initialize callback IDs to None so __del__ is safe even if __init__ fails partway
        self._cb_state = self._cb_update = self._cb_sensors = self._cb_sim_state = None
        self._cb_sim_start = self._cb_sim_stop = self._cb_render = None

        self._body_rigid_prim: RigidPrim | None = None
        self._body_rigid_prim_validated = False
        self._rigid_prims: dict[str, RigidPrim] = {}
        # Per-vehicle scratch buffers; the physics callbacks use them synchronously.
        self._force_buffer = np.empty((1, 3), dtype=np.float32)
        self._torque_buffer = np.empty((1, 3), dtype=np.float32)
        self._force_position_buffer = np.empty((1, 3), dtype=np.float32)

        # Get the current stage
        self._current_stage = omni.usd.get_context().get_stage()

        # Save the name with which the vehicle will appear in the stage
        # and the name of the .usd file that contains its description
        self._stage_prefix = get_stage_next_free_path(self._current_stage, stage_prefix, False)
        self._usd_file = usd_path

        # Get the vehicle name by taking the last part of vehicle stage prefix
        self._vehicle_name = self._stage_prefix.rpartition("/")[-1]

        # Spawn the vehicle primitive in the world's stage
        self._prim = stage_utils.define_prim(self._stage_prefix, "Xform")
        self._prim = self._current_stage.GetPrimAtPath(self._stage_prefix)
        self._prim.GetReferences().AddReference(self._usd_file)

        # Set the initial pose via the Xform API
        xformable = UsdGeom.Xformable(self._prim)
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(init_pos[0], init_pos[1], init_pos[2]))
        # Convert [qx, qy, qz, qw] → [qw, qx, qy, qz] for USD; use Quatf (single precision) to match the op type in NVIDIA USD assets
        xformable.AddOrientOp().Set(Gf.Quatf(init_orientation[3], init_orientation[0], init_orientation[1], init_orientation[2]))

        # Add the current vehicle to the vehicle manager, so that it knows
        # that a vehicle was instantiated
        VehicleManager.get_vehicle_manager().add_vehicle(self._stage_prefix, self)

        # Variable that will hold the current state of the vehicle
        self._state = State()

        # Set the flag that signals if the simulation is running or not
        self._sim_running = False

        # Register simulation start/stop callbacks
        self._cb_sim_start = SimulationManager.register_callback(self._on_sim_started, SimulationEvent.SIMULATION_STARTED)
        self._cb_sim_stop = SimulationManager.register_callback(self._on_sim_stopped, SimulationEvent.SIMULATION_STOPPED)

        # --------------------------------------------------------------------
        # -------------------- Add sensors to the vehicle --------------------
        # --------------------------------------------------------------------
        self._sensors = sensors

        for sensor in self._sensors:
            sensor.initialize(self, PegasusInterface().latitude, PegasusInterface().longitude, PegasusInterface().altitude)

        # --------------------------------------------------------------------
        # -------------------- Add the graphical sensors to the vehicle ------
        # --------------------------------------------------------------------
        self._graphical_sensors = graphical_sensors

        for graphical_sensor in self._graphical_sensors:
            graphical_sensor.initialize(self)

        # Avoid one render-frame callback per vehicle when there is no camera,
        # lidar, or other graphical sensor to update.
        if self._graphical_sensors:
            self._cb_render = RenderingManager.register_callback(
                RenderingEvent.NEW_FRAME,
                callback=self.update_graphical_sensors,
            )

        # --------------------------------------------------------------------
        # -------------------- Add the graphs to the vehicle -----------------
        # --------------------------------------------------------------------
        self._graphs = graphs

        for graph in self._graphs:
            graph.initialize(self)

        # --------------------------------------------------------------------
        # ---- Add (communication/control) backends to the vehicle -----------
        # --------------------------------------------------------------------
        self._backends = backends

        # Initialize the backends
        for backend in self._backends:
            backend.initialize(self)

        # Use one physics callback per vehicle while preserving the original
        # state -> dynamics -> sensors -> backend-state execution order.
        self._cb_update = SimulationManager.register_callback(
            self._physics_step,
            SimulationEvent.PHYSICS_POST_STEP,
        )


    def __del__(self):
        """
        Method that is invoked when a vehicle object gets destroyed. When this happens, we also invoke the
        'remove_vehicle' from the VehicleManager in order to remove the vehicle from the list of active vehicles.
        """
        # Deregister all SimulationManager callbacks
        for name in ("_cb_state", "_cb_update", "_cb_sensors", "_cb_sim_state", "_cb_sim_start", "_cb_sim_stop"):
            cb_id = getattr(self, name, None)
            if cb_id is not None:
                try:
                    SimulationManager.deregister_callback(cb_id)
                except Exception:
                    pass
                setattr(self, name, None)
        # Deregister RenderingManager callback
        if getattr(self, "_cb_render", None) is not None:
            try:
                RenderingManager.deregister_callback(self._cb_render)
            except Exception:
                pass
            self._cb_render = None

        # Remove this object from the vehicleHandler
        stage_prefix = getattr(self, "_stage_prefix", None)
        if stage_prefix is not None:
            try:
                VehicleManager.get_vehicle_manager().remove_vehicle(stage_prefix)
            except Exception:
                # The manager may already be shutting down.
                pass

    """
    Properties
    """

    @property
    def prim_path(self) -> str:
        """The USD stage path of this vehicle.

        Returns:
            str: Stage prefix string.
        """
        return self._stage_prefix

    @property
    def state(self):
        """The state of the vehicle.

        Returns:
            State: The current state of the vehicle, i.e., position, orientation, linear and angular velocities...
        """
        return self._state

    @property
    def vehicle_name(self) -> str:
        """Vehicle name.

        Returns:
            Vehicle name (str): last prim name in vehicle prim path
        """
        return self._vehicle_name

    """
    Operations
    """

    def _on_sim_started(self, *args):
        """Callback invoked when the simulation starts (after the first physics warm-up step)."""
        if not self._sim_running:
            self._sim_running = True

            # Create (or re-create) the rigid prim wrapper now that physics is ready
            self._rigid_prims.clear()
            body_path = self._stage_prefix + "/body"
            self._body_rigid_prim = RigidPrim(paths=body_path)
            self._body_rigid_prim_validated = False
            self._rigid_prims[body_path] = self._body_rigid_prim

            # Initialize the sensors
            for sensor in self._sensors:
                sensor.start()

            # Initialize the graphical sensors
            for graphical_sensor in self._graphical_sensors:
                graphical_sensor.start()

            # Initialize the communication backends
            for backend in self._backends:
                backend.start()

            self.start()

    def _on_sim_stopped(self, *args):
        """Callback invoked when the simulation stops."""
        if self._sim_running:
            self._sim_running = False

            # Release the rigid prim wrapper so it can be re-created fresh on next play
            self._body_rigid_prim = None
            self._body_rigid_prim_validated = False
            self._rigid_prims.clear()

            # Stop the sensors
            for sensor in self._sensors:
                sensor.stop()

            # Stop the graphical sensors
            for graphical_sensor in self._graphical_sensors:
                graphical_sensor.stop()

            # Signal all the backends that the simulation has stopped
            for backend in self._backends:
                backend.stop()

            self.stop()

    def _get_rigid_prim(self, body_part):
        """Reuse physics wrappers until stop/start or an invalid physics entity requires rebuilding."""
        if self._body_rigid_prim is None:
            return None

        prim_path = self._stage_prefix + body_part
        rigid_prim = self._rigid_prims.get(prim_path)
        # Tensor entities are stable for a play/stop lifecycle. Validate each
        # wrapper until it is ready, then rely on _on_sim_stopped() to clear
        # the cache before the next lifecycle.
        needs_wrapper = rigid_prim is None
        if (
            body_part == "/body"
            and rigid_prim is not None
            and not self._body_rigid_prim_validated
        ):
            if rigid_prim.is_physics_tensor_entity_valid():
                self._body_rigid_prim_validated = True
            else:
                needs_wrapper = True

        if needs_wrapper:
            rigid_prim = RigidPrim(paths=prim_path)
            if not rigid_prim.is_physics_tensor_entity_valid():
                self._rigid_prims.pop(prim_path, None)
                return None
            self._rigid_prims[prim_path] = rigid_prim
            if body_part == "/body":
                self._body_rigid_prim = rigid_prim
                self._body_rigid_prim_validated = True
        return rigid_prim

    def apply_force(self, force, pos=[0.0, 0.0, 0.0], body_part="/body"):
        """
        Method that will apply a force on the rigidbody, on the part specified in the 'body_part' at its relative position
        given by 'pos' (following a FLU) convention.

        Args:
            force (list): A 3-dimensional vector of floats with the force [Fx, Fy, Fz] on the body axis of the vehicle according to a FLU convention.
            pos (list): Position offset in local body frame. Defaults to [0.0, 0.0, 0.0].
            body_part (str): Rigid body sub-path relative to the vehicle stage prefix. Defaults to "/body".
        """
        rigid_prim = self._get_rigid_prim(body_part)
        if rigid_prim is None:
            return

        self._force_buffer[0] = force
        self._force_position_buffer[0] = pos
        rigid_prim.apply_forces_and_torques_at_pos(
            forces=self._force_buffer, positions=self._force_position_buffer, local_frame=True
        )

    def apply_torque(self, torque, body_part="/body"):
        """
        Method that when invoked applies a given torque vector to /<rigid_body_name>/"body" or to /<rigid_body_name>/<body_part>.

        Args:
            torque (list): A 3-dimensional vector of floats with the torque [Tx, Ty, Tz] on the body axis of the vehicle according to a FLU convention.
            body_part (str): Rigid body sub-path relative to the vehicle stage prefix. Defaults to "/body".
        """
        rigid_prim = self._get_rigid_prim(body_part)
        if rigid_prim is None:
            return

        self._torque_buffer[0] = torque
        rigid_prim.apply_forces_and_torques_at_pos(torques=self._torque_buffer, local_frame=True)

    def update_state(self, dt: float, context=None):
        """
        Method that is called at every physics step to retrieve and update the current state of the vehicle, i.e., get
        the current position, orientation, linear and angular velocities and acceleration of the vehicle.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
            context: Physics step context (unused, required by SimulationManager callback signature).
        """
        if self._body_rigid_prim is None:
            return

        # Get the current position and orientation from the body rigid prim
        # get_world_poses() returns (positions wp.array Nx3, orientations wp.array Nx4 in wxyz)
        positions, orientations = self._body_rigid_prim.get_world_poses()
        pos = positions.numpy()[0]        # [x, y, z]
        quat_wxyz = orientations.numpy()[0]  # [qw, qx, qy, qz]

        # Get the linear and angular velocities
        # get_velocities() returns (linear wp.array Nx3, angular wp.array Nx3) both in world frame
        linear_vels, angular_vels = self._body_rigid_prim.get_velocities()
        linear_vel = linear_vels.numpy()[0]   # [vx, vy, vz] in world frame
        ang_vel_world = angular_vels.numpy()[0]  # [wx, wy, wz] in world frame, rad/s

        # Convert warp float32 outputs to float64 so ROS 2 message fields pass PyFloat_Check
        pos = pos.astype(np.float64)
        linear_vel = linear_vel.astype(np.float64)
        ang_vel_world = ang_vel_world.astype(np.float64)

        # Get the linear acceleration of the body relative to the inertial frame, expressed in the inertial frame
        linear_acceleration = (linear_vel - self._state.linear_velocity) / dt if dt > 0.0 else np.zeros(3)

        # Update the state variable X = [x,y,z]
        self._state.position = np.array(pos)

        # Get the quaternion in the [qx, qy, qz, qw] standard
        self._state.attitude = np.array(
            [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]],
            dtype=np.float64,
        )

        # Express the velocity of the vehicle in the inertial frame X_dot = [x_dot, y_dot, z_dot]
        self._state.linear_velocity = np.array(linear_vel)

        # Reuse the same inertial-to-body rotation for both vectors. This is
        # mathematically identical for the PhysX tensor attitude above.
        inertial_to_body = Rotation.from_quat(self._state.attitude).inv()
        self._state.linear_body_velocity = inertial_to_body.apply(
            self._state.linear_velocity
        )

        # omega = [p, q, r] expressed in body frame
        self._state.angular_velocity = inertial_to_body.apply(
            np.array(ang_vel_world)
        )

        # The acceleration of the vehicle expressed in the inertial frame
        self._state.linear_acceleration = linear_acceleration

    def _physics_step(self, dt: float, context=None):
        """Run the vehicle's per-step work through one physics callback."""
        self.update_state(dt, context)
        self.update(dt, context)
        self.update_sensors(dt, context)
        self.update_sim_state(dt, context)

    def start(self):
        """
        Method that should be implemented by the class that inherits the vehicle object.
        """
        pass

    def stop(self):
        """
        Method that should be implemented by the class that inherits the vehicle object.
        """
        pass

    def update(self, dt: float, context=None):
        """
        Method that computes and applies the forces to the vehicle in
        simulation based on the motor speed. This method must be implemented
        by a class that inherits this type and it's called periodically by the physics engine.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
            context: Physics step context (unused, required by SimulationManager callback signature).
        """
        pass

    def update_sensors(self, dt: float, context=None):
        """Callback that is called at every physics steps and will call the sensor.update method to generate new
        sensor data. For each data that the sensor generates, the backend.update_sensor method will also be called for
        every backend.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
            context: Physics step context (unused, required by SimulationManager callback signature).
        """

        # Call the update method for the sensor to update its values internally (if applicable)
        for sensor in self._sensors:
            sensor_data = sensor.update(self._state, dt)

            # If some data was updated and we have a mavlink backend or ros backend (or other), then just update it
            if sensor_data is not None:
                for backend in self._backends:
                    backend.update_sensor(sensor.sensor_type, sensor_data)

    def update_graphical_sensors(self, event):
        """Callback that is called at every rendering steps and will call the graphical_sensor.update method to generate new
        sensor data.

        Args:
            event: Rendering event from RenderingManager.
        """
        # Extract dt from the event payload if available, otherwise use a sensible default
        dt = 1.0 / 60.0
        if hasattr(event, 'payload') and isinstance(event.payload, dict):
            dt = event.payload.get('dt', dt)

        for sensor in self._graphical_sensors:
            sensor_data = sensor.update(self._state, dt)

            # If some data was updated and we have a ros backend (or other), then just update it
            if sensor_data is not None:
                for backend in self._backends:
                    backend.update_graphical_sensor(sensor.sensor_type, sensor_data)

    def update_sim_state(self, dt: float, context=None):
        """
        Callback that is used to "send" the current state for each backend being used to control the vehicle. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
            context: Physics step context (unused, required by SimulationManager callback signature).
        """
        for backend in self._backends:
            backend.update_state(self._state)
