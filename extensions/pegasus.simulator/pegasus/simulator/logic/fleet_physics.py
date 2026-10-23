"""Fleet-wide PhysX tensor batching for multi-vehicle simulations."""

__all__ = ["FleetPhysicsDispatcher"]

import numpy as np

from isaacsim.core.experimental.prims import RigidPrim


class FleetPhysicsDispatcher:
    """Batch rigid-body state reads and force writes across a vehicle fleet.

    Vehicle dynamics, sensors, and backends still execute at the configured
    physics rate. Only the PhysX tensor boundary is shared: four fleet state
    arrays are copied to the CPU and one fleet force/torque operation is sent
    back, instead of repeating those operations for every body on every drone.
    """

    def __init__(self, vehicles):
        self._vehicles = list(vehicles)
        if not self._vehicles:
            raise ValueError("Fleet physics batching requires at least one vehicle")

        self._body_paths = []
        self._force_paths = []
        self._force_index = {}
        for vehicle in self._vehicles:
            self._body_paths.append(vehicle.prim_path + "/body")
            for body_part in vehicle.get_physics_body_parts():
                index = len(self._force_paths)
                self._force_paths.append(vehicle.prim_path + body_part)
                self._force_index[(id(vehicle), body_part)] = index

        body_count = len(self._force_paths)
        self._forces = np.zeros((body_count, 3), dtype=np.float32)
        self._torques = np.zeros((body_count, 3), dtype=np.float32)
        self._positions = np.zeros((body_count, 3), dtype=np.float32)
        self._has_force_position = np.zeros(body_count, dtype=bool)
        self._body_view = None
        self._force_view = None
        self._views_validated = False

    @property
    def body_paths(self):
        return tuple(self._body_paths)

    @property
    def force_paths(self):
        return tuple(self._force_paths)

    def start(self, *args):
        """Create shared tensor views after PhysX has started."""
        self._body_view = RigidPrim(paths=self._body_paths)
        self._force_view = RigidPrim(paths=self._force_paths)
        self._views_validated = False

    def stop(self, *args):
        """Release tensor views at the end of a play lifecycle."""
        self._body_view = None
        self._force_view = None
        self._views_validated = False

    def _views_ready(self):
        if self._body_view is None or self._force_view is None:
            return False
        if not self._views_validated:
            if not self._body_view.is_physics_tensor_entity_valid():
                return False
            if not self._force_view.is_physics_tensor_entity_valid():
                return False
            self._views_validated = True
        return True

    def _index_for(self, vehicle, body_part):
        try:
            return self._force_index[(id(vehicle), body_part)]
        except KeyError as exc:
            raise ValueError(
                f"{vehicle.prim_path}{body_part} was not declared for fleet physics"
            ) from exc

    def queue_force(self, vehicle, body_part, force, position):
        """Accumulate a local-frame force for the current physics step."""
        index = self._index_for(vehicle, body_part)
        position = np.asarray(position, dtype=np.float32)
        if self._has_force_position[index] and not np.array_equal(
            self._positions[index], position
        ):
            raise ValueError(
                "Fleet batching cannot combine forces at different positions "
                f"on {vehicle.prim_path}{body_part}"
            )
        self._forces[index] += force
        self._positions[index] = position
        self._has_force_position[index] = True

    def queue_torque(self, vehicle, body_part, torque):
        """Accumulate a local-frame torque for the current physics step."""
        self._torques[self._index_for(vehicle, body_part)] += torque

    def physics_step(self, dt: float, context=None):
        """Run one exact fleet physics update with shared tensor operations."""
        if not self._views_ready():
            return

        positions, orientations = self._body_view.get_world_poses()
        linear_velocities, angular_velocities = self._body_view.get_velocities()

        # Four device-to-host transfers serve the entire fleet, rather than
        # four transfers per vehicle.
        positions = positions.numpy()
        orientations = orientations.numpy()
        linear_velocities = linear_velocities.numpy()
        angular_velocities = angular_velocities.numpy()

        self._forces.fill(0.0)
        self._torques.fill(0.0)
        self._positions.fill(0.0)
        self._has_force_position.fill(False)

        for index, vehicle in enumerate(self._vehicles):
            vehicle.update_state_from_physics(
                positions[index],
                orientations[index],
                linear_velocities[index],
                angular_velocities[index],
                dt,
            )

        for vehicle in self._vehicles:
            vehicle._physics_step_after_state(dt, context)

        # One host-to-device force/torque submission covers bodies and rotors.
        self._force_view.apply_forces_and_torques_at_pos(
            forces=self._forces,
            torques=self._torques,
            positions=self._positions,
            local_frame=True,
        )
