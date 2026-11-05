"""
| File: body_interface.py
| License: BSD-3-Clause.
| Description: The rigid-body calls Pegasus made through omni.isaac.dynamic_control, which Isaac Sim 6.x
| removed, re-implemented on omni.physics.tensors.

Only what the vehicles use: a body's world pose, its world-frame linear and angular velocity, and forces and
torques in the body's own frame. Forces are ACCUMULATED per body and applied once per physics step by
flush(): a tensor view's apply call sets the body's external force for the step, so a torque call followed by
a drag call on the same body would otherwise keep only the last one (dynamic_control accumulated them).

Views are created lazily, after physics has started, and are dropped with the interface when the timeline
stops (Vehicle sets its interface to None then), because a simulation view does not survive a stop. For the
first physics steps after play PhysX has no attached stage yet and a view cannot be made; ready() is False
then and the vehicle skips the step (dynamic_control returned invalid handles at that point, silently).
"""

import carb
import numpy as np

import omni.physics.tensors as tensors


class BodyPose:
    """dynamic_control's Transform: p = position (x, y, z), r = quaternion (x, y, z, w), world frame."""

    def __init__(self, p, r):
        self.p = p
        self.r = r


class BodyInterface:

    def __init__(self):
        self._sim_view = None
        self._views = {}
        self._attempts = 0
        # path -> [force (3,), torque (3,)], both in the body's own frame
        self._pending = {}

    # ------------------------------------------------------------------ views
    def ready(self) -> bool:
        """True once a simulation view exists, i.e. PhysX is attached to the stage. Never raises."""
        if self._sim_view is not None and self._sim_view.is_valid:
            return True
        self._attempts += 1
        try:
            # Isaac's SimulationManager makes the stage's physics view when it sets physics up (warm start);
            # a view made here before or apart from that finds no attached stage. Use its numpy view.
            from isaacsim.core.simulation_manager import SimulationManager
            sim_view = SimulationManager.get_physics_sim_view()
            if sim_view is None or not sim_view.is_valid:
                raise RuntimeError("SimulationManager has no valid physics view yet")
            if SimulationManager.get_backend() != "numpy":
                raise RuntimeError("SimulationManager backend is %r, body_interface needs numpy"
                                   % SimulationManager.get_backend())
        except Exception as e:
            # Said once, then every 2000 attempts: a view that never comes means a vehicle that never flies,
            # and that must not be silent (SITL then stalls waiting for its first physics frame).
            if self._attempts == 1 or self._attempts % 2000 == 0:
                carb.log_warn("[body_interface] no physics view yet (attempt %d): %s" % (self._attempts, e))
            return False
        self._sim_view, self._views, self._pending = sim_view, {}, {}
        carb.log_warn("[body_interface] physics view ready after %d attempt(s)" % self._attempts)
        return True

    def _view(self, path: str):
        if not self.ready():
            raise RuntimeError("body_interface: PhysX is not attached to the stage yet")
        view = self._views.get(path)
        if view is None:
            view = self._sim_view.create_rigid_body_view(path)
            if view is None or view.count != 1:
                raise RuntimeError("body_interface: no single rigid body at %s" % path)
            self._views[path] = view
        return view

    # ------------------------------------------------------------------ state
    def get_pose(self, path: str) -> BodyPose:
        t = np.asarray(self._view(path).get_transforms(), dtype=np.float64).reshape(-1, 7)[0]
        return BodyPose(t[0:3].copy(), t[3:7].copy())

    def get_velocities(self, path: str):
        """(linear, angular), both world frame."""
        v = np.asarray(self._view(path).get_velocities(), dtype=np.float64).reshape(-1, 6)[0]
        return v[0:3].copy(), v[3:6].copy()

    def get_relative_poses(self, parent: str, children):
        """The children's positions in the parent's frame (dynamic_control's get_relative_body_poses)."""
        from scipy.spatial.transform import Rotation
        p = self.get_pose(parent)
        inv = Rotation.from_quat(p.r).inv()
        out = []
        for c in children:
            cp = self.get_pose(c)
            out.append(BodyPose(inv.apply(cp.p - p.p), (inv * Rotation.from_quat(cp.r)).as_quat()))
        return out

    # ----------------------------------------------------------------- forces
    def add_force(self, path: str, force, pos=(0.0, 0.0, 0.0)):
        """A force in the body frame, at pos in the body frame (dynamic_control apply_body_force, global=False)."""
        f, t = self._pending.setdefault(path, [np.zeros(3), np.zeros(3)])
        force = np.asarray(force, dtype=np.float64)
        f += force
        t += np.cross(np.asarray(pos, dtype=np.float64), force)

    def add_torque(self, path: str, torque):
        """A torque in the body frame (dynamic_control apply_body_torque, global=False)."""
        _, t = self._pending.setdefault(path, [np.zeros(3), np.zeros(3)])
        t += np.asarray(torque, dtype=np.float64)

    def flush(self):
        """Apply everything accumulated since the last flush, one call per body."""
        pending, self._pending = self._pending, {}
        idx = np.array([0], dtype=np.int32)
        for path, (f, t) in pending.items():
            self._view(path).apply_forces_and_torques_at_position(
                f.reshape(1, 3).astype(np.float32), t.reshape(1, 3).astype(np.float32), None, idx, False)
