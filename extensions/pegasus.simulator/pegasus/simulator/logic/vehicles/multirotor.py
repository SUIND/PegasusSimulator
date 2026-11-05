"""
| File: multirotor.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Definition of the Multirotor class which is used as the base for all the multirotor vehicles.
"""

import carb
import numpy as np

# The vehicle interface
from pegasus.simulator.logic.vehicles.vehicle import Vehicle

# Mavlink interface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

# Sensors and dynamics setup
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.thrusters import QuadraticThrustCurve
from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS

class MultirotorConfig:
    """
    A data class that is used for configuring a Multirotor
    """

    def __init__(self):
        """
        Initialization of the MultirotorConfig class
        """

        # Stage prefix of the vehicle when spawning in the world
        self.stage_prefix = "quadrotor"

        # The USD file that describes the visual aspect of the vehicle (and some properties such as mass and moments of inertia)
        self.usd_file = ""

        # The default thrust curve for a quadrotor and dynamics relating to drag
        self.thrust_curve = QuadraticThrustCurve()
        self.drag = LinearDrag([0.50, 0.30, 0.0])

        # The default sensors for a quadrotor
        self.sensors = [Barometer(), IMU(), Magnetometer(), GPS()]

        # The default graphical sensors for a quadrotor
        self.graphical_sensors = []

        # The default omnigraphs for a quadrotor
        self.graphs = []

        # The backends for actually sending commands to the vehicle. By default use mavlink (with default mavlink configurations)
        # [Can be None as well, if we do not desired to use PX4 with this simulated vehicle]. It can also be a ROS2 backend
        # or your own custom Backend implementation!
        self.backends = [PX4MavlinkBackend(config=PX4MavlinkBackendConfig())]


class Multirotor(Vehicle):
    """Multirotor class - It defines a base interface for creating a multirotor
    """
    def __init__(
        self,
        # Simulation specific configurations
        stage_prefix: str = "quadrotor",
        usd_file: str = "",
        vehicle_id: int = 0,
        # Spawning pose of the vehicle
        init_pos=[0.0, 0.0, 0.07],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        config=MultirotorConfig(),
    ):
        """Initializes the multirotor object

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator when spawned. Defaults to "quadrotor".
            usd_file (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            vehicle_id (int): The id to be used for the vehicle. Defaults to 0.
            init_pos (list): The initial position of the vehicle in the inertial frame (in ENU convention). Defaults to [0.0, 0.0, 0.07].
            init_orientation (list): The initial orientation of the vehicle in quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
            config (MultirotorConfig, optional): Defaults to MultirotorConfig().
        """

        # 1. Initiate the Vehicle object itself
        super().__init__(stage_prefix, usd_file, init_pos, init_orientation, config.sensors, config.graphical_sensors, config.graphs, config.backends)

        # 2. Setup the dynamics of the system - get the thrust curve of the vehicle from the configuration
        self._thrusters = config.thrust_curve
        self._drag = config.drag

    def start(self):
        """In this case we do not need to do anything extra when the simulation starts"""
        pass

    def stop(self):
        """In this case we do not need to do anything extra when the simulation stops"""
        pass

    def update(self, dt: float):
        """
        Method that computes and applies the forces to the vehicle in simulation based on the motor speed. 
        This method must be implemented by a class that inherits this type. This callback
        is called on every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # A status line every 8000 physics steps (10 s of sim at 800 Hz): whether the FDM link is up is
        # otherwise visible only as SITL silently waiting.
        self._update_calls = getattr(self, "_update_calls", 0) + 1
        if self._update_calls == 1 or self._update_calls % 8000 == 0:
            ap = getattr(self._backends[0], "ap", None) if self._backends else None
            carb.log_warn("[multirotor] update #%d: physics view %s, FDM online %s, z %.2f"
                          % (self._update_calls, self.get_dc_interface().ready(),
                             getattr(ap, "arduPilotOnline", None), self._state.position[2]))

        # No forces before PhysX is attached to the stage (the first steps after play)
        if not self.get_dc_interface().ready():
            return

        # Get the desired angular velocities for each rotor from the first backend (can be mavlink or other) expressed in rad/s
        if len(self._backends) != 0:
            desired_rotor_velocities = self._backends[0].input_reference()
        else:
            desired_rotor_velocities = [0.0 for i in range(self._thrusters._num_rotors)]

        # Input the desired rotor velocities in the thruster model
        self._thrusters.set_input_reference(desired_rotor_velocities)

        # Get the desired forces to apply to the vehicle
        forces_z, _, rolling_moment = self._thrusters.update(self._state, dt)

        # Apply force to each rotor
        for i in range(4):

            # Apply the force in Z on the rotor frame
            self.apply_force([0.0, 0.0, forces_z[i]], body_part="/rotor" + str(i))

            # The rotating propeller visual effect (handle_propeller_visual) is not driven on Isaac 6.x:
            # it set rotor joint velocities through omni.isaac.dynamic_control, which is gone.

        # Apply the torque to the body frame of the vehicle that corresponds to the rolling moment
        self.apply_torque([0.0, 0.0, rolling_moment], "/body")

        # Compute the total linear drag force to apply to the vehicle's body frame
        drag = self._drag.update(self._state, dt)
        self.apply_force(drag, body_part="/body")

        # Everything above is accumulated per body; apply it for this physics step
        self.flush_forces()

        # Call the update methods in all backends
        for backend in self._backends:
            backend.update(dt)

    def handle_propeller_visual(self, rotor_number, force: float, articulation):
        """
        Auxiliar method used to set the joint velocity of each rotor (for animation purposes) based on the 
        amount of force being applied on each joint

        Args:
            rotor_number (int): The number of the rotor to generate the rotation animation
            force (float): The force that is being applied on that rotor
            articulation (_type_): The articulation group the joints of the rotors belong to
        """

        # A no-op on Isaac 6.x: the joint velocities were set through omni.isaac.dynamic_control, which is gone.
        return

    def force_and_torques_to_velocities(self, force: float, torque: np.ndarray):
        """
        Auxiliar method used to get the target angular velocities for each rotor, given the total desired thrust [N] and
        torque [Nm] to be applied in the multirotor's body frame.

        Note: This method assumes a quadratic thrust curve. This method will be improved in a future update,
        and a general thrust allocation scheme will be adopted. For now, it is made to work with multirotors directly.

        Args:
            force (np.ndarray): A vector of the force to be applied in the body frame of the vehicle [N]
            torque (np.ndarray): A vector of the torque to be applied in the body frame of the vehicle [Nm]

        Returns:
            list: A list of angular velocities [rad/s] to apply in reach rotor to accomplish suchs forces and torques
        """

        # The relative position of the rotors with respect to the body frame of the vehicle (ignoring the orientation for now)
        relative_poses = self.get_dc_interface().get_relative_poses(
            self._stage_prefix + "/body",
            [self._stage_prefix + "/rotor" + str(i) for i in range(self._thrusters._num_rotors)])

        # Define the alocation matrix
        aloc_matrix = np.zeros((4, self._thrusters._num_rotors))
        
        # Define the first line of the matrix (T [N])
        aloc_matrix[0, :] = np.array(self._thrusters._rotor_constant)                                           

        # Define the second and third lines of the matrix (\tau_x [Nm] and \tau_y [Nm])
        aloc_matrix[1, :] = np.array([relative_poses[i].p[1] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)])
        aloc_matrix[2, :] = np.array([-relative_poses[i].p[0] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)])

        # Define the forth line of the matrix (\tau_z [Nm])
        aloc_matrix[3, :] = np.array([self._thrusters._rolling_moment_coefficient[i] * self._thrusters._rot_dir[i] for i in range(self._thrusters._num_rotors)])

        # Compute the inverse allocation matrix, so that we can get the angular velocities (squared) from the total thrust and torques
        aloc_inv = np.linalg.pinv(aloc_matrix)

        # Compute the target angular velocities (squared)
        squared_ang_vel = aloc_inv @ np.array([force, torque[0], torque[1], torque[2]])

        # Making sure that there is no negative value on the target squared angular velocities
        squared_ang_vel[squared_ang_vel < 0] = 0.0

        # ------------------------------------------------------------------------------------------------
        # Saturate the inputs while preserving their relation to each other, by performing a normalization
        # ------------------------------------------------------------------------------------------------
        max_thrust_vel_squared = np.power(self._thrusters.max_rotor_velocity[0], 2)
        max_val = np.max(squared_ang_vel)

        if max_val >= max_thrust_vel_squared:
            normalize = np.maximum(max_val / max_thrust_vel_squared, 1.0)

            squared_ang_vel = squared_ang_vel / normalize

        # Compute the angular velocities for each rotor in [rad/s]
        ang_vel = np.sqrt(squared_ang_vel)

        return ang_vel
