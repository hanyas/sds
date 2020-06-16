import numpy as np

from sds_numpy.envs.quanser.common import Base, LabeledBox, Timing


class QubeBase(Base):
    def __init__(self, fs, fs_ctrl):
        super(QubeBase, self).__init__(fs, fs_ctrl)
        self._state = None
        self.timing = Timing(fs, fs_ctrl)

        self._vis = {'vp': None, 'arm': None, 'pole': None, 'curve': None}
        self._dyn = QubeDynamics()

        self._sigma = 1e-8

        # Limits
        act_max = np.array([5.0])
        state_max = np.array([2.0, np.inf, 30.0, 40.0])
        sens_max = np.array([2.3, np.inf])

        # Spaces
        self.sensor_space = LabeledBox(
            labels=('theta', 'alpha'),
            low=-sens_max, high=sens_max, dtype=np.float64)
        self.state_space = LabeledBox(
            labels=('theta', 'alpha', 'theta_dot', 'alpha_dot'),
            low=-state_max, high=state_max, dtype=np.float64)
        self.action_space = LabeledBox(
            labels=('volts',),
            low=-act_max, high=act_max, dtype=np.float64)

        # Function to ensure that state and action constraints are satisfied
        safety_th_lim = 1.5
        self._lim_act = ActionLimiter(self.state_space,
                                      self.action_space,
                                      safety_th_lim)

        # Initialize random number generator
        self._np_random = None
        self.seed()

    def _zero_sim_step(self):
        return self._sim_step([0.0])[0]

    def _rwd(self, x, u):
        th, al, thd, ald = x
        cost = al**2 + 5e-3*ald**2 + 1e-1*th**2 + 2e-2*thd**2 + 3e-3*u[0]**2
        rwd = - cost * self.timing.dt_ctrl
        return np.float64(rwd), False

    def _calibrate(self):
        _low, _high = np.array([-0.1, -np.pi, -1., -5.]),\
                      np.array([0.1, np.pi, 1., 5.])

        self._sim_state = self._np_random.uniform(low=_low, high=_high)
        self._state = self._zero_sim_step()

    def _sim_step(self, u):
        u_cmd = self._lim_act(self._sim_state, u)
        # u_cmd = np.clip(u, self.action_space.low, self.action_space.high)

        def f(x, u):
            thdd, aldd = self._dyn(x, u)
            return np.hstack((x[2], x[3], thdd, aldd))

        c1 = f(self._sim_state, u_cmd)
        c2 = f(self._sim_state + 0.5 * self.timing.dt * c1, u_cmd)
        c3 = f(self._sim_state + 0.5 * self.timing.dt * c2, u_cmd)
        c4 = f(self._sim_state + self.timing.dt * c3, u_cmd)

        self._sim_state = self._sim_state + self.timing.dt / 6. * (c1 + 2. * c2 + 2. * c3 + c4)

        # apply state constraints
        self._sim_state = np.clip(self._sim_state, self.state_space.low, self.state_space.high)

        # add observation noise
        self._sim_state = self._sim_state + np.random.randn(self.state_space.shape[0]) * self._sigma

        return self._sim_state, u_cmd

    def reset(self):
        self._calibrate()
        if self._vis['curve'] is not None:
            self._vis['curve'].clear()
        return self.step(np.array([0.0]))[0]

    def _set_gui(self):
        scene_range = 0.2
        arm_radius = 0.003
        arm_length = 0.085
        pole_radius = 0.0045
        pole_length = 0.129
        # http://www.glowscript.org/docs/VPythonDocs/canvas.html
        self._vis['vp'].scene.width = 400
        self._vis['vp'].scene.height = 300
        self._vis['vp'].scene.background = self._vis['vp'].color.gray(0.95)
        self._vis['vp'].scene.lights = []
        self._vis['vp'].distant_light(
            direction=self._vis['vp'].vector(0.2, 0.2, 0.5),
            color=self._vis['vp'].color.white)
        self._vis['vp'].scene.up = self._vis['vp'].vector(0, 0, 1)
        self._vis['vp'].scene.range = scene_range
        self._vis['vp'].scene.center = self._vis['vp'].vector(0.04, 0, 0)
        self._vis['vp'].scene.forward = self._vis['vp'].vector(-2, 1.2, -1)
        self._vis['vp'].box(pos=self._vis['vp'].vector(0, 0, -0.07),
                            length=0.09, width=0.1, height=0.09,
                            color=self._vis['vp'].color.gray(0.5))
        self._vis['vp'].cylinder(
            axis=self._vis['vp'].vector(0, 0, -1), radius=0.005,
            length=0.03, color=self._vis['vp'].color.gray(0.5))
        # Arm
        arm = self._vis['vp'].cylinder()
        arm.radius = arm_radius
        arm.length = arm_length
        arm.color = self._vis['vp'].color.blue
        # Pole
        pole = self._vis['vp'].cylinder()
        pole.radius = pole_radius
        pole.length = pole_length
        pole.color = self._vis['vp'].color.red
        # Curve
        curve = self._vis['vp'].curve(color=self._vis['vp'].color.white,
                                      radius=0.0005, retain=2000)
        return arm, pole, curve

    def render(self, mode='human'):
        if self._vis['vp'] is None:
            import importlib
            self._vis['vp'] = importlib.import_module('vpython')
            self._vis['arm'],\
            self._vis['pole'],\
            self._vis['curve'] = self._set_gui()
        th, al, _, _ = self._state
        arm_pos = (self._dyn.Lr * np.cos(th), self._dyn.Lr * np.sin(th), 0.0)
        pole_ax = (-self._dyn.Lp * np.sin(al) * np.sin(th),
                   self._dyn.Lp * np.sin(al) * np.cos(th),
                   -self._dyn.Lp * np.cos(al))
        self._vis['arm'].axis = self._vis['vp'].vector(*arm_pos)
        self._vis['pole'].pos = self._vis['vp'].vector(*arm_pos)
        self._vis['pole'].axis = self._vis['vp'].vector(*pole_ax)
        self._vis['curve'].append(
            self._vis['pole'].pos + self._vis['pole'].axis)
        self._vis['vp'].rate(self.timing.render_rate)


class ActionLimiter:
    def __init__(self, state_space, action_space, th_lim_min):
        self._th_lim_min = th_lim_min
        self._th_lim_max = (state_space.high[0] + self._th_lim_min) / 2.0
        self._th_lim_stiffness = 0.25 * action_space.high[0] / (self._th_lim_max - self._th_lim_min)
        self._clip = lambda a: np.clip(a, action_space.low, action_space.high)
        self._relu = lambda x: x * (x > 0.0)

    def _joint_lim_violation_force(self, x):
        th, _, thd, _ = x
        up = self._relu(th - self._th_lim_max) - self._relu(th - self._th_lim_min)
        dn = -self._relu(-th - self._th_lim_max) + self._relu(-th - self._th_lim_min)
        if (th > self._th_lim_min and thd > 0.0 or
                th < -self._th_lim_min and thd < 0.0):
            force = self._th_lim_stiffness * (up + dn)
        else:
            force = 0.0
        return force

    def __call__(self, x, a):
        force = self._joint_lim_violation_force(x)
        return self._clip(force if force else a)


class QubeDynamics:
    """Solve equation M qdd + C(q, qd) = tau for qdd."""

    def __init__(self):
        # Gravity
        self.g = 9.81

        # Motor
        self.Rm = 8.4    # resistance
        self.km = 0.042  # back-emf constant (V-s/rad)

        # Rotary arm
        self.Mr = 0.095  # mass (kg)
        self.Lr = 0.085  # length (m)
        self.Dr = 5e-5   # viscous damping (N-m-s/rad), original: 0.0015

        # Pendulum link
        self.Mp = 0.024  # mass (kg)
        self.Lp = 0.129  # length (m)
        self.Dp = 1e-5   # viscous damping (N-m-s/rad), original: 0.0005

        # Init constants
        self._init_const()

    def _init_const(self):
        # Moments of inertia
        Jr = self.Mr * self.Lr ** 2 / 12  # inertia about COM (kg-m^2)
        Jp = self.Mp * self.Lp ** 2 / 12  # inertia about COM (kg-m^2)

        # Constants for equations of motion
        self._c = np.zeros(5)
        self._c[0] = Jr + self.Mp * self.Lr ** 2
        self._c[1] = 0.25 * self.Mp * self.Lp ** 2
        self._c[2] = 0.5 * self.Mp * self.Lp * self.Lr
        self._c[3] = Jp + self._c[1]
        self._c[4] = 0.5 * self.Mp * self.Lp * self.g

    @property
    def params(self):
        params = self.__dict__.copy()
        params.pop('_c')
        return params

    @params.setter
    def params(self, params):
        self.__dict__.update(params)
        self._init_const()

    def __call__(self, s, u):
        th, al, thd, ald = s
        voltage = u[0]

        # Define mass matrix M = [[a, b], [b, c]]
        a = self._c[0] + self._c[1] * np.sin(al) ** 2
        b = self._c[2] * np.cos(al)
        c = self._c[3]
        d = a * c - b * b

        # Calculate vector [x, y] = tau - C(q, qd)
        trq = self.km * (voltage - self.km * thd) / self.Rm
        c0 = self._c[1] * np.sin(2 * al) * thd * ald \
            - self._c[2] * np.sin(al) * ald * ald
        c1 = -0.5 * self._c[1] * np.sin(2 * al) * thd * thd \
            + self._c[4] * np.sin(al)
        x = trq - self.Dr * thd - c0
        y = -self.Dp * ald - c1

        # Compute M^{-1} @ [x, y]
        thdd = (c * x - b * y) / d
        aldd = (a * y - b * x) / d

        return thdd, aldd
