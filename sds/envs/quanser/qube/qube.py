import autograd.numpy as np

from sds.envs.quanser.qube.base import QubeBase, QubeDynamics


class Qube(QubeBase):
    def __init__(self, fs, fs_ctrl):
        super(Qube, self).__init__(fs, fs_ctrl)
        self._sim_state = None
        self._vis = {'vp': None, 'arm': None, 'pole': None, 'curve': None}

        self.dyn = QubeDynamics()

        self._dt = 0.01

        self.dm_state = 4
        self.dm_act = 1

        self._sigma = 1e-8

    @property
    def ulim(self):
        return self.action_space.high

    def _calibrate(self):
        _low, _high = np.array([-0.1, -np.pi, -30., -40.]),\
                      np.array([0.1, np.pi, 30., 40.])

        self._sim_state = self._np_random.uniform(low=_low, high=_high)
        self._state = self._zero_sim_step()

    def _sim_step(self, u):
        u_cmd = self._lim_act(self._sim_state, u)
        # u_cmd = np.clip(u, self.action_space.low, self.action_space.high)

        def f(x, u):
            thdd, aldd = self.dyn(x, u)
            return np.hstack((x[2], x[3], thdd, aldd))

        c1 = f(self._sim_state, u_cmd)
        c2 = f(self._sim_state + 0.5 * self.timing.dt * c1, u_cmd)
        c3 = f(self._sim_state + 0.5 * self.timing.dt * c2, u_cmd)
        c4 = f(self._sim_state + self.timing.dt * c3, u_cmd)

        self._sim_state = self._sim_state + self.timing.dt / 6. * (c1 + 2. * c2 + 2. * c3 + c4)

        # apply state constraints
        self._sim_state = np.clip(self._sim_state, self.state_space.low, self.state_space.high)

        # add observation noise
        self._sim_state = self._sim_state + np.random.randn(self.dm_state) * self._sigma

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
        arm_pos = (self.dyn.Lr * np.cos(th), self.dyn.Lr * np.sin(th), 0.0)
        pole_ax = (-self.dyn.Lp * np.sin(al) * np.sin(th),
                   self.dyn.Lp * np.sin(al) * np.cos(th),
                   -self.dyn.Lp * np.cos(al))
        self._vis['arm'].axis = self._vis['vp'].vector(*arm_pos)
        self._vis['pole'].pos = self._vis['vp'].vector(*arm_pos)
        self._vis['pole'].axis = self._vis['vp'].vector(*pole_ax)
        self._vis['curve'].append(
            self._vis['pole'].pos + self._vis['pole'].axis)
        self._vis['vp'].rate(self.timing.render_rate)
