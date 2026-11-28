"""Gate/IMU EKF and a configurable training surrogate. Requires NumPy.

Frames: world NED, body FRD. Quaternion [w,x,y,z] rotates body to world.
All times are seconds on the same monotonic/simulation clock. SI units.
This is a reference implementation, not a flight-qualified estimator.
"""
from dataclasses import dataclass, field, replace
from collections import deque
from typing import Optional, Sequence

import numpy as np


def _vec(value, n):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,) or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected {n} finite elements")
    return a.copy()


def _time(value):
    t = float(value)
    if not np.isfinite(t):
        raise ValueError("Timestamp must be finite")
    return t


def _quat(q):
    q = _vec(q, 4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Zero quaternion")
    return q / n


def _mul(a, b):
    return np.r_[a[0]*b[0] - a[1:] @ b[1:],
                 a[0]*b[1:] + b[0]*a[1:] + np.cross(a[1:], b[1:])]


def _exp(v):
    n = np.linalg.norm(v)
    if n < 1e-10:
        return _quat(np.r_[1., .5*v])
    return np.r_[np.cos(n/2), np.sin(n/2)*v/n]


def _rot(q):
    w, x, y, z = _quat(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def _sym(p):
    return (p + p.T)*.5


def _cov(value, n):
    a = np.asarray(value, dtype=float)
    if a.shape != (n,n) or not np.all(np.isfinite(a)):
        raise ValueError(f"Expected finite {n}x{n} covariance")
    if not np.allclose(a, a.T, atol=1e-10) or np.linalg.eigvalsh(a).min() < -1e-10:
        raise ValueError("Covariance must be symmetric positive semidefinite")
    return _sym(a)


def _default_p():
    return np.diag(np.r_[np.full(3,.1**2), np.full(3,.2**2),
                         np.full(4,.03**2), np.full(3,.1**2), np.full(3,.02**2)])


def _default_q():
    # Continuous-time noise spectral densities: accel, gyro, bias random walks.
    return np.diag(np.repeat([.12**2, .015**2, .01**2, .002**2], 3))


def _normalize_state(x, p):
    x = x.copy()
    q = x[6:10]
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise ValueError("Invalid orientation state")
    q = q/n
    j = np.eye(16)
    j[6:10,6:10] = (np.eye(4)-np.outer(q,q))/n
    x[6:10] = q
    return x, _sym(j@p@j.T)


def _tangent(q):
    # Jacobian of q * exp(delta_theta) at zero (body-frame rotation error).
    w, x, y, z = q
    return .5*np.array([[-x,-y,-z],[w,-z,y],[z,w,-x],[-y,x,w]])


@dataclass
class EKFConfig:
    initial_covariance: np.ndarray = field(default_factory=_default_p)
    process_noise: np.ndarray = field(default_factory=_default_q)
    gravity: float = 9.80665
    history_seconds: float = 1.0
    max_prediction_step: float = .02
    max_imu_gap: float = .15
    max_position_std: float = 1.0
    max_visual_age: float = 2.0
    nis_position: float = 16.27  # approximate chi-square 3 dof, 99.9%
    nis_pose: float = 22.46      # approximate chi-square 6 dof, 99.9%


@dataclass
class Gate:
    position: Sequence[float]
    orientation: Sequence[float]  # gate local X is plane normal


@dataclass
class MockEKFConfig:
    gates: Sequence[Gate] = field(default_factory=tuple)
    camera_position_body: Sequence[float] = (0., 0., 0.)
    camera_forward_body: Sequence[float] = (1., 0., 0.)
    camera_fov_deg: float = 120.  # circular cone approximation
    max_view_angle_deg: float = 75.
    max_gate_distance: float = 8.
    detection_probability: float = .9
    camera_hz: float = 60.
    imu_hz: float = 50.
    output_delay: float = .02
    position_noise_std: float = .015
    velocity_noise_std: float = .04
    attitude_noise_std: float = .01
    gyro_noise_std: float = .015
    accel_bias_std: float = .05
    gyro_bias_std: float = .005
    accel_bias_walk: float = .015
    gyro_bias_walk: float = .001
    position_walk: float = .015
    velocity_walk: float = .03
    attitude_walk: float = .005
    correction_gain: float = .65
    max_visual_age: float = 2.
    max_position_std: float = 1.


@dataclass
class EKFState:
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1.,0.,0.,0.]))
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    acceleration_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    angular_velocity_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    covariance: Optional[np.ndarray] = None
    timestamp: float = 0.
    is_valid: bool = False


def _copy_state(s):
    return replace(s, position=s.position.copy(), velocity=s.velocity.copy(),
                   orientation=s.orientation.copy(), angular_velocity=s.angular_velocity.copy(),
                   acceleration_bias=s.acceleration_bias.copy(),
                   angular_velocity_bias=s.angular_velocity_bias.copy(),
                   covariance=None if s.covariance is None else s.covariance.copy())


@dataclass
class _Snapshot:
    x: np.ndarray
    p: np.ndarray
    t: float
    gyro: np.ndarray
    visual_t: float
    imu_seen: bool = False
    gap: bool = False


class EKFEstimator:
    """16-component additive quaternion EKF, with bounded delayed-observation replay."""
    def __init__(self, config):
        self.c = config
        self.q = _cov(config.process_noise, 12)
        if min(config.history_seconds, config.max_prediction_step, config.max_imu_gap,
               config.max_position_std, config.max_visual_age) <= 0:
            raise ValueError("EKF time/limit parameters must be positive")
        self.base = None
        self.current = None
        self.imus = []
        self.observations = []
        self.snapshots = []
        self.serial = 0

    def reset(self, initial_state, seed=None):
        s = initial_state
        x = np.r_[_vec(s.position,3), _vec(s.velocity,3), _quat(s.orientation),
                  _vec(s.acceleration_bias,3), _vec(s.angular_velocity_bias,3)]
        p = _cov(self.c.initial_covariance if s.covariance is None else s.covariance,16)
        x,p = _normalize_state(x,p)
        t = _time(s.timestamp)
        gyro = _vec(s.angular_velocity,3)+x[13:16]
        self.base = _Snapshot(x,p,t,gyro,t if s.is_valid else -np.inf)
        self.current = self.base
        self.imus, self.observations, self.snapshots = [], [], []
        self.serial = 0

    def _require_reset(self):
        if self.current is None:
            raise RuntimeError("Call reset() before using the estimator")

    def _transition(self, x, a, gyro, dt):
        y = x.copy()
        q = _quat(x[6:10])
        omega = gyro-x[13:16]
        mid_q = _mul(q, _exp(omega*dt*.5))
        acc = _rot(mid_q)@(a-x[10:13])+np.array([0.,0.,self.c.gravity])
        y[:3] += x[3:6]*dt + .5*acc*dt*dt
        y[3:6] += acc*dt
        y[6:10] = _quat(_mul(q,_exp(omega*dt)))
        return y

    def _predict(self,s,a,gyro,t):
        x,p = s.x.copy(),s.p.copy()
        remaining = t-s.t
        if remaining < -1e-9:
            raise ValueError("Cannot predict backwards")
        while remaining > 1e-12:
            dt = min(remaining,self.c.max_prediction_step)
            y = self._transition(x,a,gyro,dt)
            # Numerical discrete Jacobian: straightforward reference implementation.
            f = np.empty((16,16))
            for i in range(16):
                delta = np.zeros(16); delta[i] = 1e-6
                f[:,i] = (self._transition(x+delta,a,gyro,dt)-
                          self._transition(x-delta,a,gyro,dt))/(2e-6)
            l = np.zeros((16,12))
            r = _rot(x[6:10])
            l[:3,:3] = -.5*dt*r
            l[3:6,:3] = -r
            l[6:10,3:6] = -_tangent(x[6:10])
            l[10:13,6:9] = np.eye(3)
            l[13:16,9:12] = np.eye(3)
            x,p = _normalize_state(y,f@p@f.T + l@self.q@l.T*dt)
            remaining -= dt
        return _Snapshot(x,p,t,gyro.copy(),s.visual_t,True,s.gap)

    def _correct(self,s,pos,ori,cov):
        x,p=s.x,s.p
        if ori is None:
            z=pos; h=np.eye(16)[:3]; threshold=self.c.nis_position
        else:
            sign = 1. if ori@x[6:10] >= 0 else -1.
            z=np.r_[pos,sign*ori]
            h=np.eye(16)[[0,1,2,6,7,8,9]]
            d=np.diag([1.,1.,1.,sign,sign,sign,sign]); cov=d@cov@d
            threshold=self.c.nis_pose
        residual=z-h@x
        innovation=_sym(h@p@h.T+cov)
        # Pseudoinverse accommodates rank-three quaternion tangent covariance.
        inv=np.linalg.pinv(innovation,hermitian=True)
        nis=float(residual@inv@residual)
        if nis > threshold:
            return s,False
        k=p@h.T@inv
        a=np.eye(16)-k@h
        xx,pp=_normalize_state(x+k@residual,a@p@a.T+k@cov@k.T)
        return replace(s,x=xx,p=pp,visual_t=s.t),True

    def _replay(self):
        s=self.base; snapshots=[]; accepted={}; j=0
        obs=self.observations
        while j<len(obs) and obs[j][0] <= s.t:
            t,i,pos,ori,cov=obs[j]
            s,accepted[i]=self._correct(s,pos,ori,cov); j+=1
        for t,a,g in self.imus:
            gap=t-s.t > self.c.max_imu_gap
            while j<len(obs) and obs[j][0] <= t:
                ot,i,pos,ori,cov=obs[j]
                s=self._predict(s,a,g,ot)
                s,accepted[i]=self._correct(s,pos,ori,cov); j+=1
            s=self._predict(s,a,g,t)
            s=replace(s,gap=gap)
            snapshots.append(s)
        self.current=s; self.snapshots=snapshots
        return accepted

    def _prune(self):
        cutoff=self.current.t-self.c.history_seconds
        n=0
        for s in self.snapshots:
            if s.t>cutoff: break
            n+=1
        if n:
            self.base=self.snapshots[n-1]
            self.imus=self.imus[n:]; self.snapshots=self.snapshots[n:]
            self.observations=[o for o in self.observations if o[0]>self.base.t]

    def update_imu(self, acceleration, angular_velocity, timestamp):
        self._require_reset()
        t=_time(timestamp); a=_vec(acceleration,3); g=_vec(angular_velocity,3)
        if t<=self.current.t:
            raise ValueError("IMU timestamps must strictly increase")
        gap=t-self.current.t>self.c.max_imu_gap
        self.imus.append((t,a,g))
        self.current=replace(self._predict(self.current,a,g,t),gap=gap)
        self.snapshots.append(self.current); self._prune()

    def update_pnp(self, position, orientation, covariance, timestamp):
        self._require_reset()
        t=_time(timestamp); pos=_vec(position,3)
        ori=None if orientation is None else _quat(orientation)
        cov=_cov(covariance,3 if ori is None else 7)
        if t<self.base.t or t>self.current.t:
            return False  # future observations must be retried by caller
        i=self.serial; self.serial+=1
        self.observations.append((t,i,pos,ori,cov))
        self.observations.sort(key=lambda o:(o[0],o[1]))
        return self._replay().get(i,False)

    def get_state(self):
        self._require_reset(); s=self.current; x=s.x
        valid=(s.imu_seen and not s.gap and s.t-s.visual_t<=self.c.max_visual_age
               and np.sqrt(max(0.,np.max(np.diag(s.p)[:3])))<=self.c.max_position_std)
        return EKFState(x[:3].copy(),x[3:6].copy(),x[6:10].copy(),
                        s.gyro-x[13:16],x[10:13].copy(),x[13:16].copy(),
                        s.p.copy(),s.t,bool(valid))


class MockEKFEstimator:
    """Cheap correlated output-error model, not an actual Kalman filter."""
    def __init__(self,config):
        self.c=config
        if min(config.camera_hz,config.imu_hz,config.max_gate_distance,
               config.max_visual_age,config.max_position_std)<=0 or config.output_delay<0:
            raise ValueError("Invalid mock timing or limits")
        if not 0<=config.detection_probability<=1 or not 0<=config.correction_gain<=1:
            raise ValueError("Probabilities/gain must be in [0,1]")
        if not 0<config.camera_fov_deg<=180 or not 0<config.max_view_angle_deg<=90:
            raise ValueError("Invalid camera/view angle")
        self.gates=[(_vec(g.position,3),_rot(g.orientation)[:,0]) for g in config.gates]
        self.cam_pos=_vec(config.camera_position_body,3)
        self.cam_forward=_vec(config.camera_forward_body,3)
        n=np.linalg.norm(self.cam_forward)
        if n<1e-12: raise ValueError("Invalid camera direction")
        self.cam_forward/=n
        self.last_t=None

    def reset(self,initial_state,seed=None):
        c=self.c; self.rng=np.random.default_rng(seed)
        self.last_t=_time(initial_state.timestamp)
        self.next_imu=self.last_t; self.next_cam=self.last_t
        self.last_sample=self.last_t
        self.last_visual=self.last_t if initial_state.is_valid else -np.inf
        self.ep=self.rng.normal(0,c.position_noise_std,3)
        self.ev=self.rng.normal(0,c.velocity_noise_std,3)
        self.eq=self.rng.normal(0,c.attitude_noise_std,3)
        self.ba=self.rng.normal(0,c.accel_bias_std,3)
        self.bg=self.rng.normal(0,c.gyro_bias_std,3)
        # Latent sensor biases and their approximate estimated counterparts.
        self.ba_hat=np.zeros(3); self.bg_hat=np.zeros(3)
        self.var=np.repeat([c.position_noise_std**2,c.velocity_noise_std**2,
                            c.attitude_noise_std**2],3)
        self.bias_var=np.repeat([c.accel_bias_std**2,c.gyro_bias_std**2],3)
        self.buffer=deque()
        self.output=EKFState(_vec(initial_state.position,3),_vec(initial_state.velocity,3),
                             _quat(initial_state.orientation),_vec(initial_state.angular_velocity,3),
                             self.ba_hat.copy(),self.bg_hat.copy(),np.zeros((16,16)),self.last_t,False)
        self.previous_truth=(self.last_t,_vec(initial_state.position,3),
                             _vec(initial_state.velocity,3),_quat(initial_state.orientation),
                             _vec(initial_state.angular_velocity,3))

    def _visible(self,pos,q):
        if not self.gates:
            return True  # geometry-free mode: constant detection probability
        r=_rot(q); origin=pos+r@self.cam_pos; axis=r@self.cam_forward
        for gp,normal in self.gates:
            d=gp-origin; distance=np.linalg.norm(d)
            if distance<1e-9 or distance>self.c.max_gate_distance: continue
            direction=d/distance
            if axis@direction < np.cos(np.deg2rad(self.c.camera_fov_deg/2)): continue
            if abs(normal@direction) < np.cos(np.deg2rad(self.c.max_view_angle_deg)): continue
            return True
        return False

    def _truth_at(self,old,new,t):
        u=np.clip((t-old[0])/max(new[0]-old[0],1e-12),0.,1.)
        q=new[3] if old[3]@new[3]>=0 else -new[3]
        return (old[1]*(1-u)+new[1]*u,old[2]*(1-u)+new[2]*u,
                _quat(old[3]*(1-u)+q*u),old[4]*(1-u)+new[4]*u)

    def _evolve(self,t,q):
        c=self.c; dt=max(0.,t-self.last_sample); root=np.sqrt(dt); r=self.rng
        acc_error=_rot(q)@(self.ba-self.ba_hat)
        self.ep+=self.ev*dt+.5*acc_error*dt*dt+r.normal(0,c.position_walk*root,3)
        self.ev+=acc_error*dt+r.normal(0,c.velocity_walk*root,3)
        self.eq+=(self.bg-self.bg_hat)*dt+r.normal(0,c.attitude_walk*root,3)
        self.ba+=r.normal(0,c.accel_bias_walk*root,3)
        self.bg+=r.normal(0,c.gyro_bias_walk*root,3)
        self.var[:3]+=self.var[3:6]*dt*dt+c.position_walk**2*dt
        self.var[3:6]+=self.bias_var[:3]*dt*dt+c.velocity_walk**2*dt
        self.var[6:9]+=self.bias_var[3:]*dt*dt+c.attitude_walk**2*dt
        self.bias_var+=np.repeat([c.accel_bias_walk**2,c.gyro_bias_walk**2],3)*dt
        self.last_sample=t

    def _camera(self,t,pos,q):
        c=self.c
        if self._visible(pos,q) and self.rng.random()<c.detection_probability:
            k=c.correction_gain; self.last_visual=t
            self.ep*=(1-k); self.ev*=(1-k); self.eq*=(1-k)
            # Surrogate for indirect bias correction; not an EKF estimate.
            self.ba_hat+=.05*k*(self.ba-self.ba_hat)
            self.bg_hat+=.05*k*(self.bg-self.bg_hat)
            self.bias_var*=(1-.05*k)**2
            floor=np.repeat([c.position_noise_std**2,c.velocity_noise_std**2,
                             c.attitude_noise_std**2],3)
            self.var=(1-k)**2*self.var+k*k*floor

    def _sample(self,t,pos,vel,q,gyro):
        c=self.c; r=self.rng
        attitude=_quat(_mul(q,_exp(self.eq+r.normal(0,c.attitude_noise_std,3))))
        p=np.zeros((16,16)); p[:6,:6]=np.diag(self.var[:6]+np.repeat(
            [c.position_noise_std**2,c.velocity_noise_std**2],3))
        j=_tangent(attitude); p[6:10,6:10]=j@np.diag(self.var[6:]+c.attitude_noise_std**2)@j.T
        p[10:,10:]=np.diag(self.bias_var)
        valid=(t-self.last_visual<=c.max_visual_age and
               np.sqrt(np.max(np.diag(p)[:3]))<=c.max_position_std)
        self.buffer.append(EKFState(pos+self.ep+r.normal(0,c.position_noise_std,3),
            vel+self.ev+r.normal(0,c.velocity_noise_std,3),attitude,
            gyro+self.bg-self.bg_hat+r.normal(0,c.gyro_noise_std,3),
            self.ba_hat.copy(),self.bg_hat.copy(),p,t,bool(valid)))

    def update_ground_truth(self,position,velocity,orientation,angular_velocity,timestamp):
        if self.last_t is None: raise RuntimeError("Call reset() first")
        t=_time(timestamp)
        if t<self.last_t: raise ValueError("Ground truth time must not go backwards")
        new=(t,_vec(position,3),_vec(velocity,3),_quat(orientation),_vec(angular_velocity,3))
        while min(self.next_imu,self.next_cam)<=t+1e-10:
            st=min(self.next_imu,self.next_cam)
            pos,vel,q,g=self._truth_at(self.previous_truth,new,st)
            self._evolve(st,q)
            if self.next_cam<=st+1e-10:
                self._camera(st,pos,q); self.next_cam+=1/self.c.camera_hz
            if self.next_imu<=st+1e-10:
                self._sample(st,pos,vel,q,g); self.next_imu+=1/self.c.imu_hz
        self.last_t=t; self.previous_truth=new
        while self.buffer and self.buffer[0].timestamp<=t-self.c.output_delay+1e-10:
            self.output=self.buffer.popleft()

    def get_state(self):
        if self.last_t is None: raise RuntimeError("Call reset() first")
        return _copy_state(self.output)


class EKF:
    """Facade. Choose mock at construction; call reset before the first update."""
    def __init__(self,config=None,mock=False,mock_config=None):
        self._mock=bool(mock)
        if self._mock:
            self._estimator=MockEKFEstimator(mock_config or MockEKFConfig())
        else:
            if mock_config is not None: raise ValueError("mock_config requires mock=True")
            self._estimator=EKFEstimator(config or EKFConfig())

    @property
    def mock(self):
        return self._mock

    def reset(self,initial_state,seed=None):
        self._estimator.reset(initial_state,seed)

    def update_imu(self,acceleration,angular_velocity,timestamp):
        if self.mock: raise RuntimeError("update_imu is only available in normal mode")
        self._estimator.update_imu(acceleration,angular_velocity,timestamp)

    def update_pnp(self,position,orientation,covariance,timestamp):
        if self.mock: raise RuntimeError("update_pnp is only available in normal mode")
        return self._estimator.update_pnp(position,orientation,covariance,timestamp)

    def update_ground_truth(self,position,velocity,orientation,angular_velocity,timestamp):
        if not self.mock: raise RuntimeError("update_ground_truth is only available in mock mode")
        self._estimator.update_ground_truth(position,velocity,orientation,angular_velocity,timestamp)

    def get_state(self):
        return self._estimator.get_state()
