"""
Offline RWM-U + MOPO-PPO env for Unitree Go2 flat-terrain velocity tracking.

Ported directly from anymal_d_flat.py with two Go2-specific changes inside
_compute_imagination_reward_terms (matching go2_manager_based_mbrl_env.py):
  1. feet_air_time threshold:  0.5  ->  0.25  (Go2 morphology calibration)
  2. undesired_contacts:       emitted (thigh contacts only), weighted only by the s2 contract
  3. s2 terms (E16, 18 Nov 2026): joint_vel, energy, joint_pos, air_time_variance and a real dof_pos_limits;
     a term counts only when its key is in reward_term_weights, so older configs are unchanged

Everything else (state layout, observation construction, command sampling,
extensions/contacts/terminations parsing, interval events, dims) is identical
because Go2 and ANYmal share the same 12-DOF quadruped WM contract:
  WM state_dim=45 (3 vel + 3 ang + 3 grav + 12 q + 12 dq + 12 tau)
  policy obs_dim=48 (state[:33] + base_vel[3] + q[12] + dq[12] + last_action[12]
                     ... see get_imagination_observation for exact layout)
  action_dim=12, contact_dim=8 (4 thigh + 4 foot), termination_dim=1
"""
from envs.base import BaseEnv
from envs.go2_kinematics import feet_slide as go2_feet_slide
import torch
from tensordict import TensorDict


class Go2FlatEnv(BaseEnv):
    # Base COM offset from the root-link origin, base frame. URDF value until
    # rsl_rl/validate_go2_feet_slide.py selects the one that matches the simulator asset.
    feet_slide_base_com_offset = (0.021112, 0.0, -0.005366)
    # Reward-contract parameters (E16, 18 Nov 2026). Defaults keep the June / Start-v0 behaviour; the s2
    # contract (configs/go2_flat_cfg.py S2_REWARD_CONTRACT) sets them from the simulator's term parameters.
    feet_air_time_threshold = 0.25
    joint_pos_stand_still_scale = 5.0          # unitree joint_position_penalty
    joint_pos_velocity_threshold = 0.3
    default_joint_pos = None                   # [12] rad, sim joint order; needed for dof_pos_limits
    soft_joint_pos_limits = None               # [12, 2] rad; None keeps dof_pos_limits = 0 (old behaviour)
    timers_before_rewards = False              # True: update the foot timers first, as the contact sensor does
    # dof_acc_l2 differences the predicted joint velocity against the previous POLICY observation, which carries
    # observation noise (+-1.5 rad/s): noise alone gives an acceleration penalty at constant velocity (review
    # 19 Nov). True: difference against the previous noise-free state instead (physical acceleration).
    acceleration_from_state = False

    def _init_additional_imagination_attributes(self):
        self.last_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.current_air_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.last_contact_time = torch.zeros(self.num_envs, 4, device=self.device)
        self.current_contact_time = torch.zeros(self.num_envs, 4, device=self.device)

    def _reset_additional_imagination_attributes(self, env_ids):
        self.last_air_time[env_ids] = 0.0
        self.current_air_time[env_ids] = 0.0
        self.last_contact_time[env_ids] = 0.0
        self.current_contact_time[env_ids] = 0.0

    def _init_imagination_command(self):
        r = torch.empty(self.num_envs, device=self.device)
        self.base_velocity = torch.zeros(self.num_envs, 3, device=self.device)
        self.is_standing_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.base_velocity[:, 0] = r.uniform_(-1.0, 1.0)
        self.base_velocity[:, 1] = r.uniform_(-1.0, 1.0)
        self.base_velocity[:, 2] = r.uniform_(-1.0, 1.0)
        self.is_standing_env[:] = r.uniform_(0.0, 1.0) <= 0.02
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.base_velocity[standing_env_ids, :] = 0.0

    def _reset_imagination_command(self, env_ids):
        r = torch.empty(len(env_ids), device=self.device)
        self.base_velocity[env_ids, 0] = r.uniform_(-1.0, 1.0)
        self.base_velocity[env_ids, 1] = r.uniform_(-1.0, 1.0)
        self.base_velocity[env_ids, 2] = r.uniform_(-1.0, 1.0)
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= 0.02
        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.base_velocity[standing_env_ids, :] = 0.0

    def get_imagination_observation(self, state_history, action_history):
        state_history_denormalized, action_history_denormalized = self.dataset.denormalize(state_history[:, -1], action_history[:, -1])
        obs_base_lin_vel = state_history_denormalized[:, 0:3]
        obs_base_ang_vel = state_history_denormalized[:, 3:6]
        obs_projected_gravity = state_history_denormalized[:, 6:9]
        obs_joint_pos = state_history_denormalized[:, 9:21]
        obs_joint_vel = state_history_denormalized[:, 21:33]
        self.last_joint_vel_state = obs_joint_vel.clone()           # before noise: the state the policy acts from
        self.obs_last_action = action_history_denormalized
        if self.observation_noise:
            obs_base_lin_vel += 2 * (torch.rand_like(obs_base_lin_vel) - 0.5) * 0.1
            obs_base_ang_vel += 2 * (torch.rand_like(obs_base_ang_vel) - 0.5) * 0.2
            obs_projected_gravity += 2 * (torch.rand_like(obs_projected_gravity) - 0.5) * 0.05
            obs_joint_pos += 2 * (torch.rand_like(obs_joint_pos) - 0.5) * 0.01
            obs_joint_vel += 2 * (torch.rand_like(obs_joint_vel) - 0.5) * 1.5
        obs = torch.cat([obs_base_lin_vel, obs_base_ang_vel, obs_projected_gravity, self.base_velocity, obs_joint_pos, obs_joint_vel, self.obs_last_action], dim=1)
        obs = TensorDict({"policy": obs}, batch_size=[self.num_envs], device=self.device)
        self.last_obs = obs
        return obs

    def _parse_imagination_states(self, imagination_states_denormalized):
        base_lin_vel = imagination_states_denormalized[:, 0:3]
        base_ang_vel = imagination_states_denormalized[:, 3:6]
        projected_gravity = imagination_states_denormalized[:, 6:9]
        joint_pos = imagination_states_denormalized[:, 9:21]
        joint_vel = imagination_states_denormalized[:, 21:33]
        joint_torque = imagination_states_denormalized[:, 33:45]

        parsed_imagination_states = {
            "base_lin_vel": base_lin_vel,
            "base_ang_vel": base_ang_vel,
            "projected_gravity": projected_gravity,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "joint_torque": joint_torque,
        }
        return parsed_imagination_states

    def _parse_extensions(self, extensions):
        if extensions is None:
            return None
        parsed_extensions = {
        }
        return parsed_extensions

    def _parse_contacts(self, contacts):
        thigh_contact = torch.sigmoid(contacts[:, 0:4]).round() if contacts is not None else None
        foot_contact = torch.sigmoid(contacts[:, 4:8]).round() if contacts is not None else None

        parsed_contacts = {
            "thigh_contact": thigh_contact,
            "foot_contact": foot_contact,
        }
        return parsed_contacts

    def _parse_terminations(self, terminations):
        parsed_terminations = torch.sigmoid(terminations).squeeze(-1).round().bool() if terminations is not None else None
        return parsed_terminations

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):

        base_lin_vel = parsed_imagination_states["base_lin_vel"]
        base_ang_vel = parsed_imagination_states["base_ang_vel"]
        projected_gravity = parsed_imagination_states["projected_gravity"]
        joint_pos = parsed_imagination_states["joint_pos"]
        joint_vel = parsed_imagination_states["joint_vel"]
        joint_torque = parsed_imagination_states["joint_torque"]
        if self.acceleration_from_state and getattr(self, "last_joint_vel_state", None) is not None:
            joint_acc = (joint_vel - self.last_joint_vel_state) / self._step_dt
        else:
            joint_acc = (joint_vel - self.last_obs["policy"][:, 24:36]) / self._step_dt
        thigh_contact = parsed_contacts["thigh_contact"]
        foot_contact = parsed_contacts["foot_contact"]

        lin_vel_error = torch.sum(torch.square(self.base_velocity[:, :2] - base_lin_vel[:, :2]), dim=1)
        ang_vel_error = torch.square(self.base_velocity[:, 2] - base_ang_vel[:, 2])

        track_lin_vel_xy_exp = torch.exp(-lin_vel_error / 0.25)
        track_ang_vel_z_exp = torch.exp(-ang_vel_error / 0.25)
        lin_vel_z_l2 = torch.square(base_lin_vel[:, 2])
        ang_vel_xy_l2 = torch.sum(torch.square(base_ang_vel[:, :2]), dim=1)
        dof_torques_l2 = torch.sum(torch.square(joint_torque), dim=1)
        dof_acc_l2 = torch.sum(torch.square(joint_acc), dim=1)
        action_rate_l2 = torch.sum(torch.square(self.obs_last_action - rollout_action), dim=1)
        if foot_contact is not None:
            if not self.timers_before_rewards:
                first_contact = (self.current_contact_time > 0.0) * (self.current_contact_time < (self._step_dt + 1.0e-8))
                # Go2 morphology: feet_air_time threshold 0.25 by default (matches UnitreeGo2FlatEnvCfg)
                feet_air_time = torch.sum((self.last_air_time - self.feet_air_time_threshold) * first_contact, dim=1) \
                    * (torch.norm(self.base_velocity[:, :2], dim=1) > 0.1)

            is_contact = foot_contact.bool()
            is_first_contact = (self.current_air_time > 0) * is_contact
            is_first_detached = (self.current_contact_time > 0) * ~is_contact
            self.last_air_time = torch.where(
                is_first_contact,
                self.current_air_time + self._step_dt,
                self.last_air_time,
            )
            self.current_air_time = torch.where(
                ~is_contact, self.current_air_time + self._step_dt, 0.0
            )
            self.last_contact_time = torch.where(
                is_first_detached,
                self.current_contact_time + self._step_dt,
                self.last_contact_time,
            )
            self.current_contact_time = torch.where(
                is_contact, self.current_contact_time + self._step_dt, 0.0
            )
            if self.timers_before_rewards:
                first_contact = (self.current_contact_time > 0.0) * (self.current_contact_time < (self._step_dt + 1.0e-8))
                feet_air_time = torch.sum((self.last_air_time - self.feet_air_time_threshold) * first_contact, dim=1) \
                    * (torch.norm(self.base_velocity[:, :2], dim=1) > 0.1)
            # unitree air_time_variance_penalty: variance over feet of the clipped last air / contact times
            air_time_variance = torch.var(torch.clip(self.last_air_time, max=0.5), dim=1) \
                + torch.var(torch.clip(self.last_contact_time, max=0.5), dim=1)
        else:
            feet_air_time = torch.zeros(self.num_envs, device=self.device)
            air_time_variance = torch.zeros(self.num_envs, device=self.device)
        # stand_still: Go2 collision proxy of the June / Start-v0 recipe (s2 removed it; see S2_REWARD_CONTRACT)
        stand_still = torch.sum(torch.abs(joint_pos), dim=1) * (torch.norm(self.base_velocity, dim=1) < 0.05)
        flat_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=1)
        if self.soft_joint_pos_limits is not None and self.default_joint_pos is not None:
            # Isaac Lab joint_pos_limits on absolute joint positions (state holds joint_pos - default)
            q = joint_pos + torch.as_tensor(self.default_joint_pos, device=self.device)
            lim = torch.as_tensor(self.soft_joint_pos_limits, device=self.device)
            dof_pos_limits = torch.sum(-(q - lim[:, 0]).clip(max=0.0) + (q - lim[:, 1]).clip(min=0.0), dim=1)
        else:
            dof_pos_limits = torch.zeros(self.num_envs, device=self.device)
        # s2 terms (unitree_rl_lab 4960b84 functions, rewards.py of mbrl): all from state, command and contacts
        joint_vel_l2 = torch.sum(torch.square(joint_vel), dim=1)
        energy = torch.sum(torch.abs(joint_vel) * torch.abs(joint_torque), dim=1)
        moving = torch.logical_or(torch.linalg.norm(self.base_velocity, dim=1) > 0.0,
                                  torch.linalg.norm(base_lin_vel[:, :2], dim=1) > self.joint_pos_velocity_threshold)
        joint_pos_dev = torch.linalg.norm(joint_pos, dim=1)
        joint_position_penalty = torch.where(moving, joint_pos_dev, self.joint_pos_stand_still_scale * joint_pos_dev)
        # undesired_contacts: the world model predicts thigh contacts only (hip and calf are not predicted)
        undesired_contacts = torch.sum(thigh_contact, dim=1) if thigh_contact is not None \
            else torch.zeros(self.num_envs, device=self.device)
        # Isaac Lab feet_slide reconstructed by leg kinematics from the imagined state and the
        # predicted foot contact (same gate as the sensor). Weight is set in go2_flat_cfg.
        if foot_contact is not None:
            feet_slide = go2_feet_slide(base_lin_vel, base_ang_vel, projected_gravity, joint_pos, joint_vel,
                                        foot_contact, base_com_offset=self.feet_slide_base_com_offset)
        else:
            feet_slide = torch.zeros(self.num_envs, device=self.device)
        self.imagination_reward_per_step = {
            "track_lin_vel_xy_exp": track_lin_vel_xy_exp,
            "track_ang_vel_z_exp": track_ang_vel_z_exp,
            "lin_vel_z_l2": lin_vel_z_l2,
            "ang_vel_xy_l2": ang_vel_xy_l2,
            "dof_torques_l2": dof_torques_l2,
            "dof_acc_l2": dof_acc_l2,
            "action_rate_l2": action_rate_l2,
            "feet_air_time": feet_air_time,
            "stand_still": stand_still,
            "flat_orientation_l2": flat_orientation_l2,
            "dof_pos_limits": dof_pos_limits,
            "feet_slide": feet_slide,
            "joint_vel": joint_vel_l2,
            "energy": energy,
            "joint_pos": joint_position_penalty,
            "air_time_variance": air_time_variance,
            "undesired_contacts": undesired_contacts,
        }
        last_obs = torch.cat([base_lin_vel, base_ang_vel, projected_gravity, self.base_velocity, joint_pos, joint_vel, rollout_action], dim=1)
        self.last_obs = TensorDict({"policy": last_obs}, batch_size=[self.num_envs], device=self.device)

    def _apply_interval_events(self, imagination_states_denormalized, parsed_imagination_states, event_ids):
        if len(event_ids) == 0:
            imagination_states, _ = self.dataset.normalize(imagination_states_denormalized, None)
        else:
            base_lin_vel = parsed_imagination_states["base_lin_vel"]
            velocity_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
            r = torch.empty(len(event_ids), device=self.device)
            base_lin_vel[event_ids, 0] += r.uniform_(*velocity_range["x"])
            base_lin_vel[event_ids, 1] += r.uniform_(*velocity_range["y"])
            imagination_states_denormalized[event_ids, 0:3] = base_lin_vel[event_ids, 0:3]
            imagination_states, _ = self.dataset.normalize(imagination_states_denormalized, None)
        return imagination_states

    @property
    def state_dim(self):
        return 45

    @property
    def observation_dim(self):
        return 48

    @property
    def action_dim(self):
        return 12
