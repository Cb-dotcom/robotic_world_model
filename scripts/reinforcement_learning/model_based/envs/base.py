import torch
from tensordict import TensorDict


class BaseEnv:
    # F1 (plan v6 §3, amendment A1, 19 Nov 2026): per-member trajectory spread penalty. Off when f1_members = 0
    # (standard RWM-U penalty, unchanged). On: every env carries K member paths (head k of the ensemble on its
    # own path, own trunk and auxiliary memory, all driven by the policy's actions); the policy follows the
    # member of its driving head, so rewards, terminations and contacts equal the standard path's; the penalty
    # term is uncertainty_penalty_weight * f1_gain * f1_scale[d] * spread, spread = sum over the normalized
    # state dims of the std over members, d = imagined states fed back since the episode start.
    f1_members = 0
    f1_scale = None          # s(d), d = 0 .. len - 1 (the last value is held beyond, never reached at d <= 255)
    f1_gain = None           # g, fixed by the declared mean-matching rule (A1 §2)

    def __init__(
        self,
        num_envs: int,
        max_episode_length: int,
        step_dt: float,
        reward_term_weights: dict,
        device: str,
        uncertainty_penalty_weight:float = -0.0,
        observation_noise: bool = False,
        command_resample_interval_range: list | None = [500, 500],
        event_interval_range: list | None = None,
        ):
        self._num_envs = num_envs
        self._max_episode_length = max_episode_length
        self._step_dt = step_dt
        self.device = device

        self.reward_term_weights = reward_term_weights
        self.uncertainty_penalty_weight = uncertainty_penalty_weight
        self.observation_noise = observation_noise
        self.command_resample_interval_range = command_resample_interval_range
        self.event_interval_range = event_interval_range
        self._init_additional_attributes()
        self.dummy_obs = TensorDict({"policy": torch.zeros(num_envs, self.observation_dim)}, batch_size=[num_envs], device=self.device)

        # set in experiment
        self.system_dynamics = None
        self.dataset = None
        self.init_dataset = None
        self.init_data_ratio = None

    def _init_additional_attributes(self):
        pass
    
    def set_system_dynamics(self, system_dynamics):
        self.system_dynamics = system_dynamics
        self.system_dynamics.eval()
        
    def set_dataset(self, dataset):
        self.dataset = dataset
        
    def set_init_dataset(self, init_dataset, init_data_ratio=0.0):
        self.init_dataset = init_dataset
        self.init_data_ratio = init_data_ratio

    def prepare_imagination(self):
        self.common_step_counter = 0
        self.system_dynamics.reset()
        self.system_dynamics_model_ids = torch.randint(0, self.system_dynamics.ensemble_size, (1, self._num_envs, 1), device=self.device)
        self._init_imagination_reward_buffer()
        self._init_intervals()
        self._init_additional_imagination_attributes()
        state_history, action_history = self._init_imagination_history()
        if self._f1_on():
            self._f1_init(state_history)
        self._init_imagination_command()
        self.last_obs = TensorDict({"policy": torch.zeros(self._num_envs, self.observation_dim)}, batch_size=[self._num_envs], device=self.device)
        self.extras = {}
        self._reset_idx(torch.arange(self._num_envs, device=self.device))
        return state_history, action_history
    
    def _reset_idx(self, env_ids):
        self.extras["log"] = dict()
        if self._f1_on():
            # memory rows of every member path of these envs; the driving head is drawn among the K members
            self.system_dynamics.reset_partial(self._f1_rows(env_ids))
            high = int(self.f1_members)
        else:
            self.system_dynamics.reset_partial(env_ids)
            high = self.system_dynamics.ensemble_size
        self.system_dynamics_model_ids[:, env_ids, :] = torch.randint(0, high, (1, len(env_ids), 1), device=self.device)
        info = self._reset_imagination_reward_buffer(env_ids)
        self.extras["log"].update(info)
        self._reset_intervals(env_ids)
        self._reset_additional_imagination_attributes(env_ids)
        self.last_obs["policy"][env_ids] = 0.0

    def _init_imagination_history(self):
        num_init_envs = int(self._num_envs * self.init_data_ratio)
        num_random_envs = self._num_envs - num_init_envs
        state_history_init, action_history_init = self.init_dataset.sample_batch(num_init_envs, normalized=True)
        state_history_random, action_history_random = self.dataset.sample_batch(num_random_envs, normalized=True)
        state_history = torch.cat([state_history_init, state_history_random], dim=0)
        action_history = torch.cat([action_history_init, action_history_random], dim=0)
        return state_history, action_history

    def _reset_imagination_history(self, env_ids, state_history, action_history):
        num_reset_envs = len(env_ids)
        num_init_envs = int(num_reset_envs * self.init_data_ratio)
        num_random_envs = num_reset_envs - num_init_envs
        state_history_init, action_history_init = self.init_dataset.sample_batch(num_init_envs, normalized=True)
        state_history_random, action_history_random = self.dataset.sample_batch(num_random_envs, normalized=True)
        state_new = torch.cat([state_history_init, state_history_random], dim=0)
        action_new = torch.cat([action_history_init, action_history_random], dim=0)
        if getattr(self, "warm_start_resets", False) and self.common_step_counter > 0:
            startup = self._startup_mask(len(env_ids))
            if startup is not None and bool(startup.any()):
                # A4 D6 (opt-in): these envs start from a recorded physical reset state with ZERO memory (reset_partial
                # in _reset_idx already zeroed it) and the recorded previous action, as a physical episode starts and
                # as the fitter's startup windows (history = the reset row only) were trained.
                rs, ra = self._startup_sample(int(startup.sum()))
                state_new = state_new.clone()
                action_new = action_new.clone()
                state_new[startup, -1] = rs
                action_new[startup, -1] = ra
            keep = slice(None) if startup is None else ~startup
            # B7 fix (opt-in): the next step feeds only the last state, so rebuild the GRU memory of the
            # reset envs from the full sampled history. Then the first prediction after a reset sees the
            # same 32-step context as the very first imagination step (and as WM training).
            if startup is None or bool(keep.any()):
                self._warm_memory(env_ids[keep], state_new[keep], action_new[keep])
        state_history[env_ids] = state_new[:, -state_history.shape[1]:]
        action_history[env_ids] = action_new[:, -action_history.shape[1]:]
        return state_history, action_history

    # A4 D6 (22 Nov 2026): startup resets. Off unless startup_reset_share > 0 and startup_reset_raw is set (raw recorded
    # reset states [R, 45] and previous actions [R, 12]); used only with warm_start_resets after the first step.
    startup_reset_share = 0.0
    startup_reset_raw = None

    def _startup_mask(self, n):
        q = float(getattr(self, "startup_reset_share", 0.0) or 0.0)
        if q <= 0.0:
            return None
        if getattr(self, "startup_reset_raw", None) is None:
            raise ValueError("startup_reset_share > 0 needs startup_reset_raw (recorded reset states)")
        m = torch.rand(n, device=self.device) < q
        log = self.extras.setdefault("log", {}) if hasattr(self, "extras") else {}
        log["startup_reset_fraction"] = float(m.float().mean()) if n else 0.0
        self.startup_reset_count = int(getattr(self, "startup_reset_count", 0)) + int(m.sum())
        return m

    def _startup_sample(self, k):
        if getattr(self, "_startup_norm", None) is None:
            st, pa = self.startup_reset_raw
            st = torch.as_tensor(st, dtype=torch.float32, device=self.device)
            pa = torch.as_tensor(pa, dtype=torch.float32, device=self.device)
            if st.ndim != 2 or st.shape[0] == 0 or st.shape[0] != pa.shape[0]:
                raise ValueError("startup_reset_raw must be two arrays [R, 45] and [R, 12] with R > 0")
            ns, na = self.dataset.normalize(st, pa)
            self._startup_norm = (ns.reshape(st.shape[0], -1), na.reshape(pa.shape[0], -1))
        st, pa = self._startup_norm
        idx = torch.randint(0, st.shape[0], (k,), device=self.device)
        return st[idx], pa[idx]

    def _warm_memory(self, env_ids, state_new, action_new):
        """Set the GRU memory of env_ids to what a zero-start pass over the sampled history leaves.

        Pairs follow imagination_step: state j goes with action j+1. The last state is fed by the next
        step together with the policy's new action, so the warm-up covers states 0..H-2.
        """
        if state_new.shape[1] < 2:
            return
        x = torch.cat([state_new[:, :-1], action_new[:, 1:]], dim=-1)
        for base in (self.system_dynamics.state_base, self.system_dynamics.auxiliary_base):
            mem = getattr(base, "memory", None)
            if mem is None:  # MLP base: no memory to warm
                continue
            if isinstance(mem.hidden_states, tuple):
                raise NotImplementedError("warm_start_resets supports GRU memory only")
            _, h = mem.rnn(x)
            if mem.hidden_states is None:
                mem.hidden_states = torch.zeros(h.shape[0], self._num_envs, h.shape[2], device=h.device, dtype=h.dtype)
            mem.hidden_states[:, env_ids] = h

    def _init_imagination_reward_buffer(self):
        self.episode_length_buf = torch.zeros(self._num_envs, device=self.device, dtype=torch.long)
        self.imagination_episode_sums = {
            term: torch.zeros(
                self._num_envs,
                device=self.device
                ) for term in self.reward_term_weights.keys()
            }
        self.imagination_episode_sums["uncertainty"] = torch.zeros(self._num_envs, device=self.device)
        self.imagination_reward_per_step = {
            term: torch.zeros(
                self._num_envs,
                device=self.device
                ) for term in self.reward_term_weights.keys()
            }
    
    def _reset_imagination_reward_buffer(self, env_ids):
        extras = {}
        self.episode_length_buf[env_ids] = 0
        for term in self.imagination_episode_sums.keys():
            episodic_sum_avg = torch.mean(self.imagination_episode_sums[term][env_ids])
            extras[term] = episodic_sum_avg / (self._max_episode_length * self._step_dt)
            self.imagination_episode_sums[term][env_ids] = 0.0
        return extras

    def _init_intervals(self):
        if self.command_resample_interval_range is None:
            return
        else:
            self.command_resample_intervals = torch.randint(self.command_resample_interval_range[0], self.command_resample_interval_range[1], (self._num_envs,), device=self.device)
        if self.event_interval_range is None:
            return
        else:
            self.event_intervals = torch.randint(self.event_interval_range[0], self.event_interval_range[1], (self._num_envs,), device=self.device)

    def _reset_intervals(self, env_ids):
        if self.command_resample_interval_range is None:
            return
        else:
            self.command_resample_intervals[env_ids] = torch.randint(self.command_resample_interval_range[0], self.command_resample_interval_range[1], (len(env_ids),), device=self.device)
        if self.event_interval_range is None:
            return
        else:
            self.event_intervals[env_ids] = torch.randint(self.event_interval_range[0], self.event_interval_range[1], (len(env_ids),), device=self.device)

    def imagination_step(self, rollout_action, state_history, action_history):
        _, rollout_action_normalized = self.dataset.normalize(None, rollout_action)
        action_history = torch.cat([action_history[:, 1:], rollout_action_normalized.unsqueeze(1)], dim=1)
        if self._f1_on():
            imagination_states, aleatoric_uncertainty, self.epistemic_uncertainty, extensions, contacts, terminations = \
                self._f1_forward(state_history, action_history)
        else:
            imagination_states, aleatoric_uncertainty, self.epistemic_uncertainty, extensions, contacts, terminations = \
                self.system_dynamics.forward(state_history, action_history, self.system_dynamics_model_ids)
        driving_prediction = imagination_states
        imagination_states_denormalized, _ = self.dataset.denormalize(imagination_states, None)
        parsed_imagination_states = self._parse_imagination_states(imagination_states_denormalized)
        parsed_extensions = self._parse_extensions(extensions)
        parsed_contacts = self._parse_contacts(contacts)
        self.termination_flags = self._parse_terminations(terminations)
        self._compute_imagination_reward_terms(parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts)
        rewards, dones, extras = self._post_imagination_step()
        command_ids = self._process_command_env_ids()
        self._reset_imagination_command(command_ids)
        event_ids = self._process_event_env_ids()
        imagination_states = self._apply_interval_events(imagination_states_denormalized, parsed_imagination_states, event_ids)
        if self._f1_on():
            self._f1_advance(driving_prediction, imagination_states, event_ids)
        state_history = torch.cat([state_history[:, 1:], imagination_states.unsqueeze(1)], dim=1)
        reset_env_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            state_history, action_history = self._reset_imagination_history(reset_env_ids, state_history, action_history)
            if self._f1_on():
                self._f1_restart(reset_env_ids, state_history)
        return self.last_obs, rewards, dones, extras, state_history, action_history, self.epistemic_uncertainty

    def _post_imagination_step(self):
        self.episode_length_buf += 1
        self.common_step_counter += 1
        rewards = torch.zeros(self._num_envs, dtype=torch.float, device=self.device)
        for term in self.imagination_episode_sums.keys():
            if term == "uncertainty":
                penalty = self.f1_penalty if self._f1_on() else self.epistemic_uncertainty
                rewards += self.uncertainty_penalty_weight * penalty * self._step_dt
                self.imagination_episode_sums[term] += self.uncertainty_penalty_weight * penalty * self._step_dt
            else:
                term_value = self.imagination_reward_per_step[term]
                rewards += self.reward_term_weights[term] * term_value * self._step_dt
                self.imagination_episode_sums[term] += self.reward_term_weights[term] * term_value * self._step_dt
        
        alive_bonus = float(getattr(self, "alive_bonus", 0.0))
        if alive_bonus:
            # Opt-in (27 Oct 2026): constant per-step survival bonus. With fixed-length episodes it
            # leaves the optimal policy unchanged; with imagined stops it makes a stop cost future reward.
            rewards = rewards + alive_bonus
        if getattr(self, "only_positive_rewards", False):
            # Opt-in (27 Oct 2026, legged_gym "only_positive_rewards"): clip the total imagined
            # reward at zero so that ending an imagined episode early can never pay.
            rewards = torch.clamp(rewards, min=0.0)
        terminated = self.termination_flags if self.termination_flags is not None else torch.zeros(self._num_envs, dtype=torch.bool, device=self.device)
        time_outs = self.episode_length_buf >= self._max_episode_length
        dones = (terminated | time_outs).to(dtype=torch.long)
        
        reset_env_ids = (terminated | time_outs).nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self._reset_idx(reset_env_ids)
        self.extras["time_outs"] = time_outs
        return rewards, dones, self.extras
    
    def _process_command_env_ids(self):
        if self.command_resample_interval_range is None:
            return torch.empty(0, dtype=torch.int, device=self.device)
        else:
            return (self.episode_length_buf % self.command_resample_intervals == 0).nonzero(as_tuple=False).squeeze(-1)

    def _process_event_env_ids(self):
        if self.event_interval_range is None:
            return torch.empty(0, dtype=torch.int, device=self.device)
        else:
            return (self.episode_length_buf % self.event_intervals == 0).nonzero(as_tuple=False).squeeze(-1)
        
    # ------------------------------------------------------------------ F1 (amendment A1)
    def _f1_on(self):
        return int(getattr(self, "f1_members", 0) or 0) > 0

    def _f1_rows(self, env_ids):
        """Rows of the member batch (member-major: row = k * num_envs + env) for env_ids."""
        k = torch.arange(int(self.f1_members), device=self.device).view(-1, 1)
        return (k * self._num_envs + env_ids.view(1, -1)).reshape(-1)

    def _f1_init(self, state_history):
        K = int(self.f1_members)
        if K > self.system_dynamics.ensemble_size:
            raise ValueError(f"f1_members {K} > ensemble size {self.system_dynamics.ensemble_size}")
        if getattr(self, "warm_start_resets", False):
            raise NotImplementedError("F1 with warm-start resets is not implemented (A1 uses paper mode)")
        rule = getattr(self, "termination_rule", None)
        if rule is not None and (rule[0] != "selected" or rule[2] is not None):
            raise NotImplementedError("F1 supports the default 'selected' termination rule only (A1)")
        if self.f1_scale is None or self.f1_gain is None:
            raise ValueError("F1 needs f1_scale (s(d)) and f1_gain (g) from the frozen calibration")
        self._f1_scale_t = torch.as_tensor(self.f1_scale, dtype=torch.float32, device=self.device).flatten()
        if self._f1_scale_t.numel() < self._max_episode_length:
            raise ValueError(f"f1_scale covers {self._f1_scale_t.numel()} depths < episode length {self._max_episode_length}")
        self.f1_states = state_history.repeat(K, 1, 1).clone()                 # [K*N, H, S], member-major
        self.f1_depth = torch.zeros(self._num_envs, dtype=torch.long, device=self.device)
        self._f1_ids = torch.arange(K, device=self.device).repeat_interleave(self._num_envs).view(1, -1, 1)
        self.f1_spread = torch.zeros(self._num_envs, device=self.device)
        self.f1_penalty = torch.zeros(self._num_envs, device=self.device)

    def _f1_forward(self, state_history, action_history):
        """One step of every member path. Returns the driving member's outputs, in the standard forward's order."""
        K, N, L = int(self.f1_members), self._num_envs, state_history.shape[1]
        xs = self.f1_states[:, -L:]                                             # same length as the driving input
        xa = action_history.repeat(K, 1, 1)
        means, aleatoric, epistemic, extensions, contacts, terminations = self.system_dynamics.forward(xs, xa, self._f1_ids)
        drive = self.system_dynamics_model_ids.view(1, N, 1)

        def pick(t):
            if t is None:
                return None
            flat = t.reshape(K, N, -1)
            return torch.gather(flat, 0, drive.expand(1, N, flat.shape[-1])).squeeze(0).reshape((N,) + tuple(t.shape[1:]))

        member_means = means.reshape(K, N, -1)
        self._f1_member_means = member_means
        self.f1_spread = member_means.std(dim=0).sum(dim=1)
        depth = self.f1_depth.clamp(max=self._f1_scale_t.numel() - 1)
        self.f1_penalty = float(self.f1_gain) * self._f1_scale_t[depth] * self.f1_spread
        return pick(means), pick(aleatoric), pick(epistemic), pick(extensions), pick(contacts), pick(terminations)

    def _f1_advance(self, driving_prediction, driving_after_events, event_ids):
        """Append each member's prediction; an interval push shifts every member by the driving path's shift."""
        member_next = self._f1_member_means.clone()
        if len(event_ids) > 0:
            member_next[:, event_ids] += (driving_after_events[event_ids] - driving_prediction[event_ids]).unsqueeze(0)
        # the driving member continues exactly as the driving path
        drive = self.system_dynamics_model_ids.view(1, -1, 1).expand(1, self._num_envs, member_next.shape[-1])
        member_next.scatter_(0, drive, driving_after_events.unsqueeze(0))
        self.f1_states = torch.cat([self.f1_states[:, 1:], member_next.reshape(-1, 1, member_next.shape[-1])], dim=1)
        self.f1_depth += 1

    def _f1_restart(self, env_ids, state_history):
        """Members of reset envs restart from the driving path's new start rows; depth 0."""
        L = state_history.shape[1]
        rows = self._f1_rows(env_ids)
        self.f1_states[rows, -L:] = state_history[env_ids].repeat(int(self.f1_members), 1, 1)
        self.f1_depth[env_ids] = 0

    @property
    def num_envs(self):
        return self._num_envs
    
    @property
    def max_episode_length(self):
        return self._max_episode_length

    def _init_additional_imagination_attributes(self):
        raise NotImplementedError

    def _reset_additional_imagination_attributes(self, env_ids):
        raise NotImplementedError

    def _init_imagination_command(self):
        raise NotImplementedError

    def _reset_imagination_command(self, env_ids):
        raise NotImplementedError

    def get_imagination_observation(self, state_history, action_history):
        raise NotImplementedError

    def _parse_imagination_states(self, imagination_states_denormalized):
        raise NotImplementedError

    def _parse_extensions(self, extensions):
        raise NotImplementedError

    def _parse_contacts(self, contacts):
        raise NotImplementedError
    
    def _parse_terminations(self, terminations):
        raise NotImplementedError

    def _compute_imagination_reward_terms(self, parsed_imagination_states, rollout_action, parsed_extensions, parsed_contacts):
        raise NotImplementedError

    def _apply_interval_events(self, imagination_states_denormalized, parsed_imagination_states, event_ids):
        raise NotImplementedError

    @property
    def state_dim(self):
        raise NotImplementedError
    
    @property
    def observation_dim(self):
        raise NotImplementedError
    
    @property
    def action_dim(self):
        raise NotImplementedError
