"""Opt-in imagination termination rule (27 Oct 2026).

ETH's offline env ends an imagined episode when the *selected* ensemble head's
termination probability rounds to 1 (p > 0.5). The pre-reset transition
diagnostic (logs/p2/transition_fidelity.ELf50z) showed that rule rarely fires on
real falls (per-head recall 6-56% at the fall step) although the probabilities
separate falls from survival by two orders of magnitude.

This module changes only that decision:
  rule "selected": selected head probability > threshold  (ETH default at 0.5)
  rule "max":      max over all heads > threshold           (pessimistic ensemble)
  rule "mean":     mean over all heads > threshold
Optional uncertainty stop (27 Oct, after the paired fall replay): also end the
imagined episode when the ensemble disagreement of the step (the same quantity
the penalty uses) exceeds ``uncertainty_threshold``. Inside imagination the fall
signal fades while the real error grows; disagreement is still high early on.
State predictions, rewards, the uncertainty penalty and the selected head are
unchanged. ``install`` must run before the training builds its environment; the
default ("selected", 0.5) installs nothing.
"""
from __future__ import annotations

import torch

RULES = ("selected", "max", "mean")
DEFAULT = ("selected", 0.5)


def ensemble_forward_with_head_terminations(self, x_state_batch, x_action_batch, model_ids=None):
    """Copy of rsl_rl_rwm 6b4eaf6 SystemDynamicsEnsemble.forward, plus one stash.

    Stores the termination logits of every head in ``self.last_head_termination_logits``
    ([ensemble, batch, termination_dim]). Outputs are identical to the original.
    """
    state_means, state_stds, extensions, contacts, terminations = [], [], [], [], []
    state_base_output = self.state_base(x_state_batch, x_action_batch)
    for head in self.state_heads:
        state_mean, state_std = head(state_base_output, x_state_batch)
        if self.prediction_type == "sequence":
            state_mean = state_mean[:, -1]
            state_std = state_std[:, -1]
        state_means.append(state_mean.unsqueeze(0))
        state_stds.append(state_std.unsqueeze(0))
    auxiliary_base_output = self.auxiliary_base(x_state_batch, x_action_batch)
    for head in self.auxiliary_heads:
        extension, contact, termination = head(auxiliary_base_output, x_state_batch)
        if self.prediction_type == "sequence":
            extension = extension[:, -1] if extension is not None else None
            contact = contact[:, -1] if contact is not None else None
            termination = termination[:, -1] if termination is not None else None
        extensions.append(extension.unsqueeze(0) if extension is not None else None)
        contacts.append(contact.unsqueeze(0) if contact is not None else None)
        terminations.append(termination.unsqueeze(0) if termination is not None else None)
    state_means = torch.cat(state_means, dim=0)
    state_stds = torch.cat(state_stds, dim=0)
    extensions = torch.cat(extensions, dim=0) if self.extension_dim > 0 else None
    contacts = torch.cat(contacts, dim=0) if self.contact_dim > 0 else None
    terminations = torch.cat(terminations, dim=0) if self.termination_dim > 0 else None
    self.last_head_termination_logits = terminations
    if model_ids is None:
        output_state_means = state_means.mean(dim=0)
        output_extensions = extensions.mean(dim=0) if extensions is not None else None
        output_contacts = contacts.mean(dim=0) if contacts is not None else None
        output_terminations = terminations.mean(dim=0) if terminations is not None else None
    else:
        output_state_means = torch.gather(state_means, 0, model_ids.repeat(1, 1, self.state_dim)).squeeze(0)
        output_extensions = torch.gather(extensions, 0, model_ids.repeat(1, 1, self.extension_dim)).squeeze(0) if extensions is not None else None
        output_contacts = torch.gather(contacts, 0, model_ids.repeat(1, 1, self.contact_dim)).squeeze(0) if contacts is not None else None
        output_terminations = torch.gather(terminations, 0, model_ids.repeat(1, 1, self.termination_dim)).squeeze(0) if terminations is not None else None
    aleatoric_uncertainty = state_stds.mean(dim=0).sum(dim=1)
    epistemic_uncertainty = state_means.std(dim=0).sum(dim=1) if self.ensemble_size > 1 else torch.zeros(output_state_means.shape[0], device=self.device)
    return output_state_means, aleatoric_uncertainty, epistemic_uncertainty, output_extensions, output_contacts, output_terminations


def decide(selected_logits, head_logits, rule, threshold, epistemic=None, uncertainty_threshold=None):
    """Termination decision [batch] (bool). ``head_logits``: [ensemble, batch, 1]."""
    stop = _term_decision(selected_logits, head_logits, rule, threshold)
    if uncertainty_threshold is not None:
        if epistemic is None or epistemic.shape[0] != stop.shape[0]:
            raise RuntimeError("Epistemic uncertainty missing or misaligned")
        stop = stop | (epistemic.reshape(-1) > uncertainty_threshold)
    return stop


def _term_decision(selected_logits, head_logits, rule, threshold):
    if threshold is None:
        return torch.zeros(selected_logits.shape[0], dtype=torch.bool, device=selected_logits.device)
    if rule == "selected":
        p = torch.sigmoid(selected_logits).squeeze(-1)
    elif rule == "max":
        p = torch.sigmoid(head_logits).squeeze(-1).max(dim=0).values
    elif rule == "mean":
        p = torch.sigmoid(head_logits).squeeze(-1).mean(dim=0)
    else:
        raise ValueError(f"Unknown termination rule {rule!r}")
    return p > threshold


def install(rule: str, threshold, env_cls=None, ensemble_cls=None, uncertainty_threshold=None):
    """Patch the ensemble forward (stash only) and the env's termination parser.

    ``threshold=None`` disables the head rule (uncertainty stop only)."""
    if rule not in RULES:
        raise ValueError(f"termination rule must be one of {RULES}")
    if threshold is not None:
        threshold = float(threshold)
        if not 0.0 < threshold < 1.0:
            raise ValueError("termination threshold must lie in (0, 1)")
    if uncertainty_threshold is not None:
        uncertainty_threshold = float(uncertainty_threshold)
        if not uncertainty_threshold > 0.0:
            raise ValueError("uncertainty threshold must be > 0")
    if threshold is None and uncertainty_threshold is None:
        raise ValueError("at least one of threshold / uncertainty_threshold is required")
    if (rule, threshold) == DEFAULT and uncertainty_threshold is None:
        return False
    if ensemble_cls is None:
        from rsl_rl.modules import SystemDynamicsEnsemble as ensemble_cls
    if env_cls is None:
        from envs.anymal_d_flat import AnymalDFlatEnv as env_cls
    ensemble_cls.forward = ensemble_forward_with_head_terminations

    def _parse_terminations(self, terminations):
        if terminations is None:
            return None
        heads = getattr(self.system_dynamics, "last_head_termination_logits", None)
        if threshold is not None and rule != "selected" and (heads is None or heads.shape[1] != terminations.shape[0]):
            raise RuntimeError("Head termination logits missing or misaligned")
        # BaseEnv.imagination_step stores this step's disagreement before parsing terminations.
        epistemic = getattr(self, "epistemic_uncertainty", None) if uncertainty_threshold is not None else None
        return decide(terminations, heads, rule, threshold, epistemic, uncertainty_threshold)

    env_cls._parse_terminations = _parse_terminations
    env_cls.termination_rule = (rule, threshold, uncertainty_threshold)
    return True
