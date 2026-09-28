from copy import deepcopy
from functools import wraps
from inspect import signature
from numbers import Integral, Real
from time import perf_counter

import torch

def _samples(result):
    value = result[0] if isinstance(result, tuple) else result
    if not isinstance(value, torch.Tensor) or value.ndim != 3:
        raise TypeError("Sampler must return a [B, K, D] tensor or a tuple beginning with it.")
    if not value.is_floating_point() or value.shape[2] == 0:
        raise ValueError("Sample coordinates must be floating point with D > 0.")
    return value

def _nonempty(result, expected):
    samples = _samples(result)
    if samples.shape[0] != expected:
        raise ValueError(f"Requested {expected} samples but sampler returned {samples.shape[0]}.")
    finite = torch.isfinite(samples).all(dim=-1)
    padding = torch.isposinf(samples).all(dim=-1) | torch.isneginf(samples).all(dim=-1)
    if not (finite | padding).all():
        raise ValueError("Malformed samples: NaN or partially nonfinite point rows; not empty sets.")
    return finite.any(dim=-1)

def _padding_value(samples):
    if torch.isposinf(samples).any():
        return float("inf")
    return float("-inf")

def _replace(result, replacement, dst, src):
    """Replace only rejected rows, retaining and aligning per-sample auxiliaries."""
    old_samples, new_samples = _samples(result), _samples(replacement)
    old_width, new_width = old_samples.shape[1], new_samples.shape[1]
    width = max(old_width, new_width)
    padding_value = _padding_value(old_samples)

    def merge(old, new, *, coordinates=False):
        if isinstance(old, torch.Tensor):
            if not isinstance(new, torch.Tensor) or old.ndim != new.ndim or old.ndim == 0:
                raise TypeError("Sampler auxiliaries must be matching per-sample tensors.")
            if old.shape[0] != old_samples.shape[0] or new.shape[0] != new_samples.shape[0]:
                raise ValueError("Sampler auxiliary tensors must have the sample batch dimension first.")
            if old.dtype != new.dtype or old.device != new.device:
                raise ValueError("Sampler changed tensor dtype or device during rejection sampling.")
            per_point = old.ndim >= 2 and old.shape[1] == old_width and new.shape[1] == new_width
            if per_point:
                if old.shape[2:] != new.shape[2:]:
                    raise ValueError("Sampler changed feature shape during rejection sampling.")
                fill = padding_value if coordinates else 0
                merged = old.new_full((old.shape[0], width, *old.shape[2:]), fill)
                merged[:, :old_width] = old
                merged[dst] = fill
                merged[dst, :new_width] = new[src]
            else:
                if old.shape[1:] != new.shape[1:]:
                    raise ValueError("Sampler changed auxiliary shape during rejection sampling.")
                merged = old.clone()
                merged[dst] = new[src]
            return merged
        if isinstance(old, tuple) and isinstance(new, tuple) and len(old) == len(new):
            return tuple(merge(a, b) for a, b in zip(old, new))
        if isinstance(old, dict) and isinstance(new, dict) and old.keys() == new.keys():
            return {key: merge(old[key], new[key]) for key in old}
        if old is None and new is None:
            return None
        raise TypeError("Sampler auxiliaries must be per-sample tensors, tuples, dictionaries, or None.")

    if isinstance(result, tuple):
        if not isinstance(replacement, tuple) or len(result) != len(replacement):
            raise TypeError("Sampler changed return type during rejection sampling.")
        return (merge(result[0], replacement[0], coordinates=True),
                *(merge(a, b) for a, b in zip(result[1:], replacement[1:])))
    if isinstance(replacement, tuple):
        raise TypeError("Sampler changed return type during rejection sampling.")
    return merge(result, replacement, coordinates=True)

def sample_nonempty(draw, num_samples, *, stats_owner=None, max_refills=100, draw_with_indices=False):

    if isinstance(num_samples, bool) or not isinstance(num_samples, Integral) or num_samples < 1:
        raise ValueError("num_samples must be a positive integer.")
    if isinstance(max_refills, bool) or not isinstance(max_refills, Integral) or max_refills < 0:
        raise ValueError("max_refills must be a nonnegative integer.")
    started = perf_counter()
    draws, stats = [], []

    def propose(n, indices):
        before = getattr(stats_owner, "last_sampling_stats", None)
        output = draw(n, indices) if draw_with_indices else draw(n)
        after = getattr(stats_owner, "last_sampling_stats", None)

        fresh = after if isinstance(after, dict) and after is not before else {}
        draws.append(n)
        stats.append(deepcopy(fresh))
        return output, _nonempty(output, n)

    def record(accepted, completed):
        if stats_owner is None:
            return
        result_stats = deepcopy(stats[0])
        cost_keys = ("initialization_nfe", "predictor_nfe", "corrector_nfe", "total", "total_nfe")
        summed_costs = {}
        for key in cost_keys:
            values = [item.get(key) for item in stats]
            if all(isinstance(value, Real) for value in values):
                summed_costs[key] = sum(values)
            else:
                summed_costs[key] = None
        totals = [item.get("total_nfe", item.get("total")) for item in stats]
        known_total = all(isinstance(value, Real) for value in totals)
        result_stats["nonempty_sampling"] = {
            "policy": "reject_empty_resample_missing",
            "requested": int(num_samples),
            "proposed": sum(draws),
            "accepted": accepted,
            "rejected": sum(draws) - accepted,
            "refill_rounds": len(draws) - 1,
            "draw_counts": draws,
            "draw_sampling_stats": stats,
            "aggregate_nfe": summed_costs,
            "network_forward_calls": sum(totals) if known_total else None,
            "sample_weighted_nfe": sum(n * cost for n, cost in zip(draws, totals)) if known_total else None,
            "seconds": perf_counter() - started,
            "completed": completed,
        }
        stats_owner.last_sampling_stats = result_stats

    result, keep = propose(int(num_samples), torch.arange(int(num_samples)))
    missing = (~keep).nonzero(as_tuple=True)[0]
    for _ in range(max_refills):
        if missing.numel() == 0:
            break
        replacement, replacement_keep = propose(int(missing.numel()), missing)
        selected = replacement_keep.nonzero(as_tuple=True)[0]
        if selected.numel():
            result = _replace(result, replacement, missing[selected], selected)
            missing = missing[~replacement_keep]
    accepted = int(num_samples - missing.numel())
    record(accepted, missing.numel() == 0)
    if missing.numel():
        raise RuntimeError(
            f"Nonempty sampling failed after {max_refills} refills: {missing.numel()} of "
            f"{num_samples} sets still missing ({sum(draws)} proposals)."
        )
    return result

def reject_empty_sets(sample_sets=None, *, batch_parameters=()):
    """Apply the shared nonempty policy without changing a method's signature."""
    if sample_sets is None:
        return lambda function: reject_empty_sets(function, batch_parameters=batch_parameters)
    method_signature = signature(sample_sets)
    @wraps(sample_sets)
    def wrapped(self, num_samples, *args, **kwargs):
        bound = method_signature.bind(self, num_samples, *args, **kwargs)
        original_parameters = {
            name: bound.arguments[name] for name in batch_parameters if name in bound.arguments
        }
        def draw(count, indices):
            bound.arguments["num_samples"] = count
            for name, value in original_parameters.items():
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    if value.numel() != num_samples:
                        raise ValueError(f"{name} must be scalar or have {num_samples} entries.")
                    bound.arguments[name] = value.reshape(-1)[indices.to(value.device)]
            return sample_sets(*bound.args, **bound.kwargs)
        result = sample_nonempty(
            draw,
            num_samples,
            stats_owner=self,
            draw_with_indices=True,
        )
        report = self.last_sampling_stats["nonempty_sampling"]
        self.last_nonempty_sampling_stats = report

        if report["refill_rounds"] == 0:
            del self.last_sampling_stats["nonempty_sampling"]
        return result
    wrapped.rejects_empty_sets = True
    return wrapped
