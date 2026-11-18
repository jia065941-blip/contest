"""Frozen CZ upper policy plus V12-R3 PPO on the final public interface.

This module intentionally keeps the old planners, selected A00--A11 candidate,
satellite schedule, strategic tree, HP tree, prior and checkpoint unchanged.
Only public observations, deployment coordinates and action rows are adapted.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from user_agents.base import BaseAgent, BaseDeployment
from .compat import CoordinateAdapter, field


FINAL_ROOT = Path(__file__).resolve().parents[2]
RELEASE = Path(os.environ.get(
    "CZ_RELEASE",
    "/home/ubuntu/yunpan/contest-cz/releases/merged_hp_seed2_20261101",
)).resolve()
APP = RELEASE / "app"
for source in (APP, APP / "deps", APP / "deps/engine", APP / "r3"):
    if str(source) not in sys.path:
        sys.path.append(str(source))

from design.planner_v2 import compile_plan, step_group as frozen_step_group  # noqa: E402

# Load the exact released red task package under a private package name.  The
# final environment owns ``user_agents`` already, so replacing that namespace
# would accidentally swap parts of the official interface.
_LEGACY_RED = APP / ".runtime/environment/user_agents/red"
_legacy_spec = importlib.util.spec_from_file_location(
    "cz_final_legacy_red",
    _LEGACY_RED / "__init__.py",
    submodule_search_locations=[str(_LEGACY_RED)],
)
_legacy_package = importlib.util.module_from_spec(_legacy_spec)
sys.modules.setdefault("cz_final_legacy_red", _legacy_package)
_legacy_spec.loader.exec_module(_legacy_package)
from cz_final_legacy_red.commander import RedBaselineCommander  # noqa: E402
from cz_final_legacy_red.contracts import Position  # noqa: E402
from cz_final_legacy_red.priors import initial_targets_from_observation  # noqa: E402
from r3_runtime.local_runtime import LocalAvoidRuntimePolicy  # noqa: E402
from harness.cooperative_system import runtime as cooperative_runtime  # noqa: E402
from harness.skill_discovery.common import digest as state_digest  # noqa: E402
from skills import cooperation as strategic_compiler  # noqa: E402
from skills import feature_projector, merged_policy, proposal  # noqa: E402
from skills.alternatives import alternatives  # noqa: E402


KIND = {21000: "H", 21001: "M", 21002: "L"}
OWN_INITIAL_CLOCK_S = 1783391450.0
_STATE = None


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


HP = _load("cz_final_frozen_hp_tree", APP / "skills/hp_tree/runtime.py")
PRIOR = _load("cz_final_frozen_prior", APP / "prior_t03_make/policy.py")
PRIOR_PARAMETERS = _read(APP / "prior_t03_make/parameters.json")
CHOICES = _read(RELEASE / "opening/frozen_choices.json")["choices"]


class StrategicOverlay:
    """The released merged strategic selector, with no evaluator instrumentation."""

    MASK = {"KEEP": 0, "A": 1, "B": 2, "AB": 3}

    def __init__(self):
        anchor, provenance = strategic_compiler.load_anchor()
        self.binder = strategic_compiler.EventBinder(anchor)
        self.provenance = provenance
        self.strategy = None
        self.stats = Counter()

    def step(self, plan, current, targets, time_s, memory):
        response = frozen_step_group(plan, current, targets, time_s, memory)
        if self.binder.event is not None or time_s > 900 or time_s % 15:
            return response
        context = strategic_compiler.project_context(
            strategic_compiler._legacy().context(
                plan, current, targets, time_s, OWN_INITIAL_CLOCK_S, response
            )
        )
        before = len(self.binder.attempts)
        try:
            event = self.binder.consider(context, response)
        except (ValueError, KeyError, TypeError) as error:
            self.stats["proposal_rejected:" + str(error)[:150]] += 1
            return response
        if len(self.binder.attempts) != before:
            self.stats["proposal_calls"] += 1
        if event is None:
            return response
        roles = copy.deepcopy(event["roles"])
        features = feature_projector.features(context, roles)
        candidates = alternatives(context, roles)
        public = [
            {"candidate_id": "C" + str(index + 1), "features": candidate["features"]}
            for index, candidate in enumerate(candidates)
        ]
        binding_decision = proposal.choose(features, public)
        tree_binding = {
            "status": "no_alternate_at_first_old_opportunity",
            "roles": None,
            "features": None,
            "prediction": None,
        }
        choices = {"merged": "KEEP"}
        if binding_decision["candidate_id"] is not None:
            index = next(
                index for index, item in enumerate(public)
                if item["candidate_id"] == binding_decision["candidate_id"]
            )
            alternate = candidates[index]
            alternate_roles = alternate["roles"]
            assert alternate_roles[0] == roles[0]
            assert alternate_roles[1]["unit"] != roles[1]["unit"]
            strategic_compiler.validate_roles(alternate_roles, context)
            for mask in (0, 1):
                assert json.dumps(strategic_compiler.merge(response, roles, mask, context)) == json.dumps(
                    strategic_compiler.merge(response, alternate_roles, mask, context)
                )
            prediction = merged_policy.predict(alternate["features"])
            choices["merged"] = prediction["choice"]
            assert choices["merged"] in ("KEEP", "A")
            tree_binding = {
                "status": "valid_alternate",
                "roles": copy.deepcopy(alternate_roles),
                "features": alternate["features"],
                "prediction": prediction,
            }
        choice = choices["merged"]
        mask = self.MASK[choice]
        self.strategy = {
            "time_s": time_s,
            "original_roles": roles,
            "legal_features": features,
            "tree_binding": tree_binding,
            "binding_decision": binding_decision,
            "choices": choices,
            "anchor_provenance": self.provenance,
            "event": copy.deepcopy(event),
            "actors_removed": 0,
            "applied_choice": choice,
            "applied_mask": mask,
            "quality": strategic_compiler.coverage_audit(response, roles, mask, context),
        }
        self.stats["strategic_events"] += 1
        return strategic_compiler.merge(response, roles, mask, context)


class Motion:
    """Unchanged V12-R3 checkpoint and mission_v12 gate."""

    def __init__(self, max_steps):
        import torch

        os.environ["LOCAL_AVOID_GATE_MODE"] = "mission_v12"
        torch.set_num_threads(1)
        self.policy = LocalAvoidRuntimePolicy(max_steps=max_steps)
        self.policy.load(str(APP / "r3/model.pt"))
        assert self.policy.model.algorithm_name == "local_avoid_set_attention_ppo_v1"
        assert self.policy.model.config.actor_obs_dim == 85
        assert self.policy.gate_mode == "mission_v12"
        self.calls = Counter()
        self.nonzero = Counter()
        self.audit = Counter()

    def action(self, agent, goal):
        observation = agent.legacy_observation
        encoded = self.policy.encode_motion_observation(
            observation,
            agent_id=agent.agent_index,
            entity_id=agent.entity_id,
            launch_step=agent.launch_step,
            maneuver_state=agent.set_acc_z_z,
            goal_position=SimpleNamespace(lon=goal[0], lat=goal[1]),
        )
        assert encoded.shape == (85,)
        mask = self.policy.prepare_action_mask(encoded, np.ones(3, dtype=np.bool_))
        selected = self.policy.select_action(encoded, mask)
        acceleration = (-1.0, 0.0, 1.0)[selected]
        agent.set_acc_z_z = int(acceleration)
        kind = KIND[observation["self"]["type"]]
        self.calls[kind] += 1
        self.nonzero[kind] += int(acceleration != 0)
        self.audit["encoded_85"] += 1
        self.audit["mask_unrestricted" if mask.all() else "mask_restricted"] += 1
        return [0, agent.entity_id, acceleration, 0]


class Group:
    def __init__(self, state, members):
        self.state = state
        self.members = frozenset(members)
        self.reports = {}
        self.prepared = -1
        self.key = state_digest(sorted(members))[:12]
        self.baseline = RedBaselineCommander(
            state.public_targets,
            "r8_satellite_packages",
            seed=state.seed,
            search_polygon=state.polygon,
        )
        for entity_id in self.members:
            self.baseline.register_platform(entity_id)
        self.plan_memory = {}
        self.prior_memory = {}
        self.prior_started = {}
        self.wave = 0
        self.satellite_requests = set()
        self.targets = {}
        self.actions = {}

    def receive(self, observation, task):
        entity_id = int(observation["entity_id"])
        if entity_id not in self.members:
            raise ValueError("Cross-group report")
        self.reports[entity_id] = (copy.deepcopy(observation), copy.deepcopy(task))

    def prepare(self, step):
        if self.prepared == step:
            return self
        self.prepared = step
        current = {
            entity_id: (observation, task)
            for entity_id, (observation, task) in self.reports.items()
            if observation["step"] == step
        }
        assert set(current) <= self.members
        self.baseline.begin_step(tuple(observation for observation, _ in current.values()))
        self.targets = {target.entity_id: target for target in self.baseline.targets}
        state = self.state
        refs = {
            target_id: state.initial_ref_by_id.get(target_id, "D" + str(target_id))
            for target_id in self.targets
        }
        ids = {reference: target_id for target_id, reference in refs.items()}
        legal_targets = {
            refs[target_id]: {
                "type": target.entity_type,
                "position": [target.position.lon, target.position.lat, target.position.alt],
                "timestamp_s": self.baseline.track_fusion._observed_at.get(target_id, 0) / 1000.0,
            }
            for target_id, target in self.targets.items()
        }
        reports = {}
        for entity_id, (observation, task) in current.items():
            own = observation["self"]
            position = own["position"]
            key = state.key_by_id[entity_id]
            life = max(0, (1800 if own["type"] == 21002 else state.max_steps) - step)
            reports[key] = {
                "position": [position["lon"], position["lat"], position["alt"]],
                "task": copy.deepcopy(task),
                "remaining_life_s": life,
                "stage": own.get("stage", 0),
                "visible": own.get("isVisible", True),
                "health": own.get("health", 0),
                "satellite_active": bool(observation.get("is_using_satellite", False)),
            }
            peer = (own.get("commRangeInfo") or {}).get(entity_id, {})
            velocity = peer.get("velEcf")
            if velocity is not None:
                reports[key]["velocity"] = [velocity[axis] for axis in ("x", "y", "z")]
        response = state.strategic.step(state.plan, reports, legal_targets, step, self.plan_memory)
        self.plan_memory = response["memory"]
        self.satellite_requests = {
            state.id_by_key[key]
            for key in response.get("request_satellite", [])
            if key in reports
        }
        proposals = {}
        self.actions = {}
        for key, order in response.get("orders", {}).items():
            if key not in reports:
                raise ValueError("Cross-group plan command")
            entity_id = state.id_by_key[key]
            task = current[entity_id][1]
            if order["kind"] == "attack":
                reference = order["target_ref"]
                if reference not in ids:
                    raise ValueError("Plan attempted unknown target")
                target_id = ids[reference]
                target_type = self.targets[target_id].entity_type
                role = KIND[current[entity_id][0]["self"]["type"]]
                if (role == "L") != (target_type == 9500):
                    raise ValueError("Illegal target type")
                if role == "L" and task.get("target_id") != target_id:
                    if task.get("target_id") is not None:
                        raise ValueError("Prior-bound L cannot abandon its target")
                    proposals.setdefault(target_id, []).append(entity_id)
                    continue
                if task.get("target_id") not in (None, target_id):
                    state.metrics["hm_reassignments"] += 1
                task.update(copy.deepcopy(order))
                target = self.targets[target_id]
                task.update(target_id=target_id, last_point=[target.position.lon, target.position.lat])
            elif order["kind"] == "search":
                if task.get("target_id") is not None:
                    raise ValueError("Cannot silently discard committed attack task")
                task.update(copy.deepcopy(order))
                task["target_id"] = None
            else:
                raise ValueError("Unsupported Astra task")
        self._dispatch_prior(step, current, proposals)
        for entity_id, (observation, task) in current.items():
            target_id = task.get("target_id")
            if target_id is not None and target_id in self.targets:
                target = self.targets[target_id]
                task["last_point"] = [target.position.lon, target.position.lat]
                task["kind"] = "attack"
                task["target_ref"] = refs[target_id]
                self.actions[entity_id] = [4, entity_id, target_id, 0]
                state.metrics["attack_coordinate_updates"] += 1
            elif target_id is not None:
                self.actions[entity_id] = None
            else:
                points = task.get("waypoints") or state.units[state.key_by_id[entity_id]]["task"].get("waypoints", [])
                if not points:
                    raise ValueError("Searching unit without executable route")
                index = min(int(task.get("waypoint_index", 0)), len(points) - 1)
                task.update(last_point=list(points[index]), waypoint_index=index)
                self.actions[entity_id] = [2, entity_id, *points[index]]
            self.reports[entity_id] = (observation, task)
        state.metrics["group_preparations"] += 1
        state.metrics["runtime_plan_calls"] += 1
        return self

    def _dispatch_prior(self, step, current, proposals):
        state = self.state
        lroles = {
            state.key_by_id[entity_id]: (entity_id, observation, task)
            for entity_id, (observation, task) in current.items()
            if observation["self"]["type"] == 21002
        }
        if not lroles:
            return
        feedback = {}
        for event, memory in self.prior_memory.items():
            target_id = int(event.split(":")[0][1:])
            start = self.prior_started[event]
            receivers = memory["receiver_roles"]
            feedback[event] = {
                "all_acknowledged": bool(receivers) and all(
                    key in lroles
                    and state.receipts.get(lroles[key][0], {}).get("ack_target") == target_id
                    and state.receipts.get(lroles[key][0], {}).get("ack_step", -1) >= start
                    for key in receivers
                ),
                "any_unavailable": any(key not in lroles for key in receivers),
                "all_exited": bool(receivers) and all(
                    key not in lroles or lroles[key][2].get("target_id") != target_id
                    for key in receivers
                ),
                "elapsed_s": step - start,
            }
        events = []
        event_targets = {}
        for target_id, entity_ids in sorted(proposals.items()):
            self.wave += 1
            event = f"D{target_id}:W{self.wave}"
            event_targets[event] = target_id
            events.append({
                "ref": event,
                "original_roles": [state.key_by_id[entity_id] for entity_id in entity_ids],
                "queue_age_s": 0,
            })
        roles = []
        for key, (entity_id, observation, task) in lroles.items():
            own = observation["self"]
            point = own["position"]
            available = task.get("target_id") is None
            roles.append({
                "ref": key,
                "eligible": available,
                "remaining_time_s": max(0, 1800 - step),
                "task_kind": "search" if available else "original",
                "original_event_refs": [] if available else ["committed:" + str(task["target_id"])],
                "distances_m": {
                    event: cooperative_runtime.distance(
                        [point["lon"], point["lat"]],
                        [self.targets[target_id].position.lon, self.targets[target_id].position.lat],
                    )
                    for event, target_id in event_targets.items()
                },
            })
        result = PRIOR.step_skill(
            {"events": events, "roles": roles, "feedback": feedback},
            self.prior_memory,
            PRIOR_PARAMETERS,
        )
        self.prior_memory = result["memory"]
        for event in self.prior_memory:
            self.prior_started.setdefault(event, step)
        for item in result["replacements"]:
            event = item["event_ref"]
            target_id = event_targets[event]
            target = self.targets[target_id]
            for key in item["roles"]:
                entity_id, _, task = lroles[key]
                assert task.get("target_id") is None
                task.update(
                    kind="attack",
                    target_id=target_id,
                    target_ref="D" + str(target_id),
                    binding="hidden_attack",
                    last_point=[target.position.lon, target.position.lat],
                    handoff_step=step,
                    handoff_event=event,
                )
                row = {
                    "step": step,
                    "group": self.key,
                    "entity_id": entity_id,
                    "role_ref": key,
                    "target_id": target_id,
                    "point": list(task["last_point"]),
                    "status": "prepared",
                }
                state.pending[entity_id] = row
                state.metrics["prior_dispatches"] += 1
            state.plan_events.append({
                "step": step,
                "group": self.key,
                "event": "hidden_handoff",
                "target_id": target_id,
                "receivers": item["roles"],
            })
        state.metrics["prior_transitions"] += len(result["transitions"])
        state.metrics["prior_deferred_proposals"] += len(events) - len(result["replacements"])


class EpisodeState:
    def __init__(self, map_name, seed, adapter, compiled, polygon):
        self.map_name = map_name
        self.seed = seed
        self.adapter = adapter
        self.compiled = compiled
        self.plan = compiled["plan"]
        self.polygon = tuple(Position(*point) for point in polygon)
        self.max_steps = 3000
        self.key_by_id = {int(value): key for key, value in compiled["unit_ids"].items()}
        self.id_by_key = {key: value for value, key in self.key_by_id.items()}
        self.units = {unit["unit_key"]: unit for unit in self.plan["units"]}
        self.initial_ref_by_id = {int(value): key for key, value in compiled["target_ids"].items()}
        self.initial_id_by_ref = {value: key for key, value in self.initial_ref_by_id.items()}
        self.public_targets = initial_targets_from_observation(compiled["legacy_initial_observation"])
        self.agent_index = {entity_id: index for index, entity_id in enumerate(sorted(self.key_by_id))}
        self.groups = {}
        self.previous_groups = {}
        self.step = -1
        self.metrics = Counter()
        self.plan_events = []
        self.receipts = {}
        self.pending = {}
        self.refcount = 0
        self.motion = Motion(self.max_steps)
        self.strategic = StrategicOverlay()
        self.hp_memory = {}
        self.boundary_clamps = Counter()
        # 仅用于赛后归因；不参与任何动作选择或状态更新。
        self.boundary_clamp_by_target = {}
        self.boundary_max_distance_m = 0.0
        self.boundary_examples = []
        self.m_target_counts = Counter()
        for unit in self.plan["units"]:
            task = unit["task"]
            if unit["role"] == "M" and task["kind"] == "attack":
                self.m_target_counts[self.initial_id_by_ref[task["target_ref"]]] += 1
        self.hp_bindings = self._hp_bindings()

    def _hp_bindings(self):
        targets = {target["ref"]: target for target in self.plan["public_targets"]}
        bindings = {}
        for unit in self.plan["units"]:
            task = unit["task"]
            reference = task.get("target_ref")
            role = unit["role"]
            target_type = targets.get(reference, {}).get("type")
            if task["kind"] == "attack" and (
                (role == "H" and target_type == 9600)
                or (role == "M" and target_type == 9400)
            ):
                entity_id = int(self.compiled["unit_ids"][unit["unit_key"]])
                bindings[entity_id] = {
                    "unit_id": entity_id,
                    "unit": unit["unit_key"],
                    "target_id": int(self.compiled["target_ids"][reference]),
                    "target": reference,
                    "role": role,
                    "public_position": list(targets[reference]["position"]),
                }
        return bindings

    def configure_agent(self, agent):
        self.refcount += 1
        unit = self.units[self.key_by_id[agent.entity_id]]
        agent.agent_index = self.agent_index[agent.entity_id]
        agent.task = copy.deepcopy(unit["task"])
        agent.task.setdefault("binding", unit.get("binding", "primary"))
        if agent.task["kind"] == "attack":
            agent.task["target_id"] = self.initial_id_by_ref[agent.task["target_ref"]]
        else:
            agent.task.update(target_id=None, waypoint_index=0)

    def receive(self, entity_id, observation, task):
        peers = {int(value) for value in (observation["self"].get("commRangeInfo") or {})} | {entity_id}
        if not peers <= set(self.key_by_id):
            raise ValueError("Unknown member in engine communication group")
        key = tuple(sorted(peers))
        step = int(observation["step"])
        if step != self.step:
            self.previous_groups = self.groups
            self.groups = {}
            self.step = step
        group = self.groups.get(key)
        if group is None:
            group = self.previous_groups.get(key) or Group(self, key)
            self.groups[key] = group
        group.receive(observation, task)
        return group

    def acknowledge(self, entity_id, point):
        row = self.pending.get(entity_id)
        if row and row["status"] == "emitted" and max(abs(a - b) for a, b in zip(point, row["point"])) < 1e-7:
            row["status"] = "native_return"
            self.receipts[entity_id] = {"ack_step": row["step"], "ack_target": row["target_id"]}
            del self.pending[entity_id]
            self.metrics["native_returns"] += 1

    def _legal_context(self, agent, binding):
        observation = agent.legacy_observation
        own = observation["self"]
        position = own["position"]
        tracks = own.get("detectInfo") or {}
        target_id = binding["target_id"]
        raw = tracks.get(target_id, tracks.get(str(target_id)))
        track = None
        source_id = None
        if raw is not None:
            lla, stamp = field(raw, "lla"), field(raw, "time")
            if lla is not None and stamp is not None:
                track = {
                    "position": [float(field(lla, axis)) for axis in ("x", "y", "z")],
                    "timestamp_s": float(stamp) / 1000.0,
                }
                source_id = field(raw, "detect_from")
                source_id = None if source_id is None else int(source_id)
        hids, mids, peer_h, peer_m = [], [], [], []
        for entity_id, (peer_observation, task) in agent.group.reports.items():
            peer_target = task.get("target_id")
            entity_type = int(peer_observation["self"]["type"])
            peer_position = peer_observation["self"]["position"]
            point = [peer_position["lon"], peer_position["lat"], peer_position["alt"]]
            if entity_type == 21000 and peer_target in self.initial_ref_by_id:
                if self.plan_target_type(peer_target) == 9600:
                    target_position = self.plan_target_position(peer_target)
                    mean_latitude = math.radians((point[1] + target_position[1]) / 2.0)
                    peer_h.append({
                        "id": int(entity_id),
                        "target_id": int(peer_target),
                        "position": point,
                        "distance_to_target_m": math.hypot(
                            (point[0] - target_position[0]) * 111320.0 * math.cos(mean_latitude),
                            (point[1] - target_position[1]) * 110570.0,
                        ),
                    })
                if peer_target == target_id:
                    hids.append(int(entity_id))
            elif entity_type == 21001:
                if peer_target == target_id:
                    mids.append(int(entity_id))
                peer_m.append({
                    "id": int(entity_id),
                    "position": point,
                    "donor": self.m_target_counts.get(peer_target, 0) >= 7,
                })
        sentry_tracks = {}
        for candidate_id, reference in self.initial_ref_by_id.items():
            if self.plan_target_type(candidate_id) != 9600:
                continue
            value = tracks.get(candidate_id, tracks.get(str(candidate_id)))
            if value is None:
                continue
            lla, stamp = field(value, "lla"), field(value, "time")
            if lla is not None and stamp is not None:
                sentry_tracks[str(candidate_id)] = {
                    "position": [float(field(lla, axis)) for axis in ("x", "y", "z")],
                    "timestamp_s": float(stamp) / 1000.0,
                }
        nearest = None
        for item in tracks.values():
            if int(field(item, "entity_type", -1)) != 24000:
                continue
            lla = field(item, "lla")
            if lla is None:
                continue
            track_lon, track_lat = float(field(lla, "x")), float(field(lla, "y"))
            mean_latitude = math.radians((position["lat"] + track_lat) / 2.0)
            distance = math.hypot(
                (position["lon"] - track_lon) * 111320.0 * math.cos(mean_latitude),
                (position["lat"] - track_lat) * 110570.0,
            )
            nearest = distance if nearest is None else min(nearest, distance)
        return {
            "role": binding["role"],
            "binding_matches": agent.task.get("target_id") == target_id,
            "step": int(observation["step"]),
            "time_s": int(observation["step"]) + OWN_INITIAL_CLOCK_S,
            "unit_id": int(agent.entity_id),
            "target_id": target_id,
            "position": [position["lon"], position["lat"], position["alt"]],
            "own_track": track,
            "satellite_active": bool(observation.get("is_using_satellite", False)),
            "track_source_id": source_id,
            "relay_verified": source_id in hids if source_id is not None else False,
            "same_target_h_ids": hids,
            "same_target_h_count": len(hids),
            "same_target_m_count": len(mids),
            "peer_h": peer_h,
            "peer_m": peer_m,
            "sentry_tracks": sentry_tracks,
            "nearest_interceptor_m": nearest,
            "public_track": {"position": binding["public_position"]},
            "map_bounds": [
                min(point.lon for point in self.polygon),
                max(point.lon for point in self.polygon),
                min(point.lat for point in self.polygon),
                max(point.lat for point in self.polygon),
            ],
        }

    def plan_target_type(self, target_id):
        reference = self.initial_ref_by_id[target_id]
        return next(target["type"] for target in self.plan["public_targets"] if target["ref"] == reference)

    def plan_target_position(self, target_id):
        reference = self.initial_ref_by_id[target_id]
        return next(target["position"] for target in self.plan["public_targets"] if target["ref"] == reference)

    def apply_hp(self, agent, actions):
        binding = self.hp_bindings.get(agent.entity_id)
        if binding is None:
            return actions
        context = self._legal_context(agent, binding)
        prior = copy.deepcopy(self.hp_memory.get(agent.entity_id, {}))
        decision, self.hp_memory[agent.entity_id] = HP.step(context, prior)
        maneuver = [index for index, action in enumerate(actions) if action[0] == 0]
        target = [
            index for index, action in enumerate(actions)
            if action[0] == 4 and int(action[2]) == binding["target_id"]
        ]
        assert len(maneuver) == 1
        result = actions.copy()
        if decision["waypoint"]["action"] == "WAYPOINT" and len(target) == 1:
            result[target[0]] = [2, agent.entity_id, *decision["waypoint"]["point"]]
            self.metrics["hp_waypoint"] += 1
        if decision["clear_maneuver"] and result[maneuver[0]][2] != 0:
            result[maneuver[0]][2] = 0.0
            agent.set_acc_z_z = 0
            self.metrics["hp_clear_maneuver"] += 1
        self.metrics["hp_observations"] += 1
        return result

    def translate_waypoint(self, agent, old_point, source, *, target_id=None):
        raw_x, raw_y = self.adapter.to_new_point(old_point)
        x, y = self.adapter.bound_new_point((raw_x, raw_y))
        if x != raw_x or y != raw_y:
            distance = math.hypot(x - raw_x, y - raw_y)
            self.metrics["public_waypoint_clamps"] += 1
            self.boundary_clamps[source] += 1
            if target_id is not None:
                row = self.boundary_clamp_by_target.setdefault(int(target_id), {
                    "count": 0,
                    "first_step": int(agent.legacy_observation["step"]),
                    "last_step": int(agent.legacy_observation["step"]),
                    "max_projection_distance_m": 0.0,
                    "unit_ids": set(),
                })
                row["count"] += 1
                row["last_step"] = int(agent.legacy_observation["step"])
                row["max_projection_distance_m"] = max(row["max_projection_distance_m"], distance)
                row["unit_ids"].add(int(agent.entity_id))
            self.boundary_max_distance_m = max(self.boundary_max_distance_m, distance)
            if len(self.boundary_examples) < 20:
                self.boundary_examples.append({
                    "step": int(agent.legacy_observation["step"]),
                    "entity_id": int(agent.entity_id),
                    "role": KIND[int(agent.legacy_observation["self"]["type"])],
                    "source": source,
                    "target_id": target_id,
                    "legacy_point": [float(value) for value in old_point],
                    "raw_cartesian_point": [float(raw_x), float(raw_y)],
                    "submitted_cartesian_point": [float(x), float(y)],
                    "projection_distance_m": distance,
                })
        return x, y

    def decide(self, agent):
        observation = agent.legacy_observation
        step = int(observation["step"])
        group = agent.group.prepare(step)
        if agent.launch_step < 0:
            agent.launch_step = step
        _, task = group.reports[agent.entity_id]
        agent.task = copy.deepcopy(task)
        actions = []
        command = group.actions.get(agent.entity_id)
        if command:
            actions.append(command)
            pending = self.pending.get(agent.entity_id)
            if pending and command[0] == 4 and int(command[2]) == pending["target_id"]:
                pending["status"] = "emitted"
        if agent.entity_id in group.satellite_requests:
            actions.append([3, agent.entity_id, 0, 0])
            agent.task["satellite_requested_step"] = step
            self.metrics["satellite_requests"] += 1
        position = observation["self"]["position"]
        goal = agent.task.get("last_point") or [position["lon"], position["lat"]]
        actions.append(self.motion.action(agent, goal))
        legacy = np.asarray(actions, dtype=np.float64).reshape(-1, 4)
        legacy = self.apply_hp(agent, legacy)
        translated = []
        acknowledgement = None
        for action in legacy:
            action_type, entity_id, value, other = action.tolist()
            if action_type == 4:
                target_id = int(value)
                target = group.targets.get(target_id)
                if target is None:
                    continue
                old_point = [target.position.lon, target.position.lat]
                x, y = self.translate_waypoint(
                    agent, old_point, "legacy_target_coordinate", target_id=target_id
                )
                translated.append([1, entity_id, x, y])
                acknowledgement = self.adapter.to_old_point([x, y, 0.0])[:2]
            elif action_type == 2:
                old_point = [value, other]
                x, y = self.translate_waypoint(agent, old_point, "legacy_waypoint")
                translated.append([1, entity_id, x, y])
                acknowledgement = self.adapter.to_old_point([x, y, 0.0])[:2]
            elif action_type == 3:
                translated.append([2, entity_id, 0, 0])
            elif action_type == 0:
                translated.append([3, entity_id, value, 0])
            else:
                raise ValueError(f"Unsupported frozen action {action_type}")
        if acknowledgement is not None:
            self.acknowledge(agent.entity_id, acknowledgement)
        self.metrics["translated_rows"] += len(translated)
        return np.asarray(translated, dtype=np.float64).reshape(-1, 4)

    def close_agent(self):
        self.refcount -= 1
        if self.refcount:
            return
        output = os.environ.get("CZ_OUTPUT_DIR")
        if not output:
            return
        path = Path(output) / "cz_adapter_audit.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "status": "complete",
            "map": self.map_name,
            "seed": self.seed,
            "choice": CHOICES[self.map_name],
            "interface_only": True,
            "official_object_address_order_retained": True,
            "algorithm": {
                "opening": "pku_initial_astra_v2",
                "strategic": "frozen merged selector",
                "hp": "frozen chronological tree",
                "motion": "V12-R3 mission_v12",
                "satellite_schedule_changed": False,
            },
            "model_sha256": _sha(APP / "r3/model.pt"),
            "counts": dict(self.metrics),
            "motion_calls": dict(self.motion.calls),
            "motion_nonzero": dict(self.motion.nonzero),
            "motion_audit": dict(self.motion.audit),
            "strategic_stats": dict(self.strategic.stats),
            "strategic_event": self.strategic.strategy,
            "pending_prior_receipts": len(self.pending),
            "new_public_interface": {
                "input_scope": "per-agent public initial state and isolated runtime observation",
                "communication_scope": "current commRangeInfo connected component",
                "output_action_types": [1, 2, 3],
                "target_id_actions_emitted": 0,
                "waypoint_boundary_projection": "clip final submitted coordinate only; retain true internal track",
                "clamp_counts": dict(self.boundary_clamps),
                "clamp_by_target": {
                    str(target_id): {
                        **{key: value for key, value in row.items() if key != "unit_ids"},
                        "unit_ids": sorted(row["unit_ids"]),
                    }
                    for target_id, row in sorted(self.boundary_clamp_by_target.items())
                },
                "max_projection_distance_m": self.boundary_max_distance_m,
                "examples": self.boundary_examples,
            },
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        temporary.replace(path)


def _compile_from_deployment(observation, map_name, adapter):
    counts = Counter(int(entity["type"]) for entity in observation["entities"].values())
    polygon = [adapter.to_old_point([point[0], point[1], 3000.0])[:2] for point in observation["coordinates"]]
    public = []
    target_rows = []
    for raw_id, entity in observation["public_targets"].items():
        old = adapter.to_old_point(entity["pos"])
        target_rows.append((int(entity["type"]), old[0], old[1], int(raw_id), entity, old))
    target_rows.sort()
    target_ids = {}
    for index, (_, _, _, target_id, entity, old) in enumerate(target_rows, 1):
        reference = f"T{index:03d}"
        target_ids[reference] = target_id
        public.append({
            "ref": reference,
            "type": int(entity["type"]),
            "position": {"x": old[0], "y": old[1], "z": old[2]},
            "initial_health": float(entity["health"]),
        })
    case = {
        "map_bounds": adapter.old_map_area,
        "deployment": {
            "coordinates": [polygon],
            "coordinatesHM": [polygon],
            "altitude": 3000.0,
            "mode": "free",
        },
        "own_inventory": [
            {"role": role, "type": entity_type, "count": counts[entity_type], "initial_health_values": []}
            for entity_type, role in KIND.items()
        ],
        "public_targets": public,
    }
    choice = CHOICES[map_name]
    plan = compile_plan(case, choice)
    unit_ids = {}
    for entity_type, role in KIND.items():
        entity_ids = sorted(
            int(entity_id) for entity_id, entity in observation["entities"].items()
            if int(entity["type"]) == entity_type
        )
        keys = sorted(unit["unit_key"] for unit in plan["units"] if unit["role"] == role)
        if len(keys) != len(entity_ids):
            raise ValueError(f"Planner inventory mismatch for {role}: {len(keys)} != {len(entity_ids)}")
        unit_ids.update(dict(zip(keys, entity_ids)))
    legacy_entities = {}
    for target in public:
        old = target["position"]
        legacy_entities[target_ids[target["ref"]]] = {
            "position": {"lon": old["x"], "lat": old["y"], "alt": old["z"]},
            "health": target["initial_health"],
            "type": target["type"],
        }
    compiled = {
        "plan": plan,
        "unit_ids": unit_ids,
        "target_ids": target_ids,
        "candidate_id": choice,
        "legacy_initial_observation": {"entities": legacy_entities},
    }
    deployment = {
        unit_ids[unit["unit_key"]]: tuple(
            round(value, 6) for value in adapter.to_new_point(unit["deployment"][:2])
        )
        for unit in plan["units"]
    }
    return compiled, deployment, polygon


class Deployment(BaseDeployment):
    def deploy(self, observation):
        global _STATE
        if _STATE is not None:
            raise RuntimeError("CZ deployment state already exists")
        map_name = os.environ.get("CZ_MAP")
        if map_name not in CHOICES:
            raise ValueError("CZ_MAP must identify one frozen launch20 map")
        seed = observation.get("seed")
        if seed != 2:
            raise ValueError(f"Frozen evaluation requires seed 2, got {seed}")
        adapter = CoordinateAdapter(map_name)
        compiled, deployment, polygon = _compile_from_deployment(observation, map_name, adapter)
        _STATE = EpisodeState(map_name, seed, adapter, compiled, polygon)
        return deployment


class RedAttack(BaseAgent):
    def __init__(self, entity_id, init_observation):
        super().__init__(entity_id, init_observation, history_length=1)
        if _STATE is None:
            raise RuntimeError("CZ Deployment must run before RedAttack construction")
        self.state = _STATE
        self.legacy_observation = None
        self.group = None
        self.task = {}
        self.launch_step = -1
        self.set_acc_z_z = 0
        self._closed = False
        self.state.configure_agent(self)

    def get_obs(self, observation):
        super().get_obs(observation)
        self.legacy_observation = self.state.adapter.observation(observation)
        self.task.update(self.state.receipts.get(self.entity_id, {}))
        self.group = self.state.receive(self.entity_id, self.legacy_observation, self.task)

    def get_action(self):
        if self.legacy_observation is None or self.group is None:
            raise RuntimeError("The environment must provide an isolated observation")
        return self.state.decide(self)

    def on_reset(self):
        # The official runner calls reset immediately after construction.  This
        # is a single-episode frozen evaluation, so no shared policy reset occurs.
        self.legacy_observation = None
        self.group = None

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.state.close_agent()
