"""The graph-study runner: plan, generate, record.

Generation and scoring are separate phases. This module never loads a scorer — an NLI
evaluator and a generation model must not be resident together, and a run whose numbers
depend on a scoring call made during generation cannot be rescored later without
regenerating.

What it produces, under ``runs/graph/<run-id>/``:

    RUN_MANIFEST.json    the durable contract, written BEFORE the first model call
    RESOLVED_CONFIG.json the fully resolved tree and its three hashes
    COHORT.json          the frozen cohort that ran
    PLAN.json            the dry-run cost model
    generations/         immutable, content-addressed shards of raw trajectories
    traces/              graph events and policy decisions
    PERFORMANCE.json     throughput, cache behaviour, defence counters
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import importlib.metadata
import json
import platform
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ...defenses.semantic_detector import SemanticConceptDetector
from ...eval.leak_at_k import decoding_hash
from ...eval.tofu_data import TofuItem
from ...graph.config import ResolvedGraphConfig
from ...graph.executor import EpisodeSpec, GraphExecutor, GraphTrajectory
from ...graph.scheduler import planned_generations
from ...graph_memory.staged_store import StagedMemory
from ...logging_utils import get_logger
from ...memory.blocklist import Blocklist, build_blocklist
from ...memory.store import MemoryStore
from ...paths import git_diff_sha256, git_dirty, git_sha
from ...runtime.backend import GenerationBackend, GenRequest
from ...runtime.batch_scheduler import BatchScheduler
from ...runtime.resource_monitor import ResourceMonitor
from .arms import ArmPlan, build_arm_runtime, build_detector, build_registry
from .cohort import (
    Cohort,
    assert_forget_policy_cohort,
    assert_policy_excludes_evaluation_concepts,
)
from .controls import ControlledChallengeSet, build_controlled_challenges, concept_control_mapping
from .evidence import ShardWriter, atomic_json, verify_ledger, verify_shards
from .response_bank import ResponseBank

__all__ = ["GraphRunner", "RunPlan", "build_baseline_memory"]

log = get_logger(__name__)


def _brief(value: object, limit: int = 160) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass(frozen=True)
class _ProbePlan:
    """One trajectory's readback probe, split so a whole wave can be dispatched at once.

    ``requests`` is exactly two prompts — with and without the new memory, same seed —
    and ``record`` is everything the retrieval decision already determined. Keeping the
    two apart is what lets the scheduler see every trajectory's prompts in one call
    instead of two at a time.
    """

    trajectory_id: str
    requests: tuple[GenRequest, ...]
    record: dict

    def resolve(self, responses: dict) -> dict:
        with_request, without_request = self.requests
        return {
            **self.record,
            "with_store_text": responses[with_request.request_id].text,
            "without_store_text": responses[without_request.request_id].text,
        }


@dataclass(frozen=True)
class RunPlan:
    """What a run intends to do, computed before anything is generated."""

    n_items: int
    n_samples: int
    arms: tuple[str, ...]
    challenges: tuple[str, ...]
    topology: str
    planned_trajectories: int
    planned_graph_generations: int
    planned_probe_generations: int

    @property
    def planned_generations_total(self) -> int:
        return self.planned_graph_generations + self.planned_probe_generations

    def to_dict(self) -> dict:
        return {
            "n_items": self.n_items,
            "n_samples": self.n_samples,
            "arms": list(self.arms),
            "challenges": list(self.challenges),
            "topology": self.topology,
            "planned_trajectories": self.planned_trajectories,
            # The GPU-readiness gate compares against this number: the graph's own model
            # calls, before the response cache and excluding the readback probe.
            "planned_graph_generations": self.planned_graph_generations,
            "planned_probe_generations": self.planned_probe_generations,
            "planned_generations_total": self.planned_generations_total,
        }


def build_baseline_memory(
    items: Sequence[TofuItem],
    *,
    blocklist_kind: str = "id",
    embedding_dim: int = 64,
    index_backend: str = "numpy",
    ingest_forget_set: bool = True,
    semantic_threshold: float = 0.75,
) -> tuple[MemoryStore, Blocklist]:
    """The post-deletion state every arm starts from.

    Identical to the two-agent setup: the forget content was ingested, was deleted
    through the memory pathway, and its ids are blocklisted. Both SBU invariants hold
    here, which is what makes anything recovered afterwards a re-derivation rather than
    a failure to delete.
    """
    store = MemoryStore(index_backend=index_backend, embedding_dim=embedding_dim)
    blocklist = build_blocklist(
        blocklist_kind,
        texts=[f"{i.question} {i.answer}" for i in items] if blocklist_kind == "semantic" else None,
        threshold=semantic_threshold,
    )
    if ingest_forget_set:
        ids = [
            store.add(
                f"{item.question} {item.answer}",
                source_agent="ingest",
                source_kind="ingest",
                turn=0,
                meta={"item_id": item.item_id},
            ).node_id
            for item in items
        ]
        for node_id in ids:
            store.delete(node_id, blocklist=blocklist)
    return store, blocklist


@dataclass
class GraphRunner:
    """One graph study run.

    **Two cohorts, deliberately.** ``cohort``/``items`` is the EVALUATION cohort — the
    questions that get asked — and ``policy_cohort``/``policy_items`` is the frozen
    FORGET cohort that defines what the system must withhold. They are the same object
    for a forget-cohort run and different objects for a retain-utility run, and the
    forget-policy role is the one that must never move:

        concept registry        always the policy cohort's concepts
        deleted baseline memory always the policy cohort's content

    Deriving either of those from the evaluation cohort is what made a retain-utility
    run register retained authors as forgotten, so ``policy_cohort`` defaults to the
    evaluation cohort only while that cohort is itself a forget cohort; a retain
    evaluation cohort with no explicit policy cohort is refused rather than guessed at.

    The policy cohort is deliberately NOT subject to ``--limit``. A limit narrows which
    questions are asked; it does not narrow what the deployed system forgot, and a
    registry that shrank with the question budget would make the guard's scope a
    function of how much GPU time was bought.
    """

    cfg: ResolvedGraphConfig
    items: Sequence[TofuItem]
    cohort: Cohort
    backend: GenerationBackend
    output: Path
    # The frozen forget cohort. ``None`` means "the evaluation cohort is itself the
    # forget cohort", which is checked rather than assumed.
    policy_cohort: Cohort | None = None
    policy_items: Sequence[TofuItem] | None = None
    challenges: tuple[str, ...] = ("natural",)
    # ``end_to_end_safety`` or ``graph_flow``. One protocol per run; the resume
    # fingerprint refuses to continue a run under a different one, because pooling the
    # two would average "the system refused the request" with "the graph contained what
    # it produced" and neither number would survive.
    protocol: str = "end_to_end_safety"
    tokenizer_revision: str | None = None
    resume: bool = False
    bank: ResponseBank = field(default_factory=ResponseBank)
    monitor: ResourceMonitor = field(default_factory=ResourceMonitor)

    # ------------------------------------------------------------------- assembly --

    def __post_init__(self) -> None:
        if self.protocol not in self.cfg.study.protocols:
            raise ValueError(
                f"protocol '{self.protocol}' is not enabled by study "
                f"'{self.cfg.study.study_id}'; available: {list(self.cfg.study.protocols)}"
            )
        # ---- the two cohorts, and the rule that keeps them apart --------------------
        # `policy_cohort` / `policy_items` are the constructor's INPUTS and may be None;
        # `forget_policy` / `forget_policy_items` are the resolved, always-present
        # attributes everything downstream reads. Keeping the two names apart is what
        # stops "did the caller pass one?" and "which cohort is the policy?" from being
        # the same question.
        if self.policy_cohort is None:
            # An unqualified cohort may serve both roles only if it is a forget cohort.
            # `assert_forget_policy_cohort` is what turns "retain cohort with no policy
            # cohort" from a silently wrong registry into a refusal.
            self.forget_policy: Cohort = self.cohort
            self.forget_policy_items: Sequence[TofuItem] = self.items
        else:
            if self.policy_items is None:
                raise ValueError("policy_cohort was given without policy_items")
            self.forget_policy = self.policy_cohort
            self.forget_policy_items = self.policy_items
        assert_forget_policy_cohort(self.forget_policy)
        assert_policy_excludes_evaluation_concepts(self.forget_policy, self.cohort)

        self.policy_concept_by_item = {
            entry.item_id: entry.concept_id for entry in self.forget_policy.items
        }
        self.evaluation_concept_by_item = {
            entry.item_id: entry.concept_id for entry in self.cohort.items
        }
        # Episodes look up the concept of the question being asked, which lives in the
        # evaluation cohort; the policy map is kept separate so a lookup can never fall
        # through from one role to the other.
        self.concept_by_item = {
            **self.policy_concept_by_item,
            **self.evaluation_concept_by_item,
        }
        # THE registry. Policy cohort only, always — see the class docstring.
        self.registry = build_registry(
            self.forget_policy_items,
            concept_of=lambda item_id: self.policy_concept_by_item[item_id],
        )
        self.calibration, self.calibration_status = self._load_calibration()
        self.detector: SemanticConceptDetector = build_detector(
            self.cfg,
            self.registry,
            calibration=self.calibration,
            status=self.calibration_status["effective_status"],
        )
        self.arm_plans: list[ArmPlan] = build_arm_runtime(
            self.cfg, self.cfg.topology, self.detector, protocol=self.protocol
        )
        self.scheduler = BatchScheduler(
            self.backend,
            cache=self.bank.cache,
            max_batch_size=self.cfg.profile.runtime.max_num_seqs,
            tokenizer_revision=self.tokenizer_revision,
            backend_version=_pkg_version(self.cfg.profile.runtime.backend),
            observer=self.bank.note,
        )
        self.mapping = concept_control_mapping(
            [self.evaluation_concept_by_item[item.item_id] for item in self.items]
        )
        self.control_source = {
            item.item_id: self.items[self.mapping.permutation[i]]
            for i, item in enumerate(self.items)
        }
        # The post-deletion state. Policy cohort only: the deleted baseline is what the
        # deployed system deleted, not what this particular run happens to ask about.
        self._base_store, self._blocklist = build_baseline_memory(
            self.forget_policy_items,
            blocklist_kind=self.cfg.study.memory.blocklist,
            ingest_forget_set=self.cfg.study.memory.ingest_forget_set,
        )
        self._baseline = self._base_store.snapshot()

    def _load_calibration(self) -> tuple[dict | None, dict]:
        """The verified detector-calibration artefact, or ``None`` for a diagnostic run.

        ``detector.status: calibrated`` with a missing, edited or failed artefact is a
        refusal, not a warning. The status is the flag the report gate reads before
        letting a run be called reportable; a flag that can be set without the evidence
        behind it is worse than no flag, because it is trusted.
        """
        spec = self.cfg.study.detector
        declared = spec.status
        if not spec.calibration_artifact:
            if declared == "calibrated":
                raise ValueError(
                    "detector.status is 'calibrated' but no calibration_artifact is set"
                )
            return None, {
                "declared_status": declared,
                "effective_status": "diagnostic",
                "covers_forget_policy": False,
                "reason": "no calibration artefact is configured",
            }
        from ...cli.calibrate_detector import load_calibration
        from ...paths import repo_root

        path = Path(spec.calibration_artifact)
        if not path.is_absolute():
            path = repo_root() / path
        try:
            payload = load_calibration(path)
        except Exception as exc:
            if declared == "calibrated":
                raise
            log.warning("ignoring unusable calibration artefact %s: %s", path, exc)
            return None, {
                "declared_status": declared,
                "effective_status": "diagnostic",
                "covers_forget_policy": False,
                "reason": f"calibration artefact unusable: {exc}",
            }
        if spec.calibration_id and payload["calibration_id"] != spec.calibration_id:
            raise ValueError(
                f"detector.calibration_id is '{spec.calibration_id}' but "
                f"{path} records '{payload['calibration_id']}'. The study file and the "
                "artefact describe different calibrations; one of them is stale."
            )
        fpr = payload.get("fpr")
        ceiling = self.cfg.study.evaluation.max_detector_fpr
        if declared == "calibrated" and (fpr is None or float(fpr) > ceiling):
            raise ValueError(
                f"{path}: measured false-positive rate {fpr} exceeds the study's "
                f"max_detector_fpr of {ceiling}. A detector over the ceiling buys its "
                "leakage number with over-blocking and cannot be reported as calibrated."
            )

        # A threshold is calibrated FOR A REGISTRY. The artefact records the forget policy
        # it was selected against, and a run over a different policy inherits neither the
        # measured FPR nor the right to call itself calibrated — the CPU stub cohort is
        # the everyday case. Rather than refuse (which would block the offline gate) or
        # pretend (which is the mislabel this whole change exists to stop), the run
        # DOWNGRADES itself to diagnostic, keeps the study's own threshold, and records
        # why. `detector_status` in the manifest is always the effective one.
        policy_fingerprint = self.forget_policy.fingerprint()
        covers = str((payload.get("registry_from") or {}).get("fingerprint")) == policy_fingerprint
        if not covers:
            reason = (
                f"calibrated on forget policy "
                f"'{(payload.get('registry_from') or {}).get('split')}' "
                f"({str((payload.get('registry_from') or {}).get('fingerprint'))[:12]}), "
                f"but this run's forget policy is '{self.forget_policy.split}' "
                f"({policy_fingerprint[:12]}). A threshold is calibrated for a registry; "
                "this one does not cover the concepts being guarded here."
            )
            log.warning("detector reported as diagnostic: %s", reason)
            return None, {
                "declared_status": declared,
                "effective_status": "diagnostic",
                "covers_forget_policy": False,
                "calibration_id": payload["calibration_id"],
                "reason": reason,
            }
        return payload, {
            "declared_status": declared,
            "effective_status": declared,
            "covers_forget_policy": True,
            "calibration_id": payload["calibration_id"],
            "reason": "",
        }

    # ----------------------------------------------------------------------- plan --

    def plan(self) -> RunPlan:
        n_samples = self.cfg.profile.sampling.n_samples
        graph_calls = 0
        trajectories = 0
        for challenge in self.challenges:
            injected = self._max_injected_nodes(challenge)
            adjusted = planned_generations(
                self.cfg.topology,
                n_items=len(self.items),
                n_samples=n_samples,
                arms=[a.name for a in self.arm_plans],
                single_agent_arms=[a.name for a in self.arm_plans if a.spec.mode == "single_agent"],
                injected_nodes_per_trajectory=injected,
            )
            graph_calls += adjusted["planned_generations"]
            trajectories += adjusted["planned_trajectories"]
        probes = 2 * trajectories if self.cfg.study.memory.later_episode_probe else 0
        return RunPlan(
            n_items=len(self.items),
            n_samples=n_samples,
            arms=tuple(a.name for a in self.arm_plans),
            challenges=self.challenges,
            topology=self.cfg.topology.name,
            planned_trajectories=trajectories,
            planned_graph_generations=graph_calls,
            planned_probe_generations=probes,
        )

    def _max_injected_nodes(self, challenge: str) -> int:
        """How many nodes emit scripted text instead of generating, for this challenge."""
        if challenge == "natural":
            return 0
        counts = [
            len(cs.injected_outputs)
            for item in self.items
            for cs in build_controlled_challenges(item, self.cfg.topology, [challenge])
        ]
        return max(counts) if counts else 0

    # ------------------------------------------------------------------ execution --

    def run(self) -> dict:
        self.output.mkdir(parents=True, exist_ok=True)
        generations = self.output / "generations"
        plan = self.plan()
        manifest = self._manifest(plan)

        # Compatibility is checked BEFORE anything is written. The previous version
        # overwrote RUN_MANIFEST.json first, so a resume under a different checkpoint,
        # topology, sample budget or protocol replaced the record of what had actually
        # produced the existing shards and then silently skipped their keys.
        previous = self._previous_manifest()
        if previous is not None:
            if not self.resume:
                raise FileExistsError(
                    f"{self.output} already holds a run manifest. Pass resume=True to "
                    "continue it, or choose a new output directory — runs are append-only."
                )
            self._assert_resumable(previous)

        writer = ShardWriter(generations, shard_size=self.cfg.profile.runtime.shard_size)
        traces = ShardWriter(
            self.output / "traces",
            prefix="graph-events",
            shard_size=self.cfg.profile.runtime.shard_size,
        )
        done = writer.completed()
        if done and not self.resume:
            raise FileExistsError(
                f"{generations} already holds {len(done)} trajectories. Pass resume=True to "
                "continue it, or choose a new output directory — runs are append-only."
            )
        if done and previous is not None:
            # The shards being resumed onto must be the ones the old manifest describes.
            #
            # Only a COMPLETE previous run has an `evidence_shards` list at all, so this
            # check alone covered nothing on the case that matters — an interrupted run.
            # The per-shard ledger (GU-0024) is what covers those, and `ShardWriter` has
            # already refused above if it did not verify. What is added here is that a
            # hash mismatch or a vanished declared shard now refuses whether or not the
            # previous run finished; only "on disk but undeclared" stays tolerated,
            # because an interrupted run declares nothing in its manifest.
            verdict = verify_shards(generations, previous.get("evidence_shards", []))
            tampered = [
                problem
                for problem in verdict["problems"]
                if "sha256 mismatch" in problem or "missing from disk" in problem
            ]
            if tampered or (not verdict["ok"] and previous.get("complete")):
                raise ValueError(
                    "refusing to resume: existing evidence shards do not match the hashes "
                    f"in the previous manifest: {(tampered or verdict['problems'])[:3]}"
                )
        # Carry the completed work forward so the final manifest describes the whole run.
        manifest["resumed_from"] = (
            {
                "trajectories": len(done),
                "previous_git_sha": previous.get("git_sha"),
                "previous_started": previous.get("started_utc"),
            }
            if previous is not None
            else None
        )
        atomic_json(self.output / "RUN_MANIFEST.json", manifest)
        atomic_json(self.output / "RESOLVED_CONFIG.json", self.cfg.to_dict())
        atomic_json(self.output / "COHORT.json", self.cohort.to_dict())
        # The forget policy travels with the run as its own artefact. Its fingerprint is
        # also in the manifest, but a reader asking "what did this run treat as
        # forgotten" should not have to reconstruct it from a hash.
        atomic_json(self.output / "FORGET_POLICY_COHORT.json", self.forget_policy.to_dict())
        atomic_json(self.output / "PLAN.json", plan.to_dict())

        wave = max(1, self.cfg.profile.runtime.max_num_seqs)
        n_samples = self.cfg.profile.sampling.n_samples
        for challenge in self.challenges:
            challenge_sets = self._challenge_sets(challenge)
            units = [(item, sample) for item in self.items for sample in range(n_samples)]
            for arm in self.arm_plans:
                self.scheduler.tag = arm.name
                executor = GraphExecutor(
                    self.cfg.topology,
                    scheduler=self.scheduler,
                    defense=arm.defense,
                    max_new_tokens=self.cfg.study.sampling.max_new_tokens,
                    temperature=self.cfg.study.sampling.temperature,
                    top_p=self.cfg.study.sampling.top_p,
                    top_k=self.cfg.study.sampling.top_k,
                    base_seed=self.cfg.study.sampling.base_seed,
                    retrieval_k=self.cfg.study.memory.retrieval_k,
                    monitor=self.monitor,
                )
                for start in range(0, len(units), wave):
                    batch = units[start : start + wave]
                    episodes: list[EpisodeSpec] = []
                    memories: list[StagedMemory] = []
                    for item, sample in batch:
                        key = (item.item_id, sample, arm.name, challenge)
                        if key in done:
                            continue
                        episode, memory = self._episode(
                            item, sample, arm, challenge, challenge_sets
                        )
                        episodes.append(episode)
                        memories.append(memory)
                    if not episodes:
                        continue
                    trajectories = list(executor.run(episodes, memories))
                    # Every probe in the wave is dispatched as ONE batch. Building the
                    # rows first and probing per row sent two prompts per trajectory:
                    # 8,000 two-prompt backend calls per protocol at 50x32, against a
                    # backend whose entire reason for being here is continuous batching.
                    probes = self._probe_wave(trajectories, executor)
                    for trajectory in trajectories:
                        writer.append(
                            self._row(
                                trajectory,
                                arm,
                                executor,
                                probes.get(trajectory.spec.trajectory_id, {}),
                            )
                        )
                        traces.append(
                            {
                                "item_id": trajectory.spec.item_id,
                                "sample_id": trajectory.spec.sample_id,
                                "arm": trajectory.spec.arm,
                                "challenge": trajectory.spec.challenge,
                                **trajectory.trace.to_dict(with_content=False),
                            }
                        )

        shard_manifest = writer.close()
        trace_manifest = traces.close()
        performance = {
            "runtime": self.monitor.to_dict(),
            "scheduler": self.scheduler.stats(),
            "response_bank": self.bank.to_dict(),
            "detector": {**self.detector.to_dict(), **self.detector.stats()},
            "defenses": {a.name: a.defense.stats() for a in self.arm_plans},
        }
        atomic_json(self.output / "PERFORMANCE.json", performance)

        manifest["complete"] = True
        manifest["evidence_shards"] = [s.to_dict() for s in shard_manifest]
        manifest["trace_shards"] = [s.to_dict() for s in trace_manifest]
        # The ledger's own verdict, so a resumed run records whether the shards it
        # inherited were ones it could check or ones it merely adopted.
        manifest["evidence_ledger"] = {
            "generations": {**verify_ledger(generations), "open_status": writer.ledger_status},
            "traces": {
                **verify_ledger(self.output / "traces", prefix="graph-events"),
                "open_status": traces.ledger_status,
            },
        }
        manifest["completed_trajectories"] = writer.n_rows
        # `actual_graph_generations` is GONE, not renamed. It held the scheduler's total
        # dispatch count — graph calls AND readback probes — under a name that says
        # "graph", so it could not be checked against `plan.planned_graph_generations`
        # and silently was not. The replacement is the per-purpose breakdown, in the
        # same shape as PERFORMANCE.json's `scheduler` block.
        manifest["actual_generations"] = {
            purpose: dict(counts) for purpose, counts in self.scheduler.by_purpose.items()
        }
        atomic_json(self.output / "RUN_MANIFEST.json", manifest)
        return manifest

    # --------------------------------------------------------------------- resume --

    # Fields that define WHICH EXPERIMENT this is. A resume that differs on any of them
    # would append trajectories from a different experiment to the same evidence file,
    # and the trajectory-key skip would hide it — the completed keys look identical
    # whichever checkpoint produced them.
    IMMUTABLE_ON_RESUME: tuple[str, ...] = (
        "study_id",
        "phase",
        "protocol",
        "study_design_hash",
        "profile_hash",
        "resolved_run_hash",
        "cohort_fingerprint",
        "cohort_split",
        # Resuming under a different forget policy would append trajectories judged
        # against a different set of forbidden concepts to the same evidence file.
        "forget_policy_fingerprint",
        "forget_policy_split",
        "challenges",
        "backend",
        "model",
        "tokenizer_revision",
        "logical_agents",
        "detector",
        "control_mapping",
        "sampling",
    )

    def _previous_manifest(self) -> dict | None:
        path = self.output / "RUN_MANIFEST.json"
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"refusing to resume: {path} is not readable JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"refusing to resume: {path} is not an object")
        return payload

    def _assert_resumable(self, previous: dict) -> None:
        current = self._manifest(self.plan())
        differences: list[str] = []
        for key in self.IMMUTABLE_ON_RESUME:
            was, now = previous.get(key), current.get(key)
            if was != now:
                differences.append(f"  {key}:\n    was: {_brief(was)}\n    now: {_brief(now)}")
        if differences:
            raise ValueError(
                "refusing to resume: this run is not the same experiment as the one in "
                f"{self.output / 'RUN_MANIFEST.json'}.\n"
                + "\n".join(differences)
                + "\nStart a new output directory. Appending to these shards would mix two "
                "experiments in one evidence file, and the completed-key skip would hide it."
            )
        # Git state is recorded rather than enforced: a resume after an unrelated commit
        # is normal, and a hard failure there would strand recoverable runs. A CODE
        # change that matters shows up in the hashes above.
        if previous.get("git_sha") != current.get("git_sha"):
            log.warning(
                "resuming a run started at git %s from git %s; the study, profile and "
                "resolved hashes all match, so the experiment is the same",
                previous.get("git_sha"),
                current.get("git_sha"),
            )

    # ------------------------------------------------------------------- episodes --

    def _challenge_sets(self, challenge: str) -> dict[str, ControlledChallengeSet]:
        if challenge == "natural":
            return {}
        out: dict[str, ControlledChallengeSet] = {}
        for item in self.items:
            built = build_controlled_challenges(item, self.cfg.topology, [challenge])
            if built:
                out[item.item_id] = built[0]
        return out

    def _episode(
        self,
        item: TofuItem,
        sample: int,
        arm: ArmPlan,
        challenge: str,
        challenge_sets: dict[str, ControlledChallengeSet],
    ) -> tuple[EpisodeSpec, StagedMemory]:
        source = self.control_source[item.item_id]
        # A cross-concept arm gets the CONTROL item's injection, not the target's.
        # Injecting the target concept into MA-CONTROL would hand the control arm the
        # treatment's content and delete the contrast — the same failure the two-agent
        # work hit when C3C and C3S stopped differing in one variable.
        injection = challenge_sets.get(
            source.item_id if arm.uses_control_question else item.item_id
        )

        store = MemoryStore(
            index_backend="numpy",
            embedding_dim=self._base_store.embedder.dim,
            deletion_mode=self._base_store.deletion_mode,
            refcount_semantics=self._base_store.refcount_semantics,
        )
        store.restore(self._baseline)
        memory = StagedMemory(
            store,
            blocklist=self._blocklist,
            visibility=self.cfg.study.memory.visibility,
        )
        if injection is not None:
            for text in injection.seeded_memory:
                # A re-entry challenge plants content in memory BEFORE the episode, so
                # what it measures is retrieval protection, not the write guard.
                memory.store.add(
                    text, source_agent="ingest", source_kind="ingest", meta={"injected": True}
                )

        episode = EpisodeSpec(
            trajectory_id=f"{arm.name}:{challenge}:{item.item_id}:{sample}",
            item_id=item.item_id,
            concept_id=self.concept_by_item[item.item_id],
            sample_id=sample,
            arm=arm.name,
            question=item.question,
            upstream_question=source.question if arm.uses_control_question else None,
            upstream_item_id=source.item_id if arm.uses_control_question else None,
            active_nodes=arm.active_nodes,
            injected_outputs=dict(injection.injected_outputs) if injection else {},
            injected_tool_inputs=dict(injection.injected_tool_inputs) if injection else {},
            node_questions=dict(injection.node_questions) if injection else {},
            challenge=challenge,
            topology=self.cfg.topology.name,
        )
        return episode, memory

    # ----------------------------------------------------------------------- rows --

    def _row(
        self,
        traj: GraphTrajectory,
        arm: ArmPlan,
        executor: GraphExecutor,
        probe: dict | None = None,
    ) -> dict:
        item = next(i for i in self.items if i.item_id == traj.spec.item_id)
        probe = probe or {}
        node_events = traj.trace.of_kind("node_executed")
        edge_events = traj.trace.of_kind("edge_decision")
        write_events = traj.trace.of_kind("memory_write")
        read_events = traj.trace.of_kind("memory_read")
        decode = GenRequest(
            request_id="decoding",
            prompt="",
            max_new_tokens=self.cfg.study.sampling.max_new_tokens,
            do_sample=True,
            temperature=self.cfg.study.sampling.temperature,
            top_p=self.cfg.study.sampling.top_p,
            top_k=self.cfg.study.sampling.top_k,
        )
        return {
            "schema": "graph-leak-row-v1",
            "trajectory_id": traj.spec.trajectory_id,
            "item_id": traj.spec.item_id,
            "concept_id": traj.spec.concept_id,
            "sample_id": traj.spec.sample_id,
            "arm": traj.spec.arm,
            "challenge": traj.spec.challenge,
            "protocol": self.protocol,
            "topology": traj.spec.topology,
            "defense": arm.defense.name,
            # The reference answer travels with the raw evidence so a pinned evaluator can
            # rescore completed generations without regenerating them.
            "reference_answer": item.answer,
            "question": item.question,
            "control_source_item_id": traj.spec.upstream_item_id,
            "final_text": traj.final_text,
            "final_status": traj.final_status,
            "raw_outputs": {
                "agent_messages": traj.agent_output_texts(),
                "released_edge_payloads": traj.released_edge_texts(),
                "memory_writes": traj.committed_write_texts(),
                "probe": probe,
            },
            "counts": {
                "generations": traj.n_generations,
                "forced_or_injected": traj.n_forced,
                "edges": len(edge_events),
                "edges_blocked": sum(
                    1 for e in edge_events if getattr(e, "release_status", "pass") != "pass"
                ),
                "edges_removed": sum(
                    1 for e in edge_events if not getattr(e, "edge_preserved", True)
                ),
                "write_candidates": len(write_events),
                "writes_allowed": sum(1 for e in write_events if getattr(e, "allowed", False)),
                "committed_writes": len(traj.committed_write_ids),
                "retrieval_withheld": sum(
                    len(getattr(e, "withheld_node_ids", []))
                    + len(getattr(e, "rescan_withheld_node_ids", []))
                    for e in read_events
                ),
                "nodes_executed": len(node_events),
                "nodes_abstained": sum(1 for e in node_events if getattr(e, "abstained", False)),
            },
            "provenance": {
                "prompt_sha256s": sorted(
                    {str(getattr(e, "prompt_sha256", "")) for e in node_events}
                ),
                "generation_seeds": [getattr(e, "generation_seed", None) for e in node_events],
                "model_revisions": sorted(
                    {
                        str(getattr(e, "model_revision", None))
                        for e in node_events
                        if getattr(e, "model_revision", None)
                    }
                ),
                "decoding_sha256": decoding_hash(decode.decoding()),
                "detector_version": self.detector.version,
                "trace_digest": traj.trace.digest(),
                "cached_nodes": sum(1 for e in node_events if getattr(e, "cached", False)),
            },
            "memory_evidence": [
                {
                    "node_id": node_id,
                    "content": node.content,
                    "content_sha256": hashlib.sha256(node.content.encode("utf-8")).hexdigest(),
                    "parent_ids": list(node.parent_ids),
                    "forget_ids": list(traj.memory.scopes.forget_ids(node_id)),
                    "is_parametric": node.is_parametric,
                }
                for node_id in traj.committed_write_ids
                if (node := traj.memory.store.get(node_id)) is not None
            ],
        }

    def _probe_wave(
        self, trajectories: Sequence[GraphTrajectory], executor: GraphExecutor
    ) -> dict[str, dict]:
        """Every trajectory's readback probe, in one dispatch.

        Returns ``{trajectory_id: probe}``. The retrieval decision is per trajectory and
        is taken here, before any generation, so the two prompts a trajectory contributes
        are fully determined by the time the batch is assembled — the batching changes
        nothing about what is asked, only how many times the backend is called.
        """
        if not self.cfg.study.memory.later_episode_probe or not trajectories:
            return {}
        plans = [self._probe_plan(traj, executor) for traj in trajectories]
        requests = [request for plan in plans for request in plan.requests]
        responses = self.scheduler.run(requests, purpose="probe") if requests else {}
        return {plan.trajectory_id: plan.resolve(responses) for plan in plans}

    def _probe_plan(self, traj: GraphTrajectory, executor: GraphExecutor) -> _ProbePlan:
        """Later-episode readback: the sink asks again, with and without the new memory.

        Two draws under the same seed. ``attributable`` needs the with-memory answer to
        leak, the without-memory answer not to, and the retrieval to have returned a node
        that actually carries the content — otherwise "memory caused it" is an assumption.
        """
        item_question = traj.spec.question
        sink = self.cfg.topology.node(self.cfg.topology.sink)
        seed = executor.seed_for_node(traj.spec, self.cfg.topology.sink) ^ 0x5EED
        guarded = traj.memory.candidates(item_question, self.cfg.study.memory.retrieval_k)
        from ...defenses.base import RetrievalContext

        verdict = executor.defense.on_retrieval(
            RetrievalContext(
                node_id=self.cfg.topology.sink,
                depth=self.cfg.topology.depth_of(self.cfg.topology.sink),
                query=item_question,
                candidates=guarded,
            )
        )
        allowed_ids = tuple(verdict.allowed_node_ids)
        memory_texts = traj.memory.texts_for(allowed_ids)

        from ...graph.prompt_builder import build_agent_prompt, role_system_prompt

        system = role_system_prompt(sink.role, sink.system_prompt)
        with_prompt = build_agent_prompt(question=item_question, memory_texts=memory_texts)
        without_prompt = build_agent_prompt(question=item_question)
        requests = [
            GenRequest(
                request_id=f"{traj.spec.trajectory_id}|probe-with",
                prompt=with_prompt,
                system=system,
                max_new_tokens=self.cfg.study.sampling.max_new_tokens,
                seed=seed,
                do_sample=True,
                temperature=self.cfg.study.sampling.temperature,
                top_p=self.cfg.study.sampling.top_p,
                top_k=self.cfg.study.sampling.top_k,
                model=sink.model,
            ),
            GenRequest(
                request_id=f"{traj.spec.trajectory_id}|probe-without",
                prompt=without_prompt,
                system=system,
                max_new_tokens=self.cfg.study.sampling.max_new_tokens,
                seed=seed,
                do_sample=True,
                temperature=self.cfg.study.sampling.temperature,
                top_p=self.cfg.study.sampling.top_p,
                top_k=self.cfg.study.sampling.top_k,
                model=sink.model,
            ),
        ]
        return _ProbePlan(
            trajectory_id=traj.spec.trajectory_id,
            requests=tuple(requests),
            record={
                "retrieved_node_ids": list(allowed_ids),
                "retrieved_texts": list(memory_texts),
                "withheld_node_ids": list(verdict.withheld_node_ids),
                "rescan_withheld_node_ids": list(verdict.rescan_withheld_node_ids),
                "probe_seed": seed,
            },
        )

    # ------------------------------------------------------------------- manifest --

    def _manifest(self, plan: RunPlan) -> dict:
        uses_gold = any(c != "natural" for c in self.challenges)
        return {
            "schema": "graph-run-manifest-v2",
            "started_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "study_id": self.cfg.study.study_id,
            "phase": self.cfg.phase,
            **self.cfg.hashes(),
            "git_sha": git_sha(),
            "git_dirty": git_dirty(),
            "git_diff_sha256": git_diff_sha256(),
            "python": platform.python_version(),
            "runtime_packages": {
                package: _pkg_version(package)
                for package in ("torch", "transformers", "datasets", "vllm", "numpy", "pydantic")
            },
            "protocol": self.protocol,
            "protocol_note": (
                "graph_flow: the request gate is held CONSTANT across every arm — no arm "
                "inspects the incoming question — so what is measured is whether forgotten "
                "information generated or introduced after the initial boundary can "
                "propagate."
                if self.protocol == "graph_flow"
                else "end_to_end_safety: the request gate is part of the defence, so a "
                "forget question is refused before the model is called. This measures "
                "whether the DEPLOYED SYSTEM releases forgotten information; it does NOT "
                "isolate the graph contribution."
            ),
            "backend": self.cfg.profile.runtime.backend,
            "model": self.cfg.model.model_dump(mode="json"),
            # What the backend ACTUALLY resolved, not what the config asked for.
            "resolved_model_revisions": _resolved_revisions(self.backend),
            "tokenizer_revision": self.tokenizer_revision,
            "shared_model_handles": getattr(self.backend, "shared_handle_ids", dict)(),
            "logical_agents": self.cfg.profile.agents.logical_count,
            "share_model_handle": self.cfg.profile.agents.share_model_handle,
            "agent_note": (
                f"{self.cfg.profile.agents.logical_count} logical agents sharing one "
                "unlearned model backend; NOT independently unlearned checkpoints"
            ),
            "topology": self.cfg.topology.to_dict(),
            "arms": [a.to_dict() for a in self.arm_plans],
            "detector": self.detector.to_dict(),
            # The EFFECTIVE status. A study may declare `calibrated`; a run only inherits
            # that when the artefact was calibrated on this run's forget policy.
            "detector_status": self.calibration_status["effective_status"],
            "detector_status_declared": self.calibration_status["declared_status"],
            "detector_calibration_covers_forget_policy": self.calibration_status[
                "covers_forget_policy"
            ],
            "detector_calibration_note": self.calibration_status["reason"],
            # The artefact behind the status, verbatim minus the sweep. `null` means the
            # run is diagnostic and says so; it is never absent.
            "detector_calibration": (
                {
                    key: value
                    for key, value in self.calibration.items()
                    if key not in ("sweep", "grid")
                }
                if self.calibration
                else None
            ),
            "concept_registry": self.registry.to_dict(),
            # ---- the two cohorts -------------------------------------------------
            # `cohort_*` names the EVALUATION cohort (the questions asked) and keeps its
            # historical key names. `forget_policy_*` names the frozen forget cohort the
            # registry and the deleted baseline memory came from. Both are recorded on
            # every run, equal or not, so a report never has to infer which was which.
            "cohort_fingerprint": self.cohort.fingerprint(),
            "cohort_split": self.cohort.split,
            "evaluation_cohort": {
                "split": self.cohort.split,
                "dataset_config": self.cohort.dataset_config,
                "dataset_revision": self.cohort.dataset_revision,
                "fingerprint": self.cohort.fingerprint(),
                "n_items": len(self.cohort.items),
                "n_concepts": len(self.cohort.concept_ids),
                "is_retain": self.cohort.is_retain,
            },
            "forget_policy_cohort": {
                "split": self.forget_policy.split,
                "dataset_config": self.forget_policy.dataset_config,
                "dataset_revision": self.forget_policy.dataset_revision,
                "fingerprint": self.forget_policy.fingerprint(),
                "n_items": len(self.forget_policy.items),
                "n_concepts": len(self.forget_policy.concept_ids),
                "is_retain": self.forget_policy.is_retain,
            },
            "forget_policy_fingerprint": self.forget_policy.fingerprint(),
            "forget_policy_split": self.forget_policy.split,
            "cohorts_separated": self.forget_policy.fingerprint() != self.cohort.fingerprint(),
            "retain_evaluation": self.cohort.is_retain,
            "cohort_note": (
                "the concept registry and the deleted baseline memory come from "
                f"'{self.forget_policy.split}' ({len(self.forget_policy.concept_ids)} "
                f"forgotten concepts); the questions come from '{self.cohort.split}' "
                f"({len(self.cohort.concept_ids)} concepts)"
            ),
            "control_mapping": self.mapping.to_dict(),
            "challenges": list(self.challenges),
            "uses_gold_answers": uses_gold,
            "uses_gold_answers_note": (
                "controlled challenges construct injected messages from the gold answer; "
                "the runtime concept registry never contains one"
                if uses_gold
                else "no injected content; every message is a model output"
            ),
            "sampling": {
                **self.cfg.study.sampling.model_dump(mode="json"),
                **self.cfg.profile.sampling.model_dump(mode="json"),
            },
            "profile_reportable": self.cfg.profile.reportable,
            "plan": plan.to_dict(),
            "offline_scoring": True,
            "semantic_scorer": None,
            "complete": False,
        }


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "absent"


def _resolved_revisions(backend: GenerationBackend) -> dict:
    """What the backend actually loaded, per model profile.

    The config records what was ASKED for. A manifest that repeats the request as though
    it were the outcome cannot detect a backend that silently fell back to a branch head,
    which is exactly the failure a pinned revision exists to prevent.
    """
    resolved = getattr(backend, "resolved_revisions", None)
    if callable(resolved):
        try:
            return dict(resolved())
        except Exception as exc:  # provenance is best-effort; never fail a finished run
            return {"error": f"{type(exc).__name__}: {exc}"}
    return {}
