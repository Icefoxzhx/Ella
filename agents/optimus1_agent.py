"""
Optimus-1 agent adapted for Ella Controlled Finals.

Adapts three core Optimus-1 mechanisms to the social simulation domain:
  1. Task Knowledge Graph (TKG) — analogous to HDKG; encodes TASK-STRUCTURE
     knowledge from scratch.json (role, party details, deadline, own group).
     Crucially, it does NOT pre-enumerate every candidate target from s_mem —
     that would be a privileged "solution list" with no counterpart in Optimus-1.
     HDKG stores domain/procedural knowledge (how crafting works), not instance
     data (who currently exists).  Candidates are derived at plan time from
     self.s_mem, which reflects only agents the agent has actually observed so
     far — the same live knowledge EllaAgent uses.
  2. Knowledge-Guided Planner — at plan time, reads the TKG + currently-known
     agents from s_mem + AMEP experience, and asks the LLM for an ordered
     subgoal list (find_and_converse, navigate_to_place, pick_item, put_item).
  3. Experience-Driven Reflector — fires every REFLECT_INTERVAL simulation
     seconds (matching Optimus-1's every-1200-steps cadence).  At each trigger
     the current frame is saved; at the NEXT trigger the previous frame becomes
     the "before" image and the current frame becomes the "after" image, giving
     a grounded visual before/after pair that is passed to the LLM alongside
     recent episodic memory (AMEP) to decide COMPLETE / CONTINUE / REPLAN.

Passive agents (empty daily_requirement) fall back to EllaAgent's schedule-based
behaviour entirely.
"""

import json
import re
import os
import numpy as np
from typing import Optional
from datetime import datetime

from .ella import EllaAgent


class Optimus1Agent(EllaAgent):

    MAX_SUBGOAL_ATTEMPTS = 240   # steps before auto-skipping a stuck subgoal
    REFLECT_INTERVAL     = 60   # simulation seconds between reflections (= Optimus-1's 1 min)
    _PLAN_PROMPT    = "agents/prompts/optimus1/prompt_plan.txt"
    _REFLECT_PROMPT = "agents/prompts/optimus1/prompt_reflect.txt"
    _REPLAN_PROMPT  = "agents/prompts/optimus1/prompt_replan.txt"
    _UTTER_PROMPT   = "agents/prompts/optimus1/prompt_utterance.txt"

    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False,
                 logger=None, lm_source="azure", lm_id="gpt-4o", max_tokens=4096,
                 temperature=0, top_p=1.0, detect_interval=1, region_layer=False,
                 enable_indoor_activities=False, model_channel=None, model_device=None):
        super().__init__(
            name, pose, info, sim_path, no_react, debug, logger,
            lm_source, lm_id, max_tokens, temperature, top_p,
            detect_interval, region_layer, enable_indoor_activities,
            model_channel, model_device,
        )

        # ── 1. Task Knowledge Graph ──────────────────────────────────────────
        self.tkg = self._build_task_knowledge_graph()

        # ── 2. Knowledge-Guided Planner state ───────────────────────────────
        self.subgoal_plan: list      = self.scratch.get("subgoal_plan", [])
        self.curr_subgoal_idx: int   = self.scratch.get("curr_subgoal_idx", 0)
        self._subgoal_attempts: dict = {}   # idx → consecutive step count

        # ── 3. Experience-Driven Reflector state ────────────────────────────
        # Restored from scratch so interval survives simulation restarts
        last_str = self.scratch.get("_last_reflect_time")
        self._last_reflect_time = (
            datetime.strptime(last_str, "%B %d, %Y, %H:%M:%S") if last_str else None
        )
        self._before_img_path: Optional[str] = self.scratch.get("_before_img_path")
        self._last_replan_reason = ""

        self.logger.info(
            f"Optimus1Agent {self.name}: active={self.tkg['is_active']}, "
            f"task_type={self.tkg.get('task_type')}, role={self.tkg.get('role')}"
        )

    # ════════════════════════════════════════════════════════════════════════
    # 1. Task Knowledge Graph
    # ════════════════════════════════════════════════════════════════════════

    def _build_task_knowledge_graph(self) -> dict:
        """
        Build the TKG from scratch.json only — no s_mem queries.

        This mirrors Optimus-1's HDKG design: the graph encodes DOMAIN /
        PROCEDURAL knowledge (task type, role, constraints, party details,
        goods requirements) derived from what the agent was explicitly told at
        the start of the simulation.  It does NOT pre-enumerate candidate
        targets from s_mem — that would give the agent a privileged "who exists"
        list before the simulation begins, which has no counterpart in HDKG and
        would be an unfair advantage over EllaAgent.

        Candidate targets are discovered during the simulation via observation
        (s_mem) and are built fresh at each call to _generate_subgoal_plan(),
        just as EllaAgent can only interact with agents it has actually seen.
        """
        groups = self.scratch.get("groups", [{}])
        my_group = groups[0] if groups else {}
        my_group_members: list = []
        for g in groups:
            my_group_members.extend(g.get("members", []))

        tkg = {
            "is_active":        False,
            "task_type":        None,    # "influence_battle" | "leadership_quest"
            "role":             None,    # "organizer" | "leader" | "member"
            "task_description": self.scratch.get("daily_requirement", ""),
            "task_details":     {},
            # ── progress tracking (updated at runtime) ──────────────────
            "completed_targets": [],
            "failed_targets":    [],
            # ── own-group context (from scratch.json — not privileged) ──
            "my_group":         my_group.get("name", ""),
            "my_group_place":   my_group.get("place", ""),
            "my_group_members": my_group_members,
        }

        task_desc = tkg["task_description"].strip()
        if not task_desc:
            return tkg

        tkg["is_active"] = True
        td_lower = task_desc.lower()

        # ── Influence Battle – organizer ────────────────────────────────
        if "party" in td_lower or "invite" in td_lower:
            tkg["task_type"] = "influence_battle"
            tkg["role"]      = "organizer"
            m = re.search(r"party at (.+?) from (\d+:\d+:\d+) to (\d+:\d+:\d+)", task_desc)
            if m:
                tkg["task_details"] = {
                    "party_place": m.group(1).strip(),
                    "party_start": m.group(2),
                    "party_end":   m.group(3),
                }

        # ── Leadership Quest – leader ────────────────────────────────────
        elif "leader" in td_lower and ("collect" in td_lower or "goods" in td_lower):
            tkg["task_type"] = "leadership_quest"
            tkg["role"]      = "leader"
            m = re.search(r"collect (.+?) from stores", task_desc)
            if m:
                tkg["task_details"]["target_goods"] = m.group(1).strip()
            tkg["task_details"]["deadline"]    = "12:00:00"
            tkg["task_details"]["group_place"] = my_group.get("place", "")

        # ── Leadership Quest – member ────────────────────────────────────
        elif "help" in td_lower and "leader" in td_lower:
            tkg["task_type"] = "leadership_quest"
            tkg["role"]      = "member"
            m = re.search(r"leader (.+?) to prepare", task_desc)
            if m:
                tkg["task_details"]["leader"] = m.group(1).strip()
            tkg["task_details"]["deadline"]    = "12:00:00"
            tkg["task_details"]["group_place"] = my_group.get("place", "")

        return tkg

    # ════════════════════════════════════════════════════════════════════════
    # 2. Knowledge-Guided Planner
    # ════════════════════════════════════════════════════════════════════════

    def _get_current_candidates(self) -> list:
        """
        Return candidate targets based on currently observed agents in s_mem.

        This is called at plan time — not at init — so it reflects only agents
        the agent has actually encountered during the simulation so far, keeping
        Optimus1Agent on equal informational footing with EllaAgent.
        """
        observed = set(self.s_mem.agents)   # agents seen / known so far
        excluded = set(self.tkg["my_group_members"]) | {self.name}
        excluded |= set(self.tkg["completed_targets"])

        role = self.tkg.get("role")
        task_type = self.tkg.get("task_type")

        if task_type == "influence_battle":
            # Invite anyone observed who is NOT in own group
            return [a for a in observed if a not in excluded]

        elif task_type == "leadership_quest" and role == "leader":
            # Coordinate with own group members who have been observed
            members = set(self.tkg["my_group_members"]) - {self.name}
            return [a for a in observed if a in members and a not in excluded]

        elif task_type == "leadership_quest" and role == "member":
            # Find the leader (if observed)
            leader = self.tkg["task_details"].get("leader", "")
            if leader and leader in observed and leader not in excluded:
                return [leader]
            return []

        return []

    def _generate_subgoal_plan(self) -> list:
        """
        Knowledge-Guided Planner: combine TKG task-structure knowledge with
        currently-observed candidates (s_mem) and AMEP episodic experience to
        produce an ordered subgoal list.

        Candidate list is derived from s_mem at THIS moment, not pre-loaded at
        init — so it only includes agents the simulation has actually shown the
        agent so far, matching EllaAgent's information level.
        """
        if self.no_react:
            return self._default_subgoal_plan()

        # AMEP retrieval — past experience relevant to the task
        experience = self.e_mem.retrieve(
            self.tkg["task_description"], None, self.curr_time, self.pose[:3], k=5
        )
        experience_str = self.describe_events(experience) if experience else "No prior experience."

        # Build candidate list from live s_mem (agents discovered so far)
        candidates_info = []
        for name in self._get_current_candidates():
            k = self.s_mem.get_knowledge(name)
            entry = {"name": name}
            if k:
                for field in ("age", "learned", "living_place", "groups"):
                    if field in k and k[field] is not None:
                        entry[field] = k[field]
            pos = self.s_mem.get_position_from_name(name)
            if pos is not None:
                entry["last_known_position"] = [round(float(p), 1) for p in pos[:2]]
            candidates_info.append(entry)

        prompt = open(self._PLAN_PROMPT).read()
        prompt = prompt.replace("$Character$",        self.get_character_description())
        prompt = prompt.replace("$Task$",             self.tkg["task_description"])
        prompt = prompt.replace("$TaskDetails$",      json.dumps(self.tkg.get("task_details", {}), indent=2))
        prompt = prompt.replace("$Candidates$",       json.dumps(candidates_info, indent=2))
        prompt = prompt.replace("$Places$",           self.get_places_description())
        prompt = prompt.replace("$Time$",             self.curr_time.strftime("%H:%M:%S"))
        prompt = prompt.replace("$Experience$",       experience_str)
        prompt = prompt.replace("$CompletedTargets$", json.dumps(self.tkg.get("completed_targets", [])))

        self.logger.debug("Generating Optimus-1 subgoal plan …")
        response = self.generator.generate(prompt, img=None, json_mode=False)

        try:
            plan = self.parse_json(prompt, response)
            assert isinstance(plan, list) and len(plan) > 0, "Plan must be a non-empty list"
            for step in plan:
                assert "type" in step
            self.logger.info(
                f"Subgoal plan ({len(plan)} steps): "
                + str([s.get("target", s.get("place", s["type"])) for s in plan])
            )
            return plan
        except Exception as e:
            self.logger.error(f"Plan generation failed ({e}); using default plan.")
            return self._default_subgoal_plan()

    def _default_subgoal_plan(self) -> list:
        """Fallback: one find_and_converse per currently-observed candidate."""
        plan = []
        for name in self._get_current_candidates():
            expected = None
            k = self.s_mem.get_knowledge(name)
            if k:
                for field in ("current_place", "living_place"):
                    if k.get(field):
                        expected = k[field]
                        break
            if not expected and k and isinstance(k.get("groups"), list) and k["groups"]:
                expected = k["groups"][0].get("place")
            plan.append({
                "type":           "find_and_converse",
                "target":         name,
                "goal":           self.tkg["task_description"],
                "expected_place": expected,
            })
        return plan

    # ════════════════════════════════════════════════════════════════════════
    # 3. Experience-Driven Reflector
    # ════════════════════════════════════════════════════════════════════════

    # ── In-context example pool (Optimus-1 AMEP analogue) ───────────────────
    # Optimus-1 stores (before_img, after_img) pairs labelled by reflection
    # outcome, retrieves one per class by fuzzy task+environment matching, and
    # injects them as raw image pairs into the GPT-4V message.
    # Here we mirror that structure but store text event narratives alongside
    # the image paths, because Ella's generator.generate() takes a single img
    # argument rather than an arbitrary multi-image message.  The text narrative
    # (from e_mem events over the interval) replaces the visual image pair as
    # the in-context signal — appropriate for a social domain where what was
    # *said* is at least as informative as what was *seen*.

    def _reflection_memory_file(self) -> str:
        """Path to the per-agent reflection example JSON (mirrors Optimus-1's per-task file)."""
        reflect_dir = os.path.join(self.storage_path, "reflection")
        os.makedirs(reflect_dir, exist_ok=True)
        return os.path.join(reflect_dir, "reflection_memory.json")

    def _save_reflection_example(self, state: str,
                                  before_img: Optional[str],
                                  after_img: Optional[str],
                                  subgoal: dict,
                                  event_narrative: str) -> None:
        """
        Save a labelled (before, after) example to the in-context pool.
        Mirrors Optimus-1's Memory.save_reflection() which appends
        [img_old, img_new] under task → environment → {done/continue/replan}.
        Key = task_type (plays the role of Optimus-1's environment biome).
        """
        task_key = self.tkg.get("task_type") or "unknown"
        path = self._reflection_memory_file()
        try:
            with open(path, "r") as f:
                mem = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            mem = {}
        mem.setdefault(task_key, {"COMPLETE": [], "CONTINUE": [], "REPLAN": []})
        mem[task_key][state].append({
            "before_img":      before_img,
            "after_img":       after_img,
            "subgoal_type":    subgoal.get("type", ""),
            "subgoal_goal":    subgoal.get("goal", subgoal.get("reason", "")),
            "event_narrative": event_narrative,
        })
        with open(path, "w") as f:
            json.dump(mem, f, indent=2)

    def _retrieve_reflection_examples(self) -> str:
        """
        Retrieve one in-context example per class from the stored pool.
        Mirrors Optimus-1's Memory.retrieve_reflection(): fuzzy-matches the
        task (here: task_type) and randomly samples one entry per class.
        Returns a formatted text block injected into the reflection prompt.
        """
        import random
        task_key = self.tkg.get("task_type") or "unknown"
        path = self._reflection_memory_file()
        try:
            with open(path, "r") as f:
                mem = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return "No prior reflection examples yet."

        # Prefer exact task_type match; fall back to any stored key
        if task_key not in mem:
            if not mem:
                return "No prior reflection examples yet."
            task_key = next(iter(mem))

        pool = mem[task_key]
        lines = ["[In-context reflection examples]"]
        for state, label in (("COMPLETE", "complete"), ("CONTINUE", "continue"), ("REPLAN", "replan")):
            examples = pool.get(state, [])
            if not examples:
                continue
            ex = random.choice(examples)
            lines.append(
                f"\n<{label}>:\n"
                f"  Subgoal type: {ex['subgoal_type']}\n"
                f"  Goal: {ex['subgoal_goal']}\n"
                f"  What happened during the interval: {ex['event_narrative']}\n"
                f"  Outcome: {state}"
            )
        if len(lines) == 1:
            return "No prior reflection examples yet."
        return "\n".join(lines)

    def _save_reflection_img(self, rgb: np.ndarray) -> Optional[str]:
        """Save an RGB frame for before/after comparison, return its path."""
        if rgb is None:
            return None
        from PIL import Image
        reflect_dir = os.path.join(self.storage_path, "reflection")
        os.makedirs(reflect_dir, exist_ok=True)
        path = os.path.join(
            reflect_dir,
            f"obs_{self.curr_time.strftime('%Y%m%d_%H%M%S')}.png"
        )
        Image.fromarray(rgb).save(path)
        return path

    def _tick_reflector(self, obs) -> None:
        """
        Time-triggered reflector — fires every REFLECT_INTERVAL simulation seconds.

        Mirrors Optimus-1's `if env.num_steps % MINUTE == 0` trigger.  At each
        firing the current frame is saved; the previous trigger's frame becomes
        the 'before' image and the current frame becomes the 'after' image,
        giving a before/after pair with REFLECT_INTERVAL seconds of separation.
        (Optimus-1 runs this asynchronously; here it is synchronous because
        Ella's step rate is orders of magnitude slower than Minecraft's 20 FPS.)
        """
        if not self.tkg["is_active"] or not self.subgoal_plan:
            return
        if self._last_reflect_time is not None:
            elapsed = (self.curr_time - self._last_reflect_time).total_seconds()
            if elapsed < self.REFLECT_INTERVAL:
                return

        self.logger.info(
            f"Reflection tick at {self.curr_time.strftime('%H:%M:%S')} "
            f"(subgoal {self.curr_subgoal_idx}/{len(self.subgoal_plan)})"
        )

        current_img_path = self._save_reflection_img(obs.get("rgb"))

        # Reflect only once we have a before-image from the previous trigger
        if self._before_img_path is not None:
            # Capture event narrative BEFORE applying reflection (still current interval)
            recent = self.e_mem.retrieve_latest_memory()
            event_narrative = self.describe_events(recent) if recent else "No events."

            result = self._reflect_on_subgoal(self._before_img_path, current_img_path)
            self._apply_reflection(result)

            # Save as in-context example for future reflections (Optimus-1 save_reflection)
            if self.subgoal_plan and self.curr_subgoal_idx > 0:
                prev_subgoal = self.subgoal_plan[self.curr_subgoal_idx - 1] if result["state"] == "COMPLETE" \
                    else (self.subgoal_plan[self.curr_subgoal_idx]
                          if self.curr_subgoal_idx < len(self.subgoal_plan) else {})
                self._save_reflection_example(
                    result["state"],
                    self._before_img_path,
                    current_img_path,
                    prev_subgoal,
                    event_narrative,
                )

        # Current frame becomes the before-image for the next interval
        self._before_img_path  = current_img_path
        self._last_reflect_time = self.curr_time

    def _reflect_on_subgoal(self, before_img_path: Optional[str],
                             after_img_path: Optional[str]) -> dict:
        """
        Experience-Driven Reflector: combine visual before/after with AMEP
        episodic retrieval to determine COMPLETE / CONTINUE / REPLAN.

        before_img_path — frame saved at the previous REFLECT_INTERVAL trigger.
        after_img_path  — frame saved at the current trigger (passed as the
                          image to the generator; the LLM sees 'now').
        """
        if not self.subgoal_plan or self.curr_subgoal_idx >= len(self.subgoal_plan):
            return {"state": "COMPLETE", "reason": "No pending subgoal."}
        if self.no_react:
            return {"state": "COMPLETE", "reason": "no_react mode."}

        curr_subgoal = self.subgoal_plan[self.curr_subgoal_idx]

        # AMEP: retrieve most recent episodic memories for context
        recent = self.e_mem.retrieve_latest_memory()
        experience_str = self.describe_events(recent) if recent else "No recent events."

        before_desc = (
            f"Saved at previous interval: {before_img_path}"
            if before_img_path else "Not available (first reflection interval)."
        )

        # Retrieve in-context examples (Optimus-1: retrieve_reflection → image pairs per class)
        in_context = self._retrieve_reflection_examples()

        prompt = open(self._REFLECT_PROMPT).read()
        prompt = prompt.replace("$Character$",         self.get_character_description())
        prompt = prompt.replace("$Task$",              self.tkg["task_description"])
        prompt = prompt.replace("$CurrentSubgoal$",    json.dumps(curr_subgoal, indent=2))
        prompt = prompt.replace("$RecentExperience$",  experience_str)
        prompt = prompt.replace("$BeforeObservation$", before_desc)
        prompt = prompt.replace("$InContextExamples$", in_context)
        prompt = prompt.replace("$CompletedTargets$",  json.dumps(self.tkg.get("completed_targets", [])))
        prompt = prompt.replace("$FailedTargets$",     json.dumps(self.tkg.get("failed_targets", [])))
        prompt = prompt.replace("$Time$",              self.curr_time.strftime("%H:%M:%S"))

        self.logger.debug(
            f"Reflecting on subgoal: {curr_subgoal.get('type')} "
            f"{curr_subgoal.get('target', '')} | before={before_img_path} after={after_img_path}"
        )
        # Pass the current (after) frame as the visual input to the LLM
        response = self.generator.generate(prompt, img=after_img_path, json_mode=False)

        try:
            result = self.parse_json(prompt, response)
            assert result["state"] in ("COMPLETE", "CONTINUE", "REPLAN"), \
                f"Invalid state: {result['state']}"
            self.logger.info(f"Reflection → {result['state']}: {result.get('reason', '')}")
            return result
        except Exception as e:
            self.logger.error(f"Reflection failed ({e}); defaulting to CONTINUE.")
            return {"state": "CONTINUE", "reason": "reflection error"}

    def _apply_reflection(self, result: dict):
        """Advance subgoal, keep going, or replan based on reflection output."""
        state  = result["state"]
        reason = result.get("reason", "")
        if self.curr_subgoal_idx < len(self.subgoal_plan):
            target = self.subgoal_plan[self.curr_subgoal_idx].get("target")
        else:
            target = None

        if state == "COMPLETE":
            if target and target not in self.tkg["completed_targets"]:
                self.tkg["completed_targets"].append(target)
            self.curr_subgoal_idx += 1
            self._subgoal_attempts.pop(self.curr_subgoal_idx - 1, None)
            self.logger.info(
                f"Subgoal COMPLETE → idx {self.curr_subgoal_idx}/{len(self.subgoal_plan)}"
            )

        elif state == "REPLAN":
            if target and target not in self.tkg["failed_targets"]:
                self.tkg["failed_targets"].append(target)
            self._last_replan_reason = reason
            new_plan = self._replan_subgoals(reason)
            if new_plan:
                self.subgoal_plan   = new_plan
                self.curr_subgoal_idx = 0
                self._subgoal_attempts = {}
                self.logger.info(f"Replanned ({len(new_plan)} steps): {reason}")
            else:
                self.curr_subgoal_idx += 1   # skip stuck subgoal

        # CONTINUE: do nothing — keep executing the current subgoal

    def _replan_subgoals(self, reason: str) -> list:
        """Replan with failure context, analogous to Optimus-1's replan prompt."""
        if self.no_react:
            return self._default_subgoal_plan()

        experience = self.e_mem.retrieve(reason, None, self.curr_time, self.pose[:3], k=5)
        experience_str = self.describe_events(experience) if experience else "No relevant experience."

        prompt = open(self._REPLAN_PROMPT).read()
        prompt = prompt.replace("$Character$",        self.get_character_description())
        prompt = prompt.replace("$Task$",             self.tkg["task_description"])
        prompt = prompt.replace("$FailReason$",       reason)
        prompt = prompt.replace("$PreviousPlan$",     json.dumps(self.subgoal_plan, indent=2))
        prompt = prompt.replace("$CompletedTargets$", json.dumps(self.tkg.get("completed_targets", [])))
        prompt = prompt.replace("$FailedTargets$",    json.dumps(self.tkg.get("failed_targets", [])))
        prompt = prompt.replace("$Experience$",       experience_str)
        prompt = prompt.replace("$Time$",             self.curr_time.strftime("%H:%M:%S"))

        response = self.generator.generate(prompt, img=None, json_mode=False)
        try:
            plan = self.parse_json(prompt, response)
            assert isinstance(plan, list)
            return plan
        except Exception as e:
            self.logger.error(f"Replan failed ({e}).")
            return []

    # ════════════════════════════════════════════════════════════════════════
    # Main decision loop
    # ════════════════════════════════════════════════════════════════════════

    def _process_obs(self, obs):
        # Reset plan on new day
        if obs.get("new_day") and self.tkg["is_active"]:
            self.subgoal_plan         = []
            self.curr_subgoal_idx     = 0
            self._subgoal_attempts    = {}
            self.tkg["completed_targets"] = []
            self.tkg["failed_targets"]    = []
        super()._process_obs(obs)

    def _act(self, obs):
        # Passive agents (no task): use EllaAgent's schedule-based loop
        if not self.tkg["is_active"]:
            return super()._act(obs)

        # ── Step 1: time-triggered reflector (Optimus-1 every-MINUTE cadence) ─
        self._tick_reflector(obs)

        # ── Step 2: continue or initiate conversation ──────────────────────
        if self.chatting_buffer:
            action = self.conversation(None)
            if action is not None:
                self.last_action = action
                return action

        # ── Step 3: generate plan if needed ───────────────────────────────
        if not self.subgoal_plan:
            self.subgoal_plan     = self._generate_subgoal_plan()
            self.curr_subgoal_idx = 0
            self._subgoal_attempts = {}

        # ── Step 4: check if all subgoals done ────────────────────────────
        if self.curr_subgoal_idx >= len(self.subgoal_plan):
            self.logger.info(f"{self.name}: all {len(self.subgoal_plan)} subgoals complete.")
            return {"type": "wait", "arg1": None}

        # ── Step 5: execute current subgoal ───────────────────────────────
        return self._execute_current_subgoal(obs)

    # ════════════════════════════════════════════════════════════════════════
    # Subgoal execution
    # ════════════════════════════════════════════════════════════════════════

    def _execute_current_subgoal(self, obs) -> dict:
        idx     = self.curr_subgoal_idx
        subgoal = self.subgoal_plan[idx]
        stype   = subgoal["type"]

        # Guard against infinite loops
        self._subgoal_attempts[idx] = self._subgoal_attempts.get(idx, 0) + 1
        if self._subgoal_attempts[idx] > self.MAX_SUBGOAL_ATTEMPTS:
            target = subgoal.get("target")
            self.logger.warning(f"Subgoal {subgoal} exceeded max attempts; skipping.")
            if target:
                self.tkg["failed_targets"].append(target)
            self.curr_subgoal_idx += 1
            self._subgoal_attempts.pop(idx, None)
            if self.curr_subgoal_idx < len(self.subgoal_plan):
                return self._execute_current_subgoal(obs)
            return {"type": "wait", "arg1": None}

        if stype == "navigate_to_place":
            return self._exec_navigate_to_place(subgoal, obs)
        elif stype in ("find_and_converse", "converse_with"):
            return self._exec_find_and_converse(subgoal, obs)
        elif stype == "pick_item":
            return self._exec_pick_item(subgoal, obs)
        elif stype == "put_item":
            return self._exec_put_item(subgoal, obs)
        else:
            self.logger.warning(f"Unknown subgoal type '{stype}'; waiting.")
            return {"type": "wait", "arg1": None}

    def _exec_navigate_to_place(self, subgoal: dict, obs) -> dict:
        goal = subgoal["place"]
        if self.current_place == goal:
            self.curr_subgoal_idx += 1
            self._subgoal_attempts.pop(self.curr_subgoal_idx - 1, None)
            return self._act(obs)
        action = self._navigate_toward_place(goal, obs)
        self.last_action = action
        return action

    def _exec_find_and_converse(self, subgoal: dict, obs) -> dict:
        target     = subgoal["target"]
        target_pos = self.s_mem.get_position_from_name(target)

        if target_pos is None:
            # Navigate to expected location derived from TKG world knowledge
            expected = subgoal.get("expected_place")
            if not expected:
                k = self.s_mem.get_knowledge(target)
                if k:
                    for field in ("current_place", "living_place"):
                        if k.get(field):
                            expected = k[field]
                            break
                if not expected and k and isinstance(k.get("groups"), list) and k["groups"]:
                    expected = k["groups"][0].get("place")
            if expected:
                action = self._navigate_toward_place(expected, obs)
                self.last_action = action
                return action
            return {"type": "wait", "arg1": None}

        dist = float(np.linalg.norm(np.array(target_pos) - np.array(self.pose[:3])))
        if dist > 10.0:
            action = self._navigate_toward_pos(np.array(target_pos[:2]), obs)
            self.last_action = action
            return action

        # Close enough: delegate to EllaAgent's conversation() for proper setup
        action = self.conversation(target)
        if action is not None:
            self.last_action = action
            return action
        return {"type": "wait", "arg1": None}

    def _exec_pick_item(self, subgoal: dict, obs) -> dict:
        item_name = subgoal.get("item")
        if item_name is None:
            self.curr_subgoal_idx += 1
            return self._act(obs)

        item_pos = self.s_mem.get_position_from_name(item_name)
        if item_pos is None:
            expected = subgoal.get("place")
            if expected:
                action = self._navigate_toward_place(expected, obs)
                self.last_action = action
                return action
            return {"type": "wait", "arg1": None}

        hand_slot = (0 if self.held_objects[0] is None
                     else (1 if self.held_objects[1] is None else None))
        if hand_slot is None:
            self.logger.info(f"Both hands full; skipping pick_item for {item_name}.")
            self.curr_subgoal_idx += 1
            return self._act(obs)

        dist = float(np.linalg.norm(np.array(item_pos) - np.array(self.pose[:3])))
        if dist > 2.0:
            action = self._navigate_toward_pos(np.array(item_pos[:2]), obs)
            self.last_action = action
            return action

        action = {"type": "pick", "arg1": hand_slot, "arg2": item_pos}
        self.last_action = action
        # Advance after issuing pick
        self.curr_subgoal_idx += 1
        self._subgoal_attempts.pop(self.curr_subgoal_idx - 1, None)
        return action

    def _exec_put_item(self, subgoal: dict, obs) -> dict:
        if self.held_objects[0] is None and self.held_objects[1] is None:
            self.curr_subgoal_idx += 1
            return self._act(obs)
        hand_slot = 0 if self.held_objects[0] is not None else 1
        action = {"type": "put", "arg1": hand_slot}
        self.last_action = action
        # Check if still holding; advance only when both hands empty
        if self.held_objects[1 - hand_slot] is None:
            self.curr_subgoal_idx += 1
            self._subgoal_attempts.pop(self.curr_subgoal_idx - 1, None)
        return action

    # ════════════════════════════════════════════════════════════════════════
    # Navigation helpers
    # ════════════════════════════════════════════════════════════════════════

    def _navigate_toward_place(self, goal_place: str, obs) -> dict:
        """High-level place navigation mirroring EllaAgent's commute logic."""
        if self.current_place == goal_place:
            return {"type": "wait", "arg1": None}
        accessible = obs.get("accessible_places", []) if obs else []
        if goal_place in accessible:
            return {"type": "enter", "arg1": goal_place}
        if self.current_place is not None:
            return {"type": "enter", "arg1": "open space"}
        # In open space: pathfind to goal
        info = self.s_mem.get_knowledge(goal_place)
        if info is None:
            self.logger.error(f"No knowledge for place '{goal_place}'.")
            return {"type": "wait", "arg1": None}
        goal_pos  = np.array([info["location"][0], info["location"][1]])
        goal_bbox = info.get("bounding_box")
        action = self.navigate(self.s_mem.get_sg(self.current_place), goal_pos, goal_bbox)
        return action if action else {"type": "wait", "arg1": None}

    def _navigate_toward_pos(self, target_pos_2d: np.ndarray, obs) -> dict:
        """Low-level position navigation (used when target agent is visible in s_mem)."""
        if self.current_place is not None:
            return {"type": "enter", "arg1": "open space"}
        action = self.navigate(self.s_mem.get_sg(self.current_place), target_pos_2d, None)
        return action if action else {"type": "wait", "arg1": None}

    # ════════════════════════════════════════════════════════════════════════
    # Task-aware conversation  (Optimus-1 goal-conditioned utterance)
    # ════════════════════════════════════════════════════════════════════════

    def generate_utterance(self, target_name: str, target_knowledge, target_experience):
        """Task-aware utterance: injects current subgoal context into generation."""
        if not self.tkg["is_active"]:
            return super().generate_utterance(target_name, target_knowledge, target_experience)

        curr_subgoal = (
            self.subgoal_plan[self.curr_subgoal_idx]
            if self.subgoal_plan and self.curr_subgoal_idx < len(self.subgoal_plan)
            else {}
        )

        conversation_history = "\n".join(
            f"{c.subject}: {c.content}" for c in self.chatting_buffer[-4:]
        ) or "No conversation history yet."

        retrieved = self.e_mem.retrieve(
            f"interactions with {target_name}", None, self.curr_time, self.pose[:3], 3
        )

        prompt = open(self._UTTER_PROMPT).read()
        prompt = prompt.replace("$Character$",           self.get_character_description())
        prompt = prompt.replace("$Time$",                self.curr_time.strftime("%H:%M:%S"))
        prompt = prompt.replace("$Place$",               self.current_place or "open space")
        prompt = prompt.replace("$Target_name$",         target_name)
        prompt = prompt.replace("$Target_knowledge$",    self.describe_knowledge(target_knowledge))
        prompt = prompt.replace("$Target_experience$",   self.describe_events(target_experience))
        prompt = prompt.replace("$TaskContext$",         json.dumps(self.tkg.get("task_details", {}), indent=2))
        prompt = prompt.replace("$CurrentSubgoal$",      json.dumps(curr_subgoal, indent=2))
        prompt = prompt.replace("$Conversation_history$", conversation_history)
        prompt = prompt.replace("$Context$",             self.describe_events(retrieved))

        self.logger.debug(f"Task-aware utterance prompt for {target_name}")
        response = self.generator.generate(prompt, img=None, json_mode=False)

        try:
            d         = self.parse_json(prompt, response)
            utterance = d["utterance"]
            self.logger.debug(f"Utterance: {utterance}  |  Reason: {d.get('reason', '')}")
        except Exception as e:
            self.logger.error(f"Utterance generation failed ({e}).")
            utterance = None
        return utterance

    # ════════════════════════════════════════════════════════════════════════
    # Persistence
    # ════════════════════════════════════════════════════════════════════════

    def save_scratch(self):
        self.scratch["subgoal_plan"]       = self.subgoal_plan
        self.scratch["curr_subgoal_idx"]   = self.curr_subgoal_idx
        self.scratch["_before_img_path"]   = self._before_img_path
        self.scratch["_last_reflect_time"] = (
            self._last_reflect_time.strftime("%B %d, %Y, %H:%M:%S")
            if self._last_reflect_time else None
        )
        super().save_scratch()
