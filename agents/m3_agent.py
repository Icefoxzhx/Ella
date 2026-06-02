import ast
import base64
import json
import sys
import os
import types
import importlib.util
import pickle
import numpy as np
from datetime import datetime, timedelta
from io import BytesIO
from PIL import Image

_M3_DIR = '/scratch/workspace/hongxinzhang_umass_edu-shared/m3-agent'

def _load_videograph():
    """Load VideoGraph directly, bypassing mmagent/__init__.py and its openai dependency."""
    # Register a minimal mmagent package so relative imports inside videograph.py resolve
    if 'mmagent' not in sys.modules:
        pkg = types.ModuleType('mmagent')
        pkg.__path__ = [os.path.join(_M3_DIR, 'mmagent')]
        pkg.__package__ = 'mmagent'
        sys.modules['mmagent'] = pkg

    if 'mmagent.memory_processing' not in sys.modules:
        stub = types.ModuleType('mmagent.memory_processing')
        stub.parse_video_caption = lambda _graph, _caption: []
        sys.modules['mmagent.memory_processing'] = stub

    if 'mmagent.videograph' not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            'mmagent.videograph',
            os.path.join(_M3_DIR, 'mmagent', 'videograph.py'),
        )
        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = 'mmagent'
        sys.modules['mmagent.videograph'] = mod
        _prev_cwd = os.getcwd()
        os.chdir(_M3_DIR)
        try:
            spec.loader.exec_module(mod)
        finally:
            os.chdir(_prev_cwd)

    return sys.modules['mmagent.videograph'].VideoGraph

VideoGraph = _load_videograph()

from vico.agents import Agent
from vico.tools.utils import is_near_goal
from tools.model_manager import global_model_manager
from .sg.builder.object import AGENT_TAGS
from .gen_agent_memory import SemanticMemory


_PROMPT_CAPTIONS = """You are given a sequence of egocentric observation frames from a simulated city environment, along with transcribed speech segments from nearby characters. Each voice feature is identified by a unique ID enclosed in angle brackets (e.g., <voice_1>, <voice_2>).

Your Task:
Generate a detailed description of the current 30-second clip. Each item must be a single atomic event or detail covering: characters' actions and movements, spoken dialogue, contextual behavior, or scene context.

Strict Requirements:
- If a character has an associated voice ID, refer to them only using that ID (e.g., <voice_1>).
- If a character does not have a voice ID, use a short descriptive phrase.
- Each description must represent a single atomic event. Do not combine multiple unrelated aspects into one line.
- Do not use pronouns. Do not invent events not grounded in the observations.
- Return only a valid JSON list of strings (starting with "[" and ending with "]").

Voice features:
"""

_PROMPT_THINKINGS = """You are given a sequence of egocentric observation frames from a simulated city environment, transcribed speech segments, and a list of clip descriptions. Each voice feature has a unique ID in angle brackets (e.g., <voice_1>).

Your Task:
Generate high-level reasoning-based conclusions across these categories:
1. Character-Level Attributes: inferred personality, role, interests, or distinctive behaviors for each character.
2. Interpersonal Relationships & Dynamics: relationships, tone, power dynamics, cooperation or conflict.
3. Scene-Level Summary: main event or theme, overall tone, cause-effect dynamics.
4. Contextual & General Knowledge: setting, cultural norms, or real-world facts inferable from the scene (e.g., "Alice market is pet-friendly").

Strict Requirements:
- Refer to characters only by their voice ID if available.
- Do not restate simple observations from the descriptions. Focus on high-level conclusions.
- Return only a valid JSON list of strings (starting with "[" and ending with "]").

Voice features:
"""


class M3Agent(Agent):
    CLIP_SECONDS = 30
    MAX_FRAMES_PER_CLIP = 5

    def __init__(self, name, pose, info, sim_path, no_react=False, debug=False, logger=None,
                 lm_source='azure', lm_id='gpt-4o', max_tokens=4096, temperature=0, top_p=1.0,
                 enable_gt_segmentation=True, model_channel=None, model_device=None):
        super().__init__(name, pose, info, sim_path, no_react, debug, logger)

        if model_channel is not None:
            global_model_manager.set_channel(model_channel, model_device)

        self.lm_source = lm_source
        self.lm_id = lm_id
        self.enable_gt_segmentation = enable_gt_segmentation

        self.generator = global_model_manager.get_generator(
            lm_source, lm_id, max_tokens, temperature, top_p, logger)
        self.generator_embedding = global_model_manager.get_generator(
            lm_source, 'text-embedding-3-small', max_tokens, temperature, top_p, logger)
        self.clip = global_model_manager.get_model("clip")

        self.s_mem = SemanticMemory(
            os.path.join(self.storage_path, "semantic_memory"),
            detect_interval=-1 if enable_gt_segmentation else 2,
            fov=self.fov,
            debug=debug,
            logger=logger,
        )

        # VideoGraph: text nodes + img_node per observed agent (character nodes)
        self.mem_path = os.path.join(self.storage_path, 'm3_memory_graph.pkl')
        if os.path.exists(self.mem_path):
            with open(self.mem_path, 'rb') as f:
                self.video_graph = pickle.load(f)
        else:
            self.video_graph = VideoGraph()

        # Rebuild character registry from loaded graph (for resume support)
        self.agent_nodes = {}
        for node_id, node in self.video_graph.nodes.items():
            if node.type == 'img' and node.metadata.get('contents'):
                self.agent_nodes[node.metadata['contents'][0]] = node_id

        # IB task fields
        self.daily_requirement = self.scratch.get("daily_requirement", "")
        self.group_name = ""
        self.group_members = []
        self.group_place = ""
        if self.scratch.get("groups"):
            g = self.scratch["groups"][0]
            self.group_name = g.get("name", "")
            self.group_members = g.get("members", [])
            self.group_place = g.get("place", "")

        # Navigation state (same pattern as GenAgent)
        self.commuting = None
        self.curr_goal_address = None
        self.curr_goal_pos = None
        self.curr_goal_bbox = None
        self.curr_goal_description = None
        self.curr_goal_duration = None
        self.curr_goal_end_time = None

        if self.scratch.get("act_address"):
            d = self.s_mem.get_knowledge(self.scratch["act_address"])
            if d:
                self.curr_goal_address = self.scratch["act_address"]
                self.curr_goal_pos = np.array([d["location"][0], d["location"][1]])
                self.curr_goal_bbox = np.array(d["bounding_box"])

        self.last_actions = [None] * 5
        self.react_freq = self.scratch.get("react_freq", 60)
        self.last_react_time = None
        self.cur_objects = []
        self.nearby_agents = []
        self.sim_step = 0
        self._prev_place = self.current_place

        # 30-second memorization buffer
        self._clip_frames = []      # list of PIL.Image
        self._clip_speech = {}      # agent_name -> list of {start_offset, end_offset, utterance}
        self._clip_start_time = None
        self._clip_id = 0

    # ── Character Nodes ──────────────────────────────────────────────────────

    def _ensure_character_node(self, agent_name, crop_rgb=None):
        """Get or create img_node for an agent; accumulate CLIP embeddings on repeat visits."""
        if agent_name in self.agent_nodes:
            node_id = self.agent_nodes[agent_name]
            if crop_rgb is not None and crop_rgb.size > 0:
                try:
                    emb = self.clip.predict_image(crop_rgb)
                    self.video_graph.update_node(node_id, {'contents': [], 'embeddings': [emb]})
                except Exception:
                    pass
            return node_id

        if crop_rgb is not None and crop_rgb.size > 0:
            try:
                emb = self.clip.predict_image(crop_rgb)
            except Exception:
                emb = np.array(self.generator_embedding.get_embedding(
                    agent_name, caller="m3_char_emb"))
        else:
            emb = np.array(self.generator_embedding.get_embedding(
                agent_name, caller="m3_char_emb"))

        node_id = self.video_graph.add_img_node({'contents': [agent_name], 'embeddings': [emb]})
        self.agent_nodes[agent_name] = node_id
        return node_id

    def _extract_agent_crop(self, agent_name, obs):
        """Return RGB crop of agent using gt_segmentation; None if unavailable."""
        if (not self.enable_gt_segmentation
                or obs.get('segmentation') is None
                or obs.get('gt_seg_idxc_to_info') is None
                or obs.get('rgb') is None):
            return None

        seg = obs['segmentation']
        for seg_id, info in enumerate(obs['gt_seg_idxc_to_info']):
            if not isinstance(info, dict):
                continue
            if info.get('type') == 'avatar' and info.get('name') == agent_name:
                mask = (seg == seg_id)
                if not mask.any():
                    continue
                rows = np.where(mask.any(axis=1))[0]
                cols = np.where(mask.any(axis=0))[0]
                if len(rows) == 0 or len(cols) == 0:
                    continue
                y1, y2 = int(rows.min()), int(rows.max()) + 1
                x1, x2 = int(cols.min()), int(cols.max()) + 1
                if y2 > y1 and x2 > x1:
                    return obs['rgb'][y1:y2, x1:x2]
        return None

    # ── Memorization (30-second clip pipeline) ───────────────────────────────

    def _add_memory_node(self, text, clip_id, node_type, voice_id_to_name):
        """Embed text, store as a memory node, and link to known character nodes."""
        emb = np.array(self.generator_embedding.get_embedding(text, caller="m3_mem_emb"))
        node_id = self.video_graph.add_text_node(
            {'contents': [text], 'embeddings': [emb]}, clip_id, node_type)
        for vid, name in voice_id_to_name.items():
            if f"<voice_{vid}>" in text and name in self.agent_nodes:
                self.video_graph.add_edge(node_id, self.agent_nodes[name])
        return node_id

    def _memorize_clip(self):
        """Run the two-step GPT-4o memorization on the buffered 30-second clip."""
        frames = self._clip_frames
        speech = self._clip_speech
        clip_id = self._clip_id

        # Build voice_id mapping: agent name -> integer id
        voice_id_to_name = {i + 1: name for i, name in enumerate(speech.keys())}
        name_to_voice_id = {name: i for i, name in voice_id_to_name.items()}

        voices_dict = {}
        for name, utterances in speech.items():
            vid = name_to_voice_id[name]
            voices_dict[f"<voice_{vid}>"] = utterances

        # Subsample frames evenly
        if frames:
            indices = np.linspace(0, len(frames) - 1, min(self.MAX_FRAMES_PER_CLIP, len(frames)), dtype=int)
            sampled = [frames[i] for i in indices]
        else:
            sampled = []

        voices_json = json.dumps(voices_dict, indent=2)

        # Step 1: episodic descriptions
        epi_prompt = _PROMPT_CAPTIONS + voices_json
        try:
            epi_raw = self.generator.generate(epi_prompt, img=sampled if sampled else None, caller="m3_memorize_epi")
            self.logger.debug(f"[m3_memorize_epi] clip {clip_id}:\n{epi_raw}")
            episodic_list = self._parse_str_list(epi_raw)
        except Exception as e:
            self.logger.warning(f"Episodic memorization failed for clip {clip_id}: {e}")
            episodic_list = []

        # Step 2: semantic conclusions
        sem_prompt = (
            _PROMPT_THINKINGS + voices_json +
            "\n\nClip descriptions:\n" + json.dumps(episodic_list, indent=2)
        )
        try:
            sem_raw = self.generator.generate(sem_prompt, img=sampled if sampled else None, caller="m3_memorize_sem")
            self.logger.debug(f"[m3_memorize_sem] clip {clip_id}:\n{sem_raw}")
            semantic_list = self._parse_str_list(sem_raw)
        except Exception as e:
            self.logger.warning(f"Semantic memorization failed for clip {clip_id}: {e}")
            semantic_list = []

        for text in episodic_list:
            self._add_memory_node(text, clip_id, 'episodic', voice_id_to_name)
        for text in semantic_list:
            self._add_memory_node(text, clip_id, 'semantic', voice_id_to_name)

        self.logger.info(f"{self.name}: memorized clip {clip_id} — {len(episodic_list)} episodic, {len(semantic_list)} semantic nodes.")
        self._clip_id += 1
        self._clip_frames = []
        self._clip_speech = {}
        self._clip_start_time = None

    def _parse_str_list(self, text):
        """Parse a JSON/Python list of strings from LLM output."""
        start = text.find('[')
        end = text.rfind(']')
        if start == -1 or end == -1:
            return []
        candidate = text[start:end + 1]
        try:
            result = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                result = ast.literal_eval(candidate)
            except Exception:
                return []
        return [s for s in result if isinstance(s, str)]

    # ── Retrieval ─────────────────────────────────────────────────────────────

    def _retrieve_context(self, query, topk=5):
        """Retrieve relevant memory texts; entity-aware if query mentions a known agent."""
        if not self.video_graph.text_nodes:
            return []

        range_nodes = [nid for name, nid in self.agent_nodes.items() if name in query]

        try:
            q_emb = np.array(self.generator_embedding.get_embedding(query, caller="m3_ret_emb"))
            results = self.video_graph.search_text_nodes([q_emb], range_nodes, mode='max')
            return [
                self.video_graph.nodes[nid].metadata['contents'][0]
                for nid, _ in results[:topk]
                if nid in self.video_graph.nodes and self.video_graph.nodes[nid].metadata.get('contents')
            ]
        except Exception as e:
            self.logger.warning(f"Retrieval failed: {e}")
            return []

    # ── Decision Making ───────────────────────────────────────────────────────

    def _get_available_places(self):
        """Return navigable place names from semantic memory (excluding personal rooms, stops, bikes)."""
        skip = {'open space'}
        places = []
        for name, data in self.s_mem.knowledge.items():
            if not isinstance(data, dict) or 'location' not in data:
                continue
            if name in skip:
                continue
            nl = name.lower()
            if any(k in nl for k in ("'s room", "bus stop", "bicycle sharing")):
                continue
            places.append(name)
        return places

    def _decide_action(self):
        """Retrieval-augmented LLM loop: pick [CONVERSE], [NAVIGATE], or [STAY]."""
        nearby_names = [a['name'] for a in self.nearby_agents if a['name'] not in self.group_members]
        time_str = self.curr_time.strftime('%H:%M') if self.curr_time else "unknown"
        task = self.daily_requirement or "Explore the city and meet new people."
        query = f"{task} {self.current_place or 'open space'} {time_str}"
        memories = self._retrieve_context(query, topk=5)
        mem_text = "\n".join(f"- {m}" for m in memories) if memories else "No memories yet."

        available_places = self._get_available_places()
        places_str = ", ".join(f'"{p}"' for p in available_places)

        prompt = (
            f"You are {self.name}. Task: {task}\n"
            f"Time: {time_str}  Location: {self.current_place or 'open space'}\n"
            f"People nearby (non-group members): {', '.join(nearby_names) if nearby_names else 'none'}\n"
            f"Retrieved memories:\n{mem_text}\n\n"
            f"Available places to navigate: {places_str}\n\n"
            "Choose one action. Reply with exactly one line using the exact place name from the list above:\n"
            "[CONVERSE] <person_name>   (must be in People nearby list)\n"
            "[NAVIGATE] <place_name>    (must be from Available places list)\n"
            "[STAY]"
        )
        self.logger.debug(f"[m3_decide] prompt:\n{prompt}")
        try:
            response = self.generator.generate(prompt, caller="m3_decide").strip()
            self.logger.debug(f"[m3_decide] response:\n{response}")
        except Exception as e:
            self.logger.warning(f"Decision failed: {e}")
            return "stay", ""

        if "[CONVERSE]" in response:
            parts = response.split("[CONVERSE]", 1)[1].strip().split()
            target = parts[0] if parts else ""
            if target not in nearby_names and nearby_names:
                target = nearby_names[0]
            if target:
                return "converse", target
        if "[NAVIGATE]" in response:
            place = response.split("[NAVIGATE]", 1)[1].strip()
            return "navigate", place
        return "stay", ""

    def _generate_utterance(self, target_name):
        """Generate a party invitation or conversation response for target_name."""
        memories = self._retrieve_context(f"{target_name} party invitation", topk=3)
        mem_ctx = " ".join(memories) if memories else ""

        prompt = (
            f"You are {self.name}. {self.get_character_description()}\n"
            f"Task: {self.daily_requirement}\n"
            f"You are talking to {target_name}. "
            + (f"Context from memory: {mem_ctx}\n" if mem_ctx else "") +
            "Generate a brief, natural party invitation (1-2 sentences). No prefix."
        )
        self.logger.debug(f"[m3_utterance] prompt:\n{prompt}")
        try:
            response = self.generator.generate(prompt, caller="m3_utterance").strip()
            self.logger.debug(f"[m3_utterance] response:\n{response}")
            return response
        except Exception as e:
            self.logger.warning(f"Utterance generation failed: {e}")
            return f"Hi {target_name}, would you like to join our party?"

    def _plan_next_goal(self):
        """Decide where to navigate next using the retrieval-augmented loop."""
        action_type, action_content = self._decide_action()
        if action_type != "navigate" or not action_content:
            return

        place_name = action_content
        place_dict = self.s_mem.get_knowledge(place_name)

        # Fallback: pick a random available place if LLM output didn't match exactly
        if place_dict is None:
            self.logger.warning(f"{self.name}: could not find place '{action_content}', picking random.")
            candidates = self._get_available_places()
            if not candidates:
                self.logger.warning(f"{self.name}: no available places in knowledge base, cannot plan goal.")
                return
            import random
            place_name = random.choice(candidates)
            place_dict = self.s_mem.get_knowledge(place_name)

        if place_dict is None:
            self.logger.warning(f"{self.name}: fallback place also not found.")
            return

        self.curr_goal_address = place_name
        self.curr_goal_pos = np.array([place_dict["location"][0], place_dict["location"][1]])
        self.curr_goal_bbox = np.array(place_dict["bounding_box"])
        self.curr_goal_description = "invite people to party"
        self.curr_goal_duration = 30
        self.curr_goal_end_time = self.curr_time + timedelta(minutes=30)
        self.commuting = True

        time_str = self.curr_time.strftime('%H:%M') if self.curr_time else "unknown"
        thought = f"At {time_str}, I decided to go to {place_name} to find people to invite."
        emb = np.array(self.generator_embedding.get_embedding(thought, caller="m3_plan_emb"))
        self.video_graph.add_text_node({'contents': [thought], 'embeddings': [emb]}, self.sim_step, 'episodic')
        self.logger.info(f"{self.name} planned to navigate to {place_name}.")

    # ── Observation Processing ────────────────────────────────────────────────

    def _process_obs(self, obs):
        self.cur_objects = self.s_mem.update_ga(obs)
        self.current_place = obs['current_place']
        self.obs = obs
        self.curr_time = obs['curr_time']
        self.sim_step += 1

        # Initialise clip window on first observation
        if self._clip_start_time is None:
            self._clip_start_time = self.curr_time

        # Buffer one frame per step
        if obs.get('rgb') is not None:
            self._clip_frames.append(Image.fromarray(obs['rgb']))

        # Build nearby_agents unified across GT-seg and non-GT modes
        self.nearby_agents = []
        if (self.enable_gt_segmentation
                and obs.get('segmentation') is not None
                and obs.get('gt_seg_idxc_to_info') is not None):
            seg = obs['segmentation']
            for seg_id in np.unique(seg).tolist():
                if seg_id == 0:
                    continue
                try:
                    info = obs['gt_seg_idxc_to_info'][seg_id]
                except (IndexError, KeyError):
                    continue
                if not isinstance(info, dict):
                    continue
                if info.get('type') == 'avatar' and info.get('name') != self.name:
                    agent_name = info['name']
                    mask = (seg == seg_id)
                    depth_val = 5.0
                    if obs.get('depth') is not None and mask.any():
                        rows, cols = np.where(mask)
                        cy, cx = int(rows.mean()), int(cols.mean())
                        depth_val = float(obs['depth'][cy, cx])
                    self.nearby_agents.append({
                        'name': agent_name,
                        'range': depth_val + 3,
                        'seg_id': seg_id,
                    })
                    crop = self._extract_agent_crop(agent_name, obs)
                    self._ensure_character_node(agent_name, crop)
        else:
            for obj in self.cur_objects:
                if obj.get("tag") in AGENT_TAGS and obj.get("name") != self.name:
                    pos = obj.get("position")
                    dist = (np.linalg.norm(np.array(pos) - np.array(self.pose[:3])) + 1
                            if pos is not None else 5.0)
                    self.nearby_agents.append({'name': obj["name"], 'range': dist})
                    self._ensure_character_node(obj["name"])

        # Buffer speech events for the current clip window
        elapsed = (self.curr_time - self._clip_start_time).total_seconds()
        for event in obs.get('events', []):
            if event.get("type") != "speech":
                continue
            speaker = event.get("subject")
            if isinstance(event.get("content"), dict):
                speaker = event["content"].get("from_name", speaker)
            if speaker is None and event.get("position"):
                speaker = self.s_mem.get_name_from_position(event["position"])
            if not speaker or speaker == self.name:
                continue
            utterance = event.get("content", "")
            if isinstance(utterance, dict):
                utterance = utterance.get("utterance", "")
            self._ensure_character_node(speaker)
            start_mm_ss = f"{int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
            end_mm_ss = f"{int((elapsed + 1) // 60):02d}:{int((elapsed + 1) % 60):02d}"
            self._clip_speech.setdefault(speaker, []).append({
                "start_time": start_mm_ss,
                "end_time": end_mm_ss,
                "content": utterance,
            })

        # Trigger memorization when the 30-second window closes
        if elapsed >= self.CLIP_SECONDS:
            self._memorize_clip()
        self._prev_place = self.current_place

    # ── Action Selection ──────────────────────────────────────────────────────

    def _act(self, obs):
        # Respond to nearby speech from non-group agents
        for event in obs.get('events', []):
            if event.get("type") != "speech":
                continue
            speaker = event.get("subject")
            if isinstance(event.get("content"), dict):
                speaker = event["content"].get("from_name", speaker)
            if speaker and speaker != self.name and speaker not in self.group_members:
                pos = event.get("position")
                if pos is not None and list(pos[:2]) != list(self.pose[:2]):
                    utt = self._generate_utterance(speaker)
                    action = {'type': 'converse', 'arg1': utt, 'arg2': 5.0}
                    self.update_action(action)
                    return action

        # React timer: invite visible agents or plan next destination
        should_react = (
            self.last_react_time is None
            or (self.curr_time - self.last_react_time).total_seconds() >= self.react_freq
        )
        if should_react:
            self.last_react_time = self.curr_time

            for agent_info in self.nearby_agents:
                if agent_info['name'] not in self.group_members:
                    utt = self._generate_utterance(agent_info['name'])
                    action = {
                        'type': 'converse',
                        'arg1': utt,
                        'arg2': agent_info.get('range', 5.0),
                    }
                    self.update_action(action)
                    return action

            if (self.curr_goal_address is None
                    or self.curr_goal_end_time is None
                    or self.curr_time >= self.curr_goal_end_time):
                self._plan_next_goal()

        # Navigation (same pattern as GenAgent._act lines 584-617)
        if self.curr_goal_address is not None:
            if obs.get('current_place') == self.curr_goal_address or self.commuting is False:
                self.update_action(None)
                return None

            if self.curr_goal_address in (obs.get('accessible_places') or []):
                action = {'type': 'enter', 'arg1': self.curr_goal_address}
                self.update_action(action)
                self.commuting = False
                return action

            if obs.get('current_place') is not None:
                action = {'type': 'enter', 'arg1': 'open space'}
                self.update_action(action)
                self.commuting = True
                return action

            cur_trans = np.array(self.pose[:2])
            if is_near_goal(cur_trans[0], cur_trans[1], self.curr_goal_bbox, self.curr_goal_pos):
                self.logger.warning(f"{self.name} near goal {self.curr_goal_address} but can't enter.")
                self.update_action(None)
                return None

            nav_action = self.navigate()
            if nav_action is not None:
                self.update_action(nav_action)
                return nav_action

        return None

    # ── Navigation ────────────────────────────────────────────────────────────

    def navigate(self):
        return self.navigate_helper(
            self.s_mem.get_sg(self.current_place), self.curr_goal_pos, self.curr_goal_bbox)

    def navigate_helper(self, sg, goal_pos, goal_bbox=None):
        from vico.tools.utils import get_bbox, get_axis_aligned_bbox
        if goal_pos is None:
            return None
        cur_trans = np.array(self.pose[:2])
        goal_bbox = get_bbox(goal_bbox, goal_pos)
        path = sg.volume_grid_builder.navigate(cur_trans, goal_bbox, self.last_path)
        self.last_path = None
        if path is None:
            self.logger.error(f"No path found for {self.name} to {goal_pos}.")
            return None

        if self.current_vehicle == "bicycle":
            nav_grid_num = int(self.BIKE_SPEED // sg.volume_grid_builder.conf.nav_grid_size)
        else:
            nav_grid_num = int(self.WALK_SPEED // sg.volume_grid_builder.conf.nav_grid_size)

        cur_goal = path[min(nav_grid_num, len(path) - 1)]
        if sg.volume_grid_builder.has_obstacle(get_axis_aligned_bbox(np.array([cur_goal, cur_trans]), None)):
            cur_goal = path[min(2, len(path) - 1)]

        target_rad = np.arctan2(cur_goal[1] - cur_trans[1], cur_goal[0] - cur_trans[0])
        delta_rad = target_rad - self.pose[-1]
        if delta_rad > np.pi:
            delta_rad -= 2 * np.pi
        elif delta_rad < -np.pi:
            delta_rad += 2 * np.pi

        if delta_rad > np.deg2rad(15):
            action = {'type': 'turn_left', 'arg1': np.rad2deg(delta_rad)}
            self.last_path = path
        elif delta_rad < -np.deg2rad(15):
            action = {'type': 'turn_right', 'arg1': np.rad2deg(-delta_rad)}
            self.last_path = path
        else:
            action = {'type': 'move_forward', 'arg1': np.linalg.norm(cur_goal - cur_trans)}

        if isinstance(action['arg1'], float) and action['arg1'] < 0.1:
            self.logger.warning(f"{self.name} move arg1 < 0.1.")
        return action

    # ── Utilities ─────────────────────────────────────────────────────────────

    def update_action(self, action):
        self.last_actions.pop(0)
        self.last_actions.append(action)

    def chat(self, content):
        """Respond to an ongoing conversation; content = other agent's name."""
        return self._generate_utterance(str(content))

    def get_character_description(self):
        return (
            f"Name: {self.name}\n"
            f"Innate traits: {self.scratch.get('innate', '')}\n"
            f"Learned traits: {self.scratch.get('learned', '')}\n"
            f"Currently: {self.scratch.get('currently', '')}\n"
            f"Groups: {self.scratch.get('groups', [])}\n"
            f"Daily plan requirement: {self.daily_requirement}\n"
        )

    def save_scratch(self):
        super().save_scratch()
        try:
            with open(self.mem_path, 'wb') as f:
                pickle.dump(self.video_graph, f)
        except Exception as e:
            self.logger.warning(f"Failed to save VideoGraph: {e}")
