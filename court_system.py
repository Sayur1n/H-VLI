import os
import json
import logging
import re
import time
from typing import List, Dict, Any, Optional
from json_repair import repair_json

from utils import encode_image, get_image_full_path
import court_prompts as prompts_mc
import court_prompts_binary as prompts_bin

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

def clean_json_like(raw):
    """
    Clean and parse a JSON-like string returned by an LLM.
    """
    if raw is None: return None
    text = str(raw).strip()

    try:
        codeblocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.IGNORECASE)
        text_to_repair = codeblocks[-1] if codeblocks else text
        return repair_json(text_to_repair, return_objects=True)
    except:
        pass

    text_clean = re.sub(r"(^|\s)//.*", "", text, flags=re.MULTILINE)
    regex_pattern = r"(\{[\s\S]*?\}|\[[\s\S]*?\])"
    matches = re.findall(regex_pattern, text_clean)
    
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except:
            continue

    try:
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text_clean)
        if match:
            return json.loads(match.group(1))
    except:
        pass

    return raw

class CourtroomSystem:
    """
    Core system for managing the ARCADE layered debate and baseline modes.
    """
    def __init__(self, client, image_base_dir: str, cache_base_dir: str = "answers_cache", dataset_name: str = "mmhs", class_mode: str = "multiclass"):
        self.client = client
        self.image_base_dir = image_base_dir
        self.cache_base_dir = cache_base_dir
        self.dataset_name = dataset_name 
        self.class_mode = class_mode

        if self.class_mode == 'binary':
            self.prompts = prompts_bin
            logger.info("Courtroom System Initialized in BINARY Mode.")
        else:
            self.prompts = prompts_mc
            logger.info("Courtroom System Initialized in MULTI-CLASS Mode.")
        
        if not os.path.exists(self.cache_base_dir):
            os.makedirs(self.cache_base_dir)

    def call_llm_raw_messages(self, messages_content, model, temperature=0.0):
        """
        Low-level LLM call with a list of messages.
        """
        if not self.client: 
            return {"error": "No Client", "refusal": True}
            
        messages = [{"role": "user", "content": messages_content}]
        try:
            raw_response = self.client.call(model_name=model, messages=messages, temperature=temperature)
            raw_text = raw_response.get("content", "") if isinstance(raw_response, dict) else raw_response
            parsed = clean_json_like(raw_text)
            
            if not isinstance(parsed, (dict, list)):
                if isinstance(parsed, str):
                    parsed = {"content": parsed, "error": "Output format is not JSON", "refusal": False}
                else:
                    parsed = {"error": "Empty Response", "refusal": True}
            
            return parsed
        except Exception as e:
            return {"error": str(e), "refusal": True}

    def call_llm(self, system_prompt: str, user_content: Any, model: str, temperature: float = 0.0) -> dict:
        """
        Standard LLM call with system prompt and user content.
        """
        if not self.client: return {"error": "No Client", "refusal": True}
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_content}]
        try:
            raw_response = self.client.call(model_name=model, messages=messages, temperature=temperature)
            raw_text = raw_response.get("content", "") if isinstance(raw_response, dict) else raw_response
            parsed = clean_json_like(raw_text)
            if not isinstance(parsed, (dict, list)):
                if isinstance(parsed, str):
                    parsed = {"content": parsed, "error": "Output format is not JSON", "refusal": False}
                else:
                    parsed = {"error": "Empty Response", "refusal": True}
            return parsed
        except Exception as e:
            return {"error": str(e), "refusal": True}

    def _prepare_multimodal_content(self, text, image_path):
        """
        Prepare content structure for multimodal LLM input (text + base64 image).
        """
        content = [{"type": "text", "text": f"Tweet Content:\n{text}"}]
        full_img_path = get_image_full_path(self.image_base_dir, image_path)
        b64 = encode_image(full_img_path)
        if b64:
            content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
            return content, True
        else:
            content.append({"type": "text", "text": "[Image missing]"})
            return content, False
    
    def _get_cache_path(self, aux_model, mode, sid):
        """
        Construct the local file path for caching results.
        """
        safe_model = aux_model.replace("/", "_").replace(":", "_")
        directory = os.path.join(self.cache_base_dir, self.dataset_name, safe_model, mode)
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"{sid}.json")

    def _load_cache(self, aux_model, mode, sid):
        """
        Load cached results for a specific sample.
        """
        path = self._get_cache_path(aux_model, mode, sid)
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f: return json.load(f)
            except: pass
        return {}

    def _save_cache(self, aux_model, mode, sid, new_data):
        """
        Merge and save new data into the existing cache.
        """
        path = self._get_cache_path(aux_model, mode, sid)
        current_data = self._load_cache(aux_model, mode, sid)
        def recursive_update(d, u):
            for k, v in u.items():
                if isinstance(v, dict): d[k] = recursive_update(d.get(k, {}), v)
                else: d[k] = v
            return d
        recursive_update(current_data, new_data)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, ensure_ascii=False, indent=2)

    def check_is_completed(self, aux_model, run_mode, sid, rounds=2):
        """
        Check if a specific sample has already been fully processed and cached.
        """
        mode_key = run_mode
        if run_mode == 'ARCADE':
            mode_key = f"ARCADE_r{rounds}" if rounds != 2 else "ARCADE"
        cached_data = self._load_cache(aux_model, mode_key, sid)
        return bool(cached_data)

    def run_baseline_none(self, model, sid, item):
        """
        Run the simple baseline classification mode.
        """
        try:
            content, _ = self._prepare_multimodal_content(item["tweet_text"], item["image_path"])
            content.insert(0, {"type": "text", "text": self.prompts.SAFETY_PROMPT})
            res = self.call_llm(system_prompt="", user_content=content, model=model, temperature=0.8)
            if not isinstance(res, dict): res = {"label": res, "reason": str(res)}
            return res
        except Exception as e:
            return {"error": str(e), "refusal": True}

    def run_arcade(self, judge_model, aux_model, sid, item, rounds=2):
        """
        Run the full ARCADE layered debate system (Direct Hate Gatekeeper -> Indirect Debate).
        """
        MODE_KEY = f"ARCADE{f'_r{rounds}' if rounds != 2 else ''}"
        cached_data = self._load_cache(aux_model, MODE_KEY, sid)
        text = item.get('tweet_text', '')
        img_path = item.get('image_path', '')
        content_base, _ = self._prepare_multimodal_content(text, img_path)
        process_log = {"stage_a": cached_data.get("stage_a", {}), "stage_b": cached_data.get("stage_b", {})}

        def is_step_failure(data):
            return isinstance(data, dict) and (data.get("refusal") or "error" in data)

        p_check = process_log["stage_a"].get("prosecutor")
        if not p_check or is_step_failure(p_check):
            content_p = list(content_base)
            content_p.append({"type": "text", "text": "\nTask: Identify DIRECT hate speech cues. Output JSON."})
            p_check = self.call_llm(self.prompts.PROSECUTOR_DIRECT_PROMPT, content_p, aux_model, temperature=0.0)
            if not is_step_failure(p_check): self._save_cache(aux_model, MODE_KEY, sid, {"stage_a": {"prosecutor": p_check}})
        
        if is_step_failure(p_check):
            return {"final_stage": "error", "verdict": {"refusal": True, "label": -1, "reason": "Stage A Failed"}, "process_details": process_log}

        if isinstance(p_check, list) and len(p_check) > 0:
            p_std = process_log["stage_a"].get("standard_p")
            if not p_std or is_step_failure(p_std):
                content_p = list(content_base)
                content_p.append({"type": "text", "text": "\nTask: Analyze accusations."})
                p_std = self.call_llm(self.prompts.PROSECUTOR_PROMPT, content_p, aux_model, temperature=0.8)
                if not is_step_failure(p_std): self._save_cache(aux_model, MODE_KEY, sid, {"stage_a": {"standard_p": p_std}})

            d_std = process_log["stage_a"].get("standard_d")
            if not d_std or is_step_failure(d_std):
                content_d = list(content_base)
                content_d.append({"type": "text", "text": f"\nAccusations: {json.dumps(p_std)}\nTask: Refute."})
                d_std = self.call_llm(self.prompts.DEFENDER_PROMPT, content_d, aux_model, temperature=0.8)
                if not is_step_failure(d_std): self._save_cache(aux_model, MODE_KEY, sid, {"stage_a": {"standard_d": d_std}})

            content_j = list(content_base)
            content_j.append({"type": "text", "text": f"\nProsecutor: {json.dumps(p_std)}\nDefense: {json.dumps(d_std)}\nVerdict."})
            verdict = self.call_llm(self.prompts.JUDGE_PROMPT, content_j, judge_model, temperature=0.1)
            return {"final_stage": "direct_standard", "prosecutor": p_std, "defense": d_std, "verdict": verdict, "process_details": process_log}

        history = []; p_prev = None; d_prev = None
        for r in range(rounds):
            rk = f"round_{r+1}"
            rc = process_log["stage_b"].get(rk, {})
            p_curr = rc.get("p")
            if not p_curr or is_step_failure(p_curr):
                content_p = list(content_base)
                prompt = self.prompts.PROSECUTOR_INDIRECT_PROMPT if r == 0 else self.prompts.PROSECUTOR_ROUND_2_PROMPT
                p_curr = self.call_llm(prompt, content_p, aux_model, temperature=0.7)
                if not is_step_failure(p_curr): self._save_cache(aux_model, MODE_KEY, sid, {"stage_b": {rk: {"p": p_curr}}})

            if r == 0 and isinstance(p_curr, list) and len(p_curr) == 0:
                return {"final_stage": "safe", "verdict": {"label": 0, "reason": "No cues found."}, "process_details": process_log}

            d_curr = rc.get("d")
            if not d_curr or is_step_failure(d_curr):
                content_d = list(content_base)
                prompt = self.prompts.DEFENDER_INDIRECT_PROMPT if r == 0 else self.prompts.DEFENSE_ROUND_2_PROMPT
                d_curr = self.call_llm(prompt, content_d, aux_model, temperature=0.7)
                if not is_step_failure(d_curr): self._save_cache(aux_model, MODE_KEY, sid, {"stage_b": {rk: {"d": d_curr}}})

            p_prev, d_prev = p_curr, d_curr
            history.append({"round": r+1, "prosecutor": p_curr, "defense": d_curr})

        content_j = list(content_base)
        content_j.append({"type": "text", "text": f"\nTranscript: {json.dumps(history)}\nVerdict."})
        verdict = self.call_llm(self.prompts.JUDGE_PROMPT_INDIRECT, content_j, judge_model, temperature=0.1)
        return {"final_stage": "indirect_multiround", "prosecutor": p_prev, "defense": d_prev, "verdict": verdict, "process_details": process_log}
