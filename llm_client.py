import os
import json
import requests
from itertools import cycle
from threading import Lock
from openai import OpenAI, BadRequestError, APITimeoutError
from dotenv import load_dotenv

load_dotenv()

class KeyManager:
    """
    Manages API keys for various LLM providers with rotation and fallback logic.
    """
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(KeyManager, cls).__new__(cls)
                cls._instance._init_keys()
        return cls._instance

    def _init_keys(self):
        self.openai_official_key = os.getenv("OPENAI_API_KEY")
        self.gemini_official_key = os.getenv("GEMINI_API_KEY")

        self.dashscope_keys = self._load_keys_from_env("DASHSCOPE_API_KEY")
        self.dashscope_cycle = cycle(self.dashscope_keys) if self.dashscope_keys else None
        self.dashscope_lock = Lock()

        self.api_yi_keys = self._load_keys_from_env("API_YI_API_KEY")
        self.api_yi_cycle = cycle(self.api_yi_keys) if self.api_yi_keys else None
        self.api_yi_lock = Lock()

        self.glm_keys = self._load_keys_from_env("GLM_API_KEY")
        self.glm_cycle = cycle(self.glm_keys) if self.glm_keys else None
        self.glm_lock = Lock()

    def _load_keys_from_env(self, base_name):
        keys = []
        i = 1
        while True:
            key = os.getenv(f"{base_name}_{i}")
            if not key: break
            keys.append(key)
            i += 1
        if not keys:
            default_key = os.getenv(base_name)
            if default_key: keys.append(default_key)
        return keys

    def get_next_key(self, model_name):
        """
        Retrieve the next available API key and endpoint for a given model.
        """
        model_lower = model_name.lower()

        if model_lower.startswith("glm"):
            url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
            if self.glm_cycle:
                with self.glm_lock: return next(self.glm_cycle), url, "glm_rest"
            return None, url, "glm_rest"

        elif model_lower.startswith("qwen"):
            url = "https://dashscope.aliyuncs.com/compatible-mode/v1"
            if self.dashscope_cycle:
                with self.dashscope_lock: return next(self.dashscope_cycle), url, "openai_sdk"
            return None, url, "openai_sdk"

        elif "gpt" in model_lower:
            if self.openai_official_key:
                return self.openai_official_key, "https://api.openai.com/v1", "openai_sdk"
            url = "https://api.apiyi.com/v1"
            if self.api_yi_cycle:
                with self.api_yi_lock: return next(self.api_yi_cycle), url, "openai_sdk"
            return None, url, "openai_sdk"

        elif "gemini" in model_lower:
            if self.gemini_official_key:
                return self.gemini_official_key, "https://generativelanguage.googleapis.com/v1beta/openai/", "openai_sdk"
            url = "https://api.apiyi.com/v1"
            if self.api_yi_cycle:
                with self.api_yi_lock: return next(self.api_yi_cycle), url, "openai_sdk"
            return None, url, "openai_sdk"

        else:
            url = "https://api.apiyi.com/v1"
            if self.api_yi_cycle:
                with self.api_yi_lock: return next(self.api_yi_cycle), url, "openai_sdk"
            return None, url, "openai_sdk"

class LLMClient:
    """
    Client for interacting with various LLM providers using standard SDKs or REST APIs.
    """
    def __init__(self):
        self.key_manager = KeyManager()

    def call(self, model_name, messages, temperature=0.0, timeout=60):
        """
        Invoke an LLM with the provided parameters.
        """
        try:
            api_key, base_url, provider = self.key_manager.get_next_key(model_name)
            if not api_key:
                return {"error": "No API Key found", "refusal": True}

            if provider == "glm_rest":
                return self._call_glm_rest(api_key, base_url, model_name, messages, temperature, timeout)
            else:
                return self._call_openai_sdk(api_key, base_url, model_name, messages, temperature, timeout)

        except Exception as e:
            return {"error": "General Client Error", "details": str(e)}

    def _call_openai_sdk(self, api_key, base_url, model_name, messages, temperature, timeout):
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                timeout=timeout
            )
            return resp.choices[0].message.content
        except APITimeoutError:
            return json.dumps({"refusal": True, "error": "Request Timed Out"})
        except BadRequestError as e:
            return json.dumps({"refusal": True, "error": f"Bad Request: {str(e)}"})
        except Exception as e:
            error_str = str(e)
            if any(x in error_str.lower() for x in ["inappropriate", "safety", "inspection"]):
                return json.dumps({"refusal": True, "error": "Safety Policy Triggered"})
            return json.dumps({"error": "OpenAI SDK Error", "details": error_str})

    def _call_glm_rest(self, api_key, url, model_name, messages, temperature, timeout):
        try:
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": model_name,
                "messages": messages,
                "stream": False,
                "do_sample": True if temperature > 0 else False,
                "temperature": temperature if temperature > 0 else 0.01,
            }
            response = requests.post(url, json=payload, headers=headers, timeout=timeout)
            if response.status_code != 200:
                err_json = response.json()
                msg = err_json.get("error", {}).get("message", response.text)
                if err_json.get("error", {}).get("code") in [1301, 1302]:
                    return json.dumps({"refusal": True, "error": f"GLM Safety: {msg}"})
                return json.dumps({"error": f"GLM API Error {response.status_code}", "details": msg})
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return json.dumps({"error": "GLM Request Exception", "details": str(e)})
