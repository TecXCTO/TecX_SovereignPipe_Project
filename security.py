import sys
import io
import pandas as pd
import numpy as np
import torch
from typing import Dict, Any

class CompileTimeSandbox:
    """Isolates untrusted AI-generated code strings to prevent runtime container takeovers."""
    @staticmethod
    def get_secure_env() -> Dict[str, Any]:
        sandbox: Dict[str, Any] = {
            "pd": pd, "np": np, "io": io, "torch": torch,
            "__builtins__": {
                "print": print, "len": len, "range": range, "str": str,
                "int": int, "float": float, "list": list, "dict": dict, "isinstance": isinstance,
                "Exception": Exception, "ValueError": ValueError, "KeyError": KeyError, "TypeError": TypeError
            }
        }
        # Sever operating system hook vulnerabilities inside running process threads
        if "os" in sys.modules: sandbox["os"] = None
        if "sys" in sys.modules: sandbox["sys"] = None
        return sandbox
      
