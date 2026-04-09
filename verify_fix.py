
import sys
from unittest.mock import MagicMock

# Mock FastAPI because it's missing in the environment
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.middleware.cors"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()

import BITGOT_ETAP4_main
from BITGOT_ETAP1_foundation import BITGOTConfig

def test_build_api_uses_config_origins():
    # 1. Setup mocks
    mock_orchestrator = MagicMock()
    mock_orchestrator.cfg = BITGOTConfig()
    mock_orchestrator.cfg.allowed_origins = ["http://trusted.com"]
    mock_metrics = MagicMock()

    # 2. Call the function
    BITGOT_ETAP4_main.build_api(mock_orchestrator, mock_metrics)

    # 3. Verify that CORSMiddleware was added with correct origins
    # We need to find the add_middleware call on the FastAPI app
    app = BITGOT_ETAP4_main.FastAPI.return_value

    # Check if add_middleware was called
    called = False
    for call in app.add_middleware.call_args_list:
        args, kwargs = call
        if kwargs.get("allow_origins") == ["http://trusted.com"]:
            called = True
            break

    if called:
        print("SUCCESS: CORSMiddleware called with correct allowed_origins")
    else:
        print("FAILURE: CORSMiddleware NOT called with correct allowed_origins")
        # Print what it WAS called with
        for i, call in enumerate(app.add_middleware.call_args_list):
            print(f"Call {i} kwargs: {call[1]}")

if __name__ == "__main__":
    try:
        test_build_api_uses_config_origins()
    except Exception as e:
        print(f"ERROR during verification: {e}")
        import traceback
        traceback.print_exc()
