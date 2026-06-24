import numpy as np
import pytest

from insertion_tactile.recorder import assemble_frame


def _frame():
    return assemble_frame(
        rgb_wrist=np.zeros((224, 224, 3), np.uint8),
        rgb_scene=np.zeros((224, 224, 3), np.uint8),
        tactile_force=np.zeros(6, np.float32),
        proprio=np.zeros(14, np.float32),
        action=np.zeros(7, np.float32),
        language="insert the connector into the socket",
    )


def test_frame_has_contract_keys():
    f = _frame()
    for k in ("observation.images.wrist", "observation.images.scene",
              "observation.state", "observation.tactile", "action", "task"):
        assert k in f


def test_action_length_validated():
    with pytest.raises(ValueError):
        assemble_frame(np.zeros((1, 1, 3), np.uint8), np.zeros((1, 1, 3), np.uint8),
                       np.zeros(6, np.float32), np.zeros(14, np.float32),
                       np.zeros(5, np.float32), "x")   # action len 5 != 7
