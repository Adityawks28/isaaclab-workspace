import numpy as np

from insertion_tactile.oracle import Waypoint, plan_insertion


def test_starts_above_socket_no_force():
    wps = plan_insertion(np.zeros(3), approach_height=0.1, insert_depth=0.04,
                         insert_force=5.0, n_steps=20)
    assert isinstance(wps[0], Waypoint)
    assert wps[0].target_pos[2] == 0.1          # approach height above socket z=0
    assert wps[0].target_force == 0.0


def test_ends_inserted_with_force():
    wps = plan_insertion(np.zeros(3), approach_height=0.1, insert_depth=0.04,
                         insert_force=5.0, n_steps=20)
    assert np.isclose(wps[-1].target_pos[2], -0.04)   # insert_depth below socket top
    assert wps[-1].target_force == 5.0


def test_descent_is_monotonic():
    wps = plan_insertion(np.zeros(3), approach_height=0.1, insert_depth=0.04,
                         insert_force=5.0, n_steps=20)
    zs = [w.target_pos[2] for w in wps]
    assert all(zs[i] >= zs[i + 1] - 1e-9 for i in range(len(zs) - 1))


def test_force_only_during_insertion():
    wps = plan_insertion(np.zeros(3), approach_height=0.1, insert_depth=0.04,
                         insert_force=5.0, n_steps=20)
    for w in wps:
        if w.target_pos[2] < -1e-9:
            assert w.target_force == 5.0
