"""The checkpoint gate's decision: stop on a plateau of passing checkpoints, stop a run that is not
learning the correction by `fail_step`, and otherwise let it train on."""

from __future__ import annotations

from tools.fr3.train_gate import GateRules, best_step, decide

RULES = GateRules()
TOTAL = 45000


def _s(along: float, cos: float = 0.9) -> dict:
    return {"along": along, "cos": cos}


def test_a_first_passing_checkpoint_trains_on_to_see_whether_the_next_one_gains():
    action, _ = decide([(10000, _s(0.6))], TOTAL, RULES)
    assert action == "continue"


def test_two_passing_checkpoints_with_little_gain_stop_and_keep_the_better_one():
    history = [(10000, _s(0.62)), (15000, _s(0.60))]
    action, _ = decide(history, TOTAL, RULES)
    assert action == "stop_pass"
    assert best_step(history, RULES) == 10000


def test_a_plateau_one_interval_before_the_end_trains_out_to_the_annealed_checkpoint():
    action, _ = decide([(35000, _s(0.92)), (40000, _s(0.70))], TOTAL, RULES)
    assert action == "continue"


def test_a_passing_checkpoint_that_still_gains_trains_on():
    action, _ = decide([(10000, _s(0.55)), (15000, _s(0.65))], TOTAL, RULES)
    assert action == "continue"


def test_passing_after_a_failing_checkpoint_is_not_yet_a_plateau():
    action, _ = decide([(10000, _s(0.30)), (15000, _s(0.52))], TOTAL, RULES)
    assert action == "continue"


def test_moving_far_enough_in_the_wrong_direction_is_not_a_pass():
    action, _ = decide([(10000, _s(0.7, cos=0.6)), (15000, _s(0.7, cos=0.6))], TOTAL, RULES)
    assert action == "continue"


def test_a_run_still_going_straight_down_at_fail_step_is_stopped():
    history = [(10000, _s(0.07)), (15000, _s(0.09)), (20000, _s(0.10))]
    action, _ = decide(history, TOTAL, RULES)
    assert action == "stop_fail"


def test_a_slow_run_below_the_gate_but_above_the_fail_line_trains_on():
    action, _ = decide([(20000, _s(0.35))], TOTAL, RULES)
    assert action == "continue"


def test_the_last_checkpoint_ends_the_gate_without_stopping_anything():
    action, _ = decide([(40000, _s(0.1)), (45000, _s(0.1))], TOTAL, RULES)
    assert action == "done"


def test_best_falls_back_to_the_furthest_toward_the_peg_when_none_pass():
    assert best_step([(10000, _s(0.1)), (15000, _s(0.3)), (20000, _s(0.2))], RULES) == 15000
