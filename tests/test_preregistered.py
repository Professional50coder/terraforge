from terraforge.training.preregistered import sacp_verdict


def _method(under, size, eff, cov=0.9):
    return {"worst_undercoverage": under, "mean_size": size, "mean_size_at_target_coverage": eff,
            "mean_coverage": cov}


def _report(sacp, novelty, entropy, ci_hi, raps=None):
    summary = {"sacp": sacp, "cond_novelty": novelty, "cond_entropy": entropy,
               "raps": raps or _method(0.05, 2.9, 2.9, 0.95)}
    vs = {b: {"size_ci": [ci_hi - 0.1, ci_hi]} for b in ("cond_novelty", "cond_entropy")}
    return {"summary": summary, "held_out": {"haze": {"0.25": {"sacp": {"vs": vs}}}}}


def test_strictly_better_method_is_supported():
    r = _report(_method(0.01, 1.5, 1.4), _method(0.03, 1.6, 1.6), _method(0.03, 1.6, 1.7), ci_hi=-0.01)
    assert sacp_verdict(r)["verdict"] == "SUPPORTED"


def test_tie_with_entropy_is_not_supported():
    # the outcome the simulation predicts: same coverage and size, no CI excluding zero
    r = _report(_method(0.03, 1.6, 1.6), _method(0.03, 1.6, 1.6), _method(0.03, 1.6, 1.6), ci_hi=0.05)
    v = sacp_verdict(r)
    assert v["verdict"] == "NOT SUPPORTED"
    assert any(x["criterion"] == 3 and not x["passed"] for x in v["rules"])


def test_a_spoiler_that_matches_at_equal_size_voids_support():
    r = _report(_method(0.01, 1.5, 1.4), _method(0.03, 1.6, 1.6), _method(0.03, 1.6, 1.7), ci_hi=-0.01,
                raps=_method(0.0, 1.4, 1.4, cov=0.95))
    assert sacp_verdict(r)["verdict"] == "NOT SUPPORTED"


def test_oversized_sets_fail_criterion_two():
    r = _report(_method(0.01, 2.0, 1.4), _method(0.03, 1.6, 1.6), _method(0.03, 1.6, 1.7), ci_hi=-0.01)
    assert sacp_verdict(r)["verdict"] == "NOT SUPPORTED"
