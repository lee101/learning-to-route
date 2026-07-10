import numpy as np

from ltr.embed import Embedder
from ltr.router import ModelSpec, Router


def make_router(backend="hash"):
    models = [ModelSpec("cheap", 0.1, 0.3), ModelSpec("big", 5.0, 30.0)]
    return Router(models, embedder=Embedder(backend))


def test_routes_cheap_by_default_with_no_data():
    r = make_router()
    assert r.route("anything")["model"] == "cheap"


def test_learns_to_escalate():
    r = make_router()
    for _ in range(5):
        r.update("implement lock free concurrent skip list", "cheap", False, 0.001)
        r.update("implement lock free concurrent skip list", "big", True, 0.01)
        r.update("reverse a string simple function", "cheap", True, 0.001)
    assert r.route("implement a lock free concurrent queue")["model"] == "big"
    assert r.route("reverse the characters in a string")["model"] == "cheap"


def test_anchor_merging():
    r = make_router()
    a1 = r.update("write a fizzbuzz function", "cheap", True, 0.001)
    a2 = r.update("write a fizzbuzz function", "cheap", True, 0.001)
    assert a1 is a2
    assert len(r.anchors) == 1
    assert a1.stats["cheap"]["n"] == 2


def test_save_load_roundtrip(tmp_path):
    r = make_router()
    r.update("parse a csv file", "cheap", True, 0.001)
    r.update("prove p equals np", "cheap", False, 0.001)
    p = tmp_path / "router.json"
    r.save(p)
    r2 = Router.load(p, embedder=r.embedder)
    assert len(r2.anchors) == 2
    q = "parse a tsv file"
    assert r.route(q) == r2.route(q)


def test_expected_pass_prior():
    r = make_router()
    ep = r.expected_pass("novel task never seen")
    assert abs(ep["cheap"] - 0.5) < 1e-9
    assert abs(ep["big"] - 0.5) < 1e-9


def test_vectors_normalized():
    e = Embedder("hash")
    v = e.encode(["hello world"])
    assert np.allclose(np.linalg.norm(v, axis=1), 1.0)
