"""Each public factory reports one start and one outcome."""
from types import SimpleNamespace

import pytest

import codesign_gym as cg


@pytest.mark.parametrize("single", [False, True])
def test_factory_status_success(monkeypatch, capsys, single):
    native = SimpleNamespace(base_embodiment="base")
    calls = []

    def build(**kwargs):
        calls.append(kwargs)
        return native

    monkeypatch.setitem(cg.REGISTRY, "Test", ("test:build", {}))
    monkeypatch.setattr(cg, "_lazy_import", lambda _: build)
    env, base = cg.make("Test") if single else cg.make_vec("Test", 3)
    factory, count = ("make", 1) if single else ("make_vec", 3)
    label = f"[codesign-gym] {factory}('Test', num_envs={count})"
    assert capsys.readouterr().out.splitlines() == [f"{label}: instantiating", f"{label}: success"]
    assert calls[0]["num_envs"] == count
    assert base == "base"
    assert (env.vector_env if single else env) is native


@pytest.mark.parametrize("single", [False, True])
@pytest.mark.parametrize("stage", ["unknown", "import", "build", "base", "wrapper"])
def test_factory_status_failure(monkeypatch, capsys, single, stage):
    if stage == "wrapper" and not single:
        pytest.skip("Vector factory does not construct SingleEnv")
    error = RuntimeError("original construction failure")

    def fail(*args, **kwargs):
        raise error

    if stage != "unknown":
        monkeypatch.setitem(cg.REGISTRY, "Test", ("test:build", {}))
    native = SimpleNamespace() if stage == "base" else SimpleNamespace(base_embodiment="base")
    monkeypatch.setattr(cg, "_lazy_import", fail if stage == "import" else
                        lambda _: fail if stage == "build" else lambda **kw: native)
    if stage == "wrapper":
        monkeypatch.setattr(cg, "SingleEnv", fail)
    with pytest.raises(Exception) as raised:
        cg.make("Test") if single else cg.make_vec("Test", 3)
    if stage in ("import", "build", "wrapper"):
        assert raised.value is error
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 2
    assert lines[0].endswith(": instantiating")
    assert lines[1].endswith(f": failed ({type(raised.value).__name__})")
    assert "success" not in lines[1]
