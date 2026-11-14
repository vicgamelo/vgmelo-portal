"""Giz → MLflow: login, cota, CORS só para a origem do Giz, validação do que chega e repasse ao MLflow."""

from __future__ import annotations

import base64
import io
import json
import os
import zipfile
from typing import Any

import httpx2
import pytest
from fastapi.testclient import TestClient

from .conftest import FakeNet, oauth_login

ML = "https://ml.vgmelo.tech"
ML_HEADERS = {"Origin": ML}


class FakeMLflow:
    """O suficiente da API REST do MLflow 3 para o portal: experimento, run, log-batch, artefatos, busca."""

    def __init__(self, net: FakeNet) -> None:
        self.experiments: dict[str, str] = {}
        self.runs: dict[str, dict[str, Any]] = {}
        self.files: dict[str, bytes] = {}
        self.fail_upload = False
        base = "/api/2.0/mlflow"
        r = net.routes
        r[("mlflow", f"{base}/experiments/get-by-name")] = self.get_by_name
        r[("mlflow", f"{base}/experiments/create")] = self.create_exp
        r[("mlflow", f"{base}/runs/create")] = self.create_run
        r[("mlflow", f"{base}/runs/log-batch")] = self.log_batch
        r[("mlflow", f"{base}/runs/update")] = self.update
        r[("mlflow", f"{base}/runs/search")] = self.search
        net.prefixes.append(("mlflow", "/api/2.0/mlflow-artifacts/artifacts/", self.put))

    def get_by_name(self, req: httpx2.Request) -> httpx2.Response:
        name = req.url.params["experiment_name"]
        if name not in self.experiments:
            return httpx2.Response(404, json={"error_code": "RESOURCE_DOES_NOT_EXIST"})
        return httpx2.Response(200, json={"experiment": {"experiment_id": self.experiments[name], "name": name}})

    def create_exp(self, req: httpx2.Request) -> httpx2.Response:
        name = json.loads(req.content)["name"]
        self.experiments[name] = str(len(self.experiments) + 1)
        return httpx2.Response(200, json={"experiment_id": self.experiments[name]})

    def create_run(self, req: httpx2.Request) -> httpx2.Response:
        body = json.loads(req.content)
        run_id = f"r{len(self.runs) + 1:031d}"
        info = {
            "run_id": run_id,
            "experiment_id": body["experiment_id"],
            "run_name": body["run_name"],
            "status": "RUNNING",
            "start_time": body["start_time"],
            "artifact_uri": f"mlflow-artifacts:/{body['experiment_id']}/{run_id}/artifacts",
        }
        tags = {t["key"]: t["value"] for t in body["tags"]}
        self.runs[run_id] = {"info": info, "tags": tags, "params": {}, "metrics": {}}
        return httpx2.Response(200, json={"run": {"info": info, "data": {}}})

    def log_batch(self, req: httpx2.Request) -> httpx2.Response:
        body = json.loads(req.content)
        run = self.runs[body["run_id"]]
        run["params"].update({p["key"]: p["value"] for p in body.get("params", [])})
        run["metrics"].update({m["key"]: m["value"] for m in body.get("metrics", [])})
        return httpx2.Response(200, json={})

    def update(self, req: httpx2.Request) -> httpx2.Response:
        body = json.loads(req.content)
        self.runs[body["run_id"]]["info"]["status"] = body["status"]
        return httpx2.Response(200, json={})

    def put(self, req: httpx2.Request) -> httpx2.Response:
        if self.fail_upload:
            return httpx2.Response(500, json={"error_code": "INTERNAL_ERROR"})
        self.files[req.url.path.removeprefix("/api/2.0/mlflow-artifacts/artifacts/")] = req.content
        return httpx2.Response(200, json={})

    def search(self, req: httpx2.Request) -> httpx2.Response:
        body = json.loads(req.content)
        user = body["filter"].split("= '")[1].rstrip("'")
        runs = [
            {
                "info": r["info"],
                "data": {
                    "metrics": [{"key": k, "value": v} for k, v in r["metrics"].items()],
                    "params": [{"key": k, "value": v} for k, v in r["params"].items()],
                },
            }
            for r in self.runs.values()
            if r["tags"].get("giz.user") == user
        ]
        return httpx2.Response(200, json={"runs": runs})


@pytest.fixture
def overrides() -> dict[str, Any]:
    return {"mlflow_url": "http://mlflow:5000", "ml_origin": ML, "quota_ml": 3}


@pytest.fixture
def mlflow(net: FakeNet) -> FakeMLflow:
    return FakeMLflow(net)


def skops_zip() -> str:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("schema.json", '{"__class__": "Pipeline"}')
    return base64.b64encode(buf.getvalue()).decode()


def body(**over: Any) -> dict[str, Any]:
    b: dict[str, Any] = {
        "run_name": "credito.csv · LightGBM",
        "params": {"target": "inadimplente", "model": "LightGBM", "model.n_estimators": "300"},
        "metrics": {"holdout_roc_auc": 0.79, "cv_roc_auc": 0.77},
        "tags": {"giz.problem": "binary"},
        "artifacts": {"receita.json": '{"steps": []}', "giz_modelo.py": "print('oi')\n"},
        "model": None,
    }
    b.update(over)
    return b


def logged(client: TestClient) -> str:
    oauth_login(client)
    return client.get("/api/me").json()["user"]["id"]


def test_precisa_de_login(client: TestClient, mlflow: FakeMLflow) -> None:
    assert client.post("/api/ml/runs", json=body(), headers=ML_HEADERS).status_code == 401
    assert mlflow.runs == {}


@pytest.mark.parametrize("overrides", [{"mlflow_url": "", "ml_origin": ML}])
def test_mlflow_desligado(client: TestClient, overrides: dict[str, Any]) -> None:
    logged(client)
    assert client.post("/api/ml/runs", json=body(), headers=ML_HEADERS).json() == {"detail": "mlflow_off"}


def test_registra_parametros_metricas_tags_e_artefatos(client: TestClient, mlflow: FakeMLflow) -> None:
    user_id = logged(client)
    spoof = body(tags={"giz.problem": "binary", "giz.user": "outra-pessoa"})
    resp = client.post("/api/ml/runs", json=spoof, headers=ML_HEADERS)
    assert resp.status_code == 201, resp.text
    assert resp.headers["access-control-allow-origin"] == ML
    assert resp.headers["access-control-allow-credentials"] == "true"
    out = resp.json()
    run = mlflow.runs[out["run_id"]]
    assert run["info"]["status"] == "FINISHED"
    assert run["tags"]["giz.user"] == user_id  # quem diz o autor é o portal
    assert run["tags"]["mlflow.user"] == f"portal:{user_id}"
    assert run["params"]["model.n_estimators"] == "300"
    assert run["metrics"]["holdout_roc_auc"] == 0.79
    exp = mlflow.experiments["giz"]
    assert mlflow.files[f"{exp}/{out['run_id']}/artifacts/giz_modelo.py"] == b"print('oi')\n"
    assert out["artifacts"] == ["receita.json", "giz_modelo.py"]
    me = client.get("/api/me").json()
    assert me["quotas"]["ml"] == {"limit": 3, "used": 1, "remaining": 2}


def test_modelo_skops_vira_artefato_e_pickle_e_recusado(client: TestClient, mlflow: FakeMLflow) -> None:
    logged(client)
    model = {"file": "modelo.skops", "b64": skops_zip(), "requirements": "scikit-learn==1.8.0\n", "signature": "{}"}
    resp = client.post("/api/ml/runs", json=body(model=model), headers=ML_HEADERS)
    assert resp.status_code == 201, resp.text
    run_id = resp.json()["run_id"]
    assert any(k.endswith(f"{run_id}/artifacts/model/modelo.skops") for k in mlflow.files)
    assert "model/requirements.txt" in resp.json()["artifacts"]

    pickle = base64.b64encode(b"\x80\x04\x95cos\nsystem\n").decode()
    bad = client.post("/api/ml/runs", json=body(model={**model, "b64": pickle}), headers=ML_HEADERS)
    assert bad.status_code == 422 and bad.json()["detail"] == "model_not_skops"
    assert client.post("/api/ml/runs", json=body(model={**model, "b64": "@@@"}), headers=ML_HEADERS).status_code == 422
    assert client.get("/api/me").json()["quotas"]["ml"]["used"] == 1  # recusado não gasta cota


def test_falha_no_mlflow_marca_run_e_devolve_cota(client: TestClient, mlflow: FakeMLflow) -> None:
    logged(client)
    mlflow.fail_upload = True
    resp = client.post("/api/ml/runs", json=body(), headers=ML_HEADERS)
    assert resp.status_code == 502
    (run,) = mlflow.runs.values()
    assert run["info"]["status"] == "FAILED"
    assert client.get("/api/me").json()["quotas"]["ml"]["used"] == 0


def test_cota_diaria(client: TestClient, mlflow: FakeMLflow) -> None:
    logged(client)
    for _ in range(3):
        assert client.post("/api/ml/runs", json=body(), headers=ML_HEADERS).status_code == 201
    assert client.post("/api/ml/runs", json=body(), headers=ML_HEADERS).status_code == 429


@pytest.mark.parametrize(
    "bad",
    [
        {"artifacts": {"dados.csv": "a,b\n1,2"}},
        {"params": {"chave com espaço": "x"}},
        {"params": {"p": "x" * 501}},
        {"tags": {"mlflow.user": "eu"}},
        {"extra": 1},
    ],
)
def test_validacao(client: TestClient, mlflow: FakeMLflow, bad: dict[str, Any]) -> None:
    logged(client)
    resp = client.post(
        "/api/ml/runs",
        content=json.dumps(body(**bad), allow_nan=True),
        headers={**ML_HEADERS, "content-type": "application/json"},
    )
    assert resp.status_code == 422, resp.text
    assert mlflow.runs == {}


def test_metrica_nao_finita_e_descartada_sem_quebrar(client: TestClient, mlflow: FakeMLflow) -> None:
    logged(client)
    raw = json.dumps(body(metrics={"auc": float("nan"), "r2": float("inf"), "ok": 0.5}), allow_nan=True)
    resp = client.post("/api/ml/runs", content=raw, headers={**ML_HEADERS, "content-type": "application/json"})
    assert resp.status_code == 201, resp.text
    (run,) = mlflow.runs.values()
    assert run["metrics"] == {"ok": 0.5}


def test_corpo_grande_passa_ate_o_limite_da_rota(client: TestClient, mlflow: FakeMLflow) -> None:
    logged(client)
    big = body(artifacts={"giz_modelo.py": "#" * 300_000})  # bem acima dos 16 KB das outras rotas
    assert client.post("/api/ml/runs", json=big, headers=ML_HEADERS).status_code == 201
    huge = "x" * (6 * 1024 * 1024 + 10)
    resp = client.post("/api/ml/runs", content=huge, headers={**ML_HEADERS, "content-type": "application/json"})
    assert resp.status_code == 413


def test_cors_so_para_a_giz_e_so_nas_rotas_dela(client: TestClient, mlflow: FakeMLflow) -> None:
    pre = client.options(
        "/api/ml/runs",
        headers={
            "Origin": ML,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert pre.status_code == 204
    assert pre.headers["access-control-allow-origin"] == ML
    assert "POST" in pre.headers["access-control-allow-methods"]
    me = client.get("/api/me", headers=ML_HEADERS)
    assert me.headers["access-control-allow-origin"] == ML

    evil = {"Origin": "https://evil.example"}
    assert "access-control-allow-origin" not in client.options("/api/ml/runs", headers=evil).headers
    logged(client)
    assert client.post("/api/ml/runs", json=body(), headers=evil).json() == {"detail": "bad_origin"}
    # o Giz não ganha acesso às outras rotas do portal
    assert client.post("/api/auth/logout", headers=ML_HEADERS).json() == {"detail": "bad_origin"}
    assert "access-control-allow-origin" not in client.get("/api/health", headers=ML_HEADERS).headers


def test_lista_so_os_runs_da_pessoa(client: TestClient, app: Any, mlflow: FakeMLflow, net: FakeNet) -> None:
    logged(client)
    assert client.post("/api/ml/runs", json=body(), headers=ML_HEADERS).status_code == 201
    from .conftest import make_client

    with make_client(app, ip="198.51.100.9") as other:
        net.google_user = {"sub": "g-2", "email": "bia@gmail.com", "email_verified": True, "name": "Bia"}
        oauth_login(other)
        assert other.get("/api/ml/runs", headers=ML_HEADERS).json() == {"runs": []}
    runs = client.get("/api/ml/runs", headers=ML_HEADERS).json()["runs"]
    assert len(runs) == 1 and runs[0]["metrics"]["holdout_roc_auc"] == 0.79


@pytest.mark.skipif(not os.getenv("MLFLOW_TEST_URL"), reason="defina MLFLOW_TEST_URL para testar num MLflow de verdade")
def test_contra_um_mlflow_de_verdade(app: Any, settings: Any, net: FakeNet) -> None:
    """Mesmo fluxo, falando com um MLflow real (docker ghcr.io/mlflow/mlflow:v3.18.0 com --serve-artifacts)."""
    import asyncio

    from vgmelo_portal.ml import MLflow

    async def go() -> dict[str, Any]:
        async with httpx2.AsyncClient() as http:
            m = MLflow(http, os.environ["MLFLOW_TEST_URL"])
            exp = await m.experiment_id("giz-teste")
            assert exp == await m.experiment_id("giz-teste")  # segunda vez acha o mesmo
            info = await m.create_run(exp, "teste", {"giz.user": "u1", "mlflow.source.name": "giz"})
            await m.log_batch(info["run_id"], {"target": "y"}, {"holdout_roc_auc": 0.8})
            await m.upload(info["artifact_uri"], "model/modelo.skops", base64.b64decode(skops_zip()), "application/zip")
            await m.finish(info["run_id"], "FINISHED")
            runs = await m.search(exp, "u1", 5)
            got = await http.get(
                f"{os.environ['MLFLOW_TEST_URL']}/api/2.0/mlflow-artifacts/artifacts/"
                f"{info['artifact_uri'].removeprefix('mlflow-artifacts:/')}/model/modelo.skops"
            )
            return {"runs": runs, "run_id": info["run_id"], "file": got.content}

    out = asyncio.run(go())
    assert any(r["info"]["run_id"] == out["run_id"] for r in out["runs"])
    assert out["file"].startswith(b"PK\x03\x04")
