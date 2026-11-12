"""Giz (estúdio de ML no navegador) → MLflow privado.

O treino acontece no navegador de quem usa. Quando a pessoa logada pede, o Giz manda só o resumo do experimento
(parâmetros, métricas, receita, código exportado e, se ela marcar, o modelo em skops) e o portal repassa ao MLflow
pela rede interna. O navegador nunca fala com o MLflow nem segura credencial dele, e o portal nunca abre o modelo:
só confere que é um zip pequeno e grava como artefato.
"""

from __future__ import annotations

import base64
import binascii
import contextlib
import logging
import math
import time
from typing import Annotated, Any, Literal

import httpx2
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, StringConstraints

from .context import Ctx, LoggedIn
from .demos import refund, reserve, upstream_error

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/ml")

Key = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_.\-]{1,120}$")]
TagKey = Annotated[str, StringConstraints(pattern=r"^giz\.[a-z0-9_.]{1,60}$")]
ArtifactName = Literal["receita.json", "giz_modelo.py", "giz_modelo.ipynb", "ranking.json", "avaliacao.json"]

ARTIFACT_MAX = 400_000  # bytes de texto por artefato
MODEL_MAX = 4 * 1024 * 1024  # bytes do modelo depois do base64
ZIP_MAGIC = b"PK\x03\x04"


class ModelArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")
    file: Literal["modelo.skops"]
    b64: str = Field(max_length=(MODEL_MAX * 4) // 3 + 8)
    requirements: str = Field(max_length=2000)
    signature: str = Field(max_length=50_000)

    def data(self) -> bytes:
        try:
            raw = base64.b64decode(self.b64, validate=True)
        except binascii.Error as exc:
            raise HTTPException(422, "model_not_base64") from exc
        if len(raw) > MODEL_MAX:
            raise HTTPException(413, "model_too_large")
        # skops grava um zip; qualquer outra coisa (um pickle, por exemplo) é recusada.
        if not raw.startswith(ZIP_MAGIC):
            raise HTTPException(422, "model_not_skops")
        return raw


class MLRun(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_name: str = Field(min_length=1, max_length=120)
    params: dict[Key, Annotated[str, Field(max_length=500)]] = Field(max_length=100)
    metrics: dict[Key, float] = Field(max_length=100)
    tags: dict[TagKey, Annotated[str, Field(max_length=5000)]] = Field(default_factory=dict, max_length=20)
    artifacts: dict[ArtifactName, Annotated[str, Field(max_length=ARTIFACT_MAX)]] = Field(max_length=5)
    model: ModelArtifact | None = None


class MLflowError(Exception):
    pass


class MLflow:
    """O mínimo da API REST do MLflow 3 (tracking + proxy de artefatos do `--serve-artifacts`)."""

    def __init__(self, http: httpx2.AsyncClient, base: str) -> None:
        self.http, self.base = http, base.rstrip("/")

    async def _call(self, method: str, path: str, **kw: Any) -> dict[str, Any]:
        try:
            resp = await self.http.request(method, f"{self.base}{path}", timeout=20, **kw)
        except httpx2.HTTPError as exc:
            raise MLflowError(f"{path}: {type(exc).__name__}") from exc
        if resp.status_code >= 400:
            raise MLflowError(f"{path}: HTTP {resp.status_code} {resp.text[:200]}")
        return resp.json() if resp.content else {}

    async def experiment_id(self, name: str) -> str:
        try:
            found = await self._call("GET", "/api/2.0/mlflow/experiments/get-by-name", params={"experiment_name": name})
            return str(found["experiment"]["experiment_id"])
        except MLflowError as exc:
            if "RESOURCE_DOES_NOT_EXIST" not in str(exc):
                raise
        try:
            made = await self._call("POST", "/api/2.0/mlflow/experiments/create", json={"name": name})
            return str(made["experiment_id"])
        except MLflowError as exc:  # outro pedido criou ao mesmo tempo
            if "RESOURCE_ALREADY_EXISTS" not in str(exc):
                raise
            found = await self._call("GET", "/api/2.0/mlflow/experiments/get-by-name", params={"experiment_name": name})
            return str(found["experiment"]["experiment_id"])

    async def create_run(self, experiment_id: str, name: str, tags: dict[str, str]) -> dict[str, Any]:
        body = {
            "experiment_id": experiment_id,
            "run_name": name,
            "start_time": int(time.time() * 1000),
            "tags": [{"key": k, "value": v} for k, v in tags.items()],
        }
        info: dict[str, Any] = (await self._call("POST", "/api/2.0/mlflow/runs/create", json=body))["run"]["info"]
        return info

    async def log_batch(self, run_id: str, params: dict[str, str], metrics: dict[str, float]) -> None:
        now = int(time.time() * 1000)
        await self._call(
            "POST",
            "/api/2.0/mlflow/runs/log-batch",
            json={
                "run_id": run_id,
                "params": [{"key": k, "value": v} for k, v in params.items()],
                "metrics": [{"key": k, "value": v, "timestamp": now, "step": 0} for k, v in metrics.items()],
            },
        )

    async def upload(self, artifact_uri: str, path: str, data: bytes, content_type: str) -> None:
        prefix = "mlflow-artifacts:/"
        if not artifact_uri.startswith(prefix):
            raise MLflowError(f"artifact_uri inesperado: {artifact_uri}")
        where = artifact_uri.removeprefix(prefix).strip("/")
        await self._call(
            "PUT",
            f"/api/2.0/mlflow-artifacts/artifacts/{where}/{path}",
            content=data,
            headers={"Content-Type": content_type},
        )

    async def finish(self, run_id: str, status: str) -> None:
        await self._call(
            "POST",
            "/api/2.0/mlflow/runs/update",
            json={"run_id": run_id, "status": status, "end_time": int(time.time() * 1000)},
        )

    async def search(self, experiment_id: str, user_id: str, limit: int) -> list[dict[str, Any]]:
        body = {
            "experiment_ids": [experiment_id],
            "filter": f"tags.`giz.user` = '{user_id}'",
            "max_results": limit,
            "order_by": ["attributes.start_time DESC"],
        }
        runs: list[dict[str, Any]] = (await self._call("POST", "/api/2.0/mlflow/runs/search", json=body)).get(
            "runs", []
        )
        return runs


CONTENT_TYPES = {".json": "application/json", ".py": "text/x-python", ".ipynb": "application/x-ipynb+json"}


def _ctype(name: str) -> str:
    return next((v for k, v in CONTENT_TYPES.items() if name.endswith(k)), "text/plain") + "; charset=utf-8"


@router.post("/runs", status_code=201)
async def log_run(body: MLRun, current: LoggedIn, ctx: Ctx) -> dict[str, Any]:
    s = ctx.settings
    if not s.mlflow_url:
        raise HTTPException(503, "mlflow_off")
    model = body.model.data() if body.model else None  # valida antes de gastar cota
    user_id = current.user.id
    day = await reserve(ctx, user_id, "ml")
    mlflow = MLflow(ctx.ahttp, s.mlflow_url)
    run_id: str | None = None
    try:
        exp = await mlflow.experiment_id(s.mlflow_experiment)
        # As tags de autoria vêm do portal, não do navegador.
        tags = {
            **body.tags,
            "giz.user": user_id,
            "mlflow.user": f"portal:{user_id}",
            "mlflow.source.name": "giz",
            "mlflow.source.type": "LOCAL",
        }
        info = await mlflow.create_run(exp, body.run_name, tags)
        run_id = info["run_id"]
        # NaN/inf não viram erro 422: o handler padrão do FastAPI ecoa o valor e quebra ao serializar NaN.
        metrics = {k: v for k, v in body.metrics.items() if math.isfinite(v)}
        await mlflow.log_batch(run_id, body.params, metrics)
        uri = info["artifact_uri"]
        for name, text in body.artifacts.items():
            await mlflow.upload(uri, name, text.encode(), _ctype(name))
        uploaded: list[str] = list(body.artifacts)
        if body.model and model is not None:
            await mlflow.upload(uri, "model/modelo.skops", model, "application/zip")
            await mlflow.upload(uri, "model/requirements.txt", body.model.requirements.encode(), _ctype(".txt"))
            await mlflow.upload(uri, "model/assinatura.json", body.model.signature.encode(), _ctype(".json"))
            uploaded += ["model/modelo.skops", "model/requirements.txt", "model/assinatura.json"]
        await mlflow.finish(run_id, "FINISHED")
    except MLflowError as exc:
        logger.warning("mlflow log_run falhou (run %s): %s", run_id, exc)
        if run_id:
            with contextlib.suppress(MLflowError):
                await mlflow.finish(run_id, "FAILED")
        await refund(ctx, user_id, day, "ml")
        raise upstream_error() from exc
    return {"run_id": run_id, "experiment_id": exp, "experiment": s.mlflow_experiment, "artifacts": uploaded}


@router.get("/runs")
async def my_runs(current: LoggedIn, ctx: Ctx) -> dict[str, Any]:
    s = ctx.settings
    if not s.mlflow_url:
        raise HTTPException(503, "mlflow_off")
    mlflow = MLflow(ctx.ahttp, s.mlflow_url)
    try:
        exp = await mlflow.experiment_id(s.mlflow_experiment)
        runs = await mlflow.search(exp, current.user.id, 20)
    except MLflowError as exc:
        logger.warning("mlflow search falhou: %s", exc)
        raise upstream_error() from exc
    out = []
    for r in runs:
        info, data = r["info"], r.get("data", {})
        metrics = {m["key"]: m["value"] for m in data.get("metrics", [])}
        params = {p["key"]: p["value"] for p in data.get("params", [])}
        out.append(
            {
                "run_id": info["run_id"],
                "name": info.get("run_name"),
                "status": info.get("status"),
                "start_time": info.get("start_time"),
                "model": params.get("model"),
                "target": params.get("target"),
                "metrics": {k: v for k, v in metrics.items() if k.startswith(("holdout_", "cv_"))},
            }
        )
    return {"runs": out}
