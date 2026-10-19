"""
Automated unit tests for Kubernetes manifests and Dockerfiles.
"""

from __future__ import annotations

import os
import yaml
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
K8S_DIR = os.path.join(PROJECT_ROOT, "k8s")


def test_dockerfiles_exist_and_valid():
    docker_api = os.path.join(PROJECT_ROOT, "Dockerfile.api")
    docker_dash = os.path.join(PROJECT_ROOT, "Dockerfile.dashboard")

    assert os.path.isfile(docker_api), "Dockerfile.api missing"
    assert os.path.isfile(docker_dash), "Dockerfile.dashboard missing"

    with open(docker_api, "r", encoding="utf-8") as f:
        content_api = f.read()
    assert "HEALTHCHECK" in content_api
    assert "EXPOSE 8000" in content_api
    assert "uvicorn" in content_api

    with open(docker_dash, "r", encoding="utf-8") as f:
        content_dash = f.read()
    assert "HEALTHCHECK" in content_dash
    assert "EXPOSE 8501" in content_dash
    assert "streamlit" in content_dash


def test_k8s_manifests_syntax_and_structure():
    expected_files = [
        "namespace.yaml",
        "configmap.yaml",
        "secrets.yaml",
        "kafka-deployment.yaml",
        "api-deployment.yaml",
        "dashboard-deployment.yaml",
        "kustomization.yaml",
    ]

    for fname in expected_files:
        path = os.path.join(K8S_DIR, fname)
        assert os.path.isfile(path), f"Missing k8s manifest: {fname}"

        with open(path, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))
            assert len(docs) >= 1, f"Empty YAML file: {fname}"

            for doc in docs:
                assert "apiVersion" in doc or "resources" in doc
                assert "kind" in doc or "resources" in doc


def test_k8s_deployments_have_probes_and_resources():
    for manifest in ["api-deployment.yaml", "dashboard-deployment.yaml"]:
        path = os.path.join(K8S_DIR, manifest)
        with open(path, "r", encoding="utf-8") as f:
            docs = list(yaml.safe_load_all(f))

        deployment_doc = next(d for d in docs if d.get("kind") == "Deployment")
        container = deployment_doc["spec"]["template"]["spec"]["containers"][0]

        assert "livenessProbe" in container, f"Missing livenessProbe in {manifest}"
        assert "resources" in container, f"Missing resources in {manifest}"
        assert "requests" in container["resources"]
        assert "limits" in container["resources"]
