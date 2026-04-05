# Bare Metal MLOps Sandbox

![Status](https://img.shields.io/badge/Status-In%20Progress-yellow)
![Phase](https://img.shields.io/badge/Phase-1%20Foundation-blue)
![OS](https://img.shields.io/badge/OS-Ubuntu%2024.04-orange?logo=ubuntu&logoColor=white)
![K3s](https://img.shields.io/badge/Orchestration-K3s-326CE5?logo=kubernetes&logoColor=white)
[![DeepX DX-M1](https://img.shields.io/badge/NPU-DeepX%20DX--M1%2025%20TOPS-blue)](https://developer.deepx.ai)
[![LattePanda](https://img.shields.io/badge/SBC-LattePanda%203%20Delta-red?logo=intel&logoColor=white)](https://www.lattepanda.com/lattepanda-3-delta)

## Project Overview

The Bare Metal MLOps Sandbox is a high-fidelity engineering environment designed to bridge the gap between theoretical machine learning and real-world deployment. While many developers rely on abstracted cloud services, this project focuses on the "metal" — the physical hardware and low-level configurations required to run AI reliably at the edge. By building a custom, multi-node cluster using LattePanda 3 Delta hardware running Ubuntu, the system replicates the complex, distributed environments used in industrial-grade pipelines but on a localized, highly efficient scale.

A core focus of the sandbox is mastering the integration of specialized hardware with modern automation. A DeepX DX-M1 AI accelerator handles intensive computer vision and NLP tasks, utilizing a custom-engineered thermal management solution to ensure stable performance under heavy workloads. This hands-on approach to hardware optimization ensures that the AI isn't just functional, but remains resilient and performant in constrained physical environments.

The entire ecosystem is orchestrated using Kubernetes (K3s), which manages the hardware nodes as a unified resource pool. Automated CI/CD pipelines trigger model quantization and containerization directly on the control plane when code changes land on GitHub. Images are stored in a private local Docker registry and deployed to the cluster with zero downtime. A Prometheus and Grafana stack provides real-time telemetry, tracking everything from NPU temperature to model inference latency.

---

## Cluster Architecture

### Hardware Configuration

| Node | Role | Storage | Key Hardware |
|---|---|---|---|
| `panda-control` | Control Plane | 512 GB | DeepX DX-M1 AI Module (25 TOPS NPU) |
| `panda-worker` | Worker | 256 GB | Intel N5105 (LattePanda 3 Delta) |

### Control Plane (`panda-control`)

- **Kubernetes API server** — accepts and validates cluster commands
- **Scheduler** — decides where workloads run across the cluster
- **Controller Manager** — maintains desired cluster state
- **etcd** — stores all cluster configuration and state
- **Docker Registry** — serves container images to both nodes
- **DX-M1 Inference Module** — runs hardware-accelerated inference workloads
- **Prometheus / Grafana** — collects and visualizes cluster and model metrics

### Worker (`panda-worker`)

- **Kubelet** — receives instructions from control plane, manages containers
- **Container Runtime** — executes containers on the node
- Pulls images from the Node 1 registry on demand
- Runs orchestration/monitoring pods if scheduled here
- Reports status back to the control plane

---

## Stack

- **Compute:** Intel Celeron N5105 (LattePanda 3 Delta) × 2 nodes
- **Acceleration:** DeepX DX-M1 (25 TOPS, 4 GB LPDDR5) for CV and NLP inference
- **Orchestration:** K3s (Lightweight Kubernetes)
- **CI/CD:** GitHub Actions with self-hosted runners
- **Registry:** Private local Docker registry on control plane
- **Observability:** Prometheus / Grafana — NPU telemetry, inference latency, cluster health
- **Networking:** Tailscale (secure overlay)
- **OS:** Ubuntu 24.04 LTS

---

## 6-Month Roadmap

### Phase 1 — Foundation (Months 1–2) — In Progress

- OS provisioning and hardening on both nodes
- K3s cluster initialization and network overlay configuration
- CI/CD handshake between GitHub Actions and the local edge environment

### Phase 2 — Acceleration & Serving (Months 3–4) — Upcoming

- Integrating the DeepX DX-M1 drivers into the K8s runtime
- Containerizing ML models (TensorFlow/PyTorch) optimized for NPU execution
- Implementing a private model registry and versioning system

### Phase 3 — Observability & Scale (Months 5–6) — Upcoming

- Deploying Prometheus/Grafana for real-time hardware and model telemetry
- Simulating high-concurrency loads and benchmarking CPU vs. NPU performance
- Secure tunneling for live demonstrations and remote access
