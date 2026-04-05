# Multi-Node Networking Verification

**Date:** 2026-04-05
**Cluster:** panda-control (control plane) + panda-worker (agent)
**Goal:** Confirm cross-node pod scheduling and Flannel VXLAN pod networking before deploying real workloads.

---

## What This Verifies

| Test | Layer | What Passes If It Works |
|---|---|---|
| Pod scheduled on panda-worker | Scheduler + kubelet | Control plane can assign and start pods on the worker |
| ClusterIP service curl from panda-control pod | Service mesh + kube-dns | DNS resolution and kube-proxy iptables rules work cross-node |
| Raw pod IP curl from panda-control pod | Flannel VXLAN (L3) | Pod-to-pod packets route across nodes via VXLAN tunnel on port 8472/UDP |

---

## Procedure

### 1 — Deploy nginx on panda-worker

```bash
kubectl run nginx-worker \
  --image=nginx:alpine \
  --restart=Never \
  --overrides='{"spec":{"nodeName":"panda-worker"}}' \
  --port=80

kubectl wait --for=condition=Ready pod/nginx-worker --timeout=60s
kubectl get pod nginx-worker -o wide
```

Expected: pod status `Running`, `NODE` column shows `panda-worker`, pod assigned a `10.42.x.x` IP.

### 2 — Expose as ClusterIP service

```bash
kubectl expose pod nginx-worker --port=80 --name=nginx-worker-svc
```

### 3 — Test cross-node service + DNS

Schedule a curl pod on panda-control and target the service by DNS name:

```bash
kubectl run curl-control \
  --image=curlimages/curl:latest \
  --restart=Never \
  --overrides='{"spec":{"nodeName":"panda-control"}}' \
  -- curl -s --max-time 10 http://nginx-worker-svc

kubectl logs curl-control
```

Expected output: nginx welcome page HTML.

> The `kubectl wait` command will time out with an error — this is expected. The pod runs the curl command and exits immediately (Completed state). Logs are still available.

### 4 — Test raw pod-to-pod IP (Flannel VXLAN)

```bash
NGINX_POD_IP=$(kubectl get pod nginx-worker -o jsonpath='{.status.podIP}')

kubectl run curl-podip \
  --image=curlimages/curl:latest \
  --restart=Never \
  --overrides="{\"spec\":{\"nodeName\":\"panda-control\"}}" \
  -- curl -s --max-time 10 http://$NGINX_POD_IP

kubectl logs curl-podip
```

Expected: same nginx welcome page HTML, confirming direct pod-to-pod routing via Flannel VXLAN.

### 5 — Clean Up

```bash
kubectl delete pod nginx-worker curl-control curl-podip
kubectl delete svc nginx-worker-svc
```

---

## Results (2026-04-05)

| Test | Result |
|---|---|
| nginx pod scheduled on panda-worker | Pass — pod IP `10.42.1.x`, node `panda-worker` |
| ClusterIP service curl via DNS (`nginx-worker-svc`) | Pass — nginx HTML returned |
| Raw pod IP curl via Flannel VXLAN | Pass — nginx HTML returned |

Cross-node pod networking confirmed working. Flannel VXLAN (port 8472/UDP) and kube-dns are both functional across panda-control and panda-worker.

---

## Notes

- Pod IPs are in the `10.42.0.0/16` range — K3s default Flannel CIDR. panda-control pods get `10.42.0.x`, panda-worker pods get `10.42.1.x`.
- `nodeName` in the pod spec overrides the scheduler and pins the pod to a specific node — useful for testing but not how production workloads should be placed. Use `nodeSelector` or taints/tolerations for real scheduling constraints.
- UFW port `8472/UDP` must be open on both nodes for Flannel VXLAN to function. It was opened during node setup.
