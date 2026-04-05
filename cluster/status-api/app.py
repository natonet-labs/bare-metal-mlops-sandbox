from fastapi import FastAPI, HTTPException
from datetime import datetime, timezone
import httpx

PROMETHEUS_URL = "http://10.43.x.x:9090"  # Prometheus ClusterIP — run: kubectl get svc -n monitoring | grep prometheus

NODE_NAMES = {
    "192.168.x.x:9100": "panda-control",  # replace with your node IPs
    "192.168.x.x:9100": "panda-worker",
}

app = FastAPI()


async def query(client: httpx.AsyncClient, promql: str) -> dict:
    r = await client.get(f"{PROMETHEUS_URL}/api/v1/query", params={"query": promql})
    r.raise_for_status()
    return {
        item["metric"]["instance"]: float(item["value"][1])
        for item in r.json()["data"]["result"]
    }


@app.get("/status")
async def status():
    async with httpx.AsyncClient(timeout=5) as client:
        try:
            cpu_data = await query(
                client,
                '100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle",job="node-exporter"}[2m])) * 100)',
            )
            temp_data = await query(
                client,
                'node_hwmon_temp_celsius{chip="platform_coretemp_0",sensor="temp1",job="node-exporter"}',
            )
            load_data = await query(client, 'node_load1{job="node-exporter"}')
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Prometheus unreachable: {e}")

    nodes = []
    for instance, name in NODE_NAMES.items():
        nodes.append({
            "name": name,
            "cpu_pct": round(cpu_data.get(instance, 0), 1),
            "temp_c": round(temp_data.get(instance, 0), 1),
            "load1": round(load_data.get(instance, 0), 2),
        })

    return {
        "nodes": nodes,
        "updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
