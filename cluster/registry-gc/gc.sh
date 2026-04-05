#!/bin/sh
set -e

REGISTRY="http://registry.default.svc.cluster.local:5000"
KEEP_TAGS=3

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $*"; }

log "Starting registry GC"

REGISTRY_POD=$(kubectl get pods -l app=registry -o jsonpath='{.items[0].metadata.name}')
log "Registry pod: $REGISTRY_POD"

# --- Tag pruning ---
REPOS=$(curl -sf "$REGISTRY/v2/_catalog" | jq -r '.repositories // [] | .[]')

if [ -z "$REPOS" ]; then
    log "No repositories found"
else
    for REPO in $REPOS; do
        log "Processing: $REPO"

        TAGS_JSON=$(curl -sf "$REGISTRY/v2/$REPO/tags/list")
        TAGS=$(echo "$TAGS_JSON" | jq -r '.tags // [] | sort | .[]')

        TAG_COUNT=0
        if [ -n "$TAGS" ]; then
            TAG_COUNT=$(echo "$TAGS" | wc -l | tr -d ' ')
        fi

        if [ "$TAG_COUNT" -le "$KEEP_TAGS" ]; then
            log "  $TAG_COUNT tag(s) — nothing to prune (keeping $KEEP_TAGS)"
            continue
        fi

        DELETE_COUNT=$((TAG_COUNT - KEEP_TAGS))
        log "  $TAG_COUNT tags — pruning $DELETE_COUNT oldest"

        echo "$TAGS" | head -n "$DELETE_COUNT" | while read -r TAG; do
            DIGEST=$(curl -sf -I \
                -H "Accept: application/vnd.docker.distribution.manifest.v2+json" \
                "$REGISTRY/v2/$REPO/manifests/$TAG" \
                | grep -i "docker-content-digest" | tr -d '\r' | awk '{print $2}')

            if [ -n "$DIGEST" ]; then
                if curl -sf -X DELETE "$REGISTRY/v2/$REPO/manifests/$DIGEST" > /dev/null; then
                    log "  Deleted $REPO:$TAG ($DIGEST)"
                else
                    log "  [WARN] Failed to delete $REPO:$TAG"
                fi
            else
                log "  [WARN] Could not get digest for $REPO:$TAG"
            fi
        done
    done
fi

# --- Garbage collect ---
log "Running garbage-collect in $REGISTRY_POD"
kubectl exec "$REGISTRY_POD" -- registry garbage-collect \
    /etc/docker/registry/config.yml --delete-untagged=true
log "GC complete"
