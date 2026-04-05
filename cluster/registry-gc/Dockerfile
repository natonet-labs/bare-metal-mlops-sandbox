FROM alpine:3.19
RUN apk add --no-cache curl jq
COPY kubectl /usr/local/bin/kubectl
COPY gc.sh /gc.sh
RUN chmod +x /usr/local/bin/kubectl /gc.sh
ENTRYPOINT ["/gc.sh"]
