
# Summary

## Configure HAProxy Ingress Controller so that both HTTP1.1 Ingress and HTTP2 Ingress and backend servers capable of speaking both HTTP1.1 and HTTP2 get serveed


https://www.haproxy.com/blog/haproxy-1-9-2-adds-grpc-support/

After the release of version 1.8, users of HAProxy could already see performance gains simply by switching on HTTP/2 in a frontend. However, protocols like gRPC require that HTTP/2 be used for the backend services as well. The open-source community and engineers at HAProxy Technologies got to work on the problem.

https://haproxy-ingress.github.io/docs/configuration/keys/#backend-protocol

Defines the HTTP protocol version of the backend. Note that HTTP/2 is only supported if HTX is enabled. A case insensitive match is used, so either h1 or H1 configures HTTP/1 protocol. A non SSL/TLS configuration does not overrides secure-backends, so h1 and secure-backends true will still configures SSL/TLS.
Options:
h1: the default value, configures HTTP/1 protocol. http is an alias to h1.
h1-ssl: configures HTTP/1 over SSL/TLS. https is an alias to h1-ssl.
h2: configures HTTP/2 protocol. grpc is an alias to h2.
h2-ssl: configures HTTP/2 over SSL/TLS. grpcs is an alias to h2-ssl.



## Configure HA Proxy with HTTP2 (H2) , UseHTX and ALPN with TLS

kubectl -n ingress-controller edit configmap haproxy-ingress

```

HAProxy Ingress Controller

# Please edit the object below. Lines beginning with a '#' will be ignored,
# and an empty file will abort the edit. If an error occurs while saving this file will be
# reopened with the relevant failures.
#
apiVersion: v1
data:
  backend-protocol: h2
  ssl-certificate: secret/test-cert
  tls-alpn: h2,http/1.1
  use-htx: "true"
kind: ConfigMap
metadata:
  creationTimestamp: "2020-08-18T10:59:12Z"
  name: haproxy-ingress
  namespace: ingress-controller
  resourceVersion: "58178062"
  selfLink: /api/v1/namespaces/ingress-controller/configmaps/haproxy-ingress
  uid: e71c2983-e5b1-4cd2-9d41-6a152a8d51c7
```



## Create Deployment and Service

Create a deployment and a service for Nginx sample (with HTTPP 1.1)

```
kubectl create deployment nginx2 --image nginx:alpine -n test
[root@green--1 ~]#  kubectl -n test expose deployment nginx2 --port=80
service/nginx2 exposed

```

Do the same for a GRPC Server (which speaks HTTP2)

```
kubectl create deployment sample-grpc  --image=alexcpn/sample-grpc-go:1.0  -n test
kubectl -n test expose deployment sample-grpc --port=50051
```
Let's use Kube proxy and test if Server is fine and Client is proper

```
In server

kubectl port-forward service/sample-grpc --address 0.0.0.0 50051:50051 --namespace test

In Client

use address = "sample-grpc.10.131.232.223.nip.io:50051"
root@docker-desktop:/go/microservice/aa_sample_service_go/test_client# go run client.go
2020/08/18 09:00:31 Greeting 2: Some Valid response from server
```

Everthing Works, Client is able to get repsonse from Server

## Create Ingress

For GRPC

```
HOST=sample-grpc.10.131.232.223.nip.io

kubectl create -f - <<EOF
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: sample-grpc
  namespace: test
spec:
  rules:
  - host: $HOST
    http:
      paths:
      - backend:
          serviceName: sample-grpc
          servicePort: 50051
        path: /
EOF

```

For Nginx

```
HOST=nginx.10.131.232.223.nip.io



kubectl create -f - <<EOF
apiVersion: extensions/v1beta1
kind: Ingress
metadata:
  name: nginx
  namespace: test
spec:
  rules:
  - host: $HOST
    http:
      paths:
      - backend:
          serviceName: nginx
          servicePort: 80
        path: /
EOF
```

## Test 

```

