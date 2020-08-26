
# Summary

## Configure HAProxy Ingress Controller so that both HTTP1.1 Ingress and HTTP2 Ingress and backend servers capable of speaking both HTTP1.1 and HTTP2 get serveed

related Git issue - https://github.com/jcmoraisjr/haproxy-ingress/issues/643

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


Repeat this for an NGINX that is configured (hacked) to speak HTTP2


```
kubectl create deployment nginx --image nginx:alpine -n test
[root@green--1 ~]#  kubectl -n test expose deployment nginx --port=80
service/nginx exposed

```

```
Configure Nginx with http2
(We want ssl to be terminated at the ingress controller. If we use https at ningx side then we need to configure certificates in niginx .

So for plain http2 serving -In nginx pod changed  /etc/nginx/conf.d/default.conf  and did nginx -t and nignx -s reload for now)
server {
    listen       80 http2;
    listen  [::]:80 http2;

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

### HTTP 1.1 Servr Does not work
```
C:\Users\acp\Documents\Coding\daas_project\infra-common>curl -kv  https://nginx2.10.131.232.223.nip.io/
*   Trying 10.131.232.223...
* TCP_NODELAY set
* Connected to nginx2.10.131.232.223.nip.io (10.131.232.223) port 443 (#0)
* schannel: SSL/TLS connection with nginx2.10.131.232.223.nip.io port 443 (step 1/3)
* schannel: disabled server certificate revocation checks
* schannel: verifyhost setting prevents Schannel from comparing the supplied target name with the subject names in server certificates.
* schannel: sending initial handshake data: sending 184 bytes...
* schannel: sent initial handshake data: sent 184 bytes
* schannel: SSL/TLS connection with nginx2.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: failed to receive handshake, need more data
* schannel: SSL/TLS connection with nginx2.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: encrypted data got 1315
* schannel: encrypted data buffer: offset 1315 length 4096
* schannel: sending next handshake data: sending 93 bytes...
* schannel: SSL/TLS connection with nginx2.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: encrypted data got 51
* schannel: encrypted data buffer: offset 51 length 4096
* schannel: SSL/TLS handshake complete
* schannel: SSL/TLS connection with nginx2.10.131.232.223.nip.io port 443 (step 3/3)
* schannel: stored credential handle in session cache
> GET / HTTP/1.1
> Host: nginx2.10.131.232.223.nip.io
> User-Agent: curl/7.55.1
> Accept: */*
>
* schannel: client wants to read 102400 bytes
* schannel: encdata_buffer resized 103424
* schannel: encrypted data buffer: offset 0 length 103424
* schannel: encrypted data got 253
* schannel: encrypted data buffer: offset 253 length 103424
* schannel: decrypted data length: 193
* schannel: decrypted data added: 193
* schannel: decrypted data cached: offset 193 length 102400
* schannel: encrypted data length: 31
* schannel: encrypted data cached: offset 31 length 103424
* schannel: server closed the connection
* schannel: schannel_recv cleanup
* schannel: decrypted data returned 193
* schannel: decrypted data buffer: offset 0 length 102400
* HTTP 1.0, assume close after body
< HTTP/1.0 503 Service Unavailable
< cache-control: no-cache
< content-type: text/html
<
<html><body><h1>503 Service Unavailable</h1>
No server is available to handle this request.
</body></html>
* schannel: client wants to read 102400 bytes
* schannel: server indicated shutdown in a prior call
* schannel: schannel_recv cleanup
* Closing connection 0
* schannel: shutting down SSL/TLS connection with nginx2.10.131.232.223.nip.io port 443
* schannel: clear security context handle
```
### HTTP2 Capable Nginx works

```

C:\Users\acp\Documents\Coding\daas_project\infra-common>curl -kv  https://nginx.10.131.232.223.nip.io/
*   Trying 10.131.232.223...
* TCP_NODELAY set
* Connected to nginx.10.131.232.223.nip.io (10.131.232.223) port 443 (#0)
* schannel: SSL/TLS connection with nginx.10.131.232.223.nip.io port 443 (step 1/3)
* schannel: disabled server certificate revocation checks
* schannel: verifyhost setting prevents Schannel from comparing the supplied target name with the subject names in server certificates.
* schannel: sending initial handshake data: sending 183 bytes...
* schannel: sent initial handshake data: sent 183 bytes
* schannel: SSL/TLS connection with nginx.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: failed to receive handshake, need more data
* schannel: SSL/TLS connection with nginx.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: encrypted data got 1315
* schannel: encrypted data buffer: offset 1315 length 4096
* schannel: sending next handshake data: sending 93 bytes...
* schannel: SSL/TLS connection with nginx.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: encrypted data got 51
* schannel: encrypted data buffer: offset 51 length 4096
* schannel: SSL/TLS handshake complete
* schannel: SSL/TLS connection with nginx.10.131.232.223.nip.io port 443 (step 3/3)
* schannel: stored credential handle in session cache
> GET / HTTP/1.1
> Host: nginx.10.131.232.223.nip.io
> User-Agent: curl/7.55.1
> Accept: */*
>
* schannel: client wants to read 102400 bytes
* schannel: encdata_buffer resized 103424
* schannel: encrypted data buffer: offset 0 length 103424
* schannel: encrypted data got 286
* schannel: encrypted data buffer: offset 286 length 103424
* schannel: decrypted data length: 257
* schannel: decrypted data added: 257
* schannel: decrypted data cached: offset 257 length 102400
* schannel: encrypted data buffer: offset 0 length 103424
* schannel: decrypted data buffer: offset 257 length 102400
* schannel: schannel_recv cleanup
* schannel: decrypted data returned 257
* schannel: decrypted data buffer: offset 0 length 102400
< HTTP/1.1 200
< server: nginx/1.19.2
< date: Wed, 26 Aug 2020 07:24:00 GMT
< content-type: text/html
< content-length: 612
< last-modified: Tue, 11 Aug 2020 15:16:45 GMT
< etag: "5f32b65d-264"
< accept-ranges: bytes
< strict-transport-security: max-age=15768000
<
* schannel: client wants to read 612 bytes
* schannel: encrypted data buffer: offset 0 length 103424
* schannel: encrypted data got 641
* schannel: encrypted data buffer: offset 641 length 103424
* schannel: decrypted data length: 612
* schannel: decrypted data added: 612
* schannel: decrypted data cached: offset 612 length 102400
* schannel: encrypted data buffer: offset 0 length 103424
* schannel: decrypted data buffer: offset 612 length 102400
* schannel: schannel_recv cleanup
* schannel: decrypted data returned 612
* schannel: decrypted data buffer: offset 0 length 102400
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
<style>
    body {
        width: 35em;
        margin: 0 auto;
        font-family: Tahoma, Verdana, Arial, sans-serif;
    }
</style>
</head>
<body>
<h1>Welcome to nginx!</h1>
<p>If you see this page, the nginx web server is successfully installed and
working. Further configuration is required.</p>

<p>For online documentation and support please refer to
<a href="http://nginx.org/">nginx.org</a>.<br/>
Commercial support is available at
<a href="http://nginx.com/">nginx.com</a>.</p>

<p><em>Thank you for using nginx.</em></p>
</body>
</html>
* Connection #0 to host nginx.10.131.232.223.nip.io left intact
```

## Note without backend set at h2

NGINX with HTTP 1.1 Works

```
C:\Users\acp\Documents\Coding\daas_project\infra-common>curl -kv  https://nginx2.10.131.232.223.nip.io/
*   Trying 10.131.232.223...
* TCP_NODELAY set
* Connected to nginx2.10.131.232.223.nip.io (10.131.232.223) port 443 (#0)
* schannel: SSL/TLS connection with nginx2.10.131.232.223.nip.io port 443 (step 1/3)
* schannel: disabled server certificate revocation checks
* schannel: verifyhost setting prevents Schannel from comparing the supplied target name with the subject names in server certificates.
* schannel: sending initial handshake data: sending 184 bytes...
* schannel: sent initial handshake data: sent 184 bytes
* schannel: SSL/TLS connection with nginx2.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: failed to receive handshake, need more data
* schannel: SSL/TLS connection with nginx2.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: encrypted data got 1315
* schannel: encrypted data buffer: offset 1315 length 4096
* schannel: sending next handshake data: sending 93 bytes...
* schannel: SSL/TLS connection with nginx2.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: encrypted data got 51
* schannel: encrypted data buffer: offset 51 length 4096
* schannel: SSL/TLS handshake complete
* schannel: SSL/TLS connection with nginx2.10.131.232.223.nip.io port 443 (step 3/3)
* schannel: stored credential handle in session cache
> GET / HTTP/1.1
> Host: nginx2.10.131.232.223.nip.io
> User-Agent: curl/7.55.1
> Accept: */*
>
* schannel: client wants to read 102400 bytes
* schannel: encdata_buffer resized 103424
* schannel: encrypted data buffer: offset 0 length 103424
* schannel: encrypted data got 929
* schannel: encrypted data buffer: offset 929 length 103424
* schannel: decrypted data length: 259
* schannel: decrypted data added: 259
* schannel: decrypted data cached: offset 259 length 102400
* schannel: encrypted data length: 641
* schannel: encrypted data cached: offset 641 length 103424
* schannel: decrypted data length: 612
* schannel: decrypted data added: 612
* schannel: decrypted data cached: offset 871 length 102400
* schannel: encrypted data buffer: offset 0 length 103424
* schannel: decrypted data buffer: offset 871 length 102400
* schannel: schannel_recv cleanup
* schannel: decrypted data returned 871
* schannel: decrypted data buffer: offset 0 length 102400
< HTTP/1.1 200 OK
< server: nginx/1.19.2
< date: Wed, 26 Aug 2020 08:59:24 GMT
< content-type: text/html
< content-length: 612
< last-modified: Tue, 11 Aug 2020 15:16:45 GMT
< etag: "5f32b65d-264"
< accept-ranges: bytes
< strict-transport-security: max-age=15768000
<
<!DOCTYPE html>
<html>
<head>
<title>Welcome to nginx!</title>
<style>
    body {
        width: 35em;
        margin: 0 auto;
        font-family: Tahoma, Verdana, Arial, sans-serif;
    }
</style>
</head>
<body>
<h1>Welcome to nginx!</h1>
<p>If you see this page, the nginx web server is successfully installed and
working. Further configuration is required.</p>

<p>For online documentation and support please refer to
<a href="http://nginx.org/">nginx.org</a>.<br/>
Commercial support is available at
<a href="http://nginx.com/">nginx.com</a>.</p>

<p><em>Thank you for using nginx.</em></p>
</body>
</html>
* Connection #0 to host nginx2.10.131.232.223.nip.io left intact
```
NGINX which is configured to Serve H2 **DOES NOT** Work

``
C:\Users\acp\Documents\Coding\daas_project\infra-common>curl -kv  https://nginx.10.131.232.223.nip.io/
*   Trying 10.131.232.223...
* TCP_NODELAY set
* Connected to nginx.10.131.232.223.nip.io (10.131.232.223) port 443 (#0)
* schannel: SSL/TLS connection with nginx.10.131.232.223.nip.io port 443 (step 1/3)
* schannel: disabled server certificate revocation checks
* schannel: verifyhost setting prevents Schannel from comparing the supplied target name with the subject names in server certificates.
* schannel: sending initial handshake data: sending 183 bytes...
* schannel: sent initial handshake data: sent 183 bytes
* schannel: SSL/TLS connection with nginx.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: failed to receive handshake, need more data
* schannel: SSL/TLS connection with nginx.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: encrypted data got 1315
* schannel: encrypted data buffer: offset 1315 length 4096
* schannel: sending next handshake data: sending 93 bytes...
* schannel: SSL/TLS connection with nginx.10.131.232.223.nip.io port 443 (step 2/3)
* schannel: encrypted data got 51
* schannel: encrypted data buffer: offset 51 length 4096
* schannel: SSL/TLS handshake complete
* schannel: SSL/TLS connection with nginx.10.131.232.223.nip.io port 443 (step 3/3)
* schannel: stored credential handle in session cache
> GET / HTTP/1.1
> Host: nginx.10.131.232.223.nip.io
> User-Agent: curl/7.55.1
> Accept: */*
>
* schannel: client wants to read 102400 bytes
* schannel: encdata_buffer resized 103424
* schannel: encrypted data buffer: offset 0 length 103424
* schannel: encrypted data got 245
* schannel: encrypted data buffer: offset 245 length 103424
* schannel: decrypted data length: 185
* schannel: decrypted data added: 185
* schannel: decrypted data cached: offset 185 length 102400
* schannel: encrypted data length: 31
* schannel: encrypted data cached: offset 31 length 103424
* schannel: server closed the connection
* schannel: schannel_recv cleanup
* schannel: decrypted data returned 185
* schannel: decrypted data buffer: offset 0 length 102400
* HTTP 1.0, assume close after body
< HTTP/1.0 502 Bad Gateway
< cache-control: no-cache
< content-type: text/html
<
<html><body><h1>502 Bad Gateway</h1>
The server returned an invalid or incomplete response.
</body></html>
* schannel: client wants to read 102400 bytes
* schannel: server indicated shutdown in a prior call
* schannel: schannel_recv cleanup
* Closing connection 0
* schannel: shutting down SSL/TLS connection with nginx.10.131.232.223.nip.io port 443
* schannel: clear security context handl
```
