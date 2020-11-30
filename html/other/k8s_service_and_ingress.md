
# Kubernetes Different Ways of exposing a Service by an example
## Port -Forwarding, Node Port, LoadBalancer and Ingress 

We will look with an example on how we can expose a service externally. For this lets use a simple example of an Nginx pod to create a deployment and a service

### Create a Deployment of NGINX

```
kubectl create deployment nginx2 --image nginx:alpine -n test
```
### Expose Nginx as a Service

```
kubectl -n test expose deployment nginx2 --port=80
```

You should see something like below

```console
root@green-1:~# kubectl -n test get svc -o wide
NAME     TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE   SELECTOR
nginx2   ClusterIP   10.98.177.254   <none>        80/TCP    48s   app=nginx2
```

```console
root@green-1:~# kubectl -n test get pod  -o wide
NAME                      READY   STATUS    RESTARTS   AGE   IP           NODE      NOMINATED NODE   READINESS GATES
nginx2-55764b6d95-c6ct9   1/1     Running   0          28h   10.244.3.3   green-4   <none>           <none>
```

You can actually check the iptables in linux and the NAT table on how rules are set for this

```
root@green-1:~# iptables -t nat -L | grep nginx
```

Cluster IP is the default way K8s Services are exposed. As you can see this internal IP of Flannel is not exposed out

```
curl -k http://127.0.0.1:80/
```

### Option 1 - Expose via Port-Forwarding

Let's test via Port-forwarding

```
kubectl port-forward service/nginx2 --address 0.0.0.0 80:80 --namespace test
```

Locally in the master Node

```
curl -k http://127.0.0.1:80/
```

Remotely - from your laptop or browser

```
curl -kv http://10.131.XX.YY:80/
curl -kv http://192.168.0.5:80/
```
These should work; If not check your IP and firewall.

Optionally you can set type as NodePort, but the port range is from 30001 to xx; So you need to access from these non standard ports (and open firewall for these ports)

## Option 2 Expose via LoadBalancer

For this we need a LoadBalancer. We will use MetalLB

### Install a Load Balancer - We will use MetalLB

Follow this - https://metallb.universe.tf/installation/ (Installation by manifest)

We will use this Confg map to provide our Node IP of a node which also has an associated Floating .(Actually we need to give a pool of IP's to MetalLB - for loadbalancing but we are not  doing it now for this example)

```
cat << EOF | kubectl apply -f - 
apiVersion: v1
kind: ConfigMap
metadata:
  namespace: metallb-system
  name: config
data:
  config: |
    address-pools:
    - name: default
      protocol: layer2
      addresses:
      - 192.168.0.5-192.168.0.5
EOF
```
Now let us change our Nginx service we created from ClusterIP to type LoadBalancer

```
kubectl  -n test get svc
NAME     TYPE        CLUSTER-IP      EXTERNAL-IP   PORT(S)   AGE
nginx2   ClusterIP   10.98.177.254   <none>        80/TCP    24h
```
```
kubectl patch svc nginx2 -n test -p '{"spec": {"type": "LoadBalancer"}}'
```
You can see that the External IP has got assigned
```
kubectl  -n test get svc
NAME     TYPE           CLUSTER-IP      EXTERNAL-IP   PORT(S)        AGE
nginx2   LoadBalancer   10.98.177.254   192.168.0.5   80:30244/TCP   24h
```

The floating IP we have assigend for 192.168.0.5 in OpenStack is 10.131.XX.YY.

We can test this out from a remote browser
http://10.131.XX.YY/ 

 You should see the  **Welcome to nginx!** screen.

 The problem with LoadBalancer is that you can have only one service latched to this port in this node. So though we get good ports compared to NodePort range, this is rather limiting.

## Option 3 - Expose via IngressControoler

An Ingress Controller acts like a revers proxy. You can define an ingress for multiple services, and as long as the ingress names are unique, you can have as many running in a single cluster, irrespective of bothering about Nodes Ports or Node IPs.

Note that Ingress Controller is a Service and that has to be exposed out, preferably as type LoadBalancer. Let's do that

### Install Nginx Ingress Controller

https://kubernetes.github.io/ingress-nginx/deploy/

We use the BareMetal Option using - NodePort 

```console
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v0.41.2/deploy/static/provider/baremetal/deploy.yaml
```

Note that the Ingress Controller is itself a Service and it is exposed as Node Port

```
 kubectl -n ingress-nginx get svc
NAME                                 TYPE        CLUSTER-IP       EXTERNAL-IP   PORT(S)                      AGE
ingress-nginx-controller             NodePort    10.102.158.192   <none>        80:30374/TCP,443:32579/TCP   3m28s
```

### Create an Ingress for Nginx service 

```console
HOST=nginx.10.131.XX.YY.nip.io
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
          serviceName: nginx2
          servicePort: 80
        path: /
EOF
```

Check if it is created `kubectl -n test get ing`

```console
Warning: extensions/v1beta1 Ingress is deprecated in v1.14+, unavailable in v1.22+; use networking.k8s.io/v1 Ingress
NAME    CLASS    HOSTS                         ADDRESS   PORTS   AGE
nginx   <none>   nginx.10.131.XX.YY.nip.io             80      132m
```

We can test accessing this Ingress externally, but using the NodePort Port of the ingress controller - 30374 that is mapped to 80 port

http://nginx.10.131.XX.YY.nip.io:30374/

and it should work. ( Note that 10.131 is the floating IP of the node)

![ingress-nodeport](https://i.imgur.com/8ECYd4b.png)

We can have many other ingress for other services like this created and all on the same port. 

---

Now lets try to change the Ingress Controller Service from Node Port to Load Balancer using the MetalLB Load Balancer

```console
kubectl patch svc ingress-nginx-controller -n ingress-nginx -p '{"spec": {"type": "LoadBalancer"}}'
```
We can see that it got the IP from MetLB as the external IP

```
alex@N-20HEPF0ZU9PR:/mnt/c/Users/acp/Documents/certs3/ee$ kubectl -n ingress-nginx get svc
NAME                                 TYPE           CLUSTER-IP       EXTERNAL-IP   PORT(S)                      AGE
ingress-nginx-controller             LoadBalancer   10.102.158.192   192.168.0.5   80:30374/TCP,443:32579/TCP   110m
ingress-nginx-controller-admission   ClusterIP      10.105.83.58     <none>        443/TCP                      110m
```

Now the Ingress can be Accessed from (80 port)

http://nginx.10.131.XX.YY.nip.io/

![ingress-lb](https://i.imgur.com/5HG9Go2.png)


Reference

Excellent article regarding Kubernetes Services - https://www.haproxy.com/blog/dissecting-the-haproxy-kubernetes-ingress-controller/

More deep details of Linux routing https://www.karlrupp.net/en/computer/nat_tutorial#:~:text=Linux%20and%20Netfilter&text=We%20will%20use%20the%20command,%3A%20PREROUTING%2C%20OUTPUT%20und%20POSTROUTING.

