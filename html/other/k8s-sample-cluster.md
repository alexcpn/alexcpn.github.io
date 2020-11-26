
# Bring up a Sample K8s cluster 
## On Bare-metal or OpenStack with Flannel (Networking), Containerd and HAProxy Ingress Controller


Setup

We have one master and three workers. All running Ubuntu-18.04. One node has a floating IP -10.131.228.167. This will be the master, rest 3 are workers.

Make all firewall ports Open. You don't want to get stuck here.


Main Reference https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/

---

## Step 1- Install Containerd in all 4 Nodes

Follow this in all your 4 nodes https://kubernetes.io/docs/setup/production-environment/container-runtimes/#containerd

We will configure systemd cgroups driver for containerd.

Note that there is a bug with the `/etc/containerd/config.toml`
[containerd-config-bug]

It may be apparent in the next steps when you try to do `kubeadm init` or in the worker nodes `kubeadm join`.

 Workaround is to delete this file and restart the containerd service and after that the kubelet service.


```
rm /etc/containerd/config.toml
systemctl restart containerd
systemctl restart kubelet
````


## Step 2  Install kubelet,kubeadm  and kubectl
-------------

Install '`kubelet kubeadm kubectl`  in all 4 nodes as is written here

https://kubernetes.io/docs/setup/production-environment/tools/kubeadm/install-kubeadm/


## Step 3 - In Master - do Kubeadm init and CNI plugin

Before that 
- Since we are using containerd we need to specify that
- Since we are using Flannel , we need a specific pod cidr to be specified

Create a config.yaml with the following

```
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
cgroupDriver: containerd
---
apiVersion: kubeadm.k8s.io/v1beta2
kind: ClusterConfiguration
networking:
  podSubnet: "10.244.0.0/16" # --pod-network-cidr
```

**In Master Node** 

```
kubeadm init --config config.yaml
```

Note - If you are getting an error as specified in the issue [containerd-config-bug], do the workaround specified

[containerd-config-bug]: https://github.com/containerd/containerd/issues/4581

On success you should get 

```
Your Kubernetes control-plane has initialized successfully!

To start using your cluster, you need to run the following as a regular user:
```
Do as it is printed
```
  mkdir -p $HOME/.kube
  sudo cp -i /etc/kubernetes/admin.conf $HOME/.kube/config
  sudo chown $(id -u):$(id -g) $HOME/.kube/config
```
Next is POD network - we will use Flannel
```

You should now deploy a pod network to the cluster.
Run "kubectl apply -f [podnetwork].yaml" with one of the options listed at:
  https://kubernetes.io/docs/concepts/cluster-administration/addons/

Then you can join any number of worker nodes by running the following on each as root:

kubeadm join 192.168.0.5:6443 --token ni79pw.tdrrw6terqlolump \
    --discovery-token-ca-cert-hash sha256:1b36a508cb....4d9bff600749d7758de9916c1
```

- Apply Container Networking in Master node

```
kubectl apply -f https://raw.githubusercontent.com/coreos/flannel/master/Documentation/kube-flannel.yml
```

## Step 4 - In all workers do `kubeadm join`

Note - If you are getting an error as specified in the issue [containerd-config-bug], do the workaround specified

In case you find problems with node not joiining and gettting the error CNI Network not available 
do the workaround 

## Step 5 - Copy kubeconfig to your local machine

Your cluster should be up. Copy the kubeconfig from the master node (cat  ~/.kube/config ) to your lapop

Since we have not configured SSL set the insercure option via kubectl in your laptop for easy working.

```
kubectl config set-cluster kubernetes-green --insecure-skip-tls-verify=true
```

That's it.

---

## Installing an Ingress Controller

### Install MetalLB first

Follow this - https://metallb.universe.tf/installation/ (Installation by manifest)

We will need this since we are installing in Baremetal (or OpenStack VMs). More details https://kubernetes.github.io/ingress-nginx/deploy/baremetal/

We will use this Confg map to provide our Floating IP to metal LB

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
      - 10.131.228.167-10.131.228.167
EOF
```

### Install HAProxy Ingress Controller

Install HAProxy Ingress Controller via Helm as written here - https://haproxy-ingress.github.io/docs/getting-started/

```
helm install haproxy-ingress haproxy-ingress/haproxy-ingress \
  --create-namespace --namespace=ingress-controller \
  --set controller.hostNetwork=true
```

For more setting (GRPC,HTTP2) see ref - https://github.com/alexcpn/alexcpn.github.io/blob/master/html/other/haproxy-grpc.md

You should see somthing like this, with the EXTERNAL-IP filled

```
kubectl -n ingress-controller get svc
NAME              TYPE           CLUSTER-IP      EXTERNAL-IP      PORT(S)                      AGE
haproxy-ingress   LoadBalancer   10.103.47.242   10.131.228.167   80:30038/TCP,443:31304/TCP   157m
```

## Testing with NGINX

```
# Create a Deployment of NGINX
kubectl create deployment nginx2 --image nginx:alpine -n test

# Expose nginx service
kubectl -n test expose deployment nginx2 --port=80
```
You should see something like below
```
kubectl -n test get svc
NAME     TYPE        CLUSTER-IP     EXTERNAL-IP   PORT(S)   AGE
nginx2   ClusterIP   10.98.234.20   <none>        80/TCP    5h23m
```

Let's test via Port-forwarding

```
kubectl port-forward service/nginx2 --address 0.0.0.0 80:80--namespace test
```
From your laptop or browser

```
curl -kv http://nginx.10.131.228.167.nip.io/
```
This should work; If not check your IP and firewall; and test first from the master node, if it is not working locally

- Create an Ingress

```
HOST=nginx.10.131.228.167.nip.io

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

Test via http://nginx.10.131.228.167.nip.io/ from a browser. You should see the  **Welcome to nginx!** screen.
