
# Bring up a Sample K8s cluster 
## On Bare-metal or OpenStack with Flannel (Networking), Containerd

Setup

We have one master and three workers. All running Ubuntu-18.04. One node has a floating IP 10.131.XX.YY and node 192.168.0.5. This will be the master, rest 3 are workers.

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
```

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

```
alex@N-20HEPF0ZU9PR:/mnt/c/Users/acp/Documents/certs3/ee$ kubectl get nodes -o wide
NAME      STATUS   ROLES    AGE   VERSION   INTERNAL-IP    EXTERNAL-IP   OS-IMAGE             KERNEL-VERSION      CONTAINER-RUNTIME
green-1   Ready    master   46h   v1.19.4   192.168.0.5    <none>        Ubuntu 18.04.4 LTS   4.15.0-96-generic   containerd://1.3.7
green-2   Ready    <none>   46h   v1.19.4   192.168.0.19   <none>        Ubuntu 18.04.4 LTS   4.15.0-96-generic   containerd://1.3.7
green-3   Ready    <none>   45h   v1.19.4   192.168.0.32   <none>        Ubuntu 18.04.4 LTS   4.15.0-96-generic   containerd://1.3.7
green-4   Ready    <none>   45h   v1.19.4   192.168.0.20   <none>        Ubuntu 18.04.4 LTS   4.15.0-96-generic   containerd://1.3.7
````

---

