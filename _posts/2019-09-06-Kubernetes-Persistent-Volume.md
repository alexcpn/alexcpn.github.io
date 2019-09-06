# Kubernetes PV,PVC
## Trying to make this work with a specific volume

Here is the Yamls for PV

```yaml
apiVersion: v1
kind: PersistentVolume
metadata:
  name: pv-specific-node
  labels:
    type: local
spec:
  storageClassName: local-storage
  capacity:
    storage: 1Gi
  accessModes:
    - ReadWriteOnce
  local:
    path: "/mnt/data2"
  nodeAffinity:
    required:
      nodeSelectorTerms:
      - matchExpressions:
        - key: kubernetes.io/hostname
          operator: In
          values:
          - alex-k8s-2.novalocal
```          

PVC

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: pvc-specific-node
spec:
  storageClassName: local-storage
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 500M
```     
 
and the POD

```yaml

apiVersion: v1
kind: Pod
metadata:
  name: task-pv-pod
spec:
  volumes:
    - name: task-pv-storage
      persistentVolumeClaim:
        claimName: pvc-specific-node
  containers:
    - name: task-pv-container
      image: nginx
      ports:
        - containerPort: 80
          name: "http-server"
      volumeMounts:
        - mountPath: "/usr/share/nginx/html"
          name: task-pv-storage
```

And here is the output

```

In Specific Node

```
core@alex-k8s-2 ~ $ vi /mnt/data2/index.html
core@alex-k8s-2 ~ $ cat /mnt/data2/index.html 
'Hello from Kubernetes Local storage'
````

PV,PVC,Status

```
core@alex-k8s-1 ~ $ kcl get pv
NAME               CAPACITY   ACCESS MODES   RECLAIM POLICY   STATUS   CLAIM                       STORAGECLASS    REASON   AGE
pv-specific-node   1Gi        RWO            Retain           Bound    default/pvc-specific-node   local-storage            9m28s
```


core@alex-k8s-1 ~ $ kcl get pvc
NAME                STATUS   VOLUME             CAPACITY   ACCESS MODES   STORAGECLASS    AGE
pvc-specific-node   Bound    pv-specific-node   1Gi        RWO            local-storage   6m53s

core@alex-k8s-1 ~ $ kcl get pods
NAME          READY   STATUS    RESTARTS   AGE
task-pv-pod   1/1     Running   0          12s

core@alex-k8s-1 ~ $ kcl exec -it task-pv-pod -- /bin/bash
root@task-pv-pod:/# ls /usr/share/nginx/html
**index.html**  test.file
