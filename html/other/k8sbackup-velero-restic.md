# Summary

## Configure Velero to take backups of Kubernetes resources including persistant volume

So that

- You can restore the cluster resources in case of problems - like namespace deleted
- You can restore the load in another cluster, say for cluster reinstallation


## Pre-requisite 

### S3 Storage

You can setup one easily with Minio.

## What is Velero ?

Velero (formerly Heptio Ark) gives you tools to back up and restore your Kubernetes cluster resources and persistent volumes. It runs as a K8s operator

## Why Restic ?

1. Because CSI Snapshot is not wokring

### Limitations of Restic ?

https://velero.io/docs/v1.4/restic/

Restic scans each file in a single thread. This means that large files (such as ones storing a database) will take a long time to scan for data deduplication, even if the actual difference is small.
If you plan to use the Velero restic integration to backup 100GB of data or more, you may need to customize the resource limits to make sure backups complete successfully.

### How to install in baremetal ?

Follow https://velero.io/docs/v1.4/basic-install/
note  --backup-location-config region=default

```

velero install \
 --provider aws \
 --plugins velero/velero-plugin-for-aws:v1.0.0,velero/velero-plugin-for-csi:v0.1.1 \
 --bucket velero2  \
 --secret-file ./credentials-velero  \
 --use-volume-snapshots=true \
 --backup-location-config region=default,s3ForcePathStyle="true",s3Url=http://192.168.0.30:7000  \
 --image velero/velero:v1.4.2  \
 --snapshot-location-config region="default" \
 --use-restic
```
Note - Even thorugh I have not used CSI - I got this error on restoring from another cluster (could be that I used CSI Snapshot in that cluster)

```
error preparing persistentvolumeclaims/test-nginx/ceph-ext: rpc error: code = Unknown desc = Volumesnapshot test-nginx/ceph-ext does not have a velero.io/csi-volumesnapshot-handle annotation
error preparing secrets/test-nginx/default-token-cqlmk: rpc error: code = Unknown desc = Volumesnapshot test-nginx/default-token-cqlmk does not have a velero.io/csi-volumesnapshot-handle annotation
error preparing serviceaccounts/test-nginx/default: rpc error: code = Unknown desc = Volumesnapshot test-nginx/default does not have a velero.io/csi-volumesnapshot-handle annotation
error preparing pods/test-nginx/nginx-test: rpc error: code = Unknown desc = Volumesnapshot test-nginx/nginx-test does not have a velero.io/csi-volumesnapshot-handle annotation
```

Install the K8s SnapShot CRD's

```
https://github.com/kubernetes-csi/external-snapshotter/issues/245

https://github.com/kubernetes-csi/external-snapshotter/blob/master/README.md#usage

--
root@k8s-storage-1:~# git clone https://github.com/kubernetes-csi/external-snapshotter.git
root@k8s-storage-1:~# cd external-snapshotter/
--
root@k8s-storage-1:~/external-snapshotter# kubectl create -f client/config/crd
customresourcedefinition.apiextensions.k8s.io/volumesnapshotclasses.snapshot.storage.k8s.io created
customresourcedefinition.apiextensions.k8s.io/volumesnapshotcontents.snapshot.storage.k8s.io created
customresourcedefinition.apiextensions.k8s.io/volumesnapshots.snapshot.storage.k8s.io created
--

root@k8s-storage-1:~/external-snapshotter# kubectl create -f deploy/kubernetes/csi-snapshotter
serviceaccount/csi-snapshotter created
clusterrole.rbac.authorization.k8s.io/external-snapshotter-runner created
clusterrolebinding.rbac.authorization.k8s.io/csi-snapshotter-role created
role.rbac.authorization.k8s.io/external-snapshotter-leaderelection created
rolebinding.rbac.authorization.k8s.io/external-snapshotter-leaderelection created
serviceaccount/csi-provisioner created
clusterrole.rbac.authorization.k8s.io/external-provisioner-runner created
clusterrolebinding.rbac.authorization.k8s.io/csi-provisioner-role created
role.rbac.authorization.k8s.io/external-provisioner-cfg created
rolebinding.rbac.authorization.k8s.io/csi-provisioner-role-cfg created
clusterrolebinding.rbac.authorization.k8s.io/csi-snapshotter-provisioner-role created
rolebinding.rbac.authorization.k8s.io/csi-snapshotter-provisioner-role-cfg created
service/csi-snapshotter created
statefulset.apps/csi-snapshotter created
```

If you want to use CSI Snapshotting feature, you need to install the VolumeSnapShotClass and VolumeSnapshot also

```
kubectl apply -f https://raw.githubusercontent.com/rook/rook/master/cluster/examples/kubernetes/ceph/csi/rbd/snapshotclass.yaml

volumesnapshotclass.snapshot.storage.k8s.io/csi-rbdplugin-snapclass created

kubectl apply -f https://raw.githubusercontent.com/rook/rook/master/cluster/examples/kubernetes/ceph/csi/rbd/snapshot.yaml

kubectl get VolumeSnapshotClass --all-namespaces
NAME                      DRIVER                       DELETIONPOLICY   AGE
csi-rbdplugin-snapclass   rook-ceph.rbd.csi.ceph.com   Delete           67s
```


Check backup locations

```
root@k8s-storage-1:~# velero backup-location get 
NAME      PROVIDER   BUCKET/PREFIX   ACCESS MODE
default   aws        velero2         ReadWrite
```

### How to backup ?

```
velero backup create test-4 --include-namespaces test-nginx --wait
````


### How to restore ?

```

[root@green--1 ~]# kubectl delete ns test-nginx

velero restore create --from-backup test-4

[root@green--1 ~]# velero get restore
NAME                    BACKUP   STATUS      ERRORS   WARNINGS   CREATED                         SELECTOR
test-4-20200831173655   test-4   Completed   0        0          2020-08-31 17:36:55 +0530 IST   <none>

[root@green--1 ~]# kubectl -n test-nginx get pvc
NAME       STATUS   VOLUME                                     CAPACITY   ACCESS MODES   STORAGECLASS      AGE
ceph-ext   Bound    pvc-0ce3ed03-e323-41d0-b4b0-a2fc865060d1   2Gi        RWO            rook-ceph-block   3m29s

```

### How to restore to another Cluster ?

Install Velero in second cluster pointing to the same S3 bucket


### How to Schedule backups ?

https://velero.io/docs/v1.4/disaster-case/

Example every five mintutes

```
velero schedule create s-test-nginx  --include-namespaces test-nginx  --schedule "*/5 * * * *"
```

If S3 is down will fail, next will be successfull
```
# velero get backups

s-test-nginx-20200902062502   Completed         0        0          2020-09-02 11:55:02 +0530 IST   29d       default            <none>
s-test-nginx-20200902062105   Failed            0        0          2020-09-02 11:51:05 +0530 IST   29d       default            <none>

```

### Can I backup only Persistent Volumes ?

Yes, you can filter

```
 velero backup create test-pv-10  --include-namespaces test-nginx  --include-resources  persistentvolumeclaims,persistentvolumes --wait
 ```

### Can I restore to another namesapce ?

Yes, namespace mapping. Note PVC mapping wont work. You can do this in another cluster

### Are backups incremental ?

Yes

### Are restore's incremental ?

No

### What if connection to S3 breaks ??


For scheduled backups the next schedule will trigger

For Inprogress, will stay in Progress forever

