import random
import os
from pathlib import Path

out = Path("k8s/hyperparametersearch")
if not os.path.isdir(out):
    os.mkdir(out)

for i in range(40, 50):
    folder = f"batch_{i}"
    if not os.path.isdir(out / folder):
        os.mkdir(out / folder)

    for j in range(50):
        optim_lr: float = random.uniform(0.01, 0.7)
        gamma: float = random.uniform(0.0, 0.3)
        score_pow: float = random.uniform(-0.5, 3)
        composite_balance: float = random.uniform(-1, 1)
        adaptive_score_offset: float = random.uniform(0.1, 1)

        yaml = f'''
        apiVersion: batch/v1
        kind: Job
        metadata:
          name: hypersearch-{i}-{j}
        spec:
          template:
            spec:
              priorityClassName: research-low
              restartPolicy: "OnFailure"
              containers:
                - name: hyperparam-test
                  image: ls6-stud-registry.informatik.uni-wuerzburg.de/studseizinger/nicer_env:0.0.4
                  workingDir: /workdir
                  imagePullPolicy: "Always"
                  env:
                    - name: HOME
                      value: "/tmp"
                  resources:
                    limits:
                      nvidia.com/gpu: "1"
                      cpu: "48"
                      memory: "10Gi"
                    requests:
                      nvidia.com/gpu: "1"
                      cpu: "16"
                      memory: "10Gi"
                  volumeMounts:
                    - mountPath: /workdir
                      name: localdir
                    - mountPath: /dataset-orig
                      name: dataset-orig
                    - mountPath: /dataset-dist
                      name: dataset-dist
                    - mountPath: /out
                      name: out
                    - mountPath: /dev/shm
                      name: dshm
                  command:
                    - python
                    - hyperparamtest.py
                    - --optim_lr
                    - '{optim_lr}'
                    - --gamma
                    - '{gamma}'
                    - --score_pow
                    - '{score_pow}'
                    - --composite_balance
                    - '{composite_balance}'
                    - --adaptive_score_offset
                    - '{adaptive_score_offset}'
              imagePullSecrets:
                - name: lsx-registry
              nodeSelector:
                gputype: rtx2080ti
              volumes:
                - name: dataset-orig
                  cephfs:
                    monitors:
                      - 132.187.14.16,132.187.14.17,132.187.14.19,132.187.14.20
                    user: studseizinger
                    path: "/home/stud/seizinger/git/NICER/datasets/pexels/images"
                    secretRef:
                      name: ceph-secret
                - name: dataset-dist
                  cephfs:
                    monitors:
                      - 132.187.14.16,132.187.14.17,132.187.14.19,132.187.14.20
                    user: studseizinger
                    path: "/home/stud/seizinger/git/NICER/datasets/pexels_dist/images"
                    secretRef:
                      name: ceph-secret
                - name: out
                  cephfs:
                    monitors:
                      - 132.187.14.16,132.187.14.17,132.187.14.19,132.187.14.20
                    user: studseizinger
                    path: "/home/stud/seizinger/git/NICER/out"
                    secretRef:
                      name: ceph-secret
                - name: localdir
                  cephfs:
                    monitors:
                      - 132.187.14.16,132.187.14.17,132.187.14.19,132.187.14.20
                    user: studseizinger
                    path: "/home/stud/seizinger/git/NICER/"
                    secretRef:
                      name: ceph-secret
                - name: dshm
                  emptyDir:
                    medium: Memory
        
        '''

        with open(out/folder/f"hypersearch-{i}-{j}.yaml", "w") as yamlfile:
            yamlfile.write(yaml)