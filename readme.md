# Bottleneck of Graph Neural Networks
This is the code for *Discovering the Representation Bottleneck of Graph Neural Networks from Multi-order Interactions*.
[arXiv](https://arxiv.org/abs/2205.07266)

Some necessary packages before running the code. 
```markdown
pip install einops
```





## Data

### Newtonian Dynamics & Hamiltonian Dynamics 
There we follow Cranmer et al. (2020) to generate the data. 
```markdown
# install necessary packages
pip install celluloid
pip install jax[cuda11_cudnn805] -f https://storage.googleapis.com/jax-releases/jax_releases.htmlpip install celluloid
pip install jaxlib
```
Note that jax.ops.index_update is deprecated at 0.2.22, and we modify the profile via `x0.at[].set()`. Moreover, it might
cause a problem with loading jax due to `Couldn't invoke ptxas`. This is because the path of `ptxas` is not available to the system.
 A possible solution is to install cuda manually using the `install_cuda_11_1.sh` file. 
```markdown
sudo bash install_cuda_11_1.sh
```

```python
python data/dataset_nbody.py
```


### Molecular Dynamics 
The MD dataset, ISO 17, is provided by the Quantum Machine organization, which is available in 
[its official website](http://quantum-machine.org/datasets/). 
```python
python data/dataset_iso17.py
```

### Molecular Property Prediction 
QM7 and QM8 datasets are also accessible in the previous link of the Quantum Machine organization. 




## Analyze the Bottleneck
### Train a Model 
```markdown
python train.py --data=qm7 --method=egnn --gpu=0,1 
```
### Calculate the Strength 
```markdown
# load a pretrained model
python test.py --data=qm7 --method=egnn --pretrain=1

# randomly initializing a model
python test.py --data=qm7 --method=egnn --pretrain=0
```






