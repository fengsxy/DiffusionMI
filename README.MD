## Diffusion Mutual Information (DMI)

DMI is a package for estimating mutual information in the context of diffusion processes, facilitating the evaluation and testing of new mutual information estimators. The package is implemented using PyTorch.

### Installation
You can install the package using:

```bash
pip install -r requirements.txt
```


```
### Usage Example
Below is an example demonstrating how to sample data and estimate mutual information using predefined tasks.

```python
import dmi

task = dmi.benchmark.BENCHMARK_TASKS['diffusion-1v1-0.75']
print(f"Task {task.name} with dimensions {task.dim_x} and {task.dim_y}")
print(f"Ground truth mutual information: {task.mutual_information:.2f}")

X, Y = task.sample(1000, seed=42)

cca = dmi.estimators.CCAMutualInformationEstimator()
print(f"Estimate by CCA: {cca.estimate(X, Y):.2f}")

ksg = dmi.estimators.KSGEnsembleFirstEstimator(neighborhoods=(5,))
print(f"Estimate by KSG: {ksg.estimate(X, Y):.2f}")
```


## List of Implemented Estimators
- [ ✅]`CPC.py`: Contrastive Predictive Coding Estimator
- [ ✅ ] `DIME.py`: Deep InfoMax Estimator
- [ ✅]`DV.py`: Donsker-Varadhan Estimator
- [ ✅] `MINDE.py`: Mutual Information Neural Discriminative Estimator
- [ ✅] `MINE.py`: Mutual Information Neural Estimation
- [ ✅]`NWJ.py`: Nguyen-Wainwright-Jordan Estimator
- [ ✅] `SMILE.py`: Scalable Mutual Information Learning Estimator



## Benchmark Suites
DMI includes several benchmark suites to evaluate mutual information estimators:

- [✅] `self-consistency`
- [✅]`stair_case`
- [✅]`bmi`

These benchmarks ensure robust evaluation of estimators across various scenarios and data distributions.

