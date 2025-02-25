# Deep Model Predictive Control For Visual Servoing

This repository is the official implementation of **Deep Model Predictive Control For Visual Servoing** which was accepted at [CoRL-2020](https://www.robot-learning.org/program/accepted-papers).

<font size="+1">[Paper](https://corlconf.github.io/corl2020/paper_448/) | [Project Page](https://robotics.iiit.ac.in/publications/2020/deep-mpc-for-visual-servoing/project-page.html)</font>

To run:

* `git clone https://github.com/pranjali-pathre/deepMPCVS`
* `cd deepMPCVS`
* `scp -r path_to_checkpoint ./`
* `conda activate deepmpcvs`
* `python run.py ./skokloster-castle 0 0 0 1 0 0 0 1 1 0`
## Citation

```
@article{pushkalCORL,
  Title = {Deep Model Predictive Control For Visual Servoing},
  Author={P. Katara and Y V S. Harish and H. Pandya and A. Gupta and A. Sanchawala and G. Kumar and K. M. Krishna and B. Bhowmick},
  Journal={Accepted at 4th Annual Conference on Robot Learning, CoRL 2020},
  year={2020},
  URL = {https://corlconf.github.io/corl2020/paper_448/}
}
```
