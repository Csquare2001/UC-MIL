## UC-MIL: A novel uncertainty-aware causal multiple instance learning for CT image-based mutation prediction
A preliminary implementation of UC-MIL.
# 1. Framework
Figure 1 presents the overall pipeline of UC-MIL. The proposed framework includes uncertainty-aware self-learning (USL) strategy and causality-informed prediction enhancement (CPE) strategy. L_CI and L_NC denote the causal invariance constraint and noncausal counterfactual constraint, respectively.
![UC-MIL](.\Figs\UC-MIL.png "Figure 1. The overall pipeline of UC-MIL")
*<p align="center">Figure 1. The overall pipeline of UC-MIL</p>*

# 2. Environmental Requirements
To run the codes, the following dependencies are required:
+ python 3.9
+ PyTorch 2.0.1
+ cuda 11.3
+ ...

# 3. Files Descriptions
```plaintext
©¸©¤©¤ UC-MIL/  
      ©À©¤©¤ train.py                     # Code to train the model
      ©À©¤©¤ datasets.py                  # 
      ©À©¤©¤ 
      ©À©¤©¤ utils.py                     # Some supporting functions
      ©¸©¤©¤ models/
        ©À©¤©¤ 
```