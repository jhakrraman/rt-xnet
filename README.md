# RT-X Net: RGB-Thermal cross attention network for Low-Light Image Enhancement       
[Raman Jha](https://jhakrraman.github.io/), [Adithya Lenka](https://www.linkedin.com/in/adithya-lenka-7517b0217/?originalSubdomain=in), [Mani Ramanagopal](https://www.linkedin.com/in/srmanikandasriram/), [Aswin Sankaranarayanan](https://www.ece.cmu.edu/directory/bios/sankaranarayanan-aswin.html), [Kaushik Mitra](https://www.ee.iitm.ac.in/kmitra/)

---

[Paper](https://arxiv.org/abs/2505.24705), [Supplementary Material](https://sigport.org/sites/default/files/docs/Supplementary_11.pdf)

---

## Model Architecture
![ ](https://github.com/jhakrraman/rt-xnet/blob/master/imgs/model_architecture.png)

---

## Qualitative Results:

![](https://github.com/jhakrraman/rt-xnet/blob/master/imgs/qualitative_results.png)

---

## V-TIEE Dataset

![Real-world V-TIEE Dataset: Co-located Visible-Thermal Image Pairs for HDR and Low-light Vision Research](https://github.com/jhakrraman/rt-xnet/blob/master/imgs/v-tiee_dataset.png)

---

### High-gain Multi-exposure Visible-Thermal Image Pairs for Test Input Scenes

![](https://github.com/jhakrraman/rt-xnet/blob/master/imgs/high_gain_v-tiee.png)

---

### Low-gain Multi-exposure Visible-Thermal Image Pairs for Reference Scenes

![](https://github.com/jhakrraman/rt-xnet/blob/master/imgs/low_gain_v-tiee.png)

---

# 1. Create Environment

We suggest you use pytorch 1.11 to re-implement the results in our ICCV 2023 paper and pytorch 2 to re-implement the results in NTIRE 2024 Challenge because pytorch 2 can save more memory in mix-precision training.

## 1.1 Install the environment with Pytorch 1.11

- **Make Conda Environment**

    conda create -n Retinexformer python=3.7  
    conda activate Retinexformer  

- **Install Dependencies**

    conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch  

    pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm  

    pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips  

- **Install BasicSR**

    python setup.py develop --no_cuda_ext  

## 1.2 Install the environment with Pytorch 2

- **Make Conda Environment**

    conda create -n torch2 python=3.9 -y  
    conda activate torch2  

- **Install Dependencies**

    conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia  

    pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm  

    pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips thop timm  

- **Install BasicSR**

    python setup.py develop --no_cuda_ext  



## If you find this code or the dataset useful for you, please cite

<pre> 
@misc{jha2025rtxnetrgbthermalcross,  
      title={RT-X Net: RGB-Thermal cross attention network for Low-Light Image Enhancement},   
      author={Raman Jha and Adithya Lenka and Mani Ramanagopal and Aswin Sankaranarayanan and Kaushik Mitra},  
      year={2025},  
      eprint={2505.24705},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV},  
      url={https://arxiv.org/abs/2505.24705},   
}
</pre>

