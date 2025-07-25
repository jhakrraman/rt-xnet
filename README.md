# RT-X Net: RGB-Thermal cross attention network for Low-Light Image Enhancement       
[Raman Jha](https://jhakrraman.github.io/), [Adithya Lenka](https://www.linkedin.com/in/adithya-lenka-7517b0217/?originalSubdomain=in), [Mani Ramanagopal](https://www.linkedin.com/in/srmanikandasriram/), [Aswin Sankaranarayanan](https://www.ece.cmu.edu/directory/bios/sankaranarayanan-aswin.html), [Kaushik Mitra](https://www.ee.iitm.ac.in/kmitra/)

---

[Paper](https://arxiv.org/abs/2505.24705), [Supplementary Material](https://sigport.org/sites/default/files/docs/Supplementary_11.pdf), [Web Page](https://sites.google.com/view/rt-xnet/home)

Our paper has been featured at the [Awesome Low Light Image Enhancement Papers List](https://github.com/zhihongz/awesome-low-light-image-enhancement).

---

## Model Architecture

![ ](https://github.com/jhakrraman/rt-xnet/blob/master/imgs/model_architecture.png)



## Qualitative Results:

![](https://github.com/jhakrraman/rt-xnet/blob/master/imgs/qualitative_results.png)



## V-TIEE Dataset

![Real-world V-TIEE Dataset: Co-located Visible-Thermal Image Pairs for HDR and Low-light Vision Research](https://github.com/jhakrraman/rt-xnet/blob/master/imgs/v-tiee_dataset.png)



### High-gain Multi-exposure Visible-Thermal Image Pairs for Test Input Scenes
---
![](https://github.com/jhakrraman/rt-xnet/blob/master/imgs/high_gain_v-tiee.png)

### Low-gain Multi-exposure Visible-Thermal Image Pairs for Reference Scenes
---
![](https://github.com/jhakrraman/rt-xnet/blob/master/imgs/low_gain_v-tiee.png)



## 1. Environment Creation

- **Make Conda Environment**

```
conda create -n rtx-net python=3.7
conda activate rtx-net
```

- **Install Dependencies**

```
conda install pytorch=1.11 torchvision cudatoolkit=11.3 -c pytorch

pip install matplotlib scikit-learn scikit-image opencv-python yacs joblib natsort h5py tqdm tensorboard

pip install einops gdown addict future lmdb numpy pyyaml requests scipy yapf lpips
```

- **Install BasicSR**

```
python setup.py develop --no_cuda_ext
```


## 2. Dataset Preparation

Download the **LLVIP dataset** here.     
[Google Drive](https://drive.google.com/file/d/1XOfve52_4zTqPBaxCBhnV74GvgIMe0rn/view?usp=sharing)
[Hugging Face](https://huggingface.co/datasets/jhakrraman/LLVIP)

You can also download the dataset directly from Hugging Face using these commands.

```
git lfs install
git clone https://huggingface.co/datasets/jhakrraman/LLVIP
```

After the download, please place it in ./data/LLVIP

Download the **V-TIEE dataset** here.   
[Google Drive](https://drive.google.com/file/d/1ce5Dq0v4ORcDzoqTxRNF602LWFfkvlpg/view?usp=sharing)
[Hugging Face](https://huggingface.co/datasets/jhakrraman/V-TIEE)

Similar to the LLVIP dataset, you can also download the dataset directly from Hugging Face using these commands.

```
git lfs install
git clone https://huggingface.co/datasets/jhakrraman/V-TIEE
```

The proposed dataset can either be used for testing the purpose of real-world low-light image enhancement or HDR image generation.    
To test the RT-X net on the V-TIEE dataset, please make the V-TIEE dataset folder structure similar to the LLVIP dataset, and choose different types of images with various noises and illumination in real-time.

## 3. Training

To perform training on the RT-X Net, use the following command.

```
# activate the environment
conda activate rtx-net

# LLVIP
python3 basicsr/train.py --opt Options/RTxNet_LLVIP.yml
```

## 4. Testing

Download our pre-trained model of the RT-X Net from [Google Drive](https://drive.google.com/file/d/1dPrlnOQpQqzih-1F-AyKlvFIPg5gQLRF/view?usp=sharing) and [Hugging Face](https://huggingface.co/jhakrraman/RT-X_Net).    
Put them in the folder pretrained_weights.

```
# activate the environment
conda activate rtx-net

# LLVIP
python3 Enhancement/test_from_dataset.py --opt Options/RTxNet_LLVIP.yml --weights pretrained_weights/LLVIP_best.pth --dataset LLVIP
```


Our work is based on [Retinexformer](https://github.com/caiyuanhao1998/Retinexformer). We thank the authors for releasing their code.

## If you find this code or the dataset useful for you, please cite  

``` 
@misc{jha2025rtxnetrgbthermalcross,  
      title={RT-X Net: RGB-Thermal cross attention network for Low-Light Image Enhancement},   
      author={Raman Jha and Adithya Lenka and Mani Ramanagopal and Aswin Sankaranarayanan and Kaushik Mitra},  
      year={2025},  
      eprint={2505.24705},  
      archivePrefix={arXiv},  
      primaryClass={cs.CV},  
      url={https://arxiv.org/abs/2505.24705},   
}
```
