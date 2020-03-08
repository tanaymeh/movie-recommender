# Movie Recommendation System using Stacked AutoEncoders in PyTorch

<figure>
    <img src="https://miro.medium.com/max/3148/1*44eDEuZBEsmG_TCAKRI3Kw@2x.png" alt="SAE" title="A Stacked AutoEncoder" />
</figure>

Movie Recommendation system made using Stacked AutoEncoders. This SAE is trained on local machine on 100K values dataset due to low available computation power. Training on 1 Million values dataset is more desirable on Cloud VMs / Azure Notebooks / Google Colab / Kaggle Kernels.
Other Details on internal working and code explanation is provided in the respective cells in the notebook.
Code exports in form of Python (.py), HTML (.html) and LaTeX (.tex) files in the [exports/](https://github.com/heytanay/movie-recommender/tree/master/exports) folder.
Both 100K ratings and 1 Million ratings dataset is in the [data/](https://github.com/heytanay/movie-recommender/tree/master/data)
folder.

## Installation Instructions
### PyTorch
For *Windows* with Ananconda / Miniconda and CUDA v10.1

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

For *Linux* with Ananconda / Miniconda and CUDA v10.1

```
sudo conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

For more instructions on installing PyTorch, visit 
[PyTorch](https://pytorch.org/)

### Other Libraries
For pip;
```
pip3 install matplotlib numpy pandas 
```

For Miniconda (Anaconda comes with them);
```
conda install matplotlib numpy pandas
```

#### Disclaimer
Movie Ratings Dataset is Open-source and is "free to use". No rights have been infringed

```
Author: Tanay Mehta
Github: @heytanay
LinkedIn: https://linkedin.com/in/tanaymehta28
Mail: heyytanay@gmail.com
```
