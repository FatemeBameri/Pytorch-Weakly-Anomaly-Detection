# Weakly_Supervised_Anomaly
In this research, we have two main parts: feature extraction and anomaly detection. In the feature extraction phase, we use four architetures with thier combinations so that we consider nine types of feature network. In the anomaly detection phase, we suppose that anomaly and normal data have different distributions. Therefore, we first estimate appropriate distribution for them and then calculate the difference of distributions between two classes, anomaly and normal, in order to detect anomaly or normal for each video input.We evaluate the proposed method on the datasets: ShanghaiTech, Chuk Avenue, and UCSD Ped2.

## Feature Extraction
<div> &#9679; [Video swin Transformer] (https://github.com/tianyu0207/RTFM) </div>
To install `pytorch-generative`, clone the repository and install the requirements:

```
git clone https://www.github.com/EugenHotaj/pytorch-generative
cd pytorch-generative
pip install -r requirements.txt
```
## Anomaly Detection

Citation
===

```text
@article{dalmasso2020cdetools,
       author = {{Dalmasso}, N. and {Pospisil}, T. and {Lee}, A.~B. and {Izbicki}, R. and
         {Freeman}, P.~E. and {Malz}, A.~I.},
        title = "{Conditional density estimation tools in python and R with applications to photometric redshifts and likelihood-free cosmological inference}",
      journal = {Astronomy and Computing},
         year = 2020,
        month = jan,
       volume = {30},
          eid = {100362},
        pages = {100362},
          doi = {10.1016/j.ascom.2019.100362}
}
```

## <a name="9"></a> Acknowledgement
Our codebase is built based on [RTFM](https://github.com/tianyu0207/RTFM). We really appreciate the authors for the nicely organized code!
