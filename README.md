# Weakly_Supervised_Anomaly
In this research, we have two main parts: feature extraction and anomaly detection. In the feature extraction phase, we use four architetures with thier combinations so that we consider nine types of feature network. In the anomaly detection phase, we suppose that anomaly and normal data have different distributions. Therefore, we first estimate appropriate distribution for them and then calculate the difference of distributions between two classes, anomaly and normal, in order to detect anomaly or normal for each video input.We evaluate the proposed method on the datasets: ShanghaiTech, Chuk Avenue, and UCSD Ped2.

## Feature Extraction
Feature extraction networks that applied in the paper:
* Swin--> Paper: [Video Swin Transformer](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html)
* ResNet3D --> Papwe:[A Closer Look at Spatiotemporal Convolutions for Action Recognition](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html)
- ReseNet3D
- S3D
- I3D
- Combination of video swin transformer with ResNet3D (in basic version)
- Combination of video swin transformer with ResNet3D (in (2+1) version)
- Combination of video swin transformer with St3D
- Combination of video swin transformer with I3D

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
