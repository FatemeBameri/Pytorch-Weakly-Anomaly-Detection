# Weakly_Supervised_Anomaly
In this research, we have two main parts: feature extraction and anomaly detection. In the feature extraction phase, we use four architetures with thier combinations so that we consider nine types of feature network. In the anomaly detection phase, we suppose that anomaly and normal data have different distributions. Therefore, we first estimate appropriate distribution for them and then calculate the difference of distributions between two classes, anomaly and normal, in order to detect anomaly or normal for each video input.We evaluate the proposed method on the datasets: ShanghaiTech, Chuk Avenue, and UCSD Ped2.

## Feature Extraction
Feature extraction networks that applied in the paper:
* Swin [[paper]](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html)
* ResNet3D [[paper]](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html)
* S3D [[paper]](https://openaccess.thecvf.com/content_ECCV_2018/html/Saining_Xie_Rethinking_Spatiotemporal_Feature_ECCV_2018_paper.html)
* I3D [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)
* Combination of swin + ResNet (R3D version)
* Combination of swin + ResNet (R(2+1)D version)
* Combination of swin + S3D
* Combination of swin + I3D

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
