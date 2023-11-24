# Weakly Supervised Aomaly Detection
In this research, we have two main parts: feature extraction and anomaly detection. In the feature extraction phase, we use four architetures with thier combinations so that we consider nine types of feature network. In the anomaly detection phase, we suppose that anomaly and normal data have different distributions. Therefore, we first estimate appropriate distribution for them and then calculate the difference of distributions between two classes, anomaly and normal, in order to detect anomaly or normal for each video input.We evaluate the proposed method on the datasets: ShanghaiTech, Chuk Avenue, and UCSD Ped2.

## Feature Extraction
Feature extraction networks applying in the paper:
* Swin [(paper)](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html)
* ResNet3D [(paper)](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html)
* S3D [(paper)](https://openaccess.thecvf.com/content_ECCV_2018/html/Saining_Xie_Rethinking_Spatiotemporal_Feature_ECCV_2018_paper.html)
* I3D [(paper)](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)
* Combination of swin + ResNet (R3D version)
* Combination of swin + ResNet (R(2+1)D version)
* Combination of swin + S3D
* Combination of swin + I3D
## Downloading Features
To extract features we started with the implementation of the [I3D Feature Extraction](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet) repository and then modified it to apply the desired architectures. So, you can download our extracted features including video swin transformer, ResNet3D, S3D, I3D, and combined networks for each datasets from the following links:
* UCSD Ped2 [(link)](https://drive.google.com/file/d/1EUgplJ9Eqt-VdsqLm9GJ35TMQYzZR0n1/view?usp=sharing)
* Chuk Avenue [(link)](https://drive.google.com/file/d/1KEXjiIsGfvsdu9Z05Yt8Cc3qZxByhcsO/view?usp=sharing)
* ShanghaiTech part1 [(link)](https://drive.google.com/file/d/1kOp-vbkK8mH8tt4FhuUD4nPN455M6qaI/view?usp=sharing)
* ShanghaiTech part2 [(link)]()

## Anomaly Detection
To detect anomaly, we first estimate probability distribution function for each class. After that, based on multi instance learning method and difference of the distribution in the training model, label of each videos can correctely be detected. For starting of implementation in this part, we use the [RTFM](https://github.com/tianyu0207/RTFM) repository.
