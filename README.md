# Weakly Supervised Aomaly Detection
Video surveillance systems are commonly employed to monitor activities and ensure the safety and security of various environments. Integrating anomaly detection into these systems enables the identification of atypical or suspicious activities. This paper proposes a novel approach for video anomaly detection that utilizes ensemble learning with multiple base models in a weakly supervised setting. The proposed method consists of a two-stage framework. In the first stage, spatiotemporal features are extracted from video data using 3D deep spatio-temporal networks, followed by a multi-scale network to further enhance feature representation. Anomalous events are then detected by analyzing discrepancies in probabilistic distributions within a graph-based structure, incorporating a multi-instance learning approach. In the second stage, the anomaly detection process is refined through ensemble learning techniques, specifically stacking and weighted averaging, to optimize the overall model performance. The effectiveness of the proposed framework is validated through extensive quantitative and qualitative experiments conducted on four benchmark datasets: UCF-Crime, ShanghaiTech, CHUK Avenue, and UCSD Ped2. The method achieves frame-level AUC scores of 97.89% on ShanghaiTech, 95.97% on CHUK Avenue, 97.38% on UCSD Ped2, and 80.86% on UCF-Crime, demonstrating competitive performance relative to state-of-the-art approaches. These results demonstrate the robustness and effectiveness of the proposed framework in diverse video anomaly detection scenarios. 
## Feature Extraction
Feature extraction networks applying in the paper:
* Swin [(paper)](https://openaccess.thecvf.com/content/CVPR2022/html/Liu_Video_Swin_Transformer_CVPR_2022_paper.html)
* ResNet3D [(paper)](https://openaccess.thecvf.com/content_cvpr_2018/html/Tran_A_Closer_Look_CVPR_2018_paper.html)
* I3D [(paper)](https://openaccess.thecvf.com/content_cvpr_2017/html/Carreira_Quo_Vadis_Action_CVPR_2017_paper.html)
## Downloading Features
To extract features we started with the implementation of the [I3D Feature Extraction](https://github.com/GowthamGottimukkala/I3D_Feature_Extraction_resnet) repository and then modified it to apply the desired architectures. So, you can download our extracted features including video swin transformer, ResNet3D, S3D, I3D, and combined networks for each datasets from the following links:
* UCSD Ped2 [(link)](https://drive.google.com/file/d/1EUgplJ9Eqt-VdsqLm9GJ35TMQYzZR0n1/view?usp=sharing)
* Chuk Avenue [(link)](https://drive.google.com/file/d/1KEXjiIsGfvsdu9Z05Yt8Cc3qZxByhcsO/view?usp=sharing)
* ShanghaiTech part1 [(link)](https://drive.google.com/file/d/1kOp-vbkK8mH8tt4FhuUD4nPN455M6qaI/view?usp=sharing)
* ShanghaiTech part2 [(link)](https://drive.google.com/file/d/1a1Y9FZfq_E1pSAaLYo8TxLwHDsDWjZxm/view?usp=sharing)

## Anomaly Detection
The proposed framework consists of two stages: the base stage and the ensemble stage. In the base stage, the proposed method includes five main stages: feature extraction, attention mechanism, probability density estimation, computation of feature differences, and classification approach . For starting of implementation in this part, we used the [RTFM](https://github.com/tianyu0207/RTFM) repository.
## Running the code
Steps for running the code:
1. Go to the Anomaly Detection folder.
2. Download the features of your desired dataset.
3. Change the default_setting.py based on information which is compatible with your desired dataset.
4. Run main.py
