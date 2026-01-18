# Enhanced RGB-D Saliency Detection through Cross-Modal Fusion and Edge Refinement (CENet)
This repository is the official implementation of the paper: *"Enhanced RGB-D Saliency Detection through Cross-Modal Fusion and Edge Refinement"*, currently submitted to *The Visual Computer*.
---
📢 > **Note:** This research as been submitted to  **The Visual Computer**.
---

## Overiew
In this paper, we propose a salient object detection method for RGB-D image pairs, which achieves competitive performance across datasets containing various challenging scenarios. In addition to conducting experiments on classical datasets, we perform generalization experiments to test our method on RGB-T datasets.
<img width="962" height="523" alt="img" src="https://github.com/user-attachments/assets/b0d60bd6-5cad-489b-8afc-187df1c9b34b" />
<img width="1386" height="285" alt="img_1" src="https://github.com/user-attachments/assets/1d89ca91-0259-424f-9cf6-b2d8076d829f" />
---
## 🛠️ Installation & Dependencies

To ensure reproducibility, please set up the environment as follows:

* **OS:** Ubuntu 20.04 or Windows 10/11
* **Python:** 3.8+
* **Framework:** PyTorch >= 1.12, torchvision
* **Dependencies:** `pip install -r requirements.txt`
---

## ▶ RUN 
- Model outputs, link to assessment tool is [PySODEvalToolkit](https://github.com/lartpang/PySODEvalToolkit).
- The comparative experimental data are available at [link]( https://pan.baidu.com/s/1wMFuN6TTMXtdaf16gGu7ZQ?pwd=6z6t) code: 6z6t
- The dataset is available at [datasets](https://pan.baidu.com/s/1gQbYM3G0xsBPx40cpsEv-g?pwd=22bu) code: 22bu
- Download the RGB-D SOD datasets (e.g., DUT, NJU2K, NLPR) and place them in the data/ directory.
- For Train:`python train_CENet.py`
- For Test:`python test_CENet.py `

---
## 📄 Citation
If you find this work helpful, please cite our paper:

`@article{CENet2026,
  title={Enhanced RGB-D Saliency Detection through Cross-Modal Fusion and Edge Refinement},
  author={DeGuo Yang ,Xinyi Zhang and JieYan},
  journal={The Visual Computer (Submitted)},
  year={2026}
}`






