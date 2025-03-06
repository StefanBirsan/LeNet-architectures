# Summer Practice Research: LeNet CNN Architecture Performance on Traffic Sign Databases

## Overview
During my summer practice, I conducted research on the performance of **LeNet-5 Convolutional Neural Network (CNN) architectures** when applied to various **traffic sign databases** from around the world. This study aimed to analyze the effectiveness of LeNet and LeNet-5 in recognizing and classifying traffic signs.

## Research Goals
- Evaluate the **accuracy and efficiency** of LeNet on multiple traffic sign datasets.
- Compare performance metrics such as **precision, recall, and inference time**.
- Identify **challenges and limitations** when applying LeNet to traffic sign recognition.

## Methodology
1. **Dataset Selection:**
   - Collected and analyzed public traffic sign databases, including:
     - German Traffic Sign Recognition Benchmark (**GTSRB**)
     - Belgian Traffic Sign Dataset (**BTSC**)
     - Chinese Traffic Sign Database (**CTSD**)

2. **Preprocessing:**
   - Resized images to match LeNet's input size.
   - Normalized pixel values for better model performance.
   - Augmented data where necessary to improve generalization.

3. **Model Training & Evaluation:**
   - Implemented LeNet CNN using **TensorFlow/Keras**.
   - Trained the model on a bigger dataset that combined all the datasets from above.
   - Evaluated **accuracy, loss, confusion matrix, and misclassification rates**.

4. **Analysis & Comparisons:**
   - Compared LeNet's performance across datasets.
   - Identified dataset-specific challenges, such as low-resolution images, occlusions, or class imbalances.
   - Explored ways to improve LeNet (**data augmentation, fine-tuning, additional layers**).

## Key Findings
- LeNet performed **remarkably well on structured datasets** like GTSRB but struggled with **datasets containing severe distortions or lighting variations**.
- **Class imbalance** in some datasets affected model performance, necessitating **oversampling/undersampling techniques**.
- **Augmenting data with rotation, contrast adjustment, and noise addition** significantly improved recognition accuracy.
- **Modifying LeNet** by adding batch normalization and dropout layers could have helped in reducing overfitting and improved generalization.

## Conclusion & Future Work
LeNet remains a **strong baseline model** for traffic sign recognition but requires **enhancements** to handle complex real-world scenarios. Future research could focus on:
- Testing LeNet with **real-time traffic sign recognition on embedded systems**.

## Acknowledgments
I would like to thank my professor and colleague who provided guidance and support during this research. Their insights helped shape the direction and execution of this study.

---
If you have any questions or suggestions regarding this research, feel free to reach out!

