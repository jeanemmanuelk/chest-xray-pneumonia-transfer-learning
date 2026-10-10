# Chest X-ray pneumonia detection: CNN from scratch vs ResNet18

A controlled comparison of three models on chest X-rays: a small CNN trained from scratch, a ResNet18 with a frozen backbone, and a ResNet18 with its last block fine-tuned. This is the medical follow-up to my [butterfly project](https://github.com/jeanemmanuelk/cnn-vs-transfer-learning-for-butterfly-image-classification).

## Results

All numbers are on the official test set. Sensitivity and specificity use a 0.5 threshold unless noted. Each model was trained once.

| Model | AUROC (95% CI) | Sensitivity | Specificity |
|---|---|---|---|
| CNN from scratch | 0.934 (0.908 to 0.956) | 0.985 | 0.697 |
| ResNet18, frozen backbone | 0.983 (0.974 to 0.990) | 0.982 | 0.846 |
| ResNet18, layer4 fine-tuned | 0.981 (0.972 to 0.989) | 0.992 | 0.782 |

Two extra experiments on the fine-tuned model:

| Variant | Sensitivity | Specificity |
|---|---|---|
| Without class weights (AUROC 0.988) | 1.000 | 0.650 |
| Threshold 0.683, chosen on the validation set to keep sensitivity at 95% or more | 0.987 | 0.838 |

> Pretrained ResNet18 clearly beat a CNN from scratch, but fine-tuning did not add anything over freezing, and the threshold mattered more than I expected.

## Why this project

In the butterfly project I learned that fine-tuning a pretrained network beats training from scratch on a small dataset. Medical images are a harder test of that idea. Accuracy is not enough, mistakes cost more in one direction, and the data is usually imbalanced. I wanted to see how the same comparison holds up when I use the metrics a clinical question would actually need.

## Dataset

I'm using PneumoniaMNIST from [MedMNIST](https://medmnist.com/), which is built from a public set of pediatric chest X-rays. Label 0 is normal and label 1 is pneumonia. It comes with an official train / validation / test split, and I load it at 224x224 so it works with ResNet18. The data is released under CC BY 4.0.

| Split | Images | Normal | Pneumonia |
|---|---|---|---|
| Train | 4,708 | 1,214 | 3,494 |
| Validation | 524 | 135 | 389 |
| Test | 624 | 234 | 390 |

## Approach

1. Load the data, check the class counts, and look at example images.
2. Handle class imbalance with class weights in the loss.
3. Train the three models. For each one I keep the epoch with the best validation AUROC, so the test set is never used to choose anything.
4. Evaluate once on the test set with AUROC (plus a bootstrap confidence interval), sensitivity and specificity.
5. Train the fine-tuned model again without class weights to see what the weights change.
6. Pick a threshold on the validation set that catches at least 95% of pneumonia cases, and check it on the test set.
7. Use Grad-CAM to look at what the model focuses on, including its mistakes.

## What I learned

1. **Starting from a pretrained model helped a lot.** The ResNet18 models got an AUROC of about 0.98, and the CNN I built from scratch got 0.934. The pretrained model had already learned basic image features, so it needed less from my small dataset.

2. **Fine-tuning did not beat freezing.** The frozen ResNet got 0.983 and the fine-tuned one got 0.981. The confidence intervals overlap, so I can't say either one is better. I only ran each model once, so a different random seed could change the order. My guess is that with this little data, training more layers doesn't add much, but I didn't test that.

3. **Accuracy alone can hide problems.** Every model had high sensitivity (0.98 or higher) but lower specificity (0.70 to 0.85). In other words, they rarely missed pneumonia but often called a healthy chest pneumonia. This matters because missing a sick patient is usually worse than a false alarm. All 3 mistakes I looked at with Grad-CAM were normal X-rays predicted as pneumonia.

4. **Class weights and the threshold change the trade-off.** Without class weights, sensitivity hit 1.0 but specificity dropped to 0.65. Picking the threshold on the validation set (0.683) kept sensitivity at 0.987 and raised specificity to 0.838. So the 0.5 default is not always the best choice.

5. **Validation scores can be too optimistic.** Validation AUROC was about 0.999, but test was about 0.98. The test set also has a different mix of normal and pneumonia images, and there are only 234 normal test images, so the specificity numbers are shaky. I would not trust the validation score by itself.

6. **Grad-CAM is useful, but it is not proof.** The heat was mostly over the lungs, which is a good sign. But it does not prove the model "reasons" like a doctor, and one image had heat near the top edge.

7. **This is a learning project.** It uses small 224 pixel images, one run per model, and one dataset. It is not a clinical tool.

## Running it

```bash
pip install -r requirements.txt
jupyter notebook pneumonia_xray_classifier.ipynb
```

A GPU helps a lot. It also runs on Google Colab with a free GPU runtime (upload the notebook and run all cells, the first cell installs `medmnist`). The dataset downloads automatically the first time.

## Notes

- This is a learning project, not a clinical tool. It should never be used to make medical decisions.
- The images are from children and come from a single source, so I would not expect the results to carry over to adult X-rays or other hospitals.
- The test set is small, which is why I report confidence intervals. Small differences between models may just be noise. Specificity rests on only 234 normal images, so it is the noisiest number.
- The class mix is different in the test set (about 38% normal) than in train and validation (about 26% normal).
- MedMNIST does not include patient IDs, so I can't check whether the same patient appears in more than one split. That could make the results look better than they really are.
