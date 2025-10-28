# Smearly

## Problem Statement

# The Global Cervical Cancer Crisis

Cervical cancer is a major public health challenge, with more than 600,000 new cases per year worldwide. The figures are even more alarming in developing countries, as 85% of deaths related to cervical cancer occur in these countries.

### Additional Key Facts:
- **90% of cervical cancer deaths** occur in low- and middle-income countries
- In sub-Saharan Africa, cervical cancer is the **leading cause of cancer death** among women
- **Age-standardized incidence rates** are up to 6 times higher in developing countries compared to developed nations
- Only **5% of women** in low-income countries have access to cervical cancer screening, compared to 85% in high-income countries
- The **5-year survival rate** drops from 92% in developed countries to less than 50% in many low-resource settings
- **1 woman dies every 2 minutes** from cervical cancer globally, with the majority in developing countries
- Current screening methods have **60%+ error rates** and require specialized infrastructure often unavailable in rural areas
- The **economic burden** includes not just healthcare costs but loss of productive years, as cervical cancer typically affects women aged 35-50

### The Opportunity:
- Cervical cancer is **99% preventable** with proper screening and early detection
- AI-powered screening could reduce costs by up to **70%** while improving accuracy
- Mobile health solutions could reach the **2.6 billion women** currently without access to screening

It is for these reasons that the use of artificial intelligence in screening seems relevant.

# Solution

Smearly uses deep learning to automatically classify Pap smear cell images, providing accurate and cost-effective cervical cancer screening that can be deployed in low-resource settings.

## Dataset

Our model was trained and evaluated using the **ISBI 2025 Pap Smear Cell Classification Challenge (PS3C)** dataset from Kaggle, which includes:

### Classes:
- **healthy**: Normal cervical cells
- **unhealthy**: Abnormal cervical cells indicating potential cancer
- **rubbish**: Poor quality images unsuitable for diagnosis
- **bothcells**: Images containing both healthy and unhealthy cells

### Data Structure:
- Training images organized by class labels
- Imbalanced dataset reflecting real-world clinical conditions
- High-resolution Pap smear cell images

## Model Architecture

### EfficientNetB0 with Custom Classification Head
```python
Base Model: EfficientNetB0 (pre-trained on ImageNet)
├── GlobalAveragePooling2D
├── Dense(128, activation='relu')
├── Dropout(0.5)
├── Dense(64, activation='relu') 
├── Dense(32, activation='relu')
└── Dense(3, activation='softmax')  # 3-class output

Kaggle challenge: https://www.kaggle.com/competitions/pap-smear-cell-classification-challenge/data 


Sources: 
https://pmc.ncbi.nlm.nih.gov/articles/PMC7400218/ 
https://www.afro.who.int/sites/default/files/2024-03 Status%20of%20the%20Cervical%20Cancer%20Elimination%20Initiative%20%20in%20WHO%20African%20Region_0.pdf 