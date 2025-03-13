from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer
from smearly.ml_logic.preprocessing import resize_pad_image_tf, image_file_to_tf
import matplotlib.pyplot as plt


img = image_file_to_tf('../raw_data/all/healthy/isbi2025_ps3c_train_image_03449.png')

# Create a custom transformer using FunctionTransformer
resize_pad_transformer = FunctionTransformer(resize_pad_image_tf)

# Create a pipeline with our custom preprocessing function and a model
pipeline = make_pipeline(resize_pad_transformer)


new_img = pipeline.transform(img)
plt.imshow(new_img);
