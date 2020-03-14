from datasets import *
import tensorflow as tf

pc = PersonalityCaptions("./data")
dl = DatasetLoader(pc)

dl.encode_images("train")
dl.encode_images("test")
dl.encode_images("val")

# import tensorflow as tf
# resnet = tf.keras.applications.resnet_v2.ResNet101V2(include_top=False, weights="imagenet")
# resnet.preprocess_input(inp)




