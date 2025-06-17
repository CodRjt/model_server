from flask import Flask,request
import tensorflow as tf
from transformers import TFViTForImageClassification
from tensorflow.keras.utils import register_keras_serializable
import numpy as np
import cv2 as cv
import gdown
# import tf.keras.utils.register_keras_serializable as dd
# model_path=hf_hub_download(repo_id="SpaceShark/my-keras-model",
#                            filename="model.keras"
#                            )
url='https://drive.google.com/uc?id=1m2VZ6uFsetNOSxD9mxHwa09weN89Hs8h'
output='model.keras'
gdown.download(url,output,quiet=False)
def preprocess_for_vit(image):
    h,w=image.shape[:2]
    if (h,w)!=(224,224):
        image = tf.image.resize(image, (224, 224))
    image=tf.cast(image,tf.float32)
    image=(image/255)*2-1
    return image
@register_keras_serializable()
class ViTClassifier(tf.keras.Model):
    def __init__(self, model_name, num_labels, id2label=None, label2id=None,**kwargs):
        super().__init__(**kwargs)
        self.hf_model = TFViTForImageClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True
        )
        self.model_name=model_name
        self.num_labels=num_labels
        self.id2label=id2label
        self.label2id=label2id
        # self.__setattr__('hf_model', hf_model)
        self.hf_model.trainable = True
    # @property
    # def id2label(self):
    #     return self.hf_model.config.id2label
    # @id2label.setter
    # def id2label(self, value):
    #     self.hf_model.config.id2label = value
    @property
    def trainable_weights(self):
       
        # Ensure hf_model exists and is built before trying to access its weights
        if hasattr(self, 'hf_model') and self.hf_model.built:
            return self.hf_model.trainable_weights
        return []

    @property
    def non_trainable_weights(self):
      
        # Ensure hf_model exists and is built before trying to access its weights
        if hasattr(self, 'hf_model') and self.hf_model.built:
            return self.hf_model.non_trainable_weights
        return []

    def call(self, images, training=False):
        x = tf.transpose(images, [0, 3, 1, 2])
        outputs = self.hf_model(pixel_values=x, training=training)
        logits = outputs.logits
        return logits
    def get_config(self):
        config=super().get_config()
        config.update({
            "model_name":self.model_name,
            "num_labels":self.num_labels,
            "id2label":self.id2label,
            "label2id":self.label2id
        })
        return config
    @classmethod
    def from_config(cls,config):
        return cls(**config)
        
    def _get_regularization_losses(self):
        # Override Keras behavior to prevent AttributeError on .regularizer
        return []

model=tf.keras.models.load_model(
            'model.keras',
            custom_objects={"ViTClassifier":ViTClassifier},
            compile=False)
# model.id2label=dict(model.id2label)
# print(type(model.hf_model.config.id2label))
# print(model.id2label)
app=Flask(__name__)
@app.route("/")
def primary():
    return "This model is not meant to be accessed via browser"

@app.route("/prediction",methods=['POST'])
def inference():
    if model is None:
        return {"error": "Model not loaded. Check server logs."}, 500
    try:
        file=request.files['file']
        img=np.frombuffer(file.read(),np.uint8)
        img_cv=cv.imdecode(img,cv.IMREAD_COLOR)
        img_rgb=cv.cvtColor(img_cv,cv.COLOR_BGR2RGB)
        img=preprocess_for_vit(img_rgb)
        img=tf.expand_dims(img,axis=0)
        logits=model(img,training=False)

        probs=tf.nn.softmax(logits,axis=1)
        id2label=dict(model.hf_model.config.id2label)
        print(id2label)
        print(type(id2label))
        pred_class_idx=tf.argmax(probs,axis=1).numpy()[0]
        # print("Type of id2label:", type(id2label))
        # print("id2label:", id2label)
        # print("Type of pred_class_idx:", type(pred_class_idx))
        # print("pred_class_idx:", pred_class_idx)

        pred_class=id2label.get(str(pred_class_idx))
        confidence=tf.reduce_max(probs,axis=1).numpy()[0]
        print(f"pred_class (index): {pred_class}")
        print(f"confidence: {confidence}")
        # print(f"Label: {label}")

        return {"pred_class":pred_class,"confidence":float(confidence)}
        # return {"none": "none"}
    except Exception as e:
        return {"error":str(e)},400
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000,debug=False)
    