import numpy as np
from detection import detect_and_crop
import classification
import cv2 as cv

classifier = classification.InferenceModel(model_path=classification.model_path,
                                           labels=classification.labels)

def run_detection(sem_img, **args):
    crops, props, mask, display = detect_and_crop(sem_img,
                                    crop_h=args.get('crop_h', None),
                                    threshold=args.get('threshold', 127))

    return crops, props, mask, display

def run_classification(images):
    clasifications = []
    print(f'{len(images)} particles detected')
    for i, particle in enumerate(images):
        clasification_dict = {}
        probs = classification.clasiffy_img(particle, classifier)
        max_prob = np.max(probs)
        pred_label = classification.labels[np.argmax(probs)]

        # clasification_dict['id'] = str(i)
        clasification_dict['prob'] = max_prob
        clasification_dict['group'] = pred_label
        clasifications.append(clasification_dict)
    return clasifications

if __name__ == '__main__':
    import plotly.express as px
    import pandas as pd

    path = '/media/lecun/HD/Expor2/Test images/EAFIT1 EAFIT0014.tif'
    sem_img = cv.imread(path, 0)
    clasifications, _ = run_detection_classification(sem_img)

    df = pd.DataFrame.from_dict(clasifications)
    print(df)

    fig = px.pie(df, names='label')
    # fig.show()
