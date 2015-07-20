# Photo OCR
1. Problem Description and Pipeline:
    - optical character recognition(read text from an image)
    - photo ocr pipeline: image -> Text detection -> Character segmentation -> Character classification(A, B, C.....), we can use some modules to apply on our data
2. Sliding Windows:
    - sliding window classifier: aspect ratio = length/width ratio
3. Geting lost of data and artificial data
    - artificial data synthesis for photo ocr(distortion, blurring)- character recoginition;
    - distortion introduced should be representation of the type of noise and distortion in the test set;
    - Discussion on getting more data:
        - Make sure you have low classifier before expending the effort(plot learning curves. e.g. keep increasing the number of features/number of hidden units in neural network until you have a low bias classifier)
        - How much work would it be to get 10x as much data as we currently have. Much better performance
            - artificial data synthesis
            - Collect/label it yourself.(#hours, may not that much, could be a hero)
            - Crowd source(e.g. amazon mechanical turk) websites hire people to label data

8. Ceiling analysis: what part of the pipeline to work on next?
    - what part of the pipeline should you spend the most time trying to improve;
    - over all system accuracy: 72%; text detection accuracy: 89%; character segmentation: 90%; character recognition: 100%, upper bound of accuracy:
         - face recognition from images: camera -> prepreocess(remove bg) -> facedetection -> eyes segmentation -> nose segmentation -> mouth segmentation -> logistics regression -> label
         - ceiling analysis: overall 85%,  preprocess 85.1%(using gound-truth labels, to see the final accuracy), face detection 91%... compute the improvement. No gut feeling.