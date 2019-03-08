"""Generate caption from webcam image."""

import math

import cv2
import tensorflow as tf

from medium_show_and_tell_caption_generator.caption_generator \
    import CaptionGenerator
from medium_show_and_tell_caption_generator.model import ShowAndTellModel
from medium_show_and_tell_caption_generator.vocabulary import Vocabulary

# %% Image from webcam
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
rgb_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
cap.release()

cv2.imwrite('imgs/webcam_test.jpg', frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), 80])

# %% Load model and vocabulary
model = ShowAndTellModel(model_path='etc/show-and-tell.pb')
vocab = Vocabulary(vocab_file_path='etc/word_counts.txt')
generator = CaptionGenerator(model, vocab)

# %% Generate caption for webcam image
with tf.gfile.GFile('imgs/webcam_test.jpg', "rb") as f:
    image = f.read()

captions = generator.beam_search(image)
for i, caption in enumerate(captions):
    # Ignore begin and end tokens <S> and </S>.
    sentence = [vocab.id_to_token(w) for w in caption.sentence[1:-1]]
    sentence = " ".join(sentence)
    print("  %d) %s (p=%f)" % (i, sentence, math.exp(caption.logprob)))
