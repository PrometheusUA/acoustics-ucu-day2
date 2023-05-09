"""
Acoustic Model. Please read the following definitions and
proceed to additional instructions at the end of the file.
You will need to install these packages: g2p-en, torch, torchaudio
"""

import torch
import torch.nn as nn
import torchaudio
from g2p_en import G2p


def make_frames(wav):
    return torchaudio.compliance.kaldi.mfcc(wav)


class LibriSpeech(torch.utils.data.Dataset):
    def __init__(self, url="dev-clean"):
        super().__init__()
        self.librispeech = torchaudio.datasets.LIBRISPEECH(".", url=url, download=True)

    def __len__(self):
        return len(self.librispeech)

    def __getitem__(self, index):
        wav, sr, text, speaker_id, chapter_id, utterance_id = self.librispeech[index]
        return make_frames(wav), text


class Encoder(nn.Module):
    def __init__(self, input_dim=13, subsample_dim=128, hidden_dim=1024):
        super().__init__()
        self.subsample = nn.Conv1d(input_dim, subsample_dim, 5, stride=4, padding=3)
        self.lstm = nn.LSTM(
            subsample_dim, hidden_dim, batch_first=True, num_layers=3, dropout=0.2
        )

    def subsampled_lengths(self, input_lengths):
        # https://github.com/vdumoulin/conv_arithmetic
        p, k, s = (
            self.subsample.padding[0],
            self.subsample.kernel_size[0],
            self.subsample.stride[0],
        )
        o = input_lengths + 2 * p - k
        o = torch.floor(o / s + 1)
        return o.int()

    def forward(self, inputs):
        x = inputs
        x = self.subsample(x.mT).mT
        x = x.relu()
        x, _ = self.lstm(x)
        return x.relu()


class Vocabulary:
    def __init__(self):
        self.g2p = G2p()

        # http://www.speech.cs.cmu.edu/cgi-bin/cmudict
        self.rdictionary = [
            "ε",  # CTC blank
            " ",
            "AA0",
            "AA1",
            "AE0",
            "AE1",
            "AH0",
            "AH1",
            "AO0",
            "AO1",
            "AW0",
            "AW1",
            "AY0",
            "AY1",
            "B",
            "CH",
            "D",
            "DH",
            "EH0",
            "EH1",
            "ER0",
            "ER1",
            "EY0",
            "EY1",
            "F",
            "G",
            "HH",
            "IH0",
            "IH1",
            "IY0",
            "IY1",
            "JH",
            "K",
            "L",
            "M",
            "N",
            "NG",
            "OW0",
            "OW1",
            "OY0",
            "OY1",
            "P",
            "R",
            "S",
            "SH",
            "T",
            "TH",
            "UH0",
            "UH1",
            "UW0",
            "UW1",
            "V",
            "W",
            "Y",
            "Z",
            "ZH",
        ]

        self.dictionary = {c: i for i, c in enumerate(self.rdictionary)}

    def __len__(self):
        return len(self.rdictionary)

    def encode(self, text):
        labels = [c.replace("2", "0") for c in self.g2p(text) if c != "'"]
        targets = torch.LongTensor([self.dictionary[phoneme] for phoneme in labels])
        return targets


class Recognizer(nn.Module):
    def __init__(self, feat_dim=1024, vocab_size=55 + 1):
        super().__init__()
        self.classifier = nn.Linear(feat_dim, vocab_size)

    def forward(self, features):
        features = self.classifier(features)
        return features.log_softmax(dim=-1)


vocab = Vocabulary()
encoder = Encoder()
recognizer = Recognizer()

#
# Download checkpoint lstm_p3_360+500.pt from https://wilab.org.ua/lstm_p3_360+500.pt
#
ckpt = torch.load("lstm_p3_360+500.pt", map_location="cpu")
encoder.load_state_dict(ckpt["encoder"])
recognizer.load_state_dict(ckpt["recognizer"])


audio_frames, text = LibriSpeech()[0]
phonemes = vocab.encode(text)

features = encoder(audio_frames)
outputs = recognizer.forward(features)  # (T, 55+1)

#
# Your task is to decode a sequence of vocabulary entries from a sequence of distributions
# over vocabulary entries (including blank ε that means "no output").
#
# outputs have dimension (T, V) where V is vocab_size+1 and T is time.
# outputs[:,0] is the log probability of a blank emission at every time step.
#
# Because of the subsampling done by Conv1d the time dimension in the outputs is 4 times smaller
# than in the inputs.
#
#
