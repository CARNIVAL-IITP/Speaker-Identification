# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from contextlib import contextmanager
from itertools import permutations
from typing import Dict, Optional, Tuple

import os
import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from packaging.version import parse as V
from typeguard import check_argument_types

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.diar.attractor.abs_attractor import AbsAttractor
from espnet2.diar.decoder.abs_decoder import AbsDecoder
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.nets_utils import to_device

if V(torch.__version__) >= V("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield

class ESPnetDiarizationModel(AbsESPnetModel):
    """Speaker Diarization model

    If "attractor" is "None", SA-EEND will be used.
    Else if "attractor" is not "None", EEND-EDA will be used.
    For the details about SA-EEND and EEND-EDA, refer to the following papers:
    SA-EEND: https://arxiv.org/pdf/1909.06247.pdf
    EEND-EDA: https://arxiv.org/pdf/2005.09921.pdf, https://arxiv.org/pdf/2106.10654.pdf
    """

    def __init__(
        self,
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        label_aggregator: torch.nn.Module,
        encoder: AbsEncoder,
        decoder: AbsDecoder,
        attractor: Optional[AbsAttractor],
        diar_weight: float = 1.0,
        attractor_weight: float = 1.0,
    ):
        assert check_argument_types()

        super().__init__()

        self.encoder = encoder
        self.normalize = normalize
        self.frontend = frontend
        self.specaug = specaug
        self.label_aggregator = label_aggregator
        self.diar_weight = diar_weight
        self.attractor_weight = attractor_weight
        self.attractor = attractor
        self.decoder = decoder

        if self.attractor is not None:
            self.decoder = None
        elif self.decoder is not None:
            self.num_spk = decoder.num_spk
        else:
            raise NotImplementedError
        self.data_path = "/home/bjwoo/PycharmProjects/data/Libri2Mix/wav16k/min/train-100/"
        self.speaker_candidate = ["s1","s2"]
        import torch.nn as nn
        self.max_pool = nn.MaxPool1d(kernel_size=64, stride=32)
        self.avg_pool1d = nn.AvgPool1d(kernel_size=128, stride=64)
        self.oracle_linear = nn.Linear(2,256)
        self.feature_len = 256
        encoder_norm = nn.LayerNorm(self.feature_len, eps=1e-5)
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.feature_len, nhead=1, dim_feedforward=512,
                                                   dropout=0.1, batch_first=True)
        self.oracle_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1, norm=encoder_norm)
        decoder_norm = nn.LayerNorm(self.feature_len, eps=1e-5)
        decoder_layer = nn.TransformerDecoderLayer(d_model=self.feature_len, nhead=1, dim_feedforward=512,
                                                   dropout=0.1, batch_first=True)
        self.oracle_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1, norm=decoder_norm)

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,

        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, samples)
            speech_lengths: (Batch,) default None for chunk interator,
                                     because the chunk-iterator does not
                                     have the speech_lengths returned.
                                     see in
                                     espnet2/iterators/chunk_iter_factory.py
            spk_labels: (Batch, )
            kwargs: "utt_id" is among the input.
        """
        assert speech.shape[0] == spk_labels.shape[0], (speech.shape, spk_labels.shape)
        batch_size = speech.shape[0]
        # 1. Encoder
        # Use bottleneck_feats if exist. Only for "enh + diar" task.
        bottleneck_feats = kwargs.get("bottleneck_feats", None)
        bottleneck_feats_lengths = kwargs.get("bottleneck_feats_lengths", None)
        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, bottleneck_feats, bottleneck_feats_lengths
        )
        if self.attractor is None:
            # 2a. Decoder (baiscally a predction layer after encoder_out)
            pred = self.decoder(encoder_out, encoder_out_lens)
        else:
            # 2b. Encoder Decoder Attractors
            # Shuffle the chronological order of encoder_out, then calculate attractor
            encoder_out_shuffled = encoder_out.clone()
            for i in range(len(encoder_out_lens)):
                encoder_out_shuffled[i, : encoder_out_lens[i], :] = encoder_out[
                    i, torch.randperm(encoder_out_lens[i]), :
                ]
            attractor, att_prob = self.attractor(
                encoder_out_shuffled,
                encoder_out_lens,
                to_device(
                    self,
                    torch.zeros(
                        encoder_out.size(0), spk_labels.size(2) + 1, encoder_out.size(2)
                    ),
                ),
            )
            # Remove the final attractor which does not correspond to a speaker
            # Then multiply the attractors and encoder_out
            pred = torch.bmm(encoder_out, attractor[:, :-1, :].permute(0, 2, 1))
        # 3. Aggregate time-domain labels

        spk_labels, spk_labels_lengths = self.label_aggregator(
            spk_labels, spk_labels_lengths
        )

        # If encoder uses conv* as input_layer (i.e., subsampling),
        # the sequence length of 'pred' might be slighly less than the
        # length of 'spk_labels'. Here we force them to be equal.
        length_diff_tolerance = 2
        length_diff = spk_labels.shape[1] - pred.shape[1]
        if length_diff > 0 and length_diff <= length_diff_tolerance:
            spk_labels = spk_labels[:, 0 : pred.shape[1], :]

        if self.attractor is None:
            loss_pit, loss_att = None, None
            loss, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )
        else:
            loss_pit, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )
            loss_att = self.attractor_loss(att_prob, spk_labels)
            loss = self.diar_weight * loss_pit + self.attractor_weight * loss_att
        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = self.calc_diarization_error(pred, label_perm, encoder_out_lens)

        # reorder_pred = torch.zeros_like(pred).to(pred.device)
        # for i in range(pred.size(0)):
        #     for j in range(2):
        #         reorder_pred[i, :, j] = pred[i, :, perm_list[perm_idx][j]]
        if speech_scored > 0 and num_frames > 0:
            sad_mr, sad_fr, mi, fa, cf, acc, der = (
                speech_miss / speech_scored,
                speech_falarm / speech_scored,
                speaker_miss / speaker_scored,
                speaker_falarm / speaker_scored,
                speaker_error / speaker_scored,
                correct / num_frames,
                (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
            )
        else:
            sad_mr, sad_fr, mi, fa, cf, acc, der = 0, 0, 0, 0, 0, 0, 0

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_pit=loss_pit.detach() if loss_pit is not None else None,
            sad_mr=sad_mr,
            sad_fr=sad_fr,
            mi=mi,
            fa=fa,
            cf=cf,
            acc=acc,
            der=der,
        )

        loss, stats, weight, pred = force_gatherable((loss, stats, batch_size, pred), loss.device)
        return loss, stats, weight ,pred

    def oracle_forward(self,speech: torch.Tensor,
        speech_lengths: torch.Tensor = None,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
        utt_id = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

                Args:
                    speech: (Batch, samples)
                    speech_lengths: (Batch,) default None for chunk interator,
                                             because the chunk-iterator does not
                                             have the speech_lengths returned.
                                             see in
                                             espnet2/iterators/chunk_iter_factory.py
                    spk_labels: (Batch, )
                    kwargs: "utt_id" is among the input.
                """
        assert speech.shape[0] == spk_labels.shape[0], (speech.shape, spk_labels.shape)
        batch_size = speech.shape[0]
        # 1. Encoder
        # Use bottleneck_feats if exist. Only for "enh + diar" task.
        bottleneck_feats = kwargs.get("bottleneck_feats", None)
        bottleneck_feats_lengths = kwargs.get("bottleneck_feats_lengths", None)
        # encoder_out, encoder_out_lens = self.oracle_encode(
        #     speech, speech_lengths, bottleneck_feats, bottleneck_feats_lengths , spk_labels, spk_labels_lengths
        # ) # B,T, 256
        encoder_out, encoder_out_lens = self.encode(
            speech, speech_lengths, bottleneck_feats, bottleneck_feats_lengths
        ) # B,T, 256

        # import random
        # speaker_select = random.choice(self.speaker_candidate)
        # utt_wav = utt_id[0]+".wav"
        # ref_audio_path = os.path.join(self.data_path,speaker_select,utt_wav)
        #
        # ref_audio , sr = torchaudio.load(ref_audio_path)
        # ref_audio = ref_audio.to("cuda")
        # ref_spectrogram = torch.stft(ref_audio,510,255,510,window=torch.hann_window(510).to("cuda"),return_complex=True)


        oracle_feat = self.oracle_linear(spk_labels)
        # oracle_feat = self.avg_pool1d(oracle_feat.transpose(1,2))
        oracle_feat = self.max_pool(oracle_feat.transpose(1,2))
        oracle_feat = self.oracle_encoder(oracle_feat.transpose(1,2))
        encoder_out = self.oracle_decoder(tgt=encoder_out, memory=oracle_feat)
        if self.attractor is None:
            # 2a. Decoder (baiscally a predction layer after encoder_out)
            pred = self.decoder(encoder_out, encoder_out_lens)
        else:
            # 2b. Encoder Decoder Attractors
            # Shuffle the chronological order of encoder_out, then calculate attractor
            encoder_out_shuffled = encoder_out.clone()
            for i in range(len(encoder_out_lens)):
                encoder_out_shuffled[i, : encoder_out_lens[i], :] = encoder_out[
                                                                    i, torch.randperm(encoder_out_lens[i]), :
                                                                    ]
            attractor, att_prob = self.attractor(
                encoder_out_shuffled,
                encoder_out_lens,
                to_device(
                    self,
                    torch.zeros(
                        encoder_out.size(0), spk_labels.size(2) + 1, encoder_out.size(2)
                    ),
                ),
            )
            # Remove the final attractor which does not correspond to a speaker
            # Then multiply the attractors and encoder_out
            pred = torch.bmm(encoder_out, attractor[:, :-1, :].permute(0, 2, 1))
        # 3. Aggregate time-domain labels
        spk_labels, spk_labels_lengths = self.label_aggregator(
            spk_labels, spk_labels_lengths
        )
        # If encoder uses conv* as input_layer (i.e., subsampling),
        # the sequence length of 'pred' might be slighly less than the
        # length of 'spk_labels'. Here we force them to be equal.
        length_diff_tolerance = 2
        length_diff = spk_labels.shape[1] - pred.shape[1]
        if length_diff > 0 and length_diff <= length_diff_tolerance:
            spk_labels = spk_labels[:, 0: pred.shape[1], :]

        if self.attractor is None:
            loss_pit, loss_att = None, None
            loss, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )
        else:
            loss_pit, perm_idx, perm_list, label_perm = self.pit_loss(
                pred, spk_labels, encoder_out_lens
            )

            loss_att = self.attractor_loss(att_prob, spk_labels)
            loss = self.diar_weight * loss_pit + self.attractor_weight * loss_att
        (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        ) = self.calc_diarization_error(pred, label_perm, encoder_out_lens)

        # reorder_pred = torch.zeros_like(pred).to(pred.device)
        # for i in range(pred.size(0)):
        #     for j in range(2):
        #         reorder_pred[i, :, j] = pred[i, :, perm_list[perm_idx][j]]

        if speech_scored > 0 and num_frames > 0:
            sad_mr, sad_fr, mi, fa, cf, acc, der = (
                speech_miss / speech_scored,
                speech_falarm / speech_scored,
                speaker_miss / speaker_scored,
                speaker_falarm / speaker_scored,
                speaker_error / speaker_scored,
                correct / num_frames,
                (speaker_miss + speaker_falarm + speaker_error) / speaker_scored,
            )
        else:
            sad_mr, sad_fr, mi, fa, cf, acc, der = 0, 0, 0, 0, 0, 0, 0

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_pit=loss_pit.detach() if loss_pit is not None else None,
            sad_mr=sad_mr,
            sad_fr=sad_fr,
            mi=mi,
            fa=fa,
            cf=cf,
            acc=acc,
            der=der,
        )

        loss, stats, weight, pred = force_gatherable((loss, stats, batch_size,pred), loss.device)
        return loss, stats, weight, pred

    def detect_speaker_change(self,spk_labels: torch.Tensor):
        # Initialize change detection tensor
        spk_labels = spk_labels.permute(0,2,1)
        summed_labels = torch.sum(spk_labels, dim=1)
        changes = torch.zeros_like(summed_labels).to(spk_labels.device)
        B,T = summed_labels.size()
        # Detect changes based on transitions between 0, 1, and 2S
        for i in range(1, len(summed_labels)):
            if ((summed_labels[i] == 0 and summed_labels[i - 1] in [1, 2]) or
                    (summed_labels[i] == 1 and summed_labels[i - 1] in [0, 2]) or
                    (summed_labels[i] == 2 and summed_labels[i - 1] in [0, 1])):
                changes[i] = 1
                if i == 0:
                    changes[i + 1]=1
                elif i == T-1:
                    changes[i - 1]=1
                else:
                    changes[i - 1] = 1
                    changes[i + 1] = 1
        return changes
    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        spk_labels: torch.Tensor = None,
        spk_labels_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        bottleneck_feats: torch.Tensor,
        bottleneck_feats_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            bottleneck_feats: (Batch, Length, ...): used for enh + diar
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # 4. Forward encoder
            # feats: (Batch, Length, Dim)
            # -> encoder_out: (Batch, Length2, Dim)
            if bottleneck_feats is None:
                encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)
            elif self.frontend is None:
                # use only bottleneck feature
                encoder_out, encoder_out_lens, _ = self.encoder(
                    bottleneck_feats, bottleneck_feats_lengths
                )
            else:
                # use both frontend and bottleneck feats
                # interpolate (copy) feats frames
                # to match the length with bottleneck_feats
                feats = F.interpolate(
                    feats.transpose(1, 2), size=bottleneck_feats.shape[1]
                ).transpose(1, 2)
                # concatenate frontend LMF feature and bottleneck feature
                encoder_out, encoder_out_lens, _ = self.encoder(
                    torch.cat((bottleneck_feats, feats), 2), bottleneck_feats_lengths
                )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def oracle_encode(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        bottleneck_feats: torch.Tensor,
        bottleneck_feats_lengths: torch.Tensor,
        spk_labels: torch.Tensor,
        spk_labels_lengths:torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch,)
            bottleneck_feats: (Batch, Length, ...): used for enh + diar
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

            # 4. Forward encoder
            # feats: (Batch, Length, Dim)
            # -> encoder_out: (Batch, Length2, Dim)
            if bottleneck_feats is None:
                encoder_out, encoder_out_lens, _ = self.encoder.oracle_forward(feats, feats_lengths,spk_labels,spk_labels_lengths)
            elif self.frontend is None:
                # use only bottleneck feature
                encoder_out, encoder_out_lens, _ = self.encoder.oracle_forward(
                    bottleneck_feats, bottleneck_feats_lengths,spk_labels,spk_labels_lengths
                )
            else:
                # use both frontend and bottleneck feats
                # interpolate (copy) feats frames
                # to match the length with bottleneck_feats
                feats = F.interpolate(
                    feats.transpose(1, 2), size=bottleneck_feats.shape[1]
                ).transpose(1, 2)
                # concatenate frontend LMF feature and bottleneck feature
                encoder_out, encoder_out_lens, _ = self.encoder.oracle_forward(
                    torch.cat((bottleneck_feats, feats), 2), bottleneck_feats_lengths,spk_labels,spk_labels_lengths
                )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = speech.shape[0]
        speech_lengths = (
            speech_lengths
            if speech_lengths is not None
            else torch.ones(batch_size).int() * speech.shape[1]
        )

        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def pit_loss_single_permute(self, pred, label, length):
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        mask = self.create_length_mask(length, label.size(1), label.size(2))
        loss = bce_loss(pred, label)
        loss = loss * mask
        loss = torch.sum(torch.mean(loss, dim=2), dim=1)
        loss = torch.unsqueeze(loss, dim=1)
        return loss

    def pit_loss(self, pred, label, lengths):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND
        num_output = label.size(2)
        permute_list = [np.array(p) for p in permutations(range(num_output))]
        loss_list = []
        for p in permute_list:
            label_perm = label[:, :, p] # batch, length, num_spk
            loss_perm = self.pit_loss_single_permute(pred, label_perm, lengths)
            loss_list.append(loss_perm)
        loss = torch.cat(loss_list, dim=1)
        min_loss, min_idx = torch.min(loss, dim=1)
        loss = torch.sum(min_loss) / torch.sum(lengths.float())
        batch_size = len(min_idx)
        label_list = []
        for i in range(batch_size):
            label_list.append(label[i, :, permute_list[min_idx[i]]].data.cpu().numpy())
        label_permute = torch.from_numpy(np.array(label_list)).float()
        return loss, min_idx, permute_list, label_permute

    def create_length_mask(self, length, max_len, num_output):
        batch_size = len(length)
        mask = torch.zeros(batch_size, max_len, num_output)
        for i in range(batch_size):
            mask[i, : length[i], :] = 1
        mask = to_device(self, mask)
        return mask

    def attractor_loss(self, att_prob, label):
        batch_size = len(label)
        bce_loss = torch.nn.BCEWithLogitsLoss(reduction="none")
        # create attractor label [1, 1, ..., 1, 0]
        # att_label: (Batch, num_spk + 1, 1)
        att_label = to_device(self, torch.zeros(batch_size, label.size(2) + 1, 1))
        att_label[:, : label.size(2), :] = 1
        loss = bce_loss(att_prob, att_label)
        loss = torch.mean(torch.mean(loss, dim=1))
        return loss

    @staticmethod
    def calc_diarization_error(pred, label, length):
        # Note (jiatong): Credit to https://github.com/hitachi-speech/EEND

        (batch_size, max_len, num_output) = label.size()
        # mask the padding part
        mask = np.zeros((batch_size, max_len, num_output))
        for i in range(batch_size):
            mask[i, : length[i], :] = 1

        # pred and label have the shape (batch_size, max_len, num_output)
        label_np = label.data.cpu().numpy().astype(int)
        pred_np = (pred.data.cpu().numpy() > 0).astype(int)
        label_np = label_np * mask
        pred_np = pred_np * mask
        length = length.data.cpu().numpy()

        # compute speech activity detection error
        n_ref = np.sum(label_np, axis=2)
        n_sys = np.sum(pred_np, axis=2)
        speech_scored = float(np.sum(n_ref > 0))
        speech_miss = float(np.sum(np.logical_and(n_ref > 0, n_sys == 0)))
        speech_falarm = float(np.sum(np.logical_and(n_ref == 0, n_sys > 0)))

        # compute speaker diarization error
        speaker_scored = float(np.sum(n_ref))
        speaker_miss = float(np.sum(np.maximum(n_ref - n_sys, 0)))
        speaker_falarm = float(np.sum(np.maximum(n_sys - n_ref, 0)))
        n_map = np.sum(np.logical_and(label_np == 1, pred_np == 1), axis=2)
        speaker_error = float(np.sum(np.minimum(n_ref, n_sys) - n_map))
        correct = float(1.0 * np.sum((label_np == pred_np) * mask) / num_output)
        num_frames = np.sum(length)
        return (
            correct,
            num_frames,
            speech_scored,
            speech_miss,
            speech_falarm,
            speaker_scored,
            speaker_miss,
            speaker_falarm,
            speaker_error,
        )
