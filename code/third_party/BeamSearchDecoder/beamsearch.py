# This file is part of BeamSearchDecoder.

# Copyright (c) 2025 IonQ, Inc., all rights reserved

# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International License (CC BY-NC-SA 4.0).

# You may obtain a copy of the License at:
# https://creativecommons.org/licenses/by-nc-sa/4.0/


import numpy as np
import scipy
import stim
from decoder.beam_search_decoder import BeamSearchDecoder
from stimbposd.dem_to_matrices import detector_error_model_to_check_matrices


def create_decoder(pcm, priors, **kwargs):
    return BeamSearchDecoder(pcm=pcm, error_channel=list(priors), **kwargs)


class BeamSearch:
    def __init__(
        self,
        model: stim.DetectorErrorModel,
        max_rounds: int = 10,
        beam_width: int = 8,
        num_results: int = 1,
        initial_iters: int = 30,
        iters_per_round: int = 20,
        **bp_kwargs,
    ):
        """Class for decoding stim circuits using belief propagation (BP).
        This class uses Joschka Roffe's BP decoder as a subroutine. For more information on the options and
        implementation of the Parallel BP subroutine, see the documentation of the LDPC library: https://roffe.eu/software/ldpc/index.html.
        Additional keyword arguments are passed to the ``bp_decoder`` class of the ldpc Python package.

        Parameters
        ----------
        model : stim.DetectorErrorModel
            The detector error model of the stim circuit to be decoded
        max_bp_iters : int, optional
            The maximum number of iterations of belief propagation to be used, by default {DEFAULT_MAX_BP_ITERS}
        """
        self._matrices = detector_error_model_to_check_matrices(model, allow_undecomposed_hyperedges=True)
        self.num_detectors = model.num_detectors
        self.num_errors = model.num_errors
        # h_shape = self._matrices.check_matrix.shape
        self._beamsearch = create_decoder(
            pcm=self._matrices.check_matrix,
            max_rounds=max_rounds,
            beam_width=beam_width,
            num_results=num_results,
            initial_iters=initial_iters,
            iters_per_round=iters_per_round,
            priors=self._matrices.priors,
            **bp_kwargs,
        )

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        """
        Decode the syndrome and return a prediction of which observables were returned

        Parameters
        ----------
        syndrome : np.ndarray
            A single shot of syndrome data. This should be a binary array with a length equal to the
            number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`. E.g. the syndrome might be
            one row of shot data sampled from a `stim.CompiledDetectorSampler`.

        Returns
        -------
        np.ndarray
            A binary numpy array `predictions` which predicts which observables were returned.
            Its length is equal to the number of observables in the `stim.Circuit` or `stim.DetectorErrorModel`.
            `predictions[i]` is 1 if the decoder predicts observable `i` was returned and 0 otherwise.
        """
        corr = self._beamsearch.decode(syndrome)
        return (self._matrices.observables_matrix @ corr) % 2

    def decode_batch(
        self,
        shots: np.ndarray,
        *,
        bit_packed_shots: bool = False,
        bit_packed_predictions: bool = False,
    ) -> np.ndarray:
        """
        Decode a batch of shots of syndrome data. This is just a helper method, equivalent to iterating over each
        shot and calling `BP.decode` on it.

        Parameters
        ----------
        shots : np.ndarray
            A binary numpy array of dtype `np.uint8` or `bool` with shape `(num_shots, num_detectors)`, where
            here `num_shots` is the number of shots and `num_detectors` is the number of detectors in the `stim.Circuit` or `stim.DetectorErrorModel`.

        Returns
        -------
        np.ndarray
            A 2D numpy array `predictions` of dtype bool, where `predictions[i, :]` is the output of
            `self.decode(shots[i, :])`.
        """

        if bit_packed_shots:
            shots = np.unpackbits(shots, axis=1, bitorder="little")[:, : self.num_detectors]
        obs: scipy._csc.csc_matrix = self._matrices.observables_matrix
        predictions = np.zeros((shots.shape[0], obs.shape[0]), dtype=bool)
        for i in range(shots.shape[0]):
            res = self.decode(shots[i, :])
            predictions[i, :] = res
        if bit_packed_predictions:
            predictions = np.packbits(predictions, axis=1, bitorder="little")
        return predictions
