#cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True, embedsignature=True
# distutils: language = c++
import numpy as np
import scipy.sparse
from typing import Optional, List, Union
import warnings
import ldpc.helpers.scipy_helpers

cdef BpSparse* Py2BpSparse(pcm):

    cdef int m
    cdef int n
    cdef int nonzero_count

    #check the parity check matrix is the right type
    if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
        pass
    else:
        raise TypeError(f"The input matrix is of an invalid type. Please input\
        a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")

    # Convert to binary sparse matrix and validate input
    pcm = ldpc.helpers.scipy_helpers.convert_to_binary_sparse(pcm)

    # get the parity check dimensions
    m, n = pcm.shape[0], pcm.shape[1]


    # get the number of nonzero entries in the parity check matrix
    if isinstance(pcm,np.ndarray):
        nonzero_count  = int(np.sum( np.count_nonzero(pcm,axis=1) ))
    elif isinstance(pcm,scipy.sparse.spmatrix):
        nonzero_count = int(pcm.nnz)

    # Matrix memory allocation
    cdef BpSparse* cpcm = new BpSparse(m,n,nonzero_count) #creates the C++ sparse matrix object

    #fill sparse matrix
    if isinstance(pcm,np.ndarray):
        for i in range(m):
            for j in range(n):
                if pcm[i,j]==1:
                    cpcm.insert_entry(i,j)
    elif isinstance(pcm,scipy.sparse.spmatrix):
        rows, cols = pcm.nonzero()
        for i in range(len(rows)):
            cpcm.insert_entry(rows[i], cols[i])

    return cpcm

cdef coords_to_scipy_sparse(vector[vector[int]]& entries, int m, int n, int entry_count):

    cdef np.ndarray[int, ndim=1] rows = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[int, ndim=1] cols = np.zeros(entry_count, dtype=np.int32)
    cdef np.ndarray[uint8_t, ndim=1] data = np.ones(entry_count, dtype=np.uint8)

    for i in range(entry_count):
        rows[i] = entries[i][0]
        cols[i] = entries[i][1]

    smat = scipy.sparse.csr_matrix((data, (rows, cols)), shape=(m, n), dtype=np.uint8)
    return smat

cdef BpSparse2Py(BpSparse* cpcm):
    cdef int i
    cdef int m = cpcm.m
    cdef int n = cpcm.n
    cdef int entry_count = cpcm.entry_count()
    cdef vector[vector[int]] entries = cpcm.nonzero_coordinates()
    smat = coords_to_scipy_sparse(entries, m, n, entry_count)
    return smat


def io_test(pcm: Union[scipy.sparse.spmatrix,np.ndarray]):
    cdef BpSparse* cpcm = Py2BpSparse(pcm)
    output = BpSparse2Py(cpcm)
    del cpcm
    return output



cdef class BeamSearchDecoderBase:

    """
    BeamSearch Decoder base class
    """

    def __cinit__(self,pcm, **kwargs):

        error_channel=kwargs.get("error_channel", None)
        max_rounds=kwargs.get("max_rounds",10)
        beam_width=kwargs.get("beam_width",8)
        num_results=kwargs.get("num_results",1)
        initial_iters=kwargs.get("initial_iters",30)
        iters_per_round=kwargs.get("iters_per_round",20)
        channel_probs = kwargs.get("channel_probs", [None])

        """
        Docstring test
        """

        cdef int i, j, nonzero_count
        self.MEMORY_ALLOCATED=False

        # Matrix memory allocation
        if isinstance(pcm, np.ndarray) or isinstance(pcm, scipy.sparse.spmatrix):
            pass
        else:
            raise TypeError(f"The input matrix is of an invalid type. Please input\
            a np.ndarray or scipy.sparse.spmatrix object, not {type(pcm)}")
        self.pcm = Py2BpSparse(pcm)

        # get the parity check dimensions
        self.m, self.n = pcm.shape[0], pcm.shape[1]

        # allocate vectors for decoder input
        self._error_channel.resize(self.n) #C++ vector for the error channel
        self._syndrome.resize(self.m) #C++ vector for the syndrome



        ## initialise the decoder with default values
        self.bpd = new BeamSearchDecoderCpp(self.pcm[0],self._error_channel,10,8,1,30,20)

        ## set the decoder parameters
        self.max_rounds = max_rounds
        self.beam_width = beam_width
        self.num_results = num_results
        self.initial_iters = initial_iters
        self.iters_per_round = iters_per_round

        if error_channel is not None:
            self.error_channel = error_channel
        else:
            raise ValueError("Please specify the error channel. error_channel:\
            list of floats of length equal to the block length of the code {self.n}.")




        self.MEMORY_ALLOCATED=True

    def __del__(self):
        if self.MEMORY_ALLOCATED:
            del self.bpd
            del self.pcm

    @property
    def error_channel(self) -> np.ndarray:
        """
        Returns the current error channel vector.

        Returns:
            np.ndarray: A numpy array containing the current error channel vector.
        """
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self.bpd.channel_probabilities[i]
        return out

    @error_channel.setter
    def error_channel(self, value: Union[Optional[List[float]],np.ndarray]) -> None:
        """
        Sets the error channel for the decoder.

        Args:
            value (Optional[List[float]]): The error channel vector to be set. Must have length equal to the block
            length of the code `self.n`.
        """
        if value is not None:
            if len(value) != self.n:
                raise ValueError(f"The error channel vector must have length {self.n}, not {len(value)}.")
            for i in range(self.n):
                self.bpd.channel_probabilities[i] = value[i]

    def update_channel_probs(self, value: Union[List[float],np.ndarray]) -> None:
        self.error_channel = value

    @property
    def channel_probs(self) -> np.ndarray:
        out = np.zeros(self.n).astype(float)
        for i in range(self.n):
            out[i] = self.bpd.channel_probabilities[i]
        return out

    @property
    def log_prob_ratios(self) -> np.ndarray:
        """
        Returns the current log probability ratio vector.

        Returns:
            np.ndarray: A numpy array containing the current log probability ratio vector.
        """
        out = np.zeros(self.n)
        for i in range(self.n):
            out[i] = self.bpd.log_prob_ratios[i]
        return out

    @property
    def converge(self) -> bool:
        """
        Returns whether the decoder has converged or not.

        Returns:
            bool: True if the decoder has converged, False otherwise.
        """
        return self.bpd.converge

    @property
    def iter(self) -> int:
        """
        Returns the number of iterations performed by the decoder.

        Returns:
            int: The number of iterations performed by the decoder.
        """
        return self.bpd.iterations


    @property
    def check_count(self) -> int:
        """
        Returns the number of rows of the parity check matrix.

        Returns:
            int: The number of rows of the parity check matrix.
        """
        return self.bpd.pcm.m

    @property
    def bit_count(self) -> int:
        """
        Returns the number of columns of the parity check matrix.

        Returns:
            int: The number of columns of the parity check matrix.
        """
        return self.bpd.pcm.n

    @property
    def max_rounds(self) -> int:
        """
        Returns the maximum rounds of branching allowed by the decoder.

        Returns:
            int: The maximum rounds of branching allowed by the decoder.
        """
        return self.bpd.max_rounds

    @max_rounds.setter
    def max_rounds(self, value: int) -> None:
        """
        Sets the maximum rounds of branching allowed by the decoder.

        Args:
            value (int): The maximum rounds of branching allowed by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("max_rounds input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"max_rounds input parameter must be a positive int. Not {value}.")
        self.bpd.max_rounds = value if value != 0 else 1

    @property
    def beam_width(self) -> int:
        """
        Returns the maximum list size allowed by the decoder.

        Returns:
            int: The maximum list size allowed by the decoder.
        """
        return self.bpd.beam_width

    @beam_width.setter
    def beam_width(self, value: int) -> None:
        """
        Sets the maximum list size allowed by the decoder.

        Args:
            value (int): The maximum list size allowed by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("beam_width input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"beam_width input parameter must be a positive int. Not {value}.")
        self.bpd.beam_width = value if value != 0 else 8

    @property
    def num_results(self) -> int:
        """
        Returns the number of solutions sought by the decoder.

        Returns:
            int: The number of solutions sought by the decoder.
        """
        return self.bpd.num_results

    @num_results.setter
    def num_results(self, value: int) -> None:
        """
        Sets the number of solutions sought by the decoder.

        Args:
            value (int): The number of solutions sought by the decoder.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("num_results input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"num_results input parameter must be a positive int. Not {value}.")
        self.bpd.num_results = value if value != 0 else 5

    @property
    def initial_iters(self) -> int:
        """
        Returns the number of iterations in preprocessing.

        Returns:
            int: The number of iterations in preprocessing.
        """
        return self.bpd.initial_iters

    @initial_iters.setter
    def initial_iters(self, value: int) -> None:
        """
        Sets the number of iterations in preprocessing.

        Args:
            value (int): The number of iterations in preprocessing.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("initial_iters input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"initial_iters input parameter must be a positive int. Not {value}.")
        self.bpd.initial_iters = value

    @property
    def iters_per_round(self) -> int:
        """
        Returns the number of iterations in each round.

        Returns:
            int: The number of iterations in each round.
        """
        return self.bpd.iters_per_round

    @iters_per_round.setter
    def iters_per_round(self, value: int) -> None:
        """
        Sets the number of iterations in each round.

        Args:
            value (int): The number of iterations in each round.

        Raises:
            ValueError: If value is not a positive integer.
        """
        if not isinstance(value, int):
            raise ValueError("iters_per_round input parameter is invalid. This must be specified as a positive int.")
        if value < 0:
            raise ValueError(f"iters_per_round input parameter must be a positive int. Not {value}.")
        self.bpd.iters_per_round = value


cdef class BeamSearchDecoder(BeamSearchDecoderBase):
    """
    Belief propagation decoder for binary linear codes.

    This class provides an implementation of belief propagation decoding for binary linear codes. The decoder uses a sparse
    parity check matrix to decode received codewords. The decoding algorithm can be configured using various parameters,
    such as the belief propagation method used, the scheduling method used, and the maximum number of iterations.

    Parameters
    ----------
    pcm : Union[np.ndarray, scipy.sparse.spmatrix]
        The parity check matrix of the binary linear code, represented as a NumPy array or a SciPy sparse matrix.
    error_channel : Optional[List[float]], optional
        The initial error channel probabilities for the decoder, by default None.
    """

    def __cinit__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix],
                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, max_rounds: Optional[int] = 10,
                 beam_width: Optional[int] = 8, num_results: Optional[int] = 1, initial_iters: Optional[int] = 30,
                 iters_per_round: Optional[int] = 20, **kwargs):

        for key in kwargs.keys():
            if key not in ["channel_probs"]:
                raise ValueError(f"Unknown parameter '{key}' passed to the BeamSearchDecoder constructor.")

        pass

    def __init__(self, pcm: Union[np.ndarray, scipy.sparse.spmatrix],
                 error_channel: Optional[Union[np.ndarray,List[float]]] = None, max_rounds: Optional[int] = 10,
                 beam_width: Optional[int] = 8, num_results: Optional[int] = 1, initial_iters: Optional[int] = 30,
                 iters_per_round: Optional[int] = 20, **kwargs):

        pass

    def decode(self, input_vector: np.ndarray) -> np.ndarray:
        """
        Decode the input input_vector using belief propagation decoding algorithm.

        Parameters
        ----------
        input_vector : numpy.ndarray
            A 1D numpy array of length equal to the number of rows in the parity check matrix.

        Returns
        -------
        numpy.ndarray
            A 1D numpy array of length equal to the number of columns in the parity check matrix.

        Raises
        ------
        ValueError
            If the length of the input input_vector does not match the number of rows in the parity check matrix.
        """

        cdef int i
        cdef bool zero_input_vector = True
        DTYPE = input_vector.dtype

        cdef int len_input_vector = len(input_vector)

        for i in range(len_input_vector):
            self._syndrome[i] = input_vector[i]
            if self._syndrome[i]: zero_input_vector = False
        if zero_input_vector:
            self.bpd.converge = True
            return np.zeros(self.bit_count,dtype=DTYPE)
        self.bpd.decode(self._syndrome)

        out = np.zeros(self.n,dtype=DTYPE)
        for i in range(self.n): out[i] = self.bpd.decoding[i]
        return out


    @property
    def decoding(self) -> np.ndarray:
        """
        Returns the current decoded output.

        Returns:
            np.ndarray: A numpy array containing the current decoded output.
        """
        out = np.zeros(self.n).astype(int)
        for i in range(self.n):
            out[i] = self.bpd.decoding[i]
        return out
