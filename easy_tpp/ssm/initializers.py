import math

import numpy as np
import numpy as onp
import torch as th
from numpy.linalg import eigh


def make_HiPPO(P):
    """Create a HiPPO-LegS matrix.
    From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        P (int32): state size
    Returns:
        P x P HiPPO LegS matrix
        
    You can see how the LegS matrix is structured in section C.1 of the Appendix of the S4 paper.
    That is what this function produces here.
    
    As a refresher, HiPPO matrix A is a provably optimal memory compression operator.
    S4's insight was to start/initialize with a theoretically optimal memory system, then relax it.
    """
    M = np.sqrt(1 + 2 * np.arange(P)) # M[i] = sqrt(1 + 2i) for row i
    A = M[:, np.newaxis] * M[np.newaxis, :] # M * M^T, which produces A[i,j] = sqrt((1+2i)(1+2j))
    A = np.tril(A) - np.diag(np.arange(P)) # keep the lower diagonal and subtract i on the diagonal
    return -A # return the negative of this.


def make_NPLR_HiPPO(P):
    """
    Makes components needed for NPLR (Normal Plus Low-Rank) representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Args:
        P (int32): state size

    Returns:
        P x P HiPPO LegS matrix, low-rank factor P, HiPPO input matrix B

    The HiPPO matrix from make_HiPPO is not normal, but by adding a rank-1 term we can make it normal.
    A = N - rr^T, where N is normal (unitarily diagonalizable) and r is a vector.
    Again, the below code is explained from section C.1 of the Appendix of the S4 paper.
    In particular, it is implementing the "Adding 1/2 * (2n+1)^{1/2} * (2n+1)^{1/2} to the whole matrix"
    
    Think of it this way, the whole point of DPLR is that A = V \Lambda V^* - P Q^T.
    Here, we are setting P = Q = r, a vector of dim P, so that P Q^T = r r^T, a rank-1 matrix.
    It is found in section C.1 that letting r_n = sqrt(2n+1), then adding (1/2)*r*r^T to A
    produces a matrix of the form (1/2)*I + S where S is skew-symmetric. This matrix will yield the same
    diagonalization as S, which is normal (by nature of its skew-symmetry).
    
    S in this case would be S_{nk} =
                                        \begin{cases}
                                        -\frac12 r_n r_k & n > k \
                                        0 & n = k \
                                        \frac12 r_n r_k & n < k
                                        \end{cases}
                                        
    HOWEVER, (1/2)*I + S = \begin{cases}
                            -\frac12 r_n r_k & n > k \
                            \frac12 & n = k \
                            \frac12 r_n r_k & n < k
                            \end{cases}
    The above is confusing, because in make_DPLR_HiPPO, we use "S" to denote A + rr^T
    """
    # Make -HiPPO
    hippo = make_HiPPO(P)

    # This is actually is actually taking the r vector stated above and dividing it by sqrt(2)
    # this makes it so that in make_DPLR_HiPPO, r r^T becomes (1/2) * r r^T as desired.
    R1 = np.sqrt(np.arange(P) + 0.5)

    # HiPPO also specifies the B matrix (Equation (29) in the HiPPO paper)
    B = np.sqrt(2 * np.arange(P) + 1.0)
    return hippo, R1, B


def make_DPLR_HiPPO(P):
    """
    Makes components needed for DPLR representation of HiPPO-LegS
     From https://github.com/srush/annotated-s4/blob/main/s4/s4.py
    Note, we will only use the diagonal part
    Args:
        P:

    Returns:
        eigenvalues Lambda, low-rank term R1, conjugated HiPPO input matrix B,
        eigenvectors V, HiPPO B pre-conjugation

    Recall in the introduction annotation of make_NPLR_HiPPO, we said that we are producing a matrix of
    the form (1/2)*I + S where S is skew-symmetric.
    Linear algebra tells us the eigenvalues of a skew-symmetric matrix are purely imaginary.
    Thus, the eigenvalues of (1/2)*I + S are -1/2 + i * b_j for real b_j.
    So the real part of the eigenvalues is just -1/2 for all j.
    """
    A, R1, B = make_NPLR_HiPPO(P)

    # this is not a smart choice of notation, we should call "S" here like "N"
    # S in the literature is supposed to signify skew-symmetric matrix, which A + R1 R1^T is not.
    # it has the same diagonalization as a skew-symmetric matrix though.
    S = A + R1[:, np.newaxis] * R1[np.newaxis, :] # S = A + PQ^T = A + R1 R1^T
                                                  # this shares the same diagonalization as a
                                                  # skew-symmetric matrix

    """
    S_diag and Lambda_real are equivalent here since all the diagonal entries
    of S are -0.5...
    
    This snippet below is from alexander rush's annotated S4 code: https://srush.github.io/annotated-s4/
    in it, he does a check that S_diag and Lambda_real are the same, which they should be.
    So truly, we can just write Lambda_real = np.diagonal(S)
    """
    S_diag = np.diagonal(S) # just a n array of -0.5's, not sure why this is being averaged below
    Lambda_real = np.mean(S_diag) * np.ones_like(S_diag)

    """
    Diagonalize S to V \Lambda V^* using linalg.eigh
    
    linalg.eigh returns the eigenvalues and eigenvectors of a complex hermitian (conjugate symmetric)
    or real symmetric matrix. It returns a 1-D array containing the eigenvalues, and a 2-D square array
    of the corresponding eigenvectors
    
    Crucially, eigh can only take Hermitian matrices as input (matrices who are equal to their 
    own conjugate transpose). The matrix S computed above is normal, but not Hermitian.
    S = (1/2)*I + Sk where Sk is skew-symmetric (-Sk = (Sk)^T).
    In our case, if we multiply S by -i, we get:
    -iS = -i*(1/2)*I - i*Sk
    We can look each of the components:
    -i*(1/2)*I is Hermitian because it is diagonal with real entries (-i/2)
    -i*Sk is also Hermitian since (-iSk)^H = iSk^H = i(-Sk) = -iSk
    Therefore, -i*S is Hermitian, and we can use eigh on it.
    In a geometric sense, multiplying by -i is a 90 degree rotation in the complex plane,
    
    It also happens that the eigenvectors produced by eigh(S * -1j) are the same as those of S:
    S*v = \lambda*v \iff -i*S*v = -i*\lambda*v
    In other words, scaling by the scalar -i doesn't change the eigenvectors, only rotates the eigenvalues
    """
    Lambda_imag, V = eigh(S * -1j)

    """_summary_
    A = V\Lambda V^H - R1 R1^T = V(\Lambda - (V^H R1)(V^H R1)^T)V^H
    """
    R1 = V.conj().T @ R1
    B_orig = B
    B = V.conj().T @ B # see Lemma 1 of section 3.1 of the S4 paper
    return (
        # note the imaginary eigenvalues are being multiplied by i to account for the -i scaling above
        th.tensor(onp.asarray(Lambda_real + 1j * Lambda_imag), dtype=th.complex64),
        th.tensor(onp.asarray(R1)),
        th.tensor(onp.asarray(B)),
        th.tensor(onp.asarray(V), dtype=th.complex64),
        th.tensor(onp.asarray(B_orig)),
    )


def init_log_steps(P, dt_min, dt_max):
    """Initialize an array of learnable timescale parameters.
    initialized uniformly in log space.
     Args:
         input:
     Returns:
         initialized array of timescales (float32): (P,)
    """
    unlog = th.rand(size=(P,))
    log = unlog * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
    return log


def lecun_normal_(tensor: th.Tensor) -> th.Tensor:
    input_size = tensor.shape[
        -1
    ]  # Assuming that the weights' input dimension is the last.
    std = math.sqrt(1 / input_size)
    with th.no_grad():
        return tensor.normal_(0, std)  # or torch.nn.init.xavier_normal_


def init_VinvB(shape, Vinv):
    """Initialize B_tilde=V^{-1}B. First samples B. Then compute V^{-1}B.
    Note we will parameterize this with two different matrices for complex

    Modified from https://github.com/lindermanlab/S5/blob/52cc7e22d6963459ad99a8674e4d3cfb0a480008/s5/ssm.py#L165

    numbers.
     Args:
         shape (tuple): desired shape  (P,H)
         Vinv: (complex64)     the inverse eigenvectors used for initialization
     Returns:
         B_tilde (complex64) of shape (P,H)
    """
    B = lecun_normal_(th.zeros(shape))
    VinvB = Vinv @ B.type(th.complex64)
    return VinvB
