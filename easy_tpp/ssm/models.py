from typing import Optional, Tuple

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .initializers import (
    make_DPLR_HiPPO,  # , lecun_normal_ #  init_VinvB, init_log_steps,
)

MATRIX_SCALING_FACTOR = 1 # not really sure what the use case for this is


class LLH(nn.Module):
    """
    This is canon:
        L -- number of layers
        N -- number of events.
        P -- Hidden dimension.  Dimensionality of x.
        H -- output dimension.  Dimensionality of y/u.
    """

    def __init__(
        self,
        P: int,
        H: int,
        dt_init_min: float = 1e-4,
        dt_init_max: float = 0.1,
        dropout_rate: float = 0.0,
        act_func: str = "gelu",  # F.gelu,
        for_loop: bool = False,
        pre_norm: bool = True,
        post_norm: bool = False, # confusing why this is defaulted to False, seeing as
                                 # in the paper they always do post-norm
        simple_mark: bool = True,
        is_first_layer: bool = False,
        relative_time: bool = False,
        complex_values: bool = True,
    ):

        super(LLH, self).__init__()

        # Inscribe the args.
        self.P = P
        self.H = H
        self.dt_init_min = dt_init_min
        self.dt_init_max = dt_init_max
        self.dropout_rate = dropout_rate
        self.complex_values = complex_values

        
        # select the activation function.

        # each layer's output can be viewed as y^(l) := LLH^(l)(u^(l), \mathcal{H})
        # where LLH is the forward pass of the LLH layer, and \mathcal{H} is the history

        # this output is then:
        # 1) fed into a non-linear activation function (chosen from below)
        # 2) added to the initial input of the layer
        # 3) (normalized if post_norm=True) before being passed into the next layer as input u^(l+1)

        # all-in-all: u^(l+1) = LayerNorm( \sigma(y^(l)) + u^(l) )

        # the if block below defines \sigma, the activation function.

        # It should be noted that this activation just deals with activations between SSM layers. 
        # There is a final activation at the end of the entire network (after all LLH layers) 
        # that is separate from this for converting u_{t-}^(L+1) into an intensity \lambda_t, namely:
        # \lambda_t = ScaledSoftplus(Linear(u_{t-}^(L+1)))

        # furthermore, the placement of nn.Dropout (masking of random units in the computation graph for the 
        # purposes of regularization) will be explained in the comments for each activation.
        if act_func == "gelu": # GELU(x) := x * Phi(x) where Phi is the CDF of the standard normal distribution
                               # it is a smooth approximation to ReLU that suggests a more probabilistic
                               # view of a neuron's output. This is the activation used in the S2P2  and 
                               # Neural hawkes papers, so it will be the primary activation function to use.
                               # if Dropout layer is D(), then D(GELU(x)) is what's going on here.
                               # it is "regularizing the features produced by the nonlinearity"
                               # GELU(D(x)) on the otherhand would randomly remove features of the input before
                               # activation, which is not what we want.
            self.act_func = nn.Sequential(nn.GELU(), nn.Dropout(p=self.dropout_rate))
        
        # from: "Language Modeling with Gated Convolutional Networks" (Dauphin et al., 2017):
        # 
        # "gating" mechanisms control the path through which information flows and is used in recurrent nn's.
        # examples like "input gates" and "forget gates" in LSTMs allow for information to flow unimpeded
        # through potentially many timesteps. without these gates, information could easily vanish or explode
        # through the transofmrations of each timestep.

        # convolutional networks typically did not use gating mechanism until Oord et al., 2016 introduced
        # an LSTM-style mechanism of tanh(X*W+b) \otimes sigmoid(X*V+c)
        # GLU is a simplifcation (removal of the tanh) of this for non-deterministic gates that reduce the 
        # vanishing gradient problem by coupling linear units to the gates. this retains the non-linear 
        # capabilities of the layer while allowing the gradient to propagate through the linear unit 
        # without scaling. read more here: 
        # https://medium.com/deeplearningmadeeasy/glu-gated-linear-unit-21e71cd52081
        # that blog post will particularly explain why the input dimension is doubled in GLU
        # (faster implementation)

        # pytorch's nn.GLU is effectively nn.Linear(X) * nn.Linear(X).sigmoid()
        # or: GLU(a,b) = a \circdot \sigmoid(b) where [a,b] = Linear(X)

        # the reason this activation is included is because it is used in the S4 paper.

        # we have full_GLU(x) = D(GLU(D(W*x + b))) where D is Dropout, W,b are weights and biases of nn.Linear
        # the first dropout applied to the linear output is to regularize the two branches of what will be 
        # entered as input in the GLU operation. the second dropout is to regularize the output features of the 
        # GLU operation itself.

        elif act_func == "full_glu": # GLU: Gated Linear Unit
            self.act_func = nn.Sequential(
                nn.Linear(self.H, 2 * self.H),
                nn.Dropout(p=self.dropout_rate),
                nn.GLU(),
                nn.Dropout(p=self.dropout_rate),
            )

        # in the S5 paper, to take advantage of the fact that the S5 SSM outputs have already been mixed
        # throughout the MIMO SSM, they use a "weighted sigmoid gated unit" (a GLU activation without an
        # additional linear transform bfore the sigmoid): GELU(y) \circdot sigmoid(W * GELU(y))
        # read "Weighted sigmoid gate unit for an activation function of deep neural network" (Tanaka, 2020)
        # for further details.
        elif (
            act_func == "half_glu"
        ):  # ref: https://github.com/lindermanlab/S5/blob/main/s5/layers.py#L76
            self.act_func1 = nn.Sequential(
                nn.Dropout(p=self.dropout_rate),
                nn.GELU(),
                nn.Linear(self.H, self.H),
            )
            self.act_func = lambda x: nn.Dropout(p=self.dropout_rate)(
                x * nn.Sigmoid()(self.act_func1(x))
            )
        else:
            raise NotImplementedError(
                "Unrecognized activation function {}".format(act_func)
            )

        # Assume we always use conjugate symmetry.
        # why? because it's the standard way to keep a diagonalized complex SSM equivalent
        # to a real-valued SSM WHILE ONLY USING HALF THE SPECTRUM
        # if x is real, then x = v + \bar{v} = 2*Re(v) where v is complex
        # this allows us to store only half the eigenvalues/eigenvectors.
        # see the final paragraph of section 3.3 of Gu's "On the parameterization and intialization
        # of diagonal state space models" for more details.

        # the reason we'd even make this adjustable and not baked into the code is possibly
        # that someone may want to experiment with a fully complex-valued SSM,
        # or to check if the only storing the half-spectrum isn't introducing 
        # errors (debugging/ablation)?
        self.conj_sym = True

        # Allow a learnable initial state initial_state_P (\tilde{x}_0 = V^{-1}x_0).
        # Needs to be =/= 0 since we take the log during parallel scan.
        # it should be noted here that initial_state_P is NOT the same as x_0.
        # x_0 is meant to stay real-valued, however after diagonalization of A,
        # we have \tilde{x}_0 = V^{-1} x_0 which is generally complex-valued.
        # this allows for state/output evolution within the eigenbasis of A.
        # this is why it looks like the state is being allowed to be complex-valued
        # even though the S2P2 paper states that x_t should be real-valued.
        # it's because this variable below is actually \tilde{x}_0, not x_0.

        # here is also where we see we may even restrict the diagonalization of A to be
        # real-valued only, by setting complex_values=False.
        # although this reduces the representational capacity of the SSM layer (i.e. removes
        # the ability to have oscillatory eigenmodes), it may be beneficial in the sense that
        # it forces the model to learn real-valued dynamics which may be more stable and 
        # easier to train. especially since in the parallel scan part, we will have to take
        # the log and exp of these states, which may be numerically unstable for complex values.

        # note the imaginary = oscillatory and real = exponential decay comes from the fact
        # that \Lambda_p = -e^{\theta_p} + i\omega_p, which implies:
        # e^{\Lambda_p*\delta t} = e^{-e^{\theta_p}*\delta t} * (cos(\omega_p*\delta t) 
        #                           + i * sin(\omega_p*\delta t))
        if self.complex_values: # allows for more expressive dynamics at computational cost
            self.initial_state_P = nn.Parameter(
                th.complex(
                    th.randn(
                        self.P,
                    ),
                    th.randn(
                        self.P,
                    ),
                )
                * 1e-3,
                requires_grad=True,
            )
        else:
            self.initial_state_P = nn.Parameter(
                th.randn(
                    self.P,
                ),
                requires_grad=True,
            )

        # remember we discussed earlier that u^(l+1) = LayerNorm( \sigma(y^(l)) + u^(l) )
        # the activation function selection block prior to this comment defines \sigma
        # this line below defines the LayerNorm part.

        # here we instantiate a LayerNorm module in the shape of the output dimension H.
        # pytorch's LayerNorm implements the operation as described in the paper "Layer
        # Normalization" (Ba et al., 2016):
        # y = (gamma * (x - mu) / sqrt(var + epsilon)) + beta
        # where gamma and beta are learnable affine transform parameters, each of shape H.
        # there is a keyword argument elementwise_affine (default=True) that controls whether
        # these parameters are learnable or not, which clearly they are here.

        # as you've seen in this code in several places, you want to instantiate pytorch modules
        # and assign them to variable names that act as layers from then on out.
        # these variables will carry the learnable parameters forward throughout the code.
        # we will see later we will pass inputs through this LayerNorm module by calling
        # self.norm(input_tensor) where input_tensor has shape (..., H)
        self.norm = nn.LayerNorm(self.H)
        self.for_loop = for_loop # boolean (default: False) to choose between for-loop or 
                                 # parallel scan implementation of the SSM state evolution

        # implementation toggles for WHERE LayerNorm is applied (it should be one or the other!)
        self.pre_norm = pre_norm # boolean (default: True) to apply LayerNorm before NEXT SSM layer
        self.post_norm = post_norm # boolean (default: False) to apply LayerNorm after CURRENT SSM layer

        self.is_first_layer = is_first_layer # boolean (default: False) to indicate if this 
                                             # is the first LLH layer

        # this corresponds to the specific extension in the paper: 
        # input-dependent rescaling of \Lambda
        # They introduce a variant where, at event time t_i, the diagonal dynamics are 
        # rescaled by a learned function of the current layer signal u_t_i:
        # \Lambda_i := diag(softplus(W * u_t_i + b)) * \Lambda
        # this is exactly "relative"time/"input-conditioned forgetting rate"

        self.relative_time = relative_time # boolean (default: False)

        self._init_ssm_params()

        # from what i can tell, simple_mark doesn't appear to be used anywhere in the codebase.
        # i believe this part can be safely ignored.
        self.simple_mark = simple_mark
        if not simple_mark:
            self.mark_a_net = nn.Linear(self.H, self.P, bias=True)
            self.mark_u_net = nn.Linear(
                self.H, self.P, bias=False
            )  # Only need one bias
            # quick note on the the syntax below:
            # "why tensor.weight.data = something intead of tensor.weight = something?"
            # this is because tensor.weight is a nn.Parameter object, which is a wrapper around a tensor
            # that tells pytorch to track gradients for it. if we were to do tensor.weight = something,
            # we would be replacing the nn.Parameter object with a new tensor, which would not be tracked
            # by pytorch for gradients. by doing tensor.weight.data = something, we are modifying
            # the underlying tensor data of the nn.Parameter object, which is still tracked by pytorch.
            # this is a common pattern in pytorch for initializing weights of layers.
            self.mark_a_net.weight.data = th.complex(
                nn.init.xavier_normal_(self.mark_a_net.weight.data) * 1e-3,
                nn.init.xavier_normal_(self.mark_a_net.weight.data) * 1e-3,
            )
            self.mark_a_net.bias.data = th.complex(
                nn.init.xavier_normal_(self.mark_a_net.bias.data) * 1e-3,
                nn.init.xavier_normal_(self.mark_a_net.bias.data) * 1e-3,
            )
            self.mark_u_net.weight.data = th.complex(
                nn.init.xavier_normal_(self.mark_u_net.weight.data) * 1e-3,
                nn.init.xavier_normal_(self.mark_u_net.weight.data) * 1e-3,
            )
            if not self.complex_values:
                self.mark_a_net.weight.data = self.mark_a_net.weight.data.real
                self.mark_a_net.bias.data = self.mark_a_net.bias.data.real
                self.mark_u_net.weight.data = self.mark_u_net.weight.data.real

    def _init_ssm_params(self):
        self._init_A()
        if not self.is_first_layer:
            self._init_B()
        self._init_C()
        if (
            not self.is_first_layer
        ):  # Could group, but left in same order to not mess with initialization
            self._init_D()
        self._init_E()

    # make_DPLR_HiPPO only produces the continuous-time diagonalization \Lambda and eigenvectors V.
    # -A = V\Lambda V^H
    # this method below sets up the time-step/time-rescaling mechanism we will need to discretize \Lambda
    # we will need to eventually compute exp(\Lambda * \delta t)
    def _init_A(self):
        # Define the initial diagonal HiPPO matrix.
        # Te throw the HiPPO B away.
        Lambda_P, _, _, V_PP, _ = make_DPLR_HiPPO(self.P)
        
        # by calling nn.Parameter on these tensors, we are telling pytorch to track gradients on them.
        # furthermore, we need to keep in mind that we will eventually be exponentiating \Lambda * \delta t,
        # so there is a risk of numerical instability if the real parts of \Lambda are not negative.
        # therefore, we seek to keep the rel part of the eigenvalues (Re(\lambda)) negative in order to maintain 
        # stability. however, since the parameters are learned via gradient descent, we can't just tell the 
        # optimizer "hey stay negative". so instead, we represent Re(\lambda) as -exp(theta) where theta is 
        # real-valued. so the ACTUAL parameter is theta = log(-Re(\lambda)). So \theta can be drawn in any 
        # direction by gradient descent, but the final transformation will ensure the resulting real part of
        # the eigenvalue is negative.
        self.Lambda_P_log_neg_real = th.nn.Parameter((-Lambda_P.real).log())
        self.Lambda_P_imag = th.nn.Parameter(Lambda_P.imag)

        # Store these for use later.
        # makes the eigenvector matrix a property of the class (layer._V_PP)
        self._V_PP = V_PP
        self._Vc_PP = V_PP.conj().T

        # We also initialize the step size.
        # relative_time has to do with whether we want to rescale \Lambda at each event time
        # based on the current input u_t_i. this is the "input-dependent dynamics" section of the paper.
        # \Lambda_i = diag(softplus(W * u_t_i + b)) * \Lambda
        if self.relative_time:
            self.delta_net = nn.Linear( # delta_net is the W * u_t_i + b part above
                self.H, self.P, bias=True
            )  # nn.Parameter(init_log_steps(self.P, self.dt_init_min, self.dt_init_max))
            with th.no_grad(): # no_grad() because we don't want to track gradients for the initialization
                self.delta_net.weight.copy_(
                    nn.init.xavier_normal_(self.delta_net.weight)
                    # xavier normal initialization randomly initializes the weights according to a 
                    # normal distribution N(0, std^2) where std = sqrt(2 / (in_features + out_features))
                    # this type of initialization was invented in order to keep the variance of the 
                    # activations the same across layers, preventing vanishing/exploding gradients.
                )
                bias = th.ones(
                    self.P,
                )
                # note that softplus(x) = log(1 + exp(x)), so below what is happening is
                # remember we are multiplying the diagonal of the result of the softplus by \Lambda
                # so at initialization we want the softplus's output to be close to 1, so \Lambda_i
                # is close to \Lambda.
                # now, we already have that the input u at initialization is 0, so the W*u term in the
                # softplus will be 0. so we are left with initializing the bias such that softplus(bias) = 1
                # this is done algebraically by first, algebra:
                # softplus(bias) = log(1 + exp(bias)) = 1 --> bias = log(exp(1) - 1) = log(expm1(1))
                # however, the bias vector needs to be initialized with a dimension of P, so they initialize
                # it as a P-dimensional tensor of ones, then add log(-expm1(-1)) to each element,
                # which results in the log(exp(1) - 1) we initially wanted.
                # furthermore, x + log(1-e^-x) is more numerically stable for small values of x than log(exp(x)-1)
                # this is actually why expm1(x) even exists as a function
                bias += th.log(-th.expm1(-bias)) # expm1(x) = exp(x) - 1, more accurate for small x
                self.delta_net.bias.copy_(bias) # this final line is to finally assign the initialized bias
                                                # to the delta_net network.
        else:
            # if we opt to not have input-dependent dynamics, we set make the time-step a learnable parameter.
            self.log_step_size_P = nn.Parameter(
                th.zeros(size=(self.P,)), requires_grad=False
            )

    # In Python, an attribute is a general term for any piece of data or method associated with an object, 
    # accessed using dot notation (e.g., object.attribute). A property is a special type of attribute that 
    # uses methods (getter, setter, deleter) to control access and modification, without changing how it is 
    # accessed syntactically. 
    # this property is established below so we can recover the \Lambda matrix of the model at any point via
    # LLH.Lambda_P
    # of course, this is made to be a property and not an attribute because we need to perform some computation
    @property
    def Lambda_P(self):
        if self.complex_values:
            return th.complex(
                -self.Lambda_P_log_neg_real.exp(),
                self.Lambda_P_imag,
            )
        else:
            return -self.Lambda_P_log_neg_real.exp()

    def _init_B(self):
        # Initialize the B outside the eigenbasis and then transform by V*.
        # the key here is that although there is a theoretical HiPPO B, it was found in 
        # "On the parameterization and initialization of diagonal state space models" (Gu et al., 2022)
        # at the beginning of page 6 that it was not necessary to initialize B according to the HiPPO, only
        # A, due to its dominant effect on the dynamics of the SSM. However, allowing B to be learned freely
        # still allows for some material gain, so they opt to initialize B randomly with Xavier normal 
        # initialization (while still multiplying it by the eigenvector matrix V).
        B = nn.init.xavier_normal_(th.zeros((self.P, self.H))) * MATRIX_SCALING_FACTOR
        B_tilde_PH = self._Vc_PP @ B.type(th.complex64)
        self.B_tilde_PH = (
            th.nn.Parameter(B_tilde_PH)
            if self.complex_values
            else th.nn.Parameter(B_tilde_PH.real)
        )

    def _init_C(self):
        # Use the "complex_normal" initialization.
        # See ~https://github.com/lindermanlab/S5/blob/52cc7e22d6963459ad99a8674e4d3cfb0a480008/s5/ssm.py#L183
        C = nn.init.xavier_normal_(th.zeros((self.H, self.P))) * MATRIX_SCALING_FACTOR
        C_tilde_HP = C.type(th.complex64) @ self._V_PP
        self.C_tilde_HP = (
            th.nn.Parameter(C_tilde_HP)
            if self.complex_values
            else th.nn.Parameter(C_tilde_HP.real)
        )
        # self.C_tilde_HP.data *= 1e-3

    def _init_D(self):
        # Initialize feedthrough (D) matrix. Note the intensity depends on all layers.
        # xavier normal intitialization is only used on weight matrices (2-dimensional =
        # mixing of features.) this D here is a vector. as such, regular normal 
        # initialization will suffice. This follows from section B.1.2 of the S5 paper.
        D_HH = th.zeros(self.H)
        nn.init.normal_(D_HH, std=1.0)
        self.D_HH = nn.Parameter(D_HH, requires_grad=True) # the requires_grad=True is redundant here
                                                           # since nn.Parameter defaults to that.

    def _init_E(self):
        E = (
            th.nn.init.xavier_normal_(th.zeros((self.P, self.H)))
            * MATRIX_SCALING_FACTOR
        )
        E_tilde_PH = self._Vc_PP @ E.type(th.complex64)
        self.E_tilde_PH = (
            th.nn.Parameter(E_tilde_PH)
            if self.complex_values
            else th.nn.Parameter(E_tilde_PH.real)
        )

    def compute_impulse(self, right_u_H, mark_embedding_H):
        """
        Compute impulse to add to left limit of x to make right limit.
        this method is called later on in the forwards() method as:
        impulse_NP=self.compute_impulse(prime_right_u_NH, mark_embedding_NH)
        you can see here that the "mark_embedding_H" notation in this method definition
        is partially misleading, it really only means that the final dimension of "mark_embedding"
        is H, it is not a Hx1 or 1xH tensor.
        this is why the einsum notation uses an ellipses (...) before h after the comma. the fact
        that impulse_NP is NP dimensional basically tells you that it's really N multiplications of E
        (which is PxH) by a Hx1 tensor \alpha_k (mark_embedding_H)
        TODO: from what it looks like, they could have just written "ph,nh->np" instead of using ellipses
              to be more explicit, however we are not taking into account the additional batch dimension,
              but that doesn't appear to be utlized in this class it seems. monte carlo samples as well.
              just something to keep in mind. the ellipses buys you that flexibility.
        """
        alpha_P = th.einsum(
            "ph,...h->...p",
            self.E_tilde_PH,
            mark_embedding_H.type(th.complex64)
            if self.complex_values
            else mark_embedding_H,
        )
        return alpha_P

    """
    remember the input-dependent scaling formula for \Lambda?
    \Lambda_i = diag(softplus(W * u_t_i + b)) * \Lambda
    this is just implementing that. the delta_net is W * u_t_i + b (if relative_time is True),
    otherwise \Lambda_i = \exp(log(\delta_t))* Lambda.
    the "diag" part is just handled by the fact that softplus(W * u_t_i + b) is ...xNxP and Lambda_P is P-dimensional vector
    there is also some logic here if the right limit of the input u is unavailable
    """
    def get_lambda(self, right_u_NH, shift_u=True):
        if self.relative_time and (right_u_NH is not None): # the input may be null for the first layer
            # shift_u is purely an alignment fix. it makes the u-value used to compute the dynamics for the interval
            # (t_{i-1},t_i] come from the LEFT ENDPOINT t_{i-1}, not the right endpoint t_i.
            # it looks like the only time this method is called with shift_U=False is when they say 
            # "U should already be shifted"
            # recall that dt_N is stored as [0,t_1-t_0,t_2-t_1,...,t_N-t_{N-1}]
            # so dt_N[i] corresponds to the interval ENDING at t_i, however we want to use the value at t_i-1
            # concretely, F.pad(right_u_NH[..., :-1, :], (0, 0, 1, 0)) is doing the following:
                # right_u_NH[..., :-1, :] drops the last event along the N dimension. N being the number of events
                # in the sequence. so right_u_NH becomes (...,N-1,H) dimensional with inputs 
                # [u_0,u_1,...,u_{N-2}]
                    # remember array[:k] means "return elements up to BUT NOT INCLUDING index k"
                    # and since python is 0-indexed, you get get k "rows" (or columns if [:,:k])
                # F.pad(...,(0,0,1,0)) gives a pad "0" on the left and "0" on the right of the last dimension H (the feature dimension),
                # and pads "1" at the top and a "0" at the bottom of dimension N (the temporal ordering of u)
                    # F.pad(input,pad) determines how many dimensions you want to pad by how many numbers you enter for 
                    # arg "pad": something like pad(input, 0,1) means you pad the last dim, whereas pad(input, 0,1,0,0)
                    # means you are padding the final 2 dims of the tensor.
            # the theoretical point of this is that the rate over the interval (t_{i-1},t_i] should be determined by the
            # information at t_{i-1}, meaning that left endpoint should be driving the behavior over the interval, not the
            # future endpoint t_i.
            if shift_u:  # during "forward" when dts = [0, t1-t0, ..., t_N-t_{N-1}]
                right_u_NH = F.pad( # F is torch.nn.functional
                    right_u_NH[..., :-1, :], (0, 0, 1, 0)
                )  # pad default 0 at beginning of second to last dim
            lambda_rescaled_NP = (
                # recall that delta_net() was initialized as nn.Linear(H,P). this internally creates:
                # - a weight meatrix W that is PxH
                # - a bias vector b that is P-dimensional 
                # - it defines a map f(x) = x * W^T + b
                # KEY POINT HERE IS IT MULTIPLIES FROM THE RIGHT, NOT THE LEFT
                # so although you input H first, then P in function call, you actually get a PxH matrix
                # here we enter right_u_NH which is a (...,N,H) tensor, so which axis W^T applied to?
                # THE LAST LAYER ONLY. so self.delta_net(right_u_NH) is actually invoking a (1xH)*(HxP) = P multiplication.
                # so the delta_net is applied to a row of the LAST DIMENSION (H) of right_u_NH at a time.
                # this is common in pytorch, much of the nn methods involving matrix multiplication with multi-dim
                # take the last 2 dimensions of the tensor playing the role matrix, and the last dimension of the tensor
                # playing the role of the vector. this is very common. so although many of these variables only end in
                # "_H" or "_NH" or "_NP", odds are they are actually (BxH) or (BxNxH) or (BxNxP) respectively, where the
                # LEADING DIMENSIONS serve as just indices specifying how many times the matrix multiplication needs to
                # be stored (per batch)
                F.softplus(self.delta_net(right_u_NH)) * self.Lambda_P 
            )  # predict delta_i from right_u_i
            return {"lambda_rescaled_NP": lambda_rescaled_NP}
        else:
            if self.relative_time:
                lambda_rescaled_P = F.softplus(self.delta_net.bias) * self.Lambda_P
            else:
                lambda_rescaled_P = th.exp(self.log_step_size_P) * self.Lambda_P
            return {"lambda_rescaled_P": lambda_rescaled_P}

    def forward(
        self,
        left_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        right_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        mark_embedding_NH: th.Tensor, # \alpha, not E or E*\alpha
        dt_N: th.Tensor, # [0, t1-t0, ..., t_N-t_{N-1}]
        initial_state_P: Optional[th.Tensor] = None,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Apply the linear SSM to the inputs.

        In the context of TPPs, this returns the right limit of the "intensity function".
        This intensity will have been passed through a non-linearity, though, and so there is no
        guarantee for it is positive.
            # what they mean here is that the "intensity" u^(L+1) = LayerNorm(\sigma(y^(L)) + u^(L))
            # is not guaranteed to be positive UNTIL it is passed thru the scaled softplus:
            # \lambda_t = ScaledSoftplus(Linear(u_{t-}^(L+1)))
            
        _ssm() is the real "LLH recurrence engine" whereas forward() is a WRAPPER that:
            1) normalizes inputs
            2) prepares the initial state
            3) applies the "depth update" u^(l+1) = LayerNorm(\sigma(y^(l)) + u^(l))
            
        _ssm() does the state evolution + linear readout part

        :param u_NH: [..., seq_len, input_dim]
        :param alpha_NP: [..., seq_len, hidden_dim]
        :param dt_N: [..., seq_len]
        :param initial_state_P: [..., hidden_dim]
        
        u^(l+1) = LayerNorm( \sigma(y^(l)) + u^(l) )
        returns right_x_NP, next_layer_left_u_NH, next_layer_right_u_NH
        so it returns 
        1) the right limits of the P states (x) across the N sequence points (per batch)
        2) the left limits of what will be used as input for the NEXT layer
        3 the right limits of what will be used as input for the NEXT layer
        it should be noted here that although we refer to "y" as the output of the LLH layer in the literature,
        the output of this LLH() code is in fact the transformation of y, that is:
        u^(l+1) = LayerNorm( \sigma(y^(l)) + u^(l) )
        however, whether pre_norm or post_norm is toggled affects whether this LayerNorm is applied at the end
        of the current pass, or the beginning of the next pass.
        """
        # Pull out the dimensions.
        # the "*" denotes python's extended iterable unpacking (packing operator)
        # this allows for if mark_embedding has more than 3 axes, that the single "leading_dims"
        # can store all the preceding dims into one list instead of leading_dims1, leading_dims2, etc.
        *leading_dims, _, _ = mark_embedding_NH.shape # this will typically just be B
        num_leading_dims = len(leading_dims)

        # the state is P-dimensional, but it needs to be broacast across the B sequences 
        # basically we just need to reformat initial_state_P as a tensor of [...,P] tensor
        # (most likely a [B,P] tensor)
        if initial_state_P is None:
            # Pad and expand to match leading dimensions of input
            initial_state_P = self.initial_state_P.view(# view() returns a new tensor with the same data but of a different
                                                        # shape, without copying memory
                *[1 for _ in range(num_leading_dims)] # the * operator is argument unpacking. the view() method takes
                                                      # arguments one number at a time, not in a list, so since we want to
                                                      # feed the numbers from the list into view() we unpack it with *
                                                      # example foo(*[1,2,3]) -> foo(1,2,3)
                , -1 # this is the last input into the view() method. it basically makes the P dimension the final 
                     # dimension of this new state tensor we are formatting
                     # so so far, if we just B as the leading dim, the state tensor's shape is now (1,P)
                     # if we had, in addition, say G, it would be [1,1,P]
            ).expand(*leading_dims, -1) # does the final broadcasting without copying data
                                        # so if the current tensor has shape (1,1,64), if we call .expand(32,100,-1)
                                        # produces a tensor of size (32,100,64) 
                                        # the "-1" denoes "keep original size", whereas the "1" dimensions are broadcasted
                                        # to larger sizes

        # Add layer norm
        # prime_u is mean to represent the LayerNorm'd u if pre_norm=True
        # the reason the original u is not overwritten is because it is needed for the residual update when
        # computing the next layer's input: u^(l+1) = \sigma(y^(l)) + u^(l)
        # note that the above lacks the LayerNorm wrapper you see in the paper, because the paper presents the
        # post_norm = True version. for pre_norm=True, the normalized u is absorbed in y^l, and we need the pure
        # unnormalized u for the residual addition.
        prime_left_u_NH = left_u_NH
        prime_right_u_NH = right_u_NH
        if prime_left_u_NH is not None:  # ONLY for backward variant
            assert all(
                u_d == a_d
                for u_d, a_d in zip(prime_left_u_NH.shape, mark_embedding_NH.shape)
            )  # All but last dimensions should match
            if self.pre_norm:
                prime_left_u_NH = self.norm(prime_left_u_NH)
        if prime_right_u_NH is not None:
            assert all(
                u_d == a_d
                for u_d, a_d in zip(prime_right_u_NH.shape, mark_embedding_NH.shape)
            )  # All but last dimensions should match
            if self.pre_norm:
                prime_right_u_NH = self.norm(prime_right_u_NH)

        right_x_NP, left_y_NH, right_y_NH = self._ssm(
            left_u_NH=prime_left_u_NH,
            right_u_NH=prime_right_u_NH,
            impulse_NP=self.compute_impulse(prime_right_u_NH, mark_embedding_NH),
            dt_N=dt_N,
            initial_state_P=initial_state_P,
        )

        # Given the following:
        # right_u: u0, u1, u2, ... <-> u_{t_0}, u_{t_1}, u_{t_2}, ...
        # left_u: u0, u1, u2, ... <-> u_{t_0-}, u_{t_1-}, u_{t_2-}, ...
        # a: a0, a1, a2, ... <-> mark embeddings for m_0, m_1, m_2, ... at times t_0, t_1, t_2
        # dt: dt0, dt1, dt2, ... <-> 0, t_1-t_0, t_2-t_1, ...
        # initial_state_p: hidden state to evolve to to compute x_{0}

        # Returns the following:
        # right_x: x0, x1, x2, ... <-> x_{t_0}, x_{t_1}, x_{t_2}, ...
        # right_y: y0, y1, y2, ... <-> y_{t_0}, y_{t_1}, y_{t_2}, ...
        # left_y: y0, y1, y2, ... <-> y_{t_0-}, y_{t_1-}, y_{t_2-}, ...

        next_layer_left_u_NH = next_layer_right_u_NH = None
        if left_y_NH is not None:
            next_layer_left_u_NH = self.act_func(left_y_NH) + (
                left_u_NH if left_u_NH is not None else 0.0
            )
            if self.post_norm:
                next_layer_left_u_NH = self.norm(next_layer_left_u_NH)
        if right_y_NH is not None:
            next_layer_right_u_NH = self.act_func(right_y_NH) + (
                right_u_NH if right_u_NH is not None else 0.0
            )
            if self.post_norm:
                next_layer_right_u_NH = self.norm(next_layer_right_u_NH)
        return right_x_NP, next_layer_left_u_NH, next_layer_right_u_NH

    def _ssm(
        self,
        left_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        right_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        impulse_NP: th.Tensor,
        dt_N: th.Tensor,  # [0, t_1 - t_0, ..., t_N - t_{N-1}]
        initial_state_P: th.Tensor,
    ):
        *leading_dims, N, P = impulse_NP.shape
        u_NH = right_u_NH  # This implementation does not use left_u, nor does it compute left_y
        if u_NH is not None:
            impulse_NP = impulse_NP + th.einsum(
                "ph,...nh->...np",
                self.B_tilde_PH,
                u_NH.type(th.complex64) if self.complex_values else u_NH,
            )
            y_u_res_NH = th.einsum(
                "...nh,h->...nh", u_NH, self.D_HH
            )  # D_HH should really be D_H
        else:
            assert self.is_first_layer
            y_u_res_NH = 0.0

        lambda_res = self.get_lambda(right_u_NH=right_u_NH, shift_u=True)
        if "lambda_rescaled_P" in lambda_res:  # original formulation
            lambda_dt_NP = th.einsum(
                "...n,p->...np", dt_N, lambda_res["lambda_rescaled_P"]
            )
        else:  # relative time
            lambda_dt_NP = th.einsum(
                "...n,...np->...np", dt_N, lambda_res["lambda_rescaled_NP"]
            )

        if self.for_loop:
            right_x_P = initial_state_P
            right_x_NP = []
            for i in range(N):
                right_x_P = (
                    lambda_dt_NP[..., i, :].exp() * right_x_P + impulse_NP[..., i, :]
                )
                right_x_NP.append(right_x_P)
            right_x_NP = th.stack(right_x_NP, dim=-2)
        else:
            # Trick inspired by: https://github.com/PeaBrane/mamba-tiny/blob/master/scans.py
            # .unsqueeze(-2) to add sequence dimension to initial state
            log_impulse_Np1_P = th.concat(
                (initial_state_P.unsqueeze(-2), impulse_NP), dim=-2
            ).log()
            lamdba_dt_star = F.pad(lambda_dt_NP.cumsum(-2), (0, 0, 1, 0))
            right_x_log_NP = (
                th.logcumsumexp(log_impulse_Np1_P - lamdba_dt_star, -2) + lamdba_dt_star
            )
            right_x_NP = right_x_log_NP.exp()[..., 1:, :]

        conj_sym_mult = 2 if self.conj_sym else 1
        y_NH = (
            conj_sym_mult
            * th.einsum("...np,hp->...nh", right_x_NP, self.C_tilde_HP).real
            + y_u_res_NH
        )

        return right_x_NP, None, y_NH

    def get_left_limit(
        self,
        right_limit_P: th.Tensor,  # Along with dt, can have any number of leading dimensions, produces a tensor of dim ...MP
        dt_G: th.Tensor,
        current_right_u_H: th.Tensor,
        next_left_u_GH: th.Tensor,
    ) -> th.Tensor:
        """
        To get the left limit, we roll on the layer for the right dt.
        Computed for a single point (vmap for multiple).

        :param right_limit_P: at [t_0, ..., t_{N-1}]
        :param dt: Length of time to roll the layer on for. at [t_1 - t_0, ..., t_N - t_{N-1}]
        :param current_right_u_H: at [t_0, ..., t_{N-1}] -- for relative-time variant
        :param next_left_u_GH: at [t_1, ..., t_N] -- for backward variant

        :return:
        """

        if current_right_u_H is not None and self.pre_norm:
            current_right_u_H = self.norm(current_right_u_H)

        lambda_res = self.get_lambda(
            current_right_u_H, shift_u=False
        )  # U should already be shifted
        if "lambda_rescaled_P" in lambda_res:
            lambda_bar_GP = th.exp(
                th.einsum("...g,p->...gp", dt_G, lambda_res["lambda_rescaled_P"])
            )
        else:
            lambda_bar_GP = th.exp(
                th.einsum("...g,...p->...gp", dt_G, lambda_res["lambda_rescaled_NP"])
            )

        return th.einsum("...p,...gp->...gp", right_limit_P, lambda_bar_GP)

    def depth_pass(
        self,
        current_left_x_P: th.Tensor,  # No leading dimensions (seq, batch, etc.) here because we accommodate any of them
        current_left_u_H: Optional[
            th.Tensor
        ],  # Just assume that x and u match in the leading dimensions. Produces y_H with equivalent leading dimensions
        prev_right_u_H: Optional[
            th.Tensor
        ],  # Just assume that x and u match in the leading dimensions. Produces y_H with equivalent leading dimensions
    ) -> th.Tensor:
        if current_left_u_H is not None:
            if self.pre_norm:
                prime_u_H = self.norm(current_left_u_H)
            else:
                prime_u_H = current_left_u_H
            y_u_res_H = th.einsum(
                "...h,h->...h", prime_u_H, self.D_HH
            )  # D_HH should really be D_H
        else:
            assert self.is_first_layer
            y_u_res_H = 0.0

        conj_sym_mult = 2 if self.conj_sym else 1
        y_H = (
            conj_sym_mult
            * th.einsum("...p,hp->...h", current_left_x_P, self.C_tilde_HP).real
            + y_u_res_H
        )

        # Apply an activation function.
        if self.post_norm:
            new_u_H = self.norm(
                self.act_func(y_H)
                + (current_left_u_H if current_left_u_H is not None else 0.0)
            )
        else:
            new_u_H = self.act_func(y_H) + (
                current_left_u_H if current_left_u_H is not None else 0.0
            )

        return new_u_H


class Int_Forward_LLH(LLH):
    # LLH but Bu_t is integrated w.r.t dt instead of dN_t
    # After discretization, when evolving x_t to x_t', applies ZOH on u_t over [t,t'] forward in time
    # (as opposed to u_{t'} backwards over [t,t'])

    def _ssm(
        self,
        left_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        right_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        impulse_NP: th.Tensor,
        dt_N: th.Tensor,
        initial_state_P: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Apply the linear SSM to the inputs.

        In the context of TPPs, this returns the right limit of the "intensity function".
        This intensity will have been passed through a non-linearity, though, and so there is no
        guarantee for it is positive.

        :param u_NH: [..., seq_len, input_dim]
        :param alpha_NP: [..., seq_len, hidden_dim]
        :param dt_N: [..., seq_len]
        :param initial_state_P: [..., hidden_dim]

        :return:
        """
        # Pull out the dimensions.
        *leading_dims, N, P = impulse_NP.shape

        lambda_res = self.get_lambda(right_u_NH=right_u_NH, shift_u=True)
        if "lambda_rescaled_P" in lambda_res:
            lambda_rescaled = lambda_res["lambda_rescaled_P"]
            lambda_dt_NP = th.einsum(
                "...n,p->...np", dt_N, lambda_res["lambda_rescaled_P"]
            )
        else:
            lambda_rescaled = lambda_res["lambda_rescaled_NP"]
            lambda_dt_NP = th.einsum(
                "...n,...np->...np", dt_N, lambda_res["lambda_rescaled_NP"]
            )

        if left_u_NH is not None:
            left_Du_NH = th.einsum(
                "...nh,h->...nh",
                left_u_NH,
                self.D_HH,
            )
        else:
            assert self.is_first_layer
            left_Du_NH = 0.0

        if right_u_NH is not None:
            right_u_NH = F.pad(right_u_NH[..., :-1, :], (0, 0, 1, 0))
            right_Bu_NP = th.einsum(
                "...np,ph,...nh->...np",
                lambda_dt_NP.exp() - 1.0,  # dts: [0, t1-t0, t2-t1, ...]
                self.B_tilde_PH,
                right_u_NH.type(th.complex64) if self.complex_values else right_u_NH,
            )
            right_Du_NH = th.einsum(
                "...nh,h->...nh",
                right_u_NH,
                self.D_HH,
            )
        else:
            assert self.is_first_layer
            right_Bu_NP = right_Du_NH = 0.0

        if self.for_loop:
            right_x_P = initial_state_P
            left_x_NP, right_x_NP = [], []
            for i in range(N):
                left_x_P = lambda_dt_NP[..., i, :].exp() * right_x_P + (
                    right_Bu_NP[..., i, :] if left_u_NH is not None else 0.0
                )
                right_x_P = left_x_P + impulse_NP[..., i, :]
                left_x_NP.append(left_x_P)
                right_x_NP.append(right_x_P)
            right_x_NP = th.stack(
                right_x_NP, dim=-2
            )  # discard initial_hidden_states, right_limit of xs for [t0, t1, ...]
            left_x_NP = th.stack(
                left_x_NP, dim=-2
            )  # discard initial_hidden_states, left_limit of xs for [t0, t1, ...]
        else:
            # Trick inspired by: https://github.com/PeaBrane/mamba-tiny/blob/master/scans.py
            # .unsqueeze(-2) to add sequence dimension to initial state
            log_impulse_Np1_P = th.concat(
                (initial_state_P.unsqueeze(-2), right_Bu_NP + impulse_NP), dim=-2
            ).log()
            lamdba_dt_star = F.pad(lambda_dt_NP.cumsum(-2), (0, 0, 1, 0))
            right_x_log_NP = (
                th.logcumsumexp(log_impulse_Np1_P - lamdba_dt_star, -2) + lamdba_dt_star
            )
            right_x_NP = right_x_log_NP.exp()  # Contains initial_state_P in index 0
            left_x_NP = (
                lambda_dt_NP.exp() * right_x_NP[..., :-1, :] + right_Bu_NP
            )  # Evolves previous hidden state forward to compute left limit
            right_x_NP = right_x_NP[..., 1:, :]

        conj_sym_mult = 2 if self.conj_sym else 1
        left_y_NH = (
            conj_sym_mult
            * th.einsum("hp,...np->...nh", self.C_tilde_HP, left_x_NP).real
            + left_Du_NH
        )  # ys for [t0, t1, ...]
        right_y_NH = (
            conj_sym_mult
            * th.einsum("hp,...np->...nh", self.C_tilde_HP, right_x_NP).real
            + right_Du_NH
        )  # ys for [t0, t1, ...]

        return right_x_NP, left_y_NH, right_y_NH

    def get_left_limit(
        self,
        right_limit_P: th.Tensor,  # Along with dt, can have any number of leading dimensions, produces a tensor of dim ...MP
        dt_G: th.Tensor,
        current_right_u_H: Optional[th.Tensor],
        next_left_u_GH: Optional[th.Tensor],
    ) -> th.Tensor:
        """
        To get the left limit, we roll on the layer for the right dt.
        Computed for a single point (vmap for multiple).

        :param right_limit_P:
        :param dt: Length of time to roll the layer on for.
        :return:
        """
        if current_right_u_H is not None and self.pre_norm:
            current_right_u_H = self.norm(current_right_u_H)

        lambda_res = self.get_lambda(
            current_right_u_H, shift_u=False
        )  # U should already be shifted
        if "lambda_rescaled_P" in lambda_res:
            lambda_bar_GP = th.exp(
                th.einsum("...g,p->...gp", dt_G, lambda_res["lambda_rescaled_P"])
            )
        else:
            lambda_bar_GP = th.exp(
                th.einsum("...g,...p->...gp", dt_G, lambda_res["lambda_rescaled_NP"])
            )

        # lambda_rescaled_P = th.exp(self.log_step_size_P) * self.Lambda_P
        # lambda_bar_GP = th.exp(th.einsum('...g,p->...gp', dt_G, lambda_rescaled_P))
        int_hidden_GP = th.einsum("...p,...gp->...gp", right_limit_P, lambda_bar_GP)

        if current_right_u_H is None:  # no Bu term
            assert self.is_first_layer
            return int_hidden_GP
        else:  # add Bu to impulse
            if self.pre_norm:
                current_right_u_H = self.norm(current_right_u_H)

            impulse_GP = th.einsum(
                "...gp,ph,...h->...gp",
                lambda_bar_GP - 1.0,
                self.B_tilde_PH,
                current_right_u_H.type(th.complex64)
                if self.complex_values
                else current_right_u_H,
            )

            return int_hidden_GP + impulse_GP


class Int_Backward_LLH(Int_Forward_LLH):
    # LLH but Bu_t is integrated w.r.t dt instead of dN_t
    # After discretization, when evolving x_t to x_t', applies ZOH on u_t' over [t,t'] backwards in time
    # (as opposed to u_{t} forwards over [t,t'])

    def _ssm(
        self,
        left_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        right_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        impulse_NP: th.Tensor,
        dt_N: th.Tensor,
        initial_state_P: th.Tensor,
    ) -> Tuple[th.Tensor, th.Tensor]:
        """
        Apply the linear SSM to the inputs.

        In the context of TPPs, this returns the right limit of the "intensity function".
        This intensity will have been passed through a non-linearity, though, and so there is no
        guarantee for it is positive.

        :param u_NH: [..., seq_len, input_dim]
        :param alpha_NP: [..., seq_len, hidden_dim]
        :param dt_N: [..., seq_len]
        :param initial_state_P: [..., hidden_dim]

        :return:
        """
        # Pull out the dimensions.
        *leading_dims, N, P = impulse_NP.shape

        # lambda_rescaled_P = th.exp(self.log_step_size_P) * self.Lambda_P
        # lambda_dt_NP = th.einsum('...n,p->...np', dt_N, lambda_rescaled_P)
        lambda_res = self.get_lambda(right_u_NH=right_u_NH, shift_u=True)
        if "lambda_rescaled_P" in lambda_res:
            lambda_dt_NP = th.einsum(
                "...n,p->...np", dt_N, lambda_res["lambda_rescaled_P"]
            )
        else:
            lambda_dt_NP = th.einsum(
                "...n,...np->...np", dt_N, lambda_res["lambda_rescaled_NP"]
            )

        if left_u_NH is not None:
            left_Bu_NP = th.einsum(
                "...np,ph,...nh->...np",
                lambda_dt_NP.exp() - 1.0,  # dts: [0, t1-t0, t2-t1, ...]
                self.B_tilde_PH,
                left_u_NH.type(th.complex64) if self.complex_values else left_u_NH,
            )
            left_Du_NH = th.einsum(
                "...nh,h->...nh",
                left_u_NH,
                self.D_HH,
            )
        else:
            assert self.is_first_layer
            left_Bu_NP = left_Du_NH = 0.0

        if right_u_NH is not None:
            right_Du_NH = th.einsum(
                "...nh,h->...nh",
                right_u_NH,
                self.D_HH,
            )
        else:
            assert self.is_first_layer
            right_Du_NH = 0.0

        if self.for_loop:
            right_x_P = initial_state_P
            left_x_NP, right_x_NP = [], []
            for i in range(N):
                left_x_P = lambda_dt_NP[..., i, :].exp() * right_x_P + (
                    left_Bu_NP[..., i, :] if left_u_NH is not None else 0.0
                )
                right_x_P = left_x_P + impulse_NP[..., i, :]
                left_x_NP.append(left_x_P)
                right_x_NP.append(right_x_P)
            right_x_NP = th.stack(
                right_x_NP, dim=-2
            )  # discard initial_hidden_states, right_limit of xs for [t0, t1, ...]
            left_x_NP = th.stack(
                left_x_NP, dim=-2
            )  # discard initial_hidden_states, left_limit of xs for [t0, t1, ...]
        else:
            # Trick inspired by: https://github.com/PeaBrane/mamba-tiny/blob/master/scans.py
            # .unsqueeze(-2) to add sequence dimension to initial state
            log_impulse_Np1_P = th.concat(
                (initial_state_P.unsqueeze(-2), left_Bu_NP + impulse_NP), dim=-2
            ).log()
            lamdba_dt_star = F.pad(lambda_dt_NP.cumsum(-2), (0, 0, 1, 0))
            right_x_log_NP = (
                th.logcumsumexp(log_impulse_Np1_P - lamdba_dt_star, -2) + lamdba_dt_star
            )
            right_x_NP = right_x_log_NP.exp()  # Contains initial_state_P in index 0
            left_x_NP = (
                lambda_dt_NP.exp() * right_x_NP[..., :-1, :] + left_Bu_NP
            )  # Evolves previous hidden state forward to compute left limit
            right_x_NP = right_x_NP[..., 1:, :]

        conj_sym_mult = 2 if self.conj_sym else 1
        left_y_NH = (
            conj_sym_mult
            * th.einsum("hp,...np->...nh", self.C_tilde_HP, left_x_NP).real
            + left_Du_NH
        )  # ys for [t0, t1, ...]
        right_y_NH = (
            conj_sym_mult
            * th.einsum("hp,...np->...nh", self.C_tilde_HP, right_x_NP).real
            + right_Du_NH
        )  # ys for [t0, t1, ...]

        return right_x_NP, left_y_NH, right_y_NH

    def get_left_limit(
        self,
        right_limit_P: th.Tensor,  # Along with dt, can have any number of leading dimensions, produces a tensor of dim ...MP
        dt_G: th.Tensor,
        current_right_u_H: th.Tensor,
        next_left_u_GH: th.Tensor,
    ) -> th.Tensor:
        """
        To get the left limit, we roll on the layer for the right dt.
        Computed for a single point (vmap for multiple).

        :param right_limit_P:
        :param dt: Length of time to roll the layer on for.
        :return:
        """

        if current_right_u_H is not None and self.pre_norm:
            current_right_u_H = self.norm(current_right_u_H)

        lambda_res = self.get_lambda(
            current_right_u_H, shift_u=False
        )  # U should already be shifted
        if "lambda_rescaled_P" in lambda_res:
            lambda_bar_GP = th.exp(
                th.einsum("...g,p->...gp", dt_G, lambda_res["lambda_rescaled_P"])
            )
        else:
            lambda_bar_GP = th.exp(
                th.einsum("...g,...p->...gp", dt_G, lambda_res["lambda_rescaled_NP"])
            )

        int_hidden_GP = th.einsum("...p,...gp->...gp", right_limit_P, lambda_bar_GP)

        if next_left_u_GH is None:  # no Bu term
            assert self.is_first_layer
            return int_hidden_GP
        else:  # add Bu to impulse
            if self.pre_norm:
                next_left_u_GH = self.norm(next_left_u_GH)

            impulse_GP = th.einsum(
                "...gp,ph,...gh->...gp",
                lambda_bar_GP - 1.0,
                self.B_tilde_PH,
                next_left_u_GH.type(th.complex64)
                if self.complex_values
                else next_left_u_GH,
            )

            return int_hidden_GP + impulse_GP
