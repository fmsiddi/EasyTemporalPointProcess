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
        # apparently the reason this optionality is provided is for backpropagation purposes. sometimes the gradient
        # is more well-behaved when the LayerNorm is applied before the SSM, sometimes after.
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
        # see B.3 of S2P2 paper: \delta_i is a time-scaling diagonal matrix that changes the relative timescale
        # of each channel of the state's influence (a scaling for each p). large values suggest time should pass quickly
        # for that channel, suggesting faster convergence to steady-state, and smaller values will cause the model to 
        # retain the influence that prior events have on future ones due to the slower perceived passage of time.
        # if relative_time = True, we make this a learnable network via diag(softplus(W * u_t_i + b)) and call it "delta_net"
        # if relative_time = False, then instead of "delta_net", we use "log_step_size_P" and initialize it to 0 and make it unlearnable.
        # why they make it unlearnable is beyond me, but chatgpt says is that nn.Parameters move to gpu/cuda devices automatically when
        # calling model.cuda()
        # it also said they could have used self.register_buffer("log_step_size_P", torch.zeros(self.P)) and gotten the same effect
        # TODO: maybe reach out to Yuxin and inquire
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
            # if we opt to not have input-dependent dynamics, we create a constant 
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
        TODO: it doesn't appear that right_u_h is needed here since all we're calculating is \tilde{E} * \alpha
              we can probably remove this as an input.
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
    \Lambda_i = diag(softplus(W * u_t_i + b)) * \Lambda (Below equation (13) in the paper)
    this is just implementing that. the delta_net is W * u_t_i + b (if relative_time is True),
    otherwise \Lambda_i = \exp(log(\delta_t))* Lambda = \delta_t * \Lambda
    the "diag" part is just handled by the fact that softplus(W * u_t_i + b) is ...xNxP and Lambda_P is P-dimensional vector
    there is also some logic here if the right limit of the input u is unavailable

    it should also be noted here that the output dimension varies baseed on relative_time flag and whether or not right_u is available.
    in the case where we are doing input-dependent dynamics with a changing delta_t, then the output tensor will be (...,N,P)
    whereas if we just doing constant delta_t, then the tensor will be just be (...,P)
    even if relative_time=Ture, if right_u is not available, then the resulting matrix will also just be (...,P)
    the names of these keys are relevant in _ssm() because they are used to check whether relative_time = TRUE and if right_u is available
    simultaneously.
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
                # - a weight matrix W that is PxH
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
            if self.relative_time: # relative_time is true but it's the first layer
                lambda_rescaled_P = F.softplus(self.delta_net.bias) * self.Lambda_P
            else:
                lambda_rescaled_P = th.exp(self.log_step_size_P) * self.Lambda_P  # remember her that log_step_size_P is just a fixed
                                                                                  # tensor of 0's. i don't really understand why they
                                                                                  # do this here and not just set equal to self.Lambda_P
                                                                                  # TODO: perhaps change this
            return {"lambda_rescaled_P": lambda_rescaled_P} # note here that the key lambda_rescaled_P differs from lambda_rescaled_NP!
                                                            # this is the distinguishing factor used in _ssm() to check if 
                                                            # relative_time=False

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

        # Add layer norm if pre_norm = True
        # prime_u is mean to represent the LayerNorm'd u if pre_norm=True
        # the reason the original u is not overwritten is because it is needed for the residual update when
        # computing the next layer's input: u^(l+1) = \sigma(y^(l)) + u^(l)
        # note that the above lacks the LayerNorm wrapper you see in the paper, because the paper presents the
        # post_norm = True version. for pre_norm=True, the normalized u is absorbed in y^l, and we need the pure
        # unnormalized u for the residual addition.
        # apparently the reason this optionality is provided is for backpropagation purposes. sometimes the gradient
        # is more well-behaved when the LayerNorm is applied before the SSM, sometimes after.
        prime_left_u_NH = left_u_NH
        prime_right_u_NH = right_u_NH
        if prime_left_u_NH is not None:  # ONLY for backward variant
            # the below assert block found in each if clause is just a shape check.
            # the zip() method pairs elements of one list with the elements of the other list in order.
            # so if x.shape = y.shape = (B,N,H), then zip(x.shape,y.shape) will produce pairs (B,B) (N,N) and (H,H)
            # the for loop checks to see that each of these dimensions do match, and the assert all() checks that every
            # output of the for loop is True. (all() returns True if every element is True, False otherwise)
            # assert will throw an error if it returns false.
            assert all(
                u_d == a_d
                for u_d, a_d in zip(prime_left_u_NH.shape, mark_embedding_NH.shape)
            )  # All but last dimensions should match
               # this comment above is misleading, because it's checking that ALL dimensions match, including the last one...
            if self.pre_norm:
                prime_left_u_NH = self.norm(prime_left_u_NH)
        if prime_right_u_NH is not None:
            assert all(
                u_d == a_d
                for u_d, a_d in zip(prime_right_u_NH.shape, mark_embedding_NH.shape)
            )  # All but last dimensions should match
               # this comment above is misleading, because it's checking that ALL dimensions match, including the last one...
            if self.pre_norm:
                prime_right_u_NH = self.norm(prime_right_u_NH)

        # this _ssm() call is what actually carries out the bulk of of Algorithm 1
        right_x_NP, left_y_NH, right_y_NH = self._ssm(
            left_u_NH=prime_left_u_NH,
            right_u_NH=prime_right_u_NH,
            impulse_NP=self.compute_impulse(prime_right_u_NH, mark_embedding_NH), # \tilde{E} * \alpha
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
                                         # left_u is used in forward variant, not here.
        right_u_NH: Optional[th.Tensor],  # Very first layer, should feed in `None`
        impulse_NP: th.Tensor,
        dt_N: th.Tensor,  # [0, t_1 - t_0, ..., t_N - t_{N-1}]
        initial_state_P: th.Tensor,
    ):
        *leading_dims, N, P = impulse_NP.shape
        u_NH = right_u_NH  # This implementation does not use left_u, nor does it compute left_y
        if u_NH is not None: # meaning: if we are not in the first layer
            # here is an important distinction between how the "vanilla" LLH class behaves vs. its more true-to-form ZOH variants below.
            # \tilde{B}*u is treated like an event-syncronous jump-like additive term, getting wrapped up with \tidle{E}*\alpha instead
            # of being discretized as shown in the paper. that is to say, this base LLH class REINTERPRETS u as an event-tied input 
            # (a learned impulse) rather thatn a continuous-time control held between events. it is NOT the ZOH discretization we
            # see in the paper.
            impulse_NP = impulse_NP + th.einsum( # \tilde{E}*\alpha + \tilde{B}*u
                "ph,...nh->...np",
                self.B_tilde_PH,
                u_NH.type(th.complex64) if self.complex_values else u_NH,
            )
            y_u_res_NH = th.einsum( # D * u
                "...nh,h->...nh", u_NH, self.D_HH
            )  # D_HH should really be D_H
        else:
            assert self.is_first_layer
            y_u_res_NH = 0.0

        # this gives us \Lambda_i * delta_t. we will need to exponentiate this to get the discretized
        # version \bar{Lambda}_i
        lambda_res = self.get_lambda(right_u_NH=right_u_NH, shift_u=True) # retrieve Lambda_i
        if "lambda_rescaled_P" in lambda_res:  # original formulation
                                               # Lambda_i=\Lambda (no rescaling) will only be P-dimensional if right_u is unavailable 
                                               # or relative_time=False
            lambda_dt_NP = th.einsum( # \Lambda_P * delta_t
                "...n,p->...np", dt_N, lambda_res["lambda_rescaled_P"]
            )
        else:  # relative_time = True and right_u is available
            lambda_dt_NP = th.einsum( # \delta_i * \Lambda * delta_t (\delta_i in the input-dependent rescaler for each channel of x)
                "...n,...np->...np", dt_N, lambda_res["lambda_rescaled_NP"]
            )

        # so far we have not touched anything that has to do with the state x yet.

        if self.for_loop: # this is the slow, recursive version. probably used for debugging/reference
            right_x_P = initial_state_P
            right_x_NP = []
            for i in range(N): # again, only A is discretized in this base LLH implementation. we compute x via
                               # x = exp(\Lambda_i * dt) * right_x + \tilde{B}*u + \tilde{E}*\alpha
                right_x_P = (
                    lambda_dt_NP[..., i, :].exp() * right_x_P + impulse_NP[..., i, :]
                )
                right_x_NP.append(right_x_P)
            right_x_NP = th.stack(right_x_NP, dim=-2)
        else:
            # Trick inspired by: https://github.com/PeaBrane/mamba-tiny/blob/master/scans.py
            # .unsqueeze(-2) to add sequence dimension to initial state
            # this is not the classic Blelloch affine-pair scan, but an algebraic transformation that
            # reduces the recurrence to a stabilized prefix-sum in log space.
            
            # as a quick review, we can perform parallel scans (running sums) when we have an associative operator \circdot
            # if we have x_i = r_i * x_{i-1} + b_i
            #            r_i = exp(\lambda_i * \delta_{t_i})
            # then we can perform a scan over an assoicative operator on affine transforms
            # we can let T_i() represent the affine map at step i: T_i(x) := r_i * x + b_i
            # recursively: T_2(T_1(x)) = r_2 * (r_1 * x + b_1) + b_2 = (r_2 * r_1) * x + (r_2 * b_1 + b_2)
            # we can define this recursion using the associative operator \circdot: 
            # (r_2, b_2) \circdot (r_1, b_1) = (r_2 * r_1, r_2 * b_1 + b_2)
            # so the "true" parallel-scan view is: (r_{0:i}, b_{0:i}) = (r_i, b_i) \circdot ... \circdot (r_0, b_0)
            
            # this code is NOT doing that as it is presented in the S2P2 paper, instead, it is drawing from the paper
            # "Efficient Parallelization of a Ubiquitous Sequential Computation" (Heinsen 2023) that uses a cumulative sum
            # and logcumsumexp (LCSE) trick. it works as follows:
            
            # assume x_t = a_t * x_{t-1} + b_t
            # the vector log x_t is computable as a composition of two cumulative (aka prefix) sums, each of which is parallelizable:
            # log x_t = a_t^* + log(x_0 + b_t^*), with the following prefix sums:
            # a_t^* = \sum_t^cum log a_t
            # b_t^* = \sum_t^cum exp(log(b_t) - a_t^*)
            # x can then be computed via element-wise exponentiation
            # x_t = exp(a_t^* + log(x_0 + b_t^*))
            # if you read the paper they eventually show that:
            # x_t = (\prod_t^{cum} a_t) \circdot (x_0 + \sum_t^cum exp(log(b_t) - \sum_t^cum log(a_t)))
            # taking the logarithm of both sides yields
            # log x_t = \sum_t^cum log a_t + log(x_0 + \sum_t^cum exp(log(b_t) - \sum_t^cum log(a_t))) 
            # this is carried out numerically via the nested function call:
            
            # log x_t = log(a_t^*) + log(tail(LCSE(cat(log x_0, log (b_t) - a_t^*))))
            
            # cat() denotes concatenation
            # tail() removes its argument's first element
            # LCSE() is the logcumsumexp function, which computes log(\sum_t^cum exp()) in a numerically stable way.
            # that is what righ_x_log_NP is below, just in the case of tensors instead of vectors.
            log_impulse_Np1_P = th.concat(
                (initial_state_P.unsqueeze(-2), impulse_NP), dim=-2 # recall initial_state_P is (B,P) and impulse_NP is (B,N,P), 
                                                                    # so we need to unsqueeze initial_state_P to make it (B,1,P) 
                                                                    # before concatenating along the sequence dimension N (-2)
            # here's an example of what this may look like: let B=2, N=3, P=2
            # let intial_state_P = [[1,2],
            #                       [3,4]] (shape (2,2))
            # then initial_state_P.unsqueeze(-2) = [[[1,2]],
            #                                       [[3,4]]] (shape (2,1,2) (note the double bracket before the comma))
            # let impulse_NP = [[[10,11],[12,13],[14,15]],
            #                   [[16,17],[18,19],[20,21]]] (shape (2,3,2))
            # you can see each state is a 2-element array, and there are 3 states per sequence, and we have 2 of these sequences in the batch
            # if you want to visually identify the axes, the P axis exists within the inner-most brackets
            # the N (sequence) dimension can be read horizontally across rows, with each element being a 2-dimensional state
            # the B (batch) dimension can be read vertically as the columns of these rows, so each new row represents another batch
            # that is, each row represents a separate sequence
            # so, when we concatenate initial_state_P.unsqueeze(-2) along dim=-2 (the sequence dimension N), we get:
            # log_impulse_Np1_PP = [[[1,2],[10,11],[12,13],[14,15]],
            #                       [[3,4],[16,17],[18,19],[20,21]]] (shape (2,4,2))
            # so we have appended the initial state to the beginning of each sequence in the batch. represented symbollically:
            # log_impulse_Np1_P = [V^{-1}*x_0, 
            #                      \tilde{B}*u_0 + \tilde{E}*\alpha_0, 
            #                      \tilde{B}*u_1 + \tilde{E}*\alpha_1, 
            #                      \tilde{B}*u_2 + \tilde{E}*\alpha_2] for each sequence in the batch
            
            # NOTE: we have one additional element in the sequence dimension now (N+1 instead of N), so we will need some padding below:
            ).log() # we take the log of this concatenated tensor to prepare for the LCSE step. this is the log(b_t) term in the LCSE formula
            lamdba_dt_star = F.pad(lambda_dt_NP.cumsum(-2), (0, 0, 1, 0)) # lambda_dt_NP is just \Lambda_i * delta_t which is (BNP)-dimensional
                                                                          # we need to compute the cumulative sum of this along the sequence 
                                                                          # dimension N (-2) while adding a pad entry of 1 at the beginning of the 
                                                                          # sequence dimension.
                                                                          # this is essentially a_t^* in the extended annotation above.
            right_x_log_NP = (
                th.logcumsumexp(log_impulse_Np1_P - lamdba_dt_star, -2) + lamdba_dt_star # the -2 denotes that the logcumsumexp is being 
                                                                                         # taken along the sequence dimension N, and the 
                                                                                         # addition of lamdba_dt_star is the final step in 
                                                                                         # the LCSE formula to recover log x_t from the 
                                                                                         # intermediate variable. see the extended annotation 
                                                                                         # above for more details.
            )
            right_x_NP = right_x_log_NP.exp()[..., 1:, :] # finally we exponentiate, ignoring the first dimension we padded earlier

        conj_sym_mult = 2 if self.conj_sym else 1
        y_NH = ( # remember we are taking the complex \tilde{x} and coverting back to real x. 
                 # if y = \tilde{C} * \tilde{x} + D * u, we use the fact that \tilde{x} = V * x, and V yields from the diagonalization of A,
                 # which we assume has conjugate symmetery if conj_sym=True, so ignoring the D*u for a bit, if we write the matrix 
                 # multiplication in summation form, we have S = \sum_1^P \tilde{C}_{hp} * \tilde{x}_p
                 # and since x MUST be real, and D*u is real, we know that the \sum_1^P \tilde{C}_{hp} * \tilde{x}_p must have cancellation
                 # of its imaginary parts. and since we know there is conjugate symmetry, then if z = \tilde{C}_{hp} * \tilde{x}_p
                 # then we have S = z + \bar{z} = 2*Re(z) = 2*Re(\tilde{C}_{hp} * \tilde{x}_p). 
            conj_sym_mult
            * th.einsum("...np,hp->...nh", right_x_NP, self.C_tilde_HP).real
            + y_u_res_NH # + D*u
        )
        # TODO: to be honest, this final line seems suspect. if conj_sym=False, why are we only keeping the real part?
        #       it also looks like we never take advantage of the fact that conj_sym=True would allows us to use half the memory,
        #       because we are still storing the full complex \tilde{C} and \tilde{x} instead of just the half that is needed?

        return right_x_NP, None, y_NH

    # the final two methods are used for approximating the integral \int_0^T \lambda_t dt in the log likelihood used in training.
    # obviously we don't have the entire continuous trajactory of \lambda_t, so we must sample G (s_{k,g}) points along the trajectory between
    # the events, evolve the state to these points, and evaluate the intensity at these points.
    # this first method evolves x to these points and returns their left limits.
    # the fact that we are doing this between event times means we have no additive event impulse \tilde{E}*\alpha or input impulse \tilde{B}*u
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

        # in _ssm(), we were finding exp(\Lambda_i * delta_t) where delta_t was the time between events.
        # here we are finding exp(\Lambda_i * delta_s) where delta_s is the time between the event and the sampled point 
        # along the trajectory, so we can evolve the state to that point.
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

        # literally all it takes to get the left limit is to compute exp(\Lambda_i * delta_s) * x_{t_i} and that's it.
        return th.einsum("...p,...gp->...gp", right_limit_P, lambda_bar_GP)

    def depth_pass( # this method is for producing the final input u^(L+1) for the interarrival points dt_G so the intensity
                    # can be computed at those points. this differs from forward() since forward is for event times, which 
                    # requires event impulse terms.
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
            # what they mean here is that the "intensity" u^(L+1) = LayerNorm(\sigma(y^(L)) + u^(L))
            # is not guaranteed to be positive UNTIL it is passed thru the scaled softplus:
            # \lambda_t = ScaledSoftplus(Linear(u_{t-}^(L+1)))
            
        Another conceptual difference here is that whereas in LLH we combine Bu and E\alpha into a single
        event impulse to get the RIGHT limit of x, here we separate them, since Bu gets us the left limit of x,
        and then adding E\alpha get us the right limit.

        :param u_NH: [..., seq_len, input_dim]
        :param alpha_NP: [..., seq_len, hidden_dim]
        :param dt_N: [..., seq_len]
        :param initial_state_P: [..., hidden_dim]

        :return:
        """
        # Pull out the dimensions.
        *leading_dims, N, P = impulse_NP.shape

        # although the base LLH class computes \tilde{B}*u first since it adds it as an event impulse,
        # here the \Lambda_i*dt is computed first since it is needed to compute the discretized version
        # of AB: \bar{AB} = (\exp(\Lambda_i * dt) - I) * \tilde{B}
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

        # calculating Bu and Du
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

        if self.for_loop: # slow recursive version, probably used for debugging/reference
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
            # parallel scan version to get right limit of x
            # Trick inspired by: https://github.com/PeaBrane/mamba-tiny/blob/master/scans.py
            # .unsqueeze(-2) to add sequence dimension to initial state
            log_impulse_Np1_P = th.concat(
                (initial_state_P.unsqueeze(-2), right_Bu_NP + impulse_NP), dim=-2
            ).log()
            lamdba_dt_star = F.pad(lambda_dt_NP.cumsum(-2), (0, 0, 1, 0))
            right_x_log_NP = (
                th.logcumsumexp(log_impulse_Np1_P - lamdba_dt_star, -2) + lamdba_dt_star
            )
            # note that in Algorithm 1, first the right limit is computed, then the left limit is computed
            # by subtracting the event impulse from the right limit:
            # x_i- = x_i+ -\tilde{E}*\alpha_i
            # here, we do not subtract the event impulse from the right limit. instead, we invoke
            # equation (15):
            # x_i- = \bar{A}*x_{i-1+} + \bar{AB}*u_{i-1+}
            # which excludes the event impulse. notice the change in subscript.
            # this explains the comment "Evolves previous..." below.
            # I believe the main reason they do this is to avoid catastrophic cancellation in the case of 
            # large event impulses, which would cause the right limit and left limit to be very close in value, 
            # and thus lead to numerical instability when computing the left limit as the difference of the right 
            # limit and the event impulse.
            right_x_NP = right_x_log_NP.exp()  # Contains initial_state_P in index 0
            left_x_NP = ( # note here that we actually compute the left-limit here unlike the LLH class
                lambda_dt_NP.exp() * right_x_NP[..., :-1, :] + right_Bu_NP # this is equation (15)
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

    # used for approximating the integral \int_0^T \lambda_t dt in the log likelihood used in training.
    # obviously we don't have the entire continuous trajactory of \lambda_t, so we must sample G (s_{k,g}) points along the trajectory 
    # between the events, evolve the state to these points, and evaluate the intensity at these points.
    # this method evolves x to these points and returns their left limits.
    # the fact that we are doing this between event times means we have no additive event impulse \tilde{E}*\alpha
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
        
        # this is unchanged from LLH
        # in _ssm(), we were finding exp(\Lambda_i * delta_t) where delta_t was the time between events.
        # here we are finding exp(\Lambda_i * delta_s) where delta_s is the time between the event and the sampled point 
        # along the trajectory, so we can evolve the state to that point.
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
        int_hidden_GP = th.einsum("...p,...gp->...gp", right_limit_P, lambda_bar_GP) # this is what LLH returns

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
