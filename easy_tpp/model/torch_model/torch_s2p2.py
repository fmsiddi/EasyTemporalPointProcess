from typing import List, Optional, Tuple, Union

import torch
from torch import nn

from easy_tpp.model.torch_model.torch_baselayer import ScaledSoftplus
from easy_tpp.model.torch_model.torch_basemodel import TorchBaseModel
from easy_tpp.ssm.models import LLH, Int_Backward_LLH, Int_Forward_LLH

# when creating any "layer" 
class ComplexEmbedding(nn.Module): # Embedding layer for complex-valued embeddings for the diagonalized SSM matrices 
                                   # (nn.Module is the base class for all neural network modules in PyTorch)
    def __init__(self, *args, **kwargs): # args and kwargs are passed to nn.Embedding
        super(ComplexEmbedding, self).__init__() # Initialize the parent nn.Module
            # super(ClassName,self) means Give me the next class in the method resolution order (MRO) after ClassName
        self.real_embedding = nn.Embedding(*args, **kwargs) 
        self.imag_embedding = nn.Embedding(*args, **kwargs)
        # ^use pytorch's "Embedding" layer to create a simple lookup table that stores embeddings of a fixed dictionary and size

        self.real_embedding.weight.data *= 1e-3 # initialize weights to small values
        self.imag_embedding.weight.data *= 1e-3

    # Forward pass to get complex embeddings (using pytorch's "complex" tensor class that creates complex tensors: 
    # https://docs.pytorch.org/docs/stable/generated/torch.complex.html)
    def forward(self, x): 
        return torch.complex(
            self.real_embedding(x),
            self.imag_embedding(x),
        )


class IntensityNet(nn.Module): # this is the equation found in section 3.4 of the S2P2 paper a little bit below equation (14)
    def __init__(self, input_dim, bias, num_event_types):
        super().__init__() # Initialize the parent nn.Module
        # nn.Linear is a linear transformation layer: y = xA^T + b
        # first input is input dimension, second is output dimension, third is whether to include bias term
        # the "net" stands for "network", even though it's just a single linear layer here
        self.intensity_net = nn.Linear(input_dim, num_event_types, bias=bias)
        self.softplus = ScaledSoftplus(num_event_types)

    def forward(self, x):
        return self.softplus(self.intensity_net(x))


class S2P2(TorchBaseModel):
    def __init__(self, model_config):
        """Initialize the model

        Args:
            model_config (EasyTPP.ModelConfig): config of model specs.
        """
        super(S2P2, self).__init__(model_config)
        self.n_layers = model_config.num_layers
        self.P = model_config.model_specs["P"]  # Hidden state dimension
        self.H = model_config.hidden_size  # Residual stream dimension
        self.beta = model_config.model_specs.get("beta", 1.0)
        self.bias = model_config.model_specs.get("bias", True)
        self.simple_mark = model_config.model_specs.get("simple_mark", True)

        layer_kwargs = dict(
            P=self.P,
            H=self.H,
            dt_init_min=model_config.model_specs.get("dt_init_min", 1e-4),
            dt_init_max=model_config.model_specs.get("dt_init_max", 0.1),
            act_func=model_config.model_specs.get("act_func", "full_glu"),
            dropout_rate=model_config.model_specs.get("dropout_rate", 0.0),
            for_loop=model_config.model_specs.get("for_loop", False),
            pre_norm=model_config.model_specs.get("pre_norm", True),
            post_norm=model_config.model_specs.get("post_norm", False),
            simple_mark=self.simple_mark,
            relative_time=model_config.model_specs.get("relative_time", False),
            complex_values=model_config.model_specs.get("complex_values", True),
        )

        # the following two variants are separate implemntations of the LLH layer used to compute the non-event integral term (\int\lambda(t) dt)
        # efficiently and without leakage by changing which direction certain "left-limit/pre-event" quantities are propagated.
        # S2P2 needs intensities at two kinds of times: event times which must use left-limit information, and sampled times inside intervals
        # for MC integration which must evolve continuously from known state events.
        # the forward variant appears to be a rewrite specialized for the MC integral path: start at the right-limit, move forward
        # by delta_t, decode.
        # the backward variant is made explicit in S2P2.forward: they layer returns next_layer_left_u_BNH, and then stores left_u_BNm1H
        # (pre-event residual stream) and uses it directly to compute intensities at event times.
        # this means the backward variant is designed to produce the pre-event ('left') residual stream u_t_i- directly, instead of
        # reconstructing left-limit information from right limits after the fact
        
        # after looking at easy_tpp.ssm.modles and looking at the LLH, Int_Forward_LLH, and Int_Backward_LLH classes,
        # the variant variables are triggers for which class to use in the if elif else block below. I will explain more down there
        int_forward_variant = model_config.model_specs.get("int_forward_variant", False)
        int_backward_variant = model_config.model_specs.get("int_backward_variant", False)
        assert (
            int_forward_variant + int_backward_variant
            
        ) <= 1  # Only one at most is allowed to be specified

        # for this next part, remember that u is not actually step-wise constant, it is an internal resudal-stream signal computed from
        # the continuously evolving state x(t) via u(t) = C x(t) + D u_prev + impulse. So u(t) can differ across the interval even if
        # there are no events. In these LLH layers, even between the events:
        # x(s) decays continuously from x_t to x_t' according to the matrix exponential of Lambda (SSM dynamics)
        # y(s) depends on x(s)
        # u(s) is obtained from y(s) by a nonlinear/residual transformation (in this code, the "depth pass" path)
        # this is why the code has two different ways to plug u into the \int e^{\Lambda(\cdot)}*B*u(s) ds term:
            # forward variant uses uses current_right_u_H (the start-of_interval value u(t+))
            # backward variant uses next_left_u_GH (the end-of-interval value u(t'-))
        # both are approximations to the integral term, but they differ in which side of the interval they use to hold u constant.
        
        # if using forward, then ZOH holds u_t constant [t,t'] forward in time: u(s) = u_t for s in [t,t']
        # here the Bu term is integrated wrt ds instead of dNt as it is in the base LLH class, 
        # so the integral term in the x(t) equation becomes (e^{\Lambda(t'-t)}-I) * \Lambda^{-1} * B * u_t
        if int_forward_variant:
            llh_layer = Int_Forward_LLH
        # if using backward, then ZOH holds u_t'- constant BACKWARDS in time: u(s) = u_t'- for s in [t, t']
        elif int_backward_variant:
            llh_layer = Int_Backward_LLH
        # in the base LLH implementation, Bu is integrated wrt dNt, so the integral term in the x(t) equation becomes B * u_t * \alpha_t
        # this means the Bu contribution only occurs at event indices, bundled into the same impulse sequence as the mark impulse.
        # this is NOT the ZOH discretization of the continuous input. In summary LLH is closer to a pure jump-driven recurrence,
        # and Int_*_LLH are closer to continuous-time SSMs with ZOH inputs.
        else:
            llh_layer = LLH

        self.backward_variant = int_backward_variant

        # this part sets up the layers of the S2P2 model according to the number of layers specified in the model config
        self.layers = nn.ModuleList(
            [
                llh_layer(**layer_kwargs, is_first_layer=i == 0)
                for i in range(self.n_layers)
            ]
        )
        
        # this initializes the mark impulse \alpha (H \times K) embedding (without E)
        # each \alpha_k is a real-value vector of dimension H, associated with event type k
        # it is not yet in state-space dimension P, the LLH later maps it via the matrix E
        # it should be noted here the event impulse matrix \alpha is globally shared across all layers,
        # however each layer has its own E^{l} matrix to map \alpha into state-space dimension P
        self.layers_mark_emb = nn.Embedding(
            self.num_event_types_pad, # K
            self.H,
        )  # One embedding to share amongst layers to be used as input into a layer-specific and input-aware impulse
        self.layer_type_emb = None  # Remove old embeddings from EasyTPP
        self.intensity_net = IntensityNet( # initializing intensity network with linear part and scaled soft-plus.
                                           # intensity_net.forward() will actually produce the intensity values
            input_dim=self.H,
            bias=self.bias,
            num_event_types=self.num_event_types,
        )

    def _get_intensity(
        self, x_LP: Union[torch.tensor, List[torch.tensor]], right_us_BNH) -> torch.Tensor:
        """
        Assume time has already been evolved, take a vertical stack of hidden states and produce intensity.
        
        the Union here is just a type hint from python's typing module, 
        saying the input x_LP can be either a torch tensor or a list of torch tensors
        output is a an intensity "tensor" of type torch.tensor
        
        The only inputs needed to compute intensity is the list of latent states x for each layer (x_LP),
        and the list of right-limit residual streams for each layer (right_us_BNH).
        From these, we can reconstruct the left-limit residual stream at the final layer (left_u_H),
        which is then decoded to intensity via the intensity network.
        
        The following suffix components encode tensor dimensions:
        B: batch dimension
        N: event index (sequence length)
        G: grid/MC sample index
        L: layer index
        P: state dimension
        H: residual/hidden stream dimension
        
        Pluralization matters:
        u = one depth signal
        us = a list (or stack) of depth signals across layers
        
        Examples:
        left_u_H = u_t-^(l) for current layer l (it shows up in a loop thru layers below) at time t- of dimension H (no time or layer axis)
        right_us_BNH = [u_t+^(1), u_t+^(2),..., u_t+^(L)] list of right-limit u's (tensors) at time t+ for all layers, each of shape (B,N,H)
            to be clear, right_us_BNH[i] = u_t+^(i+1) of shape (B,N,H). the "us" implies a list over layers
        x_LP is either:
            list of tensors [x^(1), x^(2), ..., x^(L)] each of shape P -- list over layers
            -OR-
            a single tensor with layer axis [L, P]
        u_GH = 
        
        Recall the formula for the intensity vector is ScaledSoftplus(W * u_t-^(L+1) + b)
        u_t-^(L+1) is determined by non-linear recursive stacking of all previous layers. For example, the final recursion is:
        u_t-^(L+1) = LayerNorm^(l)(\sigma(y_t^(l)) + u_t^(l))
        of course, y_t^(l) = C^(l) * x_t^(l) + D^(l) * u_t^(l), and you can recursively unpack this all the way down to the first layer
        this is why in this _get_intensity function, the layer.depth_pass() method is called that is a "depth-only" pass that 
        reconstructs u_t-^(L+1) without evolving x(t) forward in time again. Hence the "Assume time has already evolved" comment above.
        
        TODO: the layer.depth_pass() method is annotated in ssms.models.LLH class.
        """
        left_u_H = None
        for i, layer in enumerate(self.layers): # iterating through each layer, starting with initial latent state x_LP,
                                                # to get final left_u_H at the last layer, which is required to compute intensity
            if isinstance(
                x_LP, list
            ):  # Sometimes it is convenient to pass as a list over the layers rather than a single tensor
                left_u_H = layer.depth_pass(
                    x_LP[i], current_left_u_H=left_u_H, prev_right_u_H=right_us_BNH[i]
                )
            else:
                left_u_H = layer.depth_pass(
                    x_LP[..., i, :],
                    current_left_u_H=left_u_H,
                    prev_right_u_H=right_us_BNH[i],
                )

        return self.intensity_net(left_u_H)  # self.ScaledSoftplus(self.linear(left_u_H))

    def _evolve_and_get_intensity_at_sampled_dts(self, x_LP, dt_G, right_us_H):
        """
        Whereas _get_intensity assumes time has already evolved and returns a single intensity at left limit of some time
        (typically an event time, since those are the only times we have left-limit states for),
        this function allows for the computation of the integral \int_{t_i}^{t_i+1} \lambda(s) ds by evolving the latent state
        x(t) from the right-limit at t_i+ to sampled times inside the interval [t_i, t_i+1), and then decoding
        the intensity at those sampled times. _get_intensity assumes you have all the latent states at the end of the time period,
        whereas this function is for when you only have an initial latent state at the start of the time period and need to evolve forward.
        
        This function is necessary for computing the "non-event" integral term in the log-likelihood (via Monte Carlo integration)
        
        It should be noted here that x_LP are the latent states at some starting time (in practice, the right-limit at event time t_i+),
        whereas in _get_intensity, x_LP are the latent states at some ending time (in practice, the left-limit at event time t_i+1-).
        
        dt_G is a tensor of sampled time offsets \delta_t ("grid points" G) within each interval
        right_us_H is the list of right-limit residual streams at the starting time (t_i+) for each layer
        
        TODO: layer.get_left_limit() is annotated in ssms.models.LLH class.
        """
        left_u_GH = None
        for i, layer in enumerate(self.layers):
            x_GP = layer.get_left_limit(
                right_limit_P=x_LP[..., i, :],
                dt_G=dt_G,
                next_left_u_GH=left_u_GH,
                current_right_u_H=right_us_H[i],
            )
            left_u_GH = layer.depth_pass(
                current_left_x_P=x_GP,
                current_left_u_H=left_u_GH,
                prev_right_u_H=right_us_H[i],
            ) 
        return self.intensity_net(left_u_GH)  # self.ScaledSoftplus(self.linear(left_u_GH))

    def forward( # the "Optional" type hint Optional[torch.Tensor] means the argument can be either a torch.Tensor or None 
        self, batch, initial_state_BLP: Optional[torch.Tensor] = None, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch operations of self._forward, meaning it processes many sequences in parallel in one call rather than one sequence at at time
        
        The "batching" refers to how when training, we will be maximizing the log-likelihood across many sequences in parallel,
        \sum_{b=1}^{B} log p({(t_i, m_i)}_{i=1}^{N_b} | \Theta)
        So the "batch size" is just the number of sequences specified in the model config that will be used in this implementation.
        single sequence: tensors shaped like [N, ...]
        batch of sequences: tensors shaped like [B, N, ...]
        
        t_BN: event times for batch (B sequences, each of length N (the number of events))
        dt_BN: inter-event times for batch
        marks_BN: event types for batch
        batch_non_pad_mask: mask for non-padding events in batch (1 if real event, 0 if padding)
        _ : placeholder for any additional batch elements not used here
        """
        t_BN, dt_BN, marks_BN, batch_non_pad_mask, _ = batch

        right_xs_BNP = []  # list over layers: each is [B,N,P], including both t_0 and t_N
        left_xs_BNm1P = [] # list over layers: each is [B,N-1,P] (non-backward variant)
        right_us_BNH = [None]  # list over layers+1: per layer right u, element 0 is None as this is the 'input' to the first layer
        left_u_BNH, right_u_BNH = None, None # current layer's input as layer depth is traversed, initialized to None for first layer
        alpha_BNP = self.layers_mark_emb(marks_BN) # this is the mark impulse embedding \alpha_k_i for each event in the batch
                                                   # name says BNP but \alpha is actually BNH, 
                                                   # maybe they are referring to after projection by E?

        for l_i, layer in enumerate(self.layers):
            # for each event, compute the fixed impulse via alpha_m for event i of type m (i in N, m in K)
            init_state = (
                initial_state_BLP[:, l_i] if initial_state_BLP is not None else None
            )

            """
            Returns right limit of xs and us for [t0, t1, ..., tN] by doing forward pass through the current layer in the loop
            "layer" returns the right limit of xs at current layer, and us for the next layer (as transformations of ys)
            x_BNP: at time [t_0, t_1, ..., t_{N-1}, t_N]
            next_left_u_BNH: at time [t_0, t_1, ..., t_{N-1}, t_N] -- only available for backward variant
            next_right_u_BNH: at time [t_0, t_1, ..., t_{N-1}, t_N] -- always returned but only used for RT
            """
            x_BNP, next_layer_left_u_BNH, next_layer_right_u_BNH = layer.forward(
                left_u_BNH, right_u_BNH, alpha_BNP, dt_BN, init_state
            )
            
            assert next_layer_right_u_BNH is not None
            right_xs_BNP.append(x_BNP)
            
            """
            if NOT backward variant, compute left-limit x(t_i-) from right-limit x(t_{i-1}+)
            here we are creating a list of left-limit xs at each event time except the first (t_1-, t_2-, ..., t_N-) using
            the .get_left_limit() method of the layer, which evolves the right-limit state at t_{i-1}+ forward by dt_i 
            to get left-limit state at t_i-. 
            TODO: the annotated notes for this method are in ssms.models.LLH class.
            this method takes 4 elements as input:
                right_limit_P: the last timestep of x_BNP across all batches and states,
                dt_G: dt_BN[..., 1:].unsqueeze(-1),
                    dt_BN[..., 1:] has shape [B,N-1], and unsqueeze(-1) makes it [B,N-1,1] to add a trailing dimension for broadcasting
                    this is needed because in get_left_limit, dt_G is expected to have shape [B,N-1,G], where G is the number of grid points
                    (here G=1)
                current_right_u_H: th.Tensor,
                next_left_u_GH: th.Tensor,
            note the "..." operator means "slice (:) all preceding dimensions
            """
            if next_layer_left_u_BNH is None:  # NOT backward variant
                left_xs_BNm1P.append(
                    layer.get_left_limit(  # current and next at event level
                        x_BNP[..., :-1, :],  # at time [t_0, t_1, ..., t_{N-1}]
                        dt_BN[..., 1:].unsqueeze(-1),  # with dts [t1-t0, t2-t1, ..., t_N-t_{N-1}]
                                                       # recall "unsqueeze" turns an n-d tensor into an n+1-d tensor by adding
                                                       # a dimension of size 1 at the specified index (-1 means at the end here)
                        current_right_u_H = right_u_BNH if right_u_BNH is None
                                                        else right_u_BNH[..., :-1, :],  # at time [t_0, t_1, ..., t_{N-1}]
                        next_left_u_GH = left_u_BNH if left_u_BNH is None
                                                    else left_u_BNH[..., 1:, :].unsqueeze(-2),  # at time [t_1, t_2 ..., t_N]
                    ).squeeze(-2) # get it back to shape [B,N-1,P] by removing the singleton dimension added in get_left_limit
                    
                )
            right_us_BNH.append(next_layer_right_u_BNH)

            left_u_BNH, right_u_BNH = next_layer_left_u_BNH, next_layer_right_u_BNH

        right_xs_BNLP = torch.stack(right_xs_BNP, dim=-2)

        ret_val = {
            "right_xs_BNLP": right_xs_BNLP,  # [t_0, ..., t_N]
            "right_us_BNH": right_us_BNH,  # [t_0, ..., t_N]; list starting with None
        }

        if left_u_BNH is not None:  # backward variant
            ret_val["left_u_BNm1H"] = left_u_BNH[
                ..., 1:, :
            ]  # The next inputs after last layer -> transformation of ys
        else:  # NOT backward variant
            ret_val["left_xs_BNm1LP"] = torch.stack(left_xs_BNm1P, dim=-2)

        # 'seq_len - 1' left limit for [t_1, ..., t_N] for events (u if available, x if not)
        # 'seq_len' right limit for [t_0, t_1, ..., t_{N-1}, t_N] for events xs or us
        return ret_val

    def loglike_loss(self, batch, **kwargs):
        """
        this method computes the conditional log-likelihood loss for a batch of sequences. This is expressed in the paper as:
        log p({(t_i, k_i)}_{i=1}^{N} | \Theta) = \sum_{i=1}^{N} log(\lambda_{k_i}(t_i-)) - \sum_{i=1}^{N} \int_{0}^{T} \lambda_k_i(t) dt
        where the first term is the log-intensity at event times (from the left limit),
        and the second term is the integral of the intensity over the observation window (computed via Monte Carlo integration).
        
        This method is a sort of "orchestration" for the log likelihood specific to S2P2, which takes into account right/left limits,
        whereas torch_basemodel.compute_loglikelihood is a more general MATHEMATICAL method that just computes the log-likelihood.
        loglike_loss does the S2P2 forward pass, computes intensities at event times and sampled times, and the passes these agruments
        into torch_basemodel.compute_loglikelihood to get the actual log-likelihood values for the S2P2 model.

        hidden states at the left and right limits around event time; note for the shift by 1 in indices:
        consider a sequence [t0, t1, ..., tN]
        Produces the following:
        left_x: x0, x1, x2, ... <-> x_{t_1-}, x_{t_2-}, x_{t_3-}, ..., x_{t_N-} (note the shift in indices) for all layers
           OR ==>               <-> u_{t_1-}, u_{t_2-}, u_{t_3-}, ..., u_{t_N-} for last layer
        right_x: x0, x1, x2, ... <-> x_{t_0+}, x_{t_1+}, ..., x_{t_N+} for all layers
        right_u: u0, u1, u2, ... <-> u_{t_0+}, u_{t_1+}, ..., u_{t_N+} for all layers
        """
        forward_results = self.forward(batch)  # N minus 1 comparing with sequence lengths
        right_xs_BNLP, right_us_BNH = (
            forward_results["right_xs_BNLP"],
            forward_results["right_us_BNH"],
        )
        right_us_BNm1H = [
            None if right_u_BNH is None else right_u_BNH[:, :-1, :]
            for right_u_BNH in right_us_BNH
        ]

        ts_BN, dts_BN, marks_BN, batch_non_pad_mask, _ = batch

        # evaluate intensity values at each event *from the left limit*, _get_intensity: [LP] -> [M]
        # left_xs_B_Nm1_LP = left_xs_BNm1LP[:, :-1, ...]  # discard the left limit of t_N
        # Note: no need to discard the left limit of t_N because "marks_mask" will deal with it
        if "left_u_BNm1H" in forward_results:  # ONLY backward variant
            intensity_B_Nm1_M = self.intensity_net(
                forward_results["left_u_BNm1H"]
            )  # self.ScaledSoftplus(self.linear(forward_results["left_u_BNm1H"]))
        else:  # NOT backward variant
            intensity_B_Nm1_M = self._get_intensity(
                forward_results["left_xs_BNm1LP"], right_us_BNm1H
            )

        # sample dt in each interval for MC: [batch_size, num_times=N-1, num_mc_sample]
        # N-1 because we only consider the intervals between N events
        # G for grid points
        dts_sample_B_Nm1_G = self.make_dtime_loss_samples(dts_BN[:, 1:])

        # evaluate intensity at dt_samples for MC *from the left limit* after decay -> shape (B, N-1, MC, M)
        intensity_dts_B_Nm1_G_M = self._evolve_and_get_intensity_at_sampled_dts(
            right_xs_BNLP[
                :, :-1
            ],  # x_{t_i+} will evolve up to x_{t_{i+1}-} and many times between for i=0,...,N-1
            dts_sample_B_Nm1_G,
            right_us_BNm1H,
        )

        # TODO: add code annotations for the more general torch_basemodel.compute_loglikelihood
        event_ll, non_event_ll, num_events = self.compute_loglikelihood(
                lambda_at_event=intensity_B_Nm1_M,
                lambdas_loss_samples=intensity_dts_B_Nm1_G_M,
                time_delta_seq=dts_BN[:, 1:],
                seq_mask=batch_non_pad_mask[:, 1:],
                type_seq=marks_BN[:, 1:],
        )

        # compute loss to optimize
        loss = -(event_ll - non_event_ll).sum()

        return loss, num_events

    def compute_intensities_at_sample_times(
        self, event_times_BN, inter_event_times_BN, marks_BN, sample_dtimes, **kwargs
    ):
        """Compute the intensity at sampled times, not only event times.  *from the left limit*
        
        This method is a public-ish method used for prediction/simulation taht is passed into the thinning-based sampler:
        EventSampler.draw_next_time_one_step(). It's a higher-level wrapper whose job is essentially:
        "given the observed history (times & marks) and some propsed sample offsets, return intensities at those sample offsets"

        Args:
            time_seq (tensor): [batch_size, seq_len], times seqs.
            time_delta_seq (tensor): [batch_size, seq_len], time delta seqs.
            event_seq (tensor): [batch_size, seq_len], event type seqs.
            sample_dtimes (tensor): [batch_size, seq_len, num_sample], sampled inter-event timestamps.

        Returns:
            tensor: [batch_size, num_times, num_mc_sample, num_event_types],
                    intensity at each timestamp for each event type.
        """

        compute_last_step_only = kwargs.get("compute_last_step_only", False)

        # assume inter_event_times_BN always starts from 0
        _input = event_times_BN, inter_event_times_BN, marks_BN, None, None

        # 'seq_len - 1' left limit for [t_1, ..., t_N]
        # 'seq_len' right limit for [t_0, t_1, ..., t_{N-1}, t_N]

        forward_results = self.forward(
            _input
        )  # N minus 1 comparing with sequence lengths
        right_xs_BNLP, right_us_BNH = (
            forward_results["right_xs_BNLP"],
            forward_results["right_us_BNH"],
        )

        if (
            compute_last_step_only
        ):  # fix indices for right_us_BNH: list [None, tensor([BNH]), ...]
            right_us_B1H = [
                None if right_u_BNH is None else right_u_BNH[:, -1:, :]
                for right_u_BNH in right_us_BNH
            ]
            sampled_intensity = self._evolve_and_get_intensity_at_sampled_dts(
                right_xs_BNLP[:, -1:, :, :], sample_dtimes[:, -1:, :], right_us_B1H
            )  # equiv. to right_xs_BNLP[:, -1, :, :][:, None, ...]
        else:
            sampled_intensity = self._evolve_and_get_intensity_at_sampled_dts(
                right_xs_BNLP, sample_dtimes, right_us_BNH
            )
        return sampled_intensity  # [B, N, MC, M]
