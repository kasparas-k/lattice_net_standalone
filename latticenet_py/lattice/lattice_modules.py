import math
import sys

import numpy as np
import torch
from torch.nn import functional as F
import torch_scatter

from latticenet_py.backend import Lattice
import latticenet_py.lattice.lattice_funcs as lfun
import latticenet_py.utils.utils as utils
from latticenet_py.utils.utils import LinearWN


class DropoutLattice(torch.nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.dropout = torch.nn.Dropout2d(p=prob)

    def forward(self, lv):

        if (len(lv.shape)) != 2:
            sys.exit(
                "the lattice values must be two dimensional, nr_lattice vertices x val_dim.However it is",
                len(lv.shape),
            )

        # droput expect input of shape N,C,H,W and drops a full channel
        lv_drop = lv.transpose(0, 1)  # val_full_dim x nr_lattice_vertices
        lv_drop = lv_drop.unsqueeze(0).unsqueeze(3)
        lv_drop = self.dropout(lv_drop)
        lv_drop = lv_drop.squeeze(3).squeeze(0)
        lv_drop = lv_drop.transpose(0, 1)

        return lv_drop


class SplatLatticeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lattice_py, positions, values):
        lv, ls_wrap, indices, weights = lfun.SplatLattice.apply(
            lattice_py, positions, values
        )
        return lv, ls_wrap.lattice, indices, weights


class DistributeLatticeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lattice, positions, values, reset_hashmap=True):
        distributed_lattice_wrap, distributed, splatting_indices, splatting_weights = (
           lfun.DistributeLattice.apply(lattice, positions, values, reset_hashmap)
        )
        distributed_lattice = distributed_lattice_wrap.lattice

        # subsctract mean from the positions so we have something like a local laplacian as a feature
        pos_dim = positions.shape[1]
        distributed_positions = distributed[
            :, :pos_dim
        ]  # get the first 3 columns, the ones corresponding only to the xyz positions

        indices_long = splatting_indices.long()

        # some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0
        indices_long[indices_long < 0] = 0

        # if self.experiment in experiments_that_imply_no_mean_substraction:
        # pass
        mean_positions = torch_scatter.scatter_mean(
            distributed_positions, indices_long, dim=0
        )
        index = torch.tensor([0]).to("cuda")
        mean_positions = torch.index_fill(mean_positions, dim=0, index=index, value=0)
        # by setting the first row of mean_positions to 0 it means that all the point that splat onto vertex zero will have a wrong mean. We will set those distributed_mean_substracted to also zero later
        # the distributed means now has shape nr_positions x pos_dim but we want to substract each distributed position (shape  (nr_positions x m_pos_dim+1) x pos_dim   ) with its corresponding mean. We can do a index_select with splatting indices to get the means
        distributed_mean_positions = torch.index_select(mean_positions, 0, indices_long)
        distributed[:, :pos_dim] = distributed_positions - distributed_mean_positions

        # we have to set the positions that ended up in an invalid vertes or the zero one because it's also considered invalid, to zero
        positions_that_splat_onto_vertex_zero_or_are_invalid = indices_long == 0
        positions_that_splat_onto_vertex_zero_or_are_invalid = (
            positions_that_splat_onto_vertex_zero_or_are_invalid.unsqueeze(1)
        )

        distributed = distributed.masked_fill(
            positions_that_splat_onto_vertex_zero_or_are_invalid, 0
        )

        return distributed_lattice, distributed, splatting_indices, splatting_weights


class ExpandLatticeModule(
    torch.nn.Module
):  # creates lattice vertiex not directly around the positions but also further away by applying random noise to the positions
    def __init__(self, point_multiplier, noise_stddev, expand_values):
        super().__init__()
        self.point_multiplier = point_multiplier
        self.noise_stddev = noise_stddev
        self.expand_values = expand_values

    def forward(self, lattice_values, lattice_structure, positions):
        lattice_structure.set_values(lattice_values)

        lv, ls_wrap = lfun.ExpandLattice.apply(
            lattice_values,
            lattice_structure,
            positions,
            self.point_multiplier,
            self.noise_stddev,
            self.expand_values,
        )
        ls = ls_wrap.lattice

        ls.set_values(lv)

        return lv, ls


class ConvLatticeModule(torch.nn.Module):
    def __init__(self, nr_filters, neighbourhood_size, dilation=1, bias=True):
        # def __init__(self, nr_filters, neighbourhood_size, dilation=1):
        super().__init__()
        self.first_time = True
        self.weight = None
        self.bias = None
        self.neighbourhood_size = neighbourhood_size
        self.nr_filters = nr_filters
        self.dilation = dilation
        self.use_bias = bias

    # as per https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L49
    def reset_parameters(self, filter_extent):

        fan = torch.nn.init._calculate_correct_fan(self.weight, "fan_out")
        gain = torch.nn.init.calculate_gain("relu", 1)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        if self.bias is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, lattice_values, lattice_structure):

        lattice_structure.set_values(lattice_values)

        if self.first_time:
            self.first_time = False
            filter_extent = lattice_structure.get_filter_extent(self.neighbourhood_size)
            val_dim = lattice_structure.val_dim()
            self.weight = torch.nn.Parameter(
                torch.empty(filter_extent * val_dim, self.nr_filters).to("cuda")
            )  # works for ConvIm2RowLattice
            if self.use_bias:
                self.bias = torch.nn.Parameter(torch.empty(self.nr_filters).to("cuda"))
            with torch.no_grad():
                self.reset_parameters(filter_extent)

        lv, ls_wrap = lfun.ConvIm2RowLattice.apply(
            lattice_values, lattice_structure, self.weight, self.dilation
        )
        ls = ls_wrap.lattice

        if self.use_bias:
            lv += self.bias
        ls.set_values(lv)

        return lv, ls


class ConvLatticeIm2RowModule(torch.nn.Module):
    def __init__(
        self, in_channels, out_channels, neighbourhood_size, dilation=1, bias=True
    ):
        super().__init__()
        self.first_time = True
        self.weight = None
        self.bias = None
        self.neighbourhood_size = neighbourhood_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dilation = dilation
        self.use_bias = bias
        self.filter_extent = Lattice.get_expected_filter_extent(neighbourhood_size)

        # create parameters
        self.weight = torch.nn.Parameter(
            torch.empty(self.filter_extent * self.in_channels, self.out_channels).to(
                "cuda"
            )
        )  # works for ConvIm2RowLattice
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_channels).to("cuda"))
        with torch.no_grad():
            self.reset_parameters()

    # as per https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L49
    def reset_parameters(self):

        fan = torch.nn.init._calculate_correct_fan(self.weight, "fan_out")
        gain = torch.nn.init.calculate_gain("relu", 1)
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        if self.bias is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, lattice_values, lattice_structure):

        lattice_structure.set_values(lattice_values)
        filter_extent = lattice_structure.get_filter_extent(self.neighbourhood_size)
        assert (
            self.in_channels == lattice_structure.val_dim()
        ), f"In channels doesn't match the val_dim of the lattice. In channels is {self.in_channels}, while val dim is {lattice_structure.val_dim()}"

        # if(self.first_time):
        #     if self.use_bias:
        #     with torch.no_grad():

        lattice_rowified = lfun.Im2RowLattice.apply(
            lattice_values,
            lattice_structure,
            filter_extent,
            self.dilation,
            self.out_channels,
        )
        lv = lattice_rowified.mm(self.weight)

        lattice_structure_new = (
            lattice_structure.clone_lattice()
        )  # we create a new lattice because after doing convolution on this lattice, the val dim may change

        if self.use_bias:
            lv += self.bias
        lattice_structure_new.set_values(lv)

        return lv, lattice_structure_new


class CoarsenLatticeModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.first_time = True
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.neighbourhood_size = 1
        self.bias = None
        self.use_bias = bias
        self.filter_extent = Lattice.get_expected_filter_extent(self.neighbourhood_size)

        # create parameters
        self.weight = torch.nn.Parameter(
            torch.empty(self.filter_extent * self.in_channels, self.out_channels).to(
                "cuda"
            )
        )  # works for ConvIm2RowLattice
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_channels).to("cuda"))
        with torch.no_grad():
            self.reset_parameters()

    # as per https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L49
    def reset_parameters(self):

        fan = torch.nn.init._calculate_correct_fan(self.weight, "fan_out")
        # the fan out here actually refers to the fan in because we don't have a tranposed weight as pytorch usually expects it
        # we reduce the fan because actually most of the vertices don't have any nieghbours
        # half of them have filter_extent =1 and the other ones have filter extent the usual one of 9. So we settle for the middle and go for 5
        fan = fan / 2

        gain = torch.nn.init.calculate_gain("relu", 1)
        std = gain / math.sqrt(fan) * 2.0
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        if self.bias is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, lattice_fine_values, lattice_fine_structure, coarsened_lattice=None
    ):
        lattice_fine_structure.set_values(lattice_fine_values)

        assert (
            self.in_channels == lattice_fine_structure.val_dim()
        ), f"In channels doesn't match the val_dim of the lattice. In channels is {self.in_channels}, while val dim is {lattice_fine_structure.val_dim()}"

        # if(self.first_time):
        #     if self.use_bias:
        #     with torch.no_grad():

        lv, ls_wrap = lfun.CoarsenLattice.apply(
            lattice_fine_values, lattice_fine_structure, self.weight, coarsened_lattice
        )  # this just does a convolution, we also need batch norm an non linearity
        ls = ls_wrap.lattice

        if self.use_bias:
            lv += self.bias
        ls.set_values(lv)

        return lv, ls


class FinefyLatticeModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias=False):
        super().__init__()
        self.first_time = True
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.neighbourhood_size = 1
        self.bias = None
        self.use_bias = bias
        self.filter_extent = Lattice.get_expected_filter_extent(self.neighbourhood_size)

        # create parameters
        self.weight = torch.nn.Parameter(
            torch.empty(self.filter_extent * self.in_channels, self.out_channels).to(
                "cuda"
            )
        )  # works for ConvIm2RowLattice
        if self.use_bias:
            self.bias = torch.nn.Parameter(torch.empty(self.out_channels).to("cuda"))
        with torch.no_grad():
            self.reset_parameters()

    # as per https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/conv.py#L49
    def reset_parameters(self):

        fan = torch.nn.init._calculate_correct_fan(self.weight, "fan_out")
        # the fan out here actually refers to the fan in because we don't have a tranposed weight as pytorch usually expects it
        # we reduce the fan because actually most of the vertices don't have any nieghbours
        # half of them have filter_extent =1 and the other ones have filter extent the usual one of 9. So we settle for the middle and go for 5
        fan = fan / 2

        gain = torch.nn.init.calculate_gain("relu", 1)
        std = gain / math.sqrt(fan) * 2.0
        bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
        with torch.no_grad():
            self.weight.uniform_(-bound, bound)

        if self.bias is not None:
            fan_in, fan_out = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_out)
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(
        self, lattice_coarse_values, lattice_coarse_structure, lattice_fine_structure
    ):
        lattice_coarse_structure.set_values(lattice_coarse_values)

        assert (
            self.in_channels == lattice_coarse_structure.val_dim()
        ), f"In channels doesn't match the val_dim of the lattice. In channels is {self.in_channels}, while val dim is {lattice_coarse_structure.val_dim()}"

        # if(self.first_time):
        #     if self.use_bias:
        #     with torch.no_grad():

        lv, ls_wrap = lfun.FinefyLattice.apply(
            lattice_coarse_values,
            lattice_coarse_structure,
            lattice_fine_structure,
            self.weight,
        )  # this just does a convolution, we also need batch norm an non linearity
        ls = ls_wrap.lattice

        if self.use_bias:
            lv += self.bias

        ls.set_values(lv)

        return lv, ls


class SliceLatticeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        lattice_values,
        lattice_structure,
        positions,
        splatting_indices=None,
        splatting_weights=None,
    ):

        lattice_structure.set_values(lattice_values)
        return lfun.SliceLattice.apply(
            lattice_values,
            lattice_structure,
            positions,
            splatting_indices,
            splatting_weights,
        )


class GatherLatticeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, lattice_values, lattice_structure, positions):

        lattice_structure.set_values(lattice_values)
        return lfun.GatherLattice.apply(lattice_values, lattice_structure, positions)


ConvLatticeIm2RowWNModule = utils.weight_norm_wrapper(
    ConvLatticeIm2RowModule, g_dim=1, v_dim=None
)
CoarsenLatticeWNModule = utils.weight_norm_wrapper(
    CoarsenLatticeModule, g_dim=1, v_dim=None
)
FinefyLatticeWNModule = utils.weight_norm_wrapper(
    FinefyLatticeModule, g_dim=1, v_dim=None
)


# the last attempt to make a fast slice without having a gigantic feature vector for each point. Rather from the features of the vertices, we regress directly the class probabilities
# the idea is to not do it with a gather but rather with a special slicing function that also gets as input some learnable weights
class SliceFastCUDALatticeModule(torch.nn.Module):
    def __init__(self, in_channels, nr_classes, dropout_prob, experiment):
        super().__init__()
        self.in_channels = in_channels
        self.nr_classes = nr_classes
        self.bottleneck = None
        self.stepdown = torch.nn.ModuleList([])
        self.bottleneck_size = 8
        self.norm_pre_gather = None
        self.linear_pre_deltaW = None
        self.linear_deltaW = None
        self.linear_clasify = None
        self.tanh = torch.nn.Tanh()
        self.dropout_prob = dropout_prob
        if dropout_prob > 0.0:
            self.dropout = DropoutLattice(dropout_prob)

        self.experiment = experiment

        # create parameters
        cur_nr_channels = in_channels
        for i in range(2):
            nr_channels_out = int(in_channels / np.power(2, i))
            if nr_channels_out < self.bottleneck_size:
                sys.exit(
                    "We used to many linear layers an now the values are lower than the bottlenck size. Which means that the bottleneck would actually do an expansion..."
                )
            print("adding stepdown with output of ", nr_channels_out)
            self.stepdown.append(GnRelu1x1(cur_nr_channels, nr_channels_out, False))
            cur_nr_channels = nr_channels_out
        print("adding bottleneck with output of ", self.bottleneck_size)
        self.bottleneck = GnRelu1x1(cur_nr_channels, self.bottleneck_size, False)

    def forward(self, lv, ls, positions, splatting_indices, splatting_weights):

        ls.set_values(lv)

        assert (
            self.in_channels == ls.val_dim()
        ), f"In channels doesn't match the val_dim of the lattice. In channels is {self.in_channels}, while val dim is {ls.val_dim()}"

        nr_positions = positions.shape[0]
        pos_dim = positions.shape[1]
        val_dim = lv.shape[1]

        # #slowly reduce the features
        # if len(self.stepdown) is 0:
        #     for i in range(2):
        #         if nr_channels_out  < self.bottleneck_size:
        # if self.bottleneck is None:

        # apply the stepdowns
        for i in range(2):
            if i == 0:
                lv_bottleneck, ls_bottleneck = self.stepdown[i](lv, ls)
            else:
                lv_bottleneck, ls_bottleneck = self.stepdown[i](
                    lv_bottleneck, ls_bottleneck
                )
        # last bottleneck
        lv_bottleneck, ls_bottleneck = self.bottleneck(lv_bottleneck, ls_bottleneck)

        sliced_bottleneck_rowified = lfun.GatherLattice.apply(
            lv_bottleneck,
            ls_bottleneck,
            positions,
            splatting_indices,
            splatting_weights,
        )

        nr_vertices_per_simplex = ls.pos_dim() + 1
        val_dim_of_each_vertex = int(
            sliced_bottleneck_rowified.shape[1] / nr_vertices_per_simplex
        )

        # from this slice rowified we regress for each position some barycentric offest of size m_pos_dim+1
        # linear layers on the sliced rowified get us to a tensor of nr_positions x (m_pos_dim+1), this will be the weights offsets for each positions into the 4 lattice vertices
        if self.linear_deltaW is None:
            self.linear_deltaW = torch.nn.Linear(
                val_dim_of_each_vertex, 1, bias=True
            ).to("cuda")
            with torch.no_grad():
                torch.nn.init.kaiming_uniform_(
                    self.linear_deltaW.weight, mode='fan_in', nonlinearity='tanh'
                )
                self.linear_deltaW.weight *= 0.1  # make it smaller so that we start with delta weight that are close to zero
                torch.nn.init.zeros_(self.linear_deltaW.bias)
            self.gamma = torch.nn.Parameter(
                torch.ones(val_dim_of_each_vertex).to("cuda")
            )
            self.beta = torch.nn.Parameter(
                torch.zeros(val_dim_of_each_vertex).to("cuda")
            )

        # attmept 4 predict a barycentric coordinate for each lattice vertex, and then use max over all the features in the simplex like in here https://arxiv.org/pdf/1611.04500.pdf
        sliced_bottleneck_rowified = sliced_bottleneck_rowified.view(
            nr_positions, nr_vertices_per_simplex, val_dim_of_each_vertex
        )
        # max over the al the vertices in a simplex
        max_vals, _ = sliced_bottleneck_rowified.max(1)
        max_vals = max_vals.unsqueeze(1)

        sliced_bottleneck_rowified -= (
            self.gamma * max_vals + self.beta
        )  # max vals broadcasts to all the vertices in the simplex and substracts the max from them
        delta_weights = self.linear_deltaW(sliced_bottleneck_rowified)
        delta_weights = delta_weights.reshape(nr_positions, nr_vertices_per_simplex)

        if self.experiment == "slice_no_deform":
            delta_weights *= 0

        # we slice with the delta weights and we clasify at the same time
        if (
            self.linear_clasify is None
        ):  # create the linear clasify but we use the tensors directly inside our cuda kernel
            self.linear_clasify = torch.nn.Linear(
                val_dim, self.nr_classes, bias=True
            ).to("cuda")
            utils.leaky_relu_init(self.linear_clasify, 1.0)

        if self.dropout_prob > 0.0:
            lv = self.dropout(lv)

        ls.set_values(lv)

        classes_logits = lfun.SliceClassifyLattice.apply(
            lv,
            ls,
            positions,
            delta_weights,
            self.linear_clasify.weight,
            self.linear_clasify.bias,
            self.nr_classes,
            splatting_indices,
            splatting_weights,
        )

        return classes_logits


class BatchNormLatticeModule(torch.nn.Module):
    def __init__(self, nr_params, affine=True):
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(
            num_features=nr_params, momentum=0.1, affine=affine
        ).to("cuda")

    def forward(self, lattice_values, lattice_py):

        if lattice_values.dim() != 2:
            sys.exit("lattice should be 2 dimensional, nr_vertices x val_full_dim")

        lattice_values = self.bn(lattice_values)

        lattice_py.set_values(lattice_values)

        return lattice_values, lattice_py


class GroupNormLatticeModule(torch.nn.Module):
    def __init__(self, nr_params, affine=True):
        super().__init__()
        nr_groups = 32
        # if the groups is not diivsalbe so for example if we have 80 params
        if nr_params % nr_groups != 0:
            nr_groups = int(nr_params / 2)
        # if we have less than 64 we make groups of 4 channels each. Less than 4 channels per group will probably be bad
        # if nr_params<=64:

        self.gn = torch.nn.GroupNorm(nr_groups, nr_params).to(
            "cuda"
        )  # having 32 groups is the best as explained in the GroupNormalization paper

    def forward(self, lattice_values, lattice_py, do_set_values=True):

        if lattice_values.dim() != 2:
            sys.exit("lattice should be 2 dimensional, nr_vertices x val_dim")

        # group norm wants the tensor to be N, C, L  (nr_batches, channels, nr_samples)
        lattice_values = lattice_values.unsqueeze(0)
        lattice_values = lattice_values.transpose(1, 2)
        lattice_values = self.gn(lattice_values)
        lattice_values = lattice_values.transpose(1, 2)
        lattice_values = lattice_values.squeeze(0)

        if (
            do_set_values
        ):  # sometimes we don't want to set the values because it doesnt make sense, like for example when we use this in the pointnet layer
            lattice_py.set_values(lattice_values)

        return lattice_values, lattice_py


class PointNetModule(torch.nn.Module):
    def __init__(self, nr_output_channels_per_layer, nr_outputs_last_layer):
        super().__init__()
        self.first_time = True
        self.nr_output_channels_per_layer = nr_output_channels_per_layer
        self.nr_outputs_last_layer = nr_outputs_last_layer
        self.nr_linear_layers = len(self.nr_output_channels_per_layer)
        self.layers = torch.nn.ModuleList([])
        self.act = torch.nn.LeakyReLU(0.2)

        self.last_conv = ConvLatticeIm2RowWNModule(
            in_channels=nr_output_channels_per_layer[-1] * 2,
            out_channels=self.nr_outputs_last_layer,
            neighbourhood_size=1,
            dilation=1,
            bias=True,
        )  # disable the bias becuse it is followed by a gn

    # called the first time we do a forwad pass to initialize all the modules
    def init(self, distributed):
        if self.first_time:
            with torch.no_grad():
                self.first_time = False

                # get the nr of channels of the distributed tensor
                nr_input_channels = distributed.shape[1] - 1

                nr_layers = 0
                for i in range(len(self.nr_output_channels_per_layer)):
                    nr_output_channels = self.nr_output_channels_per_layer[i]
                    self.layers.append(
                        LinearWN(nr_input_channels, nr_output_channels, bias=True).to(
                            "cuda"
                        )
                    )
                    nr_input_channels = nr_output_channels
                    nr_layers = nr_layers + 1

                utils.apply_weight_init_fn(self, utils.leaky_relu_init)

    def forward(self, lattice_py, distributed, indices):
        if self.first_time:
            self.init(distributed)

        barycentric_weights = distributed[:, -1]
        distributed = distributed[
            :, : distributed.shape[1] - 1
        ]  # IGNORE the barycentric weights for the moment and lift the coordinates of only the xyz and values

        # #run the distributed through all the layers
        for i in range(len(self.layers)):
            distributed = self.layers[i](distributed)
            # if( i < len(self.layers)-1):
            distributed = self.act(distributed)

        indices_long = indices.long()

        # some indices may be -1 because they were not inserted into the hashmap, this will cause an error for scatter_max so we just set them to 0
        indices_long[indices_long < 0] = 0

        distributed_reduced, argmax = torch_scatter.scatter_max(
            distributed, indices_long, dim=0
        )

        # get also the nr of points in the lattice so the max pooled features can be different if there is 1 point then if there are 100
        ones = torch.cuda.FloatTensor(indices_long.shape[0]).fill_(1.0)
        nr_points_per_simplex = torch_scatter.scatter_add(ones, indices_long)
        nr_points_per_simplex = nr_points_per_simplex.unsqueeze(1)
        # attempt 3 just by concatenating the barycentric coords
        barycentric_reduced = torch.index_select(
            barycentric_weights, 0, argmax.flatten()
        )  # we select for each vertex the 64 barycentric weights that got selected by the scatter max
        barycentric_reduced = barycentric_reduced.view(argmax.shape[0], argmax.shape[1])
        distributed_reduced = torch.cat((distributed_reduced, barycentric_reduced), 1)

        minimum_points_per_simplex = 4
        simplexes_with_few_points = nr_points_per_simplex < minimum_points_per_simplex
        distributed_reduced = distributed_reduced.masked_fill(
            simplexes_with_few_points, 0
        )

        index = torch.tensor([0]).to("cuda")
        distributed_reduced = torch.index_fill(
            distributed_reduced, dim=0, index=index, value=0
        )  # the first row corresponds to the invalid points, the ones that had an index of -1. We set it to 0 so it doesnt affect the prediction or the batchnorm

        lattice_py.set_values(distributed_reduced)

        distributed_reduced, lattice_py = self.last_conv(
            distributed_reduced, lattice_py
        )

        distributed_reduced = self.act(distributed_reduced)

        lattice_py.set_values(distributed_reduced)

        return distributed_reduced, lattice_py


class Conv1x1WN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__()
        self.linear = LinearWN(in_channels, out_channels, bias=bias).to("cuda")

    def forward(self, lv, ls):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        # if self.norm is None:
        # with torch.no_grad():
        # https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138

        ls.set_values(lv)
        lv = self.linear(lv)
        ls.set_values(lv)
        return lv, ls


class Conv1x1WNAct(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__()
        self.act = torch.nn.LeakyReLU(0.2)
        self.linear = LinearWN(in_channels, out_channels, bias=bias).to("cuda")

    def forward(self, lv, ls):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        # if self.norm is None:
        # with torch.no_grad():
        # https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138

        ls.set_values(lv)
        lv = self.linear(lv)
        lv = self.act(lv)
        ls.set_values(lv)
        return lv, ls


class Conv1x1(torch.nn.Module):
    def __init__(self, out_channels, bias):
        super().__init__()
        self.out_channels = out_channels
        self.linear = None
        self.use_bias = bias

    def forward(self, lv):

        # similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.linear is None:
            self.linear = torch.nn.Linear(
                lv.shape[1], self.out_channels, bias=self.use_bias
            ).to("cuda")
            with torch.no_grad():
                # https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138
                torch.nn.init.kaiming_normal_(
                    self.linear.weight, mode='fan_in', nonlinearity='relu'
                )

        lv = self.linear(lv)
        return lv


class GnRelu1x1(torch.nn.Module):
    def __init__(self, in_channels, out_channels, bias):
        super().__init__()
        self.norm = GroupNormLatticeModule(in_channels)
        self.relu = torch.nn.ReLU(inplace=False)
        self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias).to("cuda")
        torch.nn.init.kaiming_normal_(
            self.linear.weight, mode='fan_in', nonlinearity='relu'
        )

    def forward(self, lv, ls):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        # if self.norm is None:
        # with torch.no_grad():
        # https://towardsdatascience.com/understand-kaiming-initialization-and-implementation-detail-in-pytorch-f7aa967e9138

        lv, ls = self.norm(lv, ls)
        lv = self.relu(lv)
        ls.set_values(lv)
        lv = self.linear(lv)
        ls.set_values(lv)
        return lv, ls


class GnGelu1x1(torch.nn.Module):
    def __init__(self, out_channels, bias):
        super().__init__()
        self.out_channels = out_channels
        self.norm = None
        self.relu = torch.nn.ReLU(inplace=False)
        self.linear = None
        self.use_bias = bias

    def forward(self, lv, ls):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
            self.linear = torch.nn.Linear(
                lv.shape[1], self.out_channels, bias=self.use_bias
            ).to("cuda")
            with torch.no_grad():
                torch.nn.init.kaiming_normal_(
                    self.linear.weight, mode='fan_in', nonlinearity='relu'
                )

        lv, ls = self.norm(lv, ls)
        lv = F.gelu(lv)
        ls.set_values(lv)
        lv = self.linear(lv)
        ls.set_values(lv)
        return lv, ls


class Gn(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = None

    def forward(self, lv, ls):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls = self.norm(lv, ls)
        ls.set_values(lv)
        return lv, ls


class ConvAct(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation, bias, with_dropout):
        super().__init__()
        self.conv = ConvLatticeIm2RowModule(
            in_channels=in_channels,
            out_channels=out_channels,
            neighbourhood_size=1,
            dilation=dilation,
            bias=bias,
        )
        self.act = torch.nn.LeakyReLU(0.2)
        self.with_dropout = with_dropout
        if with_dropout:
            self.drop = DropoutLattice(0.2)

    def forward(self, lv, ls):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.with_dropout:
            lv = self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)
        lv_1 = self.act(lv_1)
        ls_1.set_values(lv_1)

        return lv_1, ls_1


class GnReluConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, dilation, bias, with_dropout):
        super().__init__()
        self.conv = ConvLatticeIm2RowModule(
            in_channels=in_channels,
            out_channels=out_channels,
            neighbourhood_size=1,
            dilation=dilation,
            bias=bias,
        )
        self.norm = GroupNormLatticeModule(in_channels)
        self.relu = torch.nn.ReLU(inplace=False)
        self.with_dropout = with_dropout
        if with_dropout:
            self.drop = DropoutLattice(0.2)

    def forward(self, lv, ls):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        lv, ls = self.norm(lv, ls)
        lv = self.relu(lv)
        if self.with_dropout:
            lv = self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)
        ls_1.set_values(lv_1)

        return lv_1, ls_1


class GnGeluConv(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias, with_dropout):
        super().__init__()
        self.nr_filters = nr_filters
        self.conv = ConvLatticeModule(
            nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias
        )
        self.norm = None
        self.with_dropout = with_dropout
        if with_dropout:
            self.drop = DropoutLattice(0.2)

    def forward(self, lv, ls):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls = self.norm(lv, ls)
        lv = F.gelu(lv)
        if self.with_dropout:
            lv = self.drop(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)
        ls_1.set_values(lv_1)

        return lv_1, ls_1


class BnReluConv(torch.nn.Module):
    def __init__(self, nr_filters, dilation, bias):
        super().__init__()
        self.nr_filters = nr_filters
        self.conv = ConvLatticeModule(
            nr_filters=nr_filters, neighbourhood_size=1, dilation=dilation, bias=bias
        )
        self.bn = None
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, lv, ls):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv https://arxiv.org/pdf/1603.05027.pdf
        if self.bn is None:
            self.bn = BatchNormLatticeModule(lv.shape[1])
        lv, ls = self.bn(lv, ls)
        lv = self.relu(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.conv(lv, ls)
        ls_1.set_values(lv_1)

        return lv_1, ls_1


class CoarsenAct(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coarse = CoarsenLatticeModule(
            in_channels=in_channels, out_channels=out_channels
        )
        self.act = torch.nn.LeakyReLU(0.2)

    def forward(self, lv, ls, concat_connection=None):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv
        # if self.norm is None:
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        lv_1 = self.act(lv_1)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1 = torch.cat((lv_1, concat_connection), 1)
            ls_1.set_values(lv_1)

        return lv_1, ls_1


class GnCoarsen(torch.nn.Module):
    def __init__(self, nr_filters):
        super().__init__()
        self.nr_filters = nr_filters
        self.coarse = CoarsenLatticeModule(nr_filters=nr_filters)
        self.norm = None

    def forward(self, lv, ls, concat_connection=None):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls = self.norm(lv, ls)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1 = torch.cat((lv_1, concat_connection), 1)
            ls_1.set_values(lv_1)

        return lv_1, ls_1


class GnReluCoarsen(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.coarse = CoarsenLatticeModule(
            in_channels=in_channels, out_channels=out_channels
        )
        self.norm = GroupNormLatticeModule(in_channels)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, lv, ls, concat_connection=None):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv
        # if self.norm is None:
        lv, ls = self.norm(lv, ls)
        lv = self.relu(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1 = torch.cat((lv_1, concat_connection), 1)
            ls_1.set_values(lv_1)

        return lv_1, ls_1


class GnGeluCoarsen(torch.nn.Module):
    def __init__(self, nr_filters):
        super().__init__()
        self.nr_filters = nr_filters
        self.coarse = CoarsenLatticeModule(nr_filters=nr_filters)
        self.norm = None

    def forward(self, lv, ls, concat_connection=None):

        ls.set_values(lv)

        # similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv.shape[1])
        lv, ls = self.norm(lv, ls)
        lv = F.gelu(lv)
        ls.set_values(lv)
        lv_1, ls_1 = self.coarse(lv, ls)
        ls_1.set_values(lv_1)

        if concat_connection is not None:
            lv_1 = torch.cat((lv_1, concat_connection), 1)
            ls_1.set_values(lv_1)

        return lv_1, ls_1


class FinefyAct(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fine = FinefyLatticeModule(
            in_channels=in_channels, out_channels=out_channels
        )
        self.act = torch.nn.LeakyReLU(0.2)

    def forward(self, lv_coarse, ls_coarse, ls_fine):

        ls_coarse.set_values(lv_coarse)

        # similar to densenet and resnet: bn, relu, conv
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        lv_1 = self.act(lv_1)
        ls_1.set_values(lv_1)

        return lv_1, ls_1


class GnReluFinefy(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.fine = FinefyLatticeModule(
            in_channels=in_channels, out_channels=out_channels
        )
        self.norm = GroupNormLatticeModule(in_channels)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, lv_coarse, ls_coarse, ls_fine):

        ls_coarse.set_values(lv_coarse)

        # similar to densenet and resnet: bn, relu, conv
        # if self.norm is None:
        lv_coarse, ls_coarse = self.norm(lv_coarse, ls_coarse)
        lv_coarse = self.relu(lv_coarse)
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        ls_1.set_values(lv_1)

        return lv_1, ls_1


class GnGeluFinefy(torch.nn.Module):
    def __init__(self, nr_filters):
        super().__init__()
        self.nr_filters = nr_filters
        self.fine = FinefyLatticeModule(nr_filters=nr_filters)
        self.norm = None

    def forward(self, lv_coarse, ls_coarse, ls_fine):

        ls_coarse.set_values(lv_coarse)

        # similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv_coarse.shape[1])
        lv_coarse, ls_coarse = self.norm(lv_coarse, ls_coarse)
        lv_coarse = F.gelu(lv_coarse)
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        ls_1.set_values(lv_1)

        return lv_1, ls_1


class GnFinefy(torch.nn.Module):
    def __init__(self, nr_filters):
        super().__init__()
        self.nr_filters = nr_filters
        self.fine = FinefyLatticeModule(nr_filters=nr_filters)
        self.norm = None

    def forward(self, lv_coarse, ls_coarse, ls_fine):

        ls_coarse.set_values(lv_coarse)

        # similar to densenet and resnet: bn, relu, conv
        if self.norm is None:
            self.norm = GroupNormLatticeModule(lv_coarse.shape[1])
        lv_coarse, ls_coarse = self.norm(lv_coarse, ls_coarse)
        ls_coarse.set_values(lv_coarse)
        lv_1, ls_1 = self.fine(lv_coarse, ls_coarse, ls_fine)
        ls_1.set_values(lv_1)

        return lv_1, ls_1


class TwoConv(torch.nn.Module):

    def __init__(self, in_channels, out_channels, dilations, biases, with_dropout):
        super().__init__()

        # again with bn-relu-conv

        self.conv1 = ConvAct(
            in_channels, out_channels, dilations[0], biases[0], with_dropout=False
        )
        self.conv2 = ConvAct(
            in_channels,
            out_channels,
            dilations[1],
            biases[1],
            with_dropout=with_dropout,
        )

        utils.apply_weight_init_fn(self, utils.swish_init)

    def forward(self, lv, ls):

        ls.set_values(lv)

        lv, ls = self.conv1(lv, ls)
        lv, ls = self.conv2(lv, ls)
        ls.set_values(lv)
        return lv, ls


# similar to https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
class ResnetBlock(torch.nn.Module):

    def __init__(self, in_channels, out_channels, dilations, biases, with_dropout):
        super().__init__()

        # again with bn-relu-conv
        self.conv1 = GnReluConv(
            in_channels, out_channels, dilations[0], biases[0], with_dropout=False
        )
        self.conv2 = GnReluConv(
            in_channels,
            out_channels,
            dilations[1],
            biases[1],
            with_dropout=with_dropout,
        )

    def forward(self, lv, ls):

        identity = lv

        ls.set_values(lv)

        lv, ls = self.conv1(lv, ls)
        lv, ls = self.conv2(lv, ls)
        lv += identity
        ls.set_values(lv)
        return lv, ls


# similar to convnext https://arxiv.org/pdf/2201.03545.pdf
class ResnetBlock2(torch.nn.Module):

    def __init__(self, in_channels, out_channels, dilations, biases, with_dropout):
        super().__init__()

        # again with bn-relu-conv

        self.conv1 = ConvLatticeIm2RowModule(
            in_channels=in_channels,
            out_channels=out_channels,
            neighbourhood_size=1,
            dilation=dilations[0],
            bias=biases[0],
        )
        self.norm = torch.nn.GroupNorm(
            1, out_channels
        )  # all channel sinto one group (Layer norm)
        self.conv2 = ConvLatticeIm2RowModule(
            in_channels=out_channels,
            out_channels=out_channels,
            neighbourhood_size=1,
            dilation=dilations[1],
            bias=biases[1],
        )
        self.act = torch.nn.LeakyReLU(0.2)

    def forward(self, lv, ls):

        identity = lv

        ls.set_values(lv)

        lv, ls = self.conv1(lv, ls)
        lv = self.norm(lv)
        ls.set_values(lv)
        lv, ls = self.conv2(lv, ls)
        lv = self.act(lv)
        lv += identity
        ls.set_values(lv)
        return lv, ls


# inspired from https://github.com/kuangliu/pytorch-cifar/blob/master/models/preact_resnet.py
class BottleneckBlock(torch.nn.Module):
    '''Pre-activation version of the original Bottleneck module.'''

    def __init__(self, in_channels, out_channels, biases):
        super().__init__()
        self.downsample = 4
        self.contract = GnRelu1x1(
            in_channels=in_channels,
            out_channels=int(out_channels / self.downsample),
            bias=biases[0],
        )
        self.conv = GnReluConv(
            in_channels=int(out_channels / self.downsample),
            out_channels=int(out_channels / self.downsample),
            dilation=1,
            bias=biases[1],
            with_dropout=False,
        )
        self.expand = GnRelu1x1(
            in_channels=int(out_channels / self.downsample),
            out_channels=out_channels,
            bias=biases[2],
        )

    def forward(self, lv, ls):

        ls.set_values(lv)

        identity = lv
        lv, ls = self.contract(lv, ls)
        lv, ls = self.conv(lv, ls)
        lv, ls = self.expand(lv, ls)
        lv += identity
        ls.set_values(lv)
        return lv, ls


# a bit of a naive implementation of densenet which is not very memory efficient. Idealy the storage should be shared as explained in the suplementary material of densenet
class DensenetBlock(torch.nn.Module):

    def __init__(self, nr_filters, dilation_list, nr_layers):
        super().__init__()
        self.nr_filters = nr_filters
        self.layers = torch.nn.ModuleList([])
        for i in range(nr_layers):
            self.layers.append(GnReluConv(nr_filters, dilation_list[i]))

    def forward(self, lv, ls):

        ls.set_values(lv)

        stack = lv
        output = []

        for i in range(len(self.layers)):
            lv_new, ls = self.layers[i](stack, ls)
            stack = torch.cat((stack, lv_new), 1)
            output.append(lv_new)

        output_concatenated = torch.cat(output, 1)
        ls.set_values(output_concatenated)
        return output_concatenated, ls
