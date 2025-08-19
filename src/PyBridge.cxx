#include "lattice_net/PyBridge.h"

#include <torch/extension.h>
#include "torch/torch.h"
#include "torch/csrc/utils/pybind.h"

//my stuff
#include "lattice_net/Lattice.cuh"
#include "lattice_net/HashTable.cuh"


namespace py = pybind11;




PYBIND11_MODULE(latticenet, m) {
    py::class_<HashTable, std::shared_ptr<HashTable>   > (m, "HashTable")
    .def_readonly("m_keys_tensor", &HashTable::m_keys_tensor) //careful when using this because setting it and not using update_impl is a big bug
    .def_readonly("m_nr_filled_tensor", &HashTable::m_nr_filled_tensor) ////careful when using this because setting it and not using update_impl is a big bug
    ;

    py::class_<Lattice, std::shared_ptr<Lattice>   > (m, "Lattice")
    .def_static("create", &Lattice::create<const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def_static("create", &Lattice::create<const std::string, const std::string> ) //for templated methods like this one we need to explicitly instantiate one of the arguments
    .def("begin_splat",  &Lattice::begin_splat )
    .def("splat_standalone",  &Lattice::splat_standalone )
    .def("distribute",  &Lattice::distribute )
    .def("expand",  &Lattice::expand )
    .def("convolve_im2row_standalone",  &Lattice::convolve_im2row_standalone )
    .def("im2row",  &Lattice::im2row )
    .def("row2im",  &Lattice::row2im )
    .def("im2rowindices",  &Lattice::im2rowindices )
    .def("just_create_verts",  &Lattice::just_create_verts )
    .def("create_coarse_verts",  &Lattice::create_coarse_verts )
    .def("create_coarse_verts_naive",  &Lattice::create_coarse_verts_naive )
    .def("slice_standalone_with_precomputation", &Lattice::slice_standalone_with_precomputation )
    .def("slice_standalone_no_precomputation", &Lattice::slice_standalone_no_precomputation )
    .def("slice_classify_no_precomputation", &Lattice::slice_classify_no_precomputation )
    .def("slice_classify_with_precomputation", &Lattice::slice_classify_with_precomputation )
    .def("gather_standalone_no_precomputation", &Lattice::gather_standalone_no_precomputation )
    .def("gather_standalone_with_precomputation", &Lattice::gather_standalone_with_precomputation )
    .def("slice_backwards_standalone_with_precomputation", &Lattice::slice_backwards_standalone_with_precomputation )
    .def("slice_backwards_standalone_with_precomputation_no_homogeneous", &Lattice::slice_backwards_standalone_with_precomputation_no_homogeneous )
    .def("slice_classify_backwards_with_precomputation", &Lattice::slice_classify_backwards_with_precomputation )
    .def("gather_backwards_standalone_with_precomputation", &Lattice::gather_backwards_standalone_with_precomputation )
    .def("get_filter_extent", &Lattice::get_filter_extent )
    .def_static("get_expected_filter_extent", &Lattice::get_expected_filter_extent )
    .def("val_dim", &Lattice::val_dim )
    .def("pos_dim", &Lattice::pos_dim )
    .def("name", &Lattice::name )
    .def("nr_lattice_vertices", &Lattice::nr_lattice_vertices )
    .def("capacity", &Lattice::capacity )
    .def("positions", &Lattice::positions )
    .def("sigmas_tensor", &Lattice::sigmas_tensor)
    .def("hash_table", &Lattice::hash_table)
    .def("values", &Lattice::values)
    .def("set_values", &Lattice::set_values)
    .def("set_positions", &Lattice::set_positions)
    .def("clone_lattice", &Lattice::clone_lattice)
    .def("increase_sigmas", &Lattice::increase_sigmas)
    .def("set_sigma", &Lattice::set_sigma)
    ;

}
