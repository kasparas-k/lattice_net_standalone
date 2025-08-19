#include "lattice_net/Lattice.cuh"

//c++
#include <string>

#include "EasyPytorch/UtilsPytorch.h" //contains torch so it has to be added BEFORE any other include because the other ones might include loguru which gets screwed up if torch was included before it
#include "EasyCuda/UtilsCuda.h"
#include "string_utils.h"

//my stuff
#include "lattice_net/HashTable.cuh"
#include "lattice_net/kernels/LatticeGPU.cuh"

//jitify
#define JITIFY_PRINT_INSTANTIATION 1
#define JITIFY_PRINT_SOURCE 1
#define JITIFY_PRINT_LOG 1
#define JITIFY_PRINT_PTX 1
#define JITIFY_PRINT_LAUNCH 1

//loguru
#define LOGURU_REPLACE_GLOG 1
#include "loguru/loguru.hpp" //needs to be added after torch.h otherwise loguru stops printing for some reason

//configuru
#define CONFIGURU_IMPLEMENTATION 1
#define CONFIGURU_WITH_EIGEN 1
#define CONFIGURU_IMPLICIT_CONVERSIONS 1
#include "configuru/configuru.hpp"
using namespace configuru;
//Add this header after we add all cuda stuff because we need the profiler to have cudaDeviceSyncronize defined
#define ENABLE_CUDA_PROFILING 1

using torch::Tensor;
using namespace radu::utils;


int Lattice::m_expected_position_dimensions = -1;



//CPU code that calls the kernels
Lattice::Lattice(const std::string config_file):
    m_impl( new LatticeGPU() ),
    m_lvl(1)
    {

    init_params(config_file);
    VLOG(3) << "Creating lattice";

}

Lattice::Lattice(const std::string config_file, const std::string name):
    m_impl( new LatticeGPU() ),
    m_name(name),
    m_lvl(1)
    {

    init_params(config_file);


    VLOG(3) << "Creating lattice: " <<name;

}

Lattice::Lattice(Lattice* other):
    m_impl( new LatticeGPU() ),
    m_lvl(1)
    {
        m_lvl=other->m_lvl;
        m_sigmas=other->m_sigmas;
        m_sigmas_tensor=other->m_sigmas_tensor.clone(); //deep copy
        m_expected_position_dimensions=other->m_expected_position_dimensions;
        m_positions=other->m_positions; //shallow copy

        //hashtable
        m_hash_table=std::make_shared<HashTable>(other->hash_table()->capacity() );
        m_hash_table->m_keys_tensor=other->m_hash_table->m_keys_tensor;
        m_hash_table->m_values_tensor=other->m_hash_table->m_values_tensor;
        m_hash_table->m_entries_tensor=other->m_hash_table->m_entries_tensor;
        m_hash_table->m_nr_filled_tensor=other->m_hash_table->m_nr_filled_tensor.clone(); //deep copy for this one as the new lattice may have different number of vertices
        m_hash_table->m_nr_filled=m_hash_table->m_nr_filled;
        m_hash_table->m_nr_filled_is_dirty=m_hash_table->m_nr_filled_is_dirty;
        m_hash_table->update_impl();

}

Lattice::~Lattice(){
    // LOG(WARNING) << "Deleting lattice: " << m_name;
}

void Lattice::init_params(const std::string config_file){
    std::string config_file_abs;
    config_file_abs=config_file;
    Config cfg = configuru::parse_file(config_file_abs, CFG);
    Config lattice_config=cfg["lattice_gpu"];
    int hash_table_capacity = lattice_config["hash_table_capacity"];
    m_hash_table=std::make_shared<HashTable>(hash_table_capacity);

    int nr_sigmas=lattice_config["nr_sigmas"]; //nr of is sigma values we have. Each one affecting a different number of dimensions of the positions
    for (int i=0; i < nr_sigmas; i++) {
        std::string param_name="sigma_"+std::to_string(i);
        std::string sigma_val_and_extent = (std::string)lattice_config[param_name];
        std::vector<std::string> tokenized = radu::utils::split(sigma_val_and_extent, " ");
        CHECK(tokenized.size()==2) << "For each sigma we must define its value and the extent(nr of dimensions it affects) in space separated string. So the nr of tokens split string should have would be 1. However the nr of tokens we have is" << tokenized.size();
        std::pair<float, int> sigma_params = std::make_pair<float,int> (  std::stof(tokenized[0]), std::stof(tokenized[1]) );
        m_sigmas_val_and_extent.push_back(sigma_params);
    }
    set_sigmas(m_sigmas_val_and_extent);
}

void Lattice::set_sigmas(std::initializer_list<  std::pair<float, int> > sigmas_list){
    m_sigmas.clear();
    for(auto sigma_pair : sigmas_list){
        float sigma=sigma_pair.first; //value of the sigma
        int nr_dim=sigma_pair.second; //how many dimensions are affected by this sigma
        for(int i=0; i < nr_dim; i++){
            m_sigmas.push_back(sigma);
        }
    }
    m_expected_position_dimensions=m_sigmas.size();

    m_sigmas_tensor=vec2tensor(m_sigmas);
}

void Lattice::set_sigmas(std::vector<  std::pair<float, int> > sigmas_list){
    m_sigmas.clear();
    for(auto sigma_pair : sigmas_list){
        float sigma=sigma_pair.first; //value of the sigma
        int nr_dim=sigma_pair.second; //how many dimensions are affected by this sigma
        for(int i=0; i < nr_dim; i++){
            m_sigmas.push_back(sigma);
        }
    }
    m_expected_position_dimensions=m_sigmas.size();

    m_sigmas_tensor=vec2tensor(m_sigmas);
}

void Lattice::check_positions(const torch::Tensor& positions_raw){
    CHECK(positions_raw.scalar_type()==at::kFloat) << "positions should be of type float";
    CHECK(positions_raw.dim()==2) << "positions should have dim 2 correspondin to HW. However it has sizes" << positions_raw.sizes();
    int pos_dim=positions_raw.size(1);
    CHECK(m_sigmas.size()==pos_dim) <<"One must set sigmas for each dimension of the positions. Use set_sigmas. m_sigmas is " << m_sigmas.size() << " m_pos dim is " <<pos_dim;
    CHECK(positions_raw.is_contiguous()) << "Positions raw is not contiguous. Please call .contiguous() on it";
    CHECK(pos_dim==m_expected_position_dimensions) << "The pos dim should be the same as the expected positions dimensions given by the sigmas. Pos dim is " << pos_dim << " m_expected_position_dimensions " << m_expected_position_dimensions;

}
void Lattice::check_values(const torch::Tensor& values){
    CHECK(values.scalar_type()==at::kFloat) << "values should be of type float";
    CHECK(values.dim()==2) << "values should have dim 2 correspondin to HW. However it has sizes" << values.sizes();
    CHECK(values.is_contiguous()) << "Values is not contiguous. Please call .contiguous() on it";
}
void Lattice::check_positions_and_values(const torch::Tensor& positions_raw, const torch::Tensor& values){
    //check input
    CHECK(positions_raw.size(0)==values.size(0)) << "Sizes of positions and values should match. Meaning that that there should be a value for each position. Positions_raw has sizes "<<positions_raw.sizes() << " and the values has size " << values.sizes();
    check_positions(positions_raw);
    check_values(positions_raw);
}



void Lattice::begin_splat(const bool reset_hashmap ){
    if(reset_hashmap)   {
        m_hash_table->clear();
    }else {
        m_hash_table->clear_only_values();
    }
    m_hash_table->m_nr_filled_is_dirty=true;
}


std::tuple<torch::Tensor, torch::Tensor> Lattice::splat_standalone(torch::Tensor& positions_raw, torch::Tensor& values ){
    check_positions_and_values(positions_raw, values);
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    int val_dim=values.size(1);

    m_positions=positions_raw; //raw positions which created this lattice


    //if it's not initialized to the correct values we intialize the hashtable
    if( !m_hash_table->m_keys_tensor.defined() ){
        m_hash_table->init(pos_dim, val_dim);
        m_hash_table->to(torch::kCUDA);
    }

    // if( !m_splatting_indices_tensor.defined() || m_splatting_indices_tensor.size(0)!=nr_positions*(m_pos_dim+1)  ){
    Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    splatting_indices_tensor.fill_(-1);
    splatting_weights_tensor.fill_(-1);


    //to cuda
    positions_raw=positions_raw.to("cuda");
    values=values.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");

    Tensor positions=positions_raw/m_sigmas_tensor;

    m_impl->splat_standalone(positions.data_ptr<float>(), values.data_ptr<float>(), nr_positions, pos_dim, val_dim,
                            splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(),  *(m_hash_table->m_impl) );
    m_hash_table->m_nr_filled_is_dirty=true;



    auto ret = std::make_tuple (splatting_indices_tensor, splatting_weights_tensor );
    return ret;

}


std::tuple<torch::Tensor, torch::Tensor> Lattice::just_create_verts(torch::Tensor& positions_raw, const bool return_indices_and_weights ){
    check_positions(positions_raw);
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);

    //if it's not initialized to the correct values we intialize the hashtable
    if( !m_hash_table->m_keys_tensor.defined() ){
        m_hash_table->init(pos_dim, 1 );
        m_hash_table->to(torch::kCUDA);
    }


    Tensor splatting_indices_tensor;
    Tensor splatting_weights_tensor;
    if (return_indices_and_weights){
        splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
        splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
        splatting_indices_tensor.fill_(-1);
        splatting_weights_tensor.fill_(-1);
    }


    //to cuda
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");

    Tensor positions=positions_raw/m_sigmas_tensor;

    if (return_indices_and_weights){
        m_impl->just_create_verts(positions.data_ptr<float>(), nr_positions, this->pos_dim(), this->val_dim(),
                                return_indices_and_weights,
                                splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );
    }else{
        m_impl->just_create_verts(positions.data_ptr<float>(), nr_positions, this->pos_dim(), this->val_dim(),
                                return_indices_and_weights,
                                nullptr, nullptr, *(m_hash_table->m_impl) );
    }

    m_hash_table->m_nr_filled_is_dirty=true;


    auto ret = std::make_tuple (splatting_indices_tensor, splatting_weights_tensor );
    return ret;
}

std::shared_ptr<Lattice> Lattice::expand(torch::Tensor& positions_raw, const int point_multiplier, const float noise_stddev, const bool expand_values ){
    check_positions(positions_raw);
    int pos_dim=positions_raw.size(1);

    //if it's not initialized to the correct values we intialize the hashtable
    if( !m_hash_table->m_keys_tensor.defined() ){
        m_hash_table->init(pos_dim, 1 );
        m_hash_table->to(torch::kCUDA);
    }

    //to cuda
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");

    //expand the positopns
    Tensor positions_expanded=positions_raw.repeat({point_multiplier, 1});

    //noise
    Tensor noise = torch::randn({ positions_expanded.size(0), positions_expanded.size(1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    noise=noise*noise_stddev;
    positions_expanded+=noise;


    std::shared_ptr<Lattice> expanded_lattice=create(this); //create a lattice with no config but takes the config from this one
    expanded_lattice->m_name="expanded_lattice";
    expanded_lattice->m_hash_table->m_values_tensor=torch::zeros({1, this->val_dim()}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //we just create some dummy values just so that the clear that we will do not will not destroy the current values. We will create the values when we know how many vertices we have
    expanded_lattice->m_hash_table->m_keys_tensor= this->m_hash_table->m_keys_tensor.clone();
    expanded_lattice->m_hash_table->m_entries_tensor= this->m_hash_table->m_entries_tensor.clone();
    expanded_lattice->m_hash_table->m_nr_filled_tensor= this->m_hash_table->m_nr_filled_tensor.clone();
    expanded_lattice->m_hash_table->update_impl();

    expanded_lattice->just_create_verts(positions_expanded, false );
    expanded_lattice->m_hash_table->m_nr_filled_is_dirty=true;


    if (expand_values){
        int nr_values_diff= expanded_lattice->nr_lattice_vertices() - nr_lattice_vertices();
        CHECK(nr_values_diff>=0) << "Nr of values in the difference is negative, we should always create more vertices, never substract so this doesnt make sense. In the current lattice we have " << nr_lattice_vertices() << " and in the expanded one we have " <<  expanded_lattice->nr_lattice_vertices();

        std::vector<int64_t> pad_values { 0,0,0,nr_values_diff  }; //left, right, top, bottom
        torch::nn::functional::PadFuncOptions option (pad_values);
        option.mode(torch::kConstant);
        Tensor expanded_values = torch::nn::functional::pad( values(), option);
        expanded_lattice->set_values(expanded_values);

        CHECK(expanded_lattice->values().size(0) == expanded_lattice->nr_lattice_vertices() ) << "The nr of lattice vertices and the nr of rows in the values should be the same. However we have nr of vertices " << expanded_lattice->nr_lattice_vertices() << " and the values have nr of rows " <<expanded_lattice->values().size(0);

    }


    return expanded_lattice;

}


std::tuple<std::shared_ptr<Lattice>, torch::Tensor, torch::Tensor, torch::Tensor> Lattice::distribute(torch::Tensor& positions_raw, torch::Tensor& values, const bool reset_hashmap){
    check_positions_and_values(positions_raw, values);
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    int val_dim=values.size(1);

    m_positions=positions_raw; //raw positions which created this lattice



    //if it's not initialized to the correct values we intialize the hashtable
    if(!m_hash_table->m_keys_tensor.defined()){
        m_hash_table->init(pos_dim, val_dim);
    }


    Tensor distributed_tensor = torch::zeros({ nr_positions *(pos_dim+1) , pos_dim + val_dim +1 }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    splatting_indices_tensor.fill_(-1);
    splatting_weights_tensor.fill_(-1);


    std::shared_ptr<Lattice> distributed_lattice=create(this); //create a lattice with no config but takes the config from this one
    distributed_lattice->m_hash_table->m_keys_tensor=this->m_hash_table->m_keys_tensor.clone();
    distributed_lattice->m_hash_table->m_entries_tensor=this->m_hash_table->m_entries_tensor.clone();
    if ( this->m_hash_table->m_values_tensor.defined()){
        distributed_lattice->m_hash_table->m_values_tensor=this->m_hash_table->m_values_tensor.clone();
    }
    distributed_lattice->m_name="distributed_lattice";
    distributed_lattice->m_hash_table->update_impl();


    if(reset_hashmap)   {
        distributed_lattice->m_hash_table->clear();
    }else {
        distributed_lattice->m_hash_table->clear_only_values();
    }


    //to cuda
    positions_raw=positions_raw.to("cuda");
    values=values.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");

    Tensor positions=positions_raw/m_sigmas_tensor;

    m_impl->distribute(positions.data_ptr<float>(), values.data_ptr<float>(), distributed_tensor.data_ptr<float>(), nr_positions, pos_dim, val_dim,
                            splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(distributed_lattice->m_hash_table->m_impl) );
    distributed_lattice->m_hash_table->m_nr_filled_is_dirty=true;

    VLOG(3) << "after distributing nr_verts is " << distributed_lattice->nr_lattice_vertices();

    auto ret = std::make_tuple (distributed_lattice, distributed_tensor, splatting_indices_tensor, splatting_weights_tensor );
    return ret;

}



std::shared_ptr<Lattice> Lattice::convolve_im2row_standalone(torch::Tensor& filter_bank, const int dilation, std::shared_ptr<Lattice> lattice_neighbours,  const bool flip_neighbours){

    if (!lattice_neighbours){
        lattice_neighbours=shared_from_this();
    }

    CHECK(filter_bank.defined()) << "Filter bank is undefined";
    CHECK(filter_bank.dim()==2) << "Filter bank should have dimension 2, corresponding with (filter_extent * val_dim) x nr_filters.  However it has dimension: " << filter_bank.dim();
    filter_bank=filter_bank.contiguous();

    int nr_filters=filter_bank.size(1) ;
    int filter_extent=filter_bank.size(0) / lattice_neighbours->val_dim();
    CHECK(filter_extent == get_filter_extent(1) ) << "Filters should convolve over all the neighbours in the 1 hop plus the center vertex lattice. So the filter extent should be " << get_filter_extent(1) << ". However it is" << filter_extent << "val dim is " << lattice_neighbours->val_dim();

    //this lattice should be coarser (so a higher lvl) or finer(lower lvl) or at least at the same lvl as the lattice neigbhours. But the differnce should be at most 1 level
    CHECK(std::abs(m_lvl-lattice_neighbours->m_lvl)<=1) << "the difference in levels between query and neigbhours lattice should be only 1 or zero, so the query should be corser by 1 level or finer by 1 lvl with respect to the neighbours. Or if they are at the same level then nothing needs to be done. However the current lattice lvl is " << m_lvl << " and the neighbours lvl is " << lattice_neighbours->m_lvl;

    CHECK(nr_lattice_vertices()!=0) << "Why does this current lattice have zero nr_filled?";
    int nr_vertices=nr_lattice_vertices();
    int cur_values_size=m_hash_table->m_values_tensor.size(0);

    std::shared_ptr<Lattice> convolved_lattice=create(this); //create a lattice with no config but takes the config from this one
    convolved_lattice->m_name="convolved_lattice";


    filter_bank=filter_bank.to("cuda");


    Tensor lattice_rowified=torch::zeros({nr_vertices, filter_extent* lattice_neighbours->val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );


    m_impl->im2row(nr_vertices, this->pos_dim(), lattice_neighbours->val_dim(), dilation, lattice_rowified.data_ptr<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, flip_neighbours, false);



    //multiply each patch with the filter bank
    Tensor convolved= lattice_rowified.mm(filter_bank);

    convolved_lattice->m_hash_table->set_values(convolved);

    CHECK(convolved_lattice->nr_lattice_vertices()!=0) << "Why does this convolved lattice has zero nr_filled?";

    return convolved_lattice;

}


torch::Tensor Lattice::im2rowindices(std::shared_ptr<Lattice> lattice_neighbours, const int filter_extent, const int dilation, const bool flip_neighbours){

    if (!lattice_neighbours){
        lattice_neighbours=shared_from_this();
    }

    CHECK(filter_extent == get_filter_extent(1) ) << "Filters should convolve over all the neighbours in the 1 hop plus the center vertex lattice. So the filter extent should be " << get_filter_extent(1) << ". However it is" << filter_extent;



    //this lattice should be coarser (so a higher lvl) or finer(lower lvl) or at least at the same lvl as the lattice neigbhours. But the differnce should be at most 1 level
    CHECK(std::abs(m_lvl-lattice_neighbours->m_lvl)<=1) << "the difference in levels between query and neigbhours lattice should be only 1 or zero, so the query should be corser by 1 level or finer by 1 lvl with respect to the neighbours. Or if they are at the same level then nothing needs to be done. However the current lattice lvl is " << m_lvl << " and the neighbours lvl is " << lattice_neighbours->m_lvl;

    CHECK(nr_lattice_vertices()!=0) << "Why does this current lattice have zero nr_filled?";
    int nr_vertices=nr_lattice_vertices();
    int cur_values_size=m_hash_table->m_values_tensor.size(0);


    Tensor lattice_rowified=torch::zeros({nr_vertices, filter_extent* lattice_neighbours->val_dim() }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );


    m_impl->im2rowindices(nr_vertices, this->pos_dim(), lattice_neighbours->val_dim(), dilation, lattice_rowified.data_ptr<int>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, flip_neighbours, false);

    return lattice_rowified;

}

torch::Tensor Lattice::im2row(std::shared_ptr<Lattice> lattice_neighbours, const int filter_extent, const int dilation, const bool flip_neighbours){

    if (!lattice_neighbours){
        lattice_neighbours=shared_from_this();
    }

    CHECK(filter_extent == get_filter_extent(1) ) << "Filters should convolve over all the neighbours in the 1 hop plus the center vertex lattice. So the filter extent should be " << get_filter_extent(1) << ". However it is" << filter_extent;



    //this lattice should be coarser (so a higher lvl) or finer(lower lvl) or at least at the same lvl as the lattice neigbhours. But the differnce should be at most 1 level
    CHECK(std::abs(m_lvl-lattice_neighbours->m_lvl)<=1) << "the difference in levels between query and neigbhours lattice should be only 1 or zero, so the query should be corser by 1 level or finer by 1 lvl with respect to the neighbours. Or if they are at the same level then nothing needs to be done. However the current lattice lvl is " << m_lvl << " and the neighbours lvl is " << lattice_neighbours->m_lvl;

    CHECK(nr_lattice_vertices()!=0) << "Why does this current lattice have zero nr_filled?";
    int nr_vertices=nr_lattice_vertices();
    int cur_values_size=m_hash_table->m_values_tensor.size(0);


    Tensor lattice_rowified=torch::zeros({nr_vertices, filter_extent* lattice_neighbours->val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    m_impl->im2row(nr_vertices, this->pos_dim(), lattice_neighbours->val_dim(), dilation, lattice_rowified.data_ptr<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, flip_neighbours, false);

    return lattice_rowified;

}

torch::Tensor Lattice::row2im(const torch::Tensor& lattice_rowified,  const int dilation, const int filter_extent, const int nr_filters, std::shared_ptr<Lattice> lattice_neighbours){

    CHECK(lattice_rowified.is_contiguous()) << "lattice rowified is not contiguous. Please call .contiguous() on it";
    CHECK(lattice_rowified.size(1)/filter_extent == val_dim() ) << "Each row of the lattice rowified shold be of size val_dim*filter_extent. But the row size is " << lattice_rowified.size(1) << " and th val dim is " << val_dim();

    if (!lattice_neighbours){
        lattice_neighbours=shared_from_this();
    }

    int nr_vertices=nr_lattice_vertices();
    m_hash_table->m_values_tensor=torch::zeros({nr_vertices, val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    m_hash_table->update_impl();

    CHECK(nr_lattice_vertices()!=0) <<"Something went wrong because have zero lattice vertices";


    m_impl->row2im(m_hash_table->capacity(), this->pos_dim(), lattice_neighbours->val_dim(), dilation, lattice_rowified.data_ptr<float>(), filter_extent, *(m_hash_table->m_impl), *(lattice_neighbours->m_hash_table->m_impl), m_lvl, lattice_neighbours->m_lvl, /*do_test*/false);

    return m_hash_table->m_values_tensor;
}


std::shared_ptr<Lattice> Lattice::create_coarse_verts(){

    int capacity=m_hash_table->capacity();
    int val_dim=m_hash_table->val_dim();
    int pos_dim=m_hash_table->pos_dim();

    std::shared_ptr<Lattice> coarse_lattice=create(this); //create a lattice with no config but takes the config from this one
    coarse_lattice->m_name="coarse_lattice";
    coarse_lattice->m_lvl=m_lvl+1;
    coarse_lattice->m_sigmas_tensor=m_sigmas_tensor.clone()*2.0; //the sigma for the coarser one is double. This is done so if we slice at this lattice we scale the positions with the correct sigma
    for(size_t i=0; i<m_sigmas.size(); i++){
        coarse_lattice->m_sigmas[i]=m_sigmas[i]*2.0;
    }
    coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({1, val_dim }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //we just create some dummy values just so that the clear that we will do not will not destroy the current values. We will create the values when we know how many vertices we have
    coarse_lattice->m_hash_table->m_keys_tensor=torch::zeros({capacity, pos_dim}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->m_entries_tensor=torch::zeros({capacity}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) ) ;
    coarse_lattice->m_hash_table->m_nr_filled_tensor=torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->m_nr_filled_is_dirty=true;
    coarse_lattice->m_hash_table->clear();
    coarse_lattice->m_hash_table->update_impl();

    m_impl->coarsen(capacity, pos_dim, *(m_hash_table->m_impl), *(coarse_lattice->m_hash_table->m_impl)  );

    int nr_vertices=coarse_lattice->nr_lattice_vertices();
    VLOG(3) << "after coarsening nr_verts of the coarse lattice is " << nr_vertices;

    coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({nr_vertices, val_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  ); //we create exactly the values required for he vertices that were allocated
    coarse_lattice->m_hash_table->update_impl();

    return coarse_lattice;

}


std::shared_ptr<Lattice> Lattice::create_coarse_verts_naive(torch::Tensor& positions_raw){

    check_positions(positions_raw);

    int capacity=m_hash_table->capacity();
    int val_dim=m_hash_table->val_dim();
    int pos_dim=m_hash_table->pos_dim();


    std::shared_ptr<Lattice> coarse_lattice=create(this); //create a lattice with no config but takes the config from this one
    coarse_lattice->m_name="coarse_lattice";
    coarse_lattice->m_lvl=m_lvl+1;
    coarse_lattice->m_sigmas_tensor=m_sigmas_tensor.clone()*2.0; //the sigma for the coarser one is double. This is done so if we slice at this lattice we scale the positions with the correct sigma
    coarse_lattice->m_sigmas=m_sigmas;
    for(size_t i=0; i<m_sigmas.size(); i++){
        coarse_lattice->m_sigmas[i]=m_sigmas[i]*2.0;
    }
    coarse_lattice->m_hash_table->m_values_tensor=torch::zeros({1, val_dim}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) ); //we just create some dummy values just so that the clear that we will do not will not destroy the current values. We will create the values when we know how many vertices we have
    coarse_lattice->m_hash_table->m_keys_tensor=torch::zeros({capacity, pos_dim}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->m_entries_tensor=torch::zeros({capacity}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) ) ;
    coarse_lattice->m_hash_table->m_nr_filled_tensor=torch::zeros({1}, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    coarse_lattice->m_hash_table->m_nr_filled_is_dirty=true;
    coarse_lattice->m_hash_table->clear();
    coarse_lattice->m_hash_table->update_impl();


    coarse_lattice->begin_splat();
    coarse_lattice->m_hash_table->update_impl();

    coarse_lattice->just_create_verts(positions_raw, false);


    return coarse_lattice;

}



torch::Tensor Lattice::slice_standalone_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw);
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    splatting_indices_tensor=splatting_indices_tensor.contiguous();
    splatting_weights_tensor=splatting_weights_tensor.contiguous();


     //to cuda
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");

    VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;

    //initialize the output values to zero
    Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );


    //assume we have already splatting weight and indices
    if( !splatting_indices_tensor.defined() || !splatting_weights_tensor.defined()  || splatting_indices_tensor.size(0)!=nr_positions*(this->pos_dim()+1) ||  splatting_weights_tensor.size(0)!=nr_positions*(this->pos_dim()+1)  ){
        LOG(FATAL) << "Indices or wegiths tensor is not created or doesnt have the correct size. We are assuming it has size " << nr_positions*(this->pos_dim()+1) << "but indices has size " << splatting_indices_tensor.sizes() << " m_splatting_weights_tensor have size "  << splatting_weights_tensor.sizes();
    }
    m_hash_table->update_impl();


    m_impl->slice_standalone_with_precomputation( positions.data_ptr<float>(), sliced_values_hom_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );

    return sliced_values_hom_tensor;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::slice_standalone_no_precomputation(torch::Tensor& positions_raw){

    check_positions(positions_raw);
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


     //to cuda
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");

    VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;

    //initialize the output values to zero
    Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, val_dim() }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    //recalculate the splatting indices and weight for the backward pass of the slice
    Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    splatting_indices_tensor.fill_(-1);
    splatting_weights_tensor.fill_(-1);

    m_hash_table->update_impl();


    m_impl->slice_standalone_no_precomputation( positions.data_ptr<float>(), sliced_values_hom_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );

    auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor );
    return ret;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Lattice::gather_standalone_no_precomputation(torch::Tensor& positions_raw){

    check_positions(positions_raw);
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";


     //to cuda
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");

    VLOG(3) << "gather standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;

    //initialize the output values to zero
    int row_size_gathered=(pos_dim+1)*(val_dim()+1); //we have m_pos_dim+1 vertices in a lattice and each has values of m_val_full_dim plus a barycentric coord
    Tensor gathered_values_tensor=torch::zeros({nr_positions, row_size_gathered}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    //recalculate the splatting indices and weight for the backward pass of the gather
    Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    splatting_indices_tensor.fill_(-1);
    splatting_weights_tensor.fill_(-1);
    m_hash_table->update_impl();


    m_impl->gather_standalone_no_precomputation( positions.data_ptr<float>(), gathered_values_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );

    auto ret = std::make_tuple (gathered_values_tensor, splatting_indices_tensor, splatting_weights_tensor );
    return ret;
}


torch::Tensor Lattice::gather_standalone_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw);
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    splatting_indices_tensor=splatting_indices_tensor.contiguous();
    splatting_weights_tensor=splatting_weights_tensor.contiguous();


     //to cuda
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");

    VLOG(3) << "gather standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;

    //initialize the output values to zero
    int row_size_gathered=(this->pos_dim()+1)*(this->val_dim()+1); //we have m_pos_dim+1 vertices in a lattice and each has values of m_val_full_dim plus a barycentric coord
    Tensor gathered_values_tensor=torch::zeros({nr_positions, row_size_gathered}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );

    //assume we have already splatting weight and indices
    if( !splatting_indices_tensor.defined() || !splatting_weights_tensor.defined()  || splatting_indices_tensor.size(0)!=nr_positions*(this->pos_dim()+1) ||  splatting_weights_tensor.size(0)!=nr_positions*(this->pos_dim()+1)  ){
        LOG(FATAL) << "Indices or wegiths tensor is not created or doesnt have the correct size. We are assuming it has size " << nr_positions*(this->pos_dim()+1) << "but indices has size " << splatting_indices_tensor.sizes() << " m_splatting_weights_tensor have size "  << splatting_weights_tensor.sizes();
    }
    m_hash_table->update_impl();


    m_impl->gather_standalone_with_precomputation( positions.data_ptr<float>(), gathered_values_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(),  nr_positions, splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), *(m_hash_table->m_impl) );

    return gathered_values_tensor;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>  Lattice::slice_classify_no_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes){


    check_positions(positions_raw);
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    delta_weights=delta_weights.contiguous();
    linear_clasify_weight=linear_clasify_weight.contiguous();
    linear_clasify_bias=linear_clasify_bias.contiguous();



     //to cuda
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    delta_weights=delta_weights.to("cuda");
    linear_clasify_weight=linear_clasify_weight.to("cuda");
    linear_clasify_bias=linear_clasify_bias.to("cuda");

    VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;

    //we store here the class logits directly
    Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, nr_classes}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );


    //recalculate the splatting indices and weight for the backward pass of the slice
    Tensor splatting_indices_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kInt32).device(torch::kCUDA, 0) );
    Tensor splatting_weights_tensor = torch::empty({nr_positions*(pos_dim+1) }, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );
    splatting_indices_tensor.fill_(-1);
    splatting_weights_tensor.fill_(-1);
    m_hash_table->update_impl();


    m_impl->slice_classify_no_precomputation( positions.data_ptr<float>(),
                                              sliced_values_hom_tensor.data_ptr<float>(),
                                              delta_weights.data_ptr<float>(),
                                              linear_clasify_weight.data_ptr<float>(),
                                              linear_clasify_bias.data_ptr<float>(),
                                              nr_classes,
                                              this->pos_dim(),
                                              this->val_dim(),
                                              nr_positions,
                                              splatting_indices_tensor.data_ptr<int>(),
                                              splatting_weights_tensor.data_ptr<float>(),
                                              *(m_hash_table->m_impl) );

    auto ret = std::make_tuple (sliced_values_hom_tensor, splatting_indices_tensor, splatting_weights_tensor );
    return ret;
}


torch::Tensor Lattice::slice_classify_with_precomputation(torch::Tensor& positions_raw, torch::Tensor& delta_weights, torch::Tensor& linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw);
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    delta_weights=delta_weights.contiguous();
    linear_clasify_weight=linear_clasify_weight.contiguous();
    linear_clasify_bias=linear_clasify_bias.contiguous();
    splatting_indices_tensor=splatting_indices_tensor.contiguous();
    splatting_weights_tensor=splatting_weights_tensor.contiguous();



     //to cuda
    positions_raw=positions_raw.to("cuda");
    m_sigmas_tensor=m_sigmas_tensor.to("cuda");
    delta_weights=delta_weights.to("cuda");
    linear_clasify_weight=linear_clasify_weight.to("cuda");
    linear_clasify_bias=linear_clasify_bias.to("cuda");

    VLOG(3) << "slice standalone scaling by a sigma of " << m_sigmas_tensor;
    Tensor positions=positions_raw/m_sigmas_tensor;

    //we store here the class logits directly
    Tensor sliced_values_hom_tensor=torch::zeros({nr_positions, nr_classes}, torch::dtype(torch::kFloat32).device(torch::kCUDA, 0) );


    //assume we have already splatting weight and indices
    if( !splatting_indices_tensor.defined() || !splatting_weights_tensor.defined()  || splatting_indices_tensor.size(0)!=nr_positions*(this->pos_dim()+1) ||  splatting_weights_tensor.size(0)!=nr_positions*(this->pos_dim()+1)  ){
        LOG(FATAL) << "Indices or wegiths tensor is not created or doesnt have the correct size. We are assuming it has size " << nr_positions*(this->pos_dim()+1) << "but indices has size " << splatting_indices_tensor.sizes() << " m_splatting_weights_tensor have size "  << splatting_weights_tensor.sizes();
    }
    m_hash_table->update_impl();


    m_impl->slice_classify_with_precomputation( positions.data_ptr<float>(),
                                              sliced_values_hom_tensor.data_ptr<float>(),
                                              delta_weights.data_ptr<float>(),
                                              linear_clasify_weight.data_ptr<float>(),
                                              linear_clasify_bias.data_ptr<float>(),
                                              nr_classes,
                                              this->pos_dim(),
                                              this->val_dim(),
                                              nr_positions,
                                              splatting_indices_tensor.data_ptr<int>(),
                                              splatting_weights_tensor.data_ptr<float>(),
                                              *(m_hash_table->m_impl) );

    return sliced_values_hom_tensor;

}





void Lattice::slice_backwards_standalone_with_precomputation(torch::Tensor& positions_raw, const torch::Tensor& sliced_values_hom, const Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw);
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_sliced_values.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
    CHECK(sliced_values_hom.is_contiguous()) << "sliced_values_hom needs to be contiguous. Please call .contiguous() on it";
    splatting_indices_tensor=splatting_indices_tensor.contiguous();
    splatting_weights_tensor=splatting_weights_tensor.contiguous();




    m_impl->slice_backwards_standalone_with_precomputation( sliced_values_hom.data_ptr<float>(), grad_sliced_values.data_ptr<float>(), splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(), nr_positions, *(m_hash_table->m_impl) );

}


void Lattice::slice_backwards_standalone_with_precomputation_no_homogeneous(torch::Tensor& positions_raw, const Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw);
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_sliced_values.dim()==2) <<"grad_sliced_values should be nr_positions x m_val_dim, so it should have 2 dimensions. However it has "<< grad_sliced_values.dim();
    CHECK(grad_sliced_values.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
    splatting_indices_tensor=splatting_indices_tensor.contiguous();
    splatting_weights_tensor=splatting_weights_tensor.contiguous();

    m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), grad_sliced_values.size(1)},  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    m_hash_table->update_impl();



    m_impl->slice_backwards_standalone_with_precomputation_no_homogeneous(grad_sliced_values.data_ptr<float>(), splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(), nr_positions, *(m_hash_table->m_impl) );

}


void Lattice::slice_classify_backwards_with_precomputation(const torch::Tensor& grad_class_logits, torch::Tensor& positions_raw, torch::Tensor& initial_values, torch::Tensor& delta_weights, torch::Tensor&  linear_clasify_weight, torch::Tensor& linear_clasify_bias, const int nr_classes, torch::Tensor& grad_lattice_values, torch::Tensor& grad_delta_weights, torch::Tensor& grad_linear_clasify_weight, torch::Tensor& grad_linear_clasify_bias, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw);
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_class_logits.dim()==2) <<"grad_class_logits should be  nr_positions x nr_classes, so it should have 2 dimensions. However it has "<< grad_class_logits.dim();
    CHECK(grad_class_logits.is_contiguous()) << "grad_class_logits needs to be contiguous. Please call .contiguous() on it";
    initial_values=initial_values.contiguous();
    delta_weights=delta_weights.contiguous();
    linear_clasify_weight=linear_clasify_weight.contiguous();
    linear_clasify_bias=linear_clasify_bias.contiguous();
    splatting_indices_tensor=splatting_indices_tensor.contiguous();
    splatting_weights_tensor=splatting_weights_tensor.contiguous();


    m_impl->slice_classify_backwards_with_precomputation(grad_class_logits.data_ptr<float>(), initial_values.data_ptr<float>(),  splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(), this->val_dim(), nr_positions,
    delta_weights.data_ptr<float>(), linear_clasify_weight.data_ptr<float>(), linear_clasify_bias.data_ptr<float>(), nr_classes, grad_lattice_values.data_ptr<float>(), grad_delta_weights.data_ptr<float>(), grad_linear_clasify_weight.data_ptr<float>(),grad_linear_clasify_bias.data_ptr<float>(),
     *(m_hash_table->m_impl) );

}

void Lattice::gather_backwards_standalone_with_precomputation(const torch::Tensor& positions_raw, const Tensor& grad_sliced_values, torch::Tensor& splatting_indices_tensor, torch::Tensor& splatting_weights_tensor){

    check_positions(positions_raw);
    CHECK(val_dim()>0) << "m_val_dim is 0 or lwoer. We have to splat something first so that we have values from where to slice. Val dim is " << val_dim();
    int nr_positions=positions_raw.size(0);
    int pos_dim=positions_raw.size(1);
    int val_dim=grad_sliced_values.size(1)/(pos_dim+1)-1; //we will acumulate the gradient into the value tensor. And it should have the same val_dim as the values that were in the lattice_we_gathered from
    CHECK(pos_dim==this->pos_dim()) << " The position dimension do not coreespond with the ones we used for creating the lattice";
    CHECK(grad_sliced_values.dim()==2) <<"grad_sliced_values should be nr_positions x ((m_val_dim+1)*(m_pos_dim+1)), so it should have 2 dimensions. However it has "<< grad_sliced_values.dim();
    CHECK(grad_sliced_values.is_contiguous()) << "Grad sliced values needs to be contiguous. Please call .contiguous() on it";
    splatting_indices_tensor=splatting_indices_tensor.contiguous();
    splatting_weights_tensor=splatting_weights_tensor.contiguous();



    m_hash_table->m_values_tensor=torch::zeros({nr_lattice_vertices(), val_dim },  torch::dtype(torch::kFloat32).device(torch::kCUDA, 0)  );
    m_hash_table->update_impl();



    m_impl->gather_backwards_standalone_with_precomputation(grad_sliced_values.data_ptr<float>(), splatting_indices_tensor.data_ptr<int>(), splatting_weights_tensor.data_ptr<float>(), this->pos_dim(),  this->val_dim(), nr_positions, *(m_hash_table->m_impl) );


}



std::shared_ptr<Lattice> Lattice::clone_lattice(){
    std::shared_ptr<Lattice> new_lattice=create(this); //create a lattice with no config but takes the config from this one
    return new_lattice;
}


void Lattice::increase_sigmas(const float stepsize){
    for(size_t i=0; i<m_sigmas.size(); i++){
        m_sigmas[i]+=stepsize;
    }

    m_sigmas_tensor=vec2tensor(m_sigmas);

}



//getters
int Lattice::val_dim(){
    return m_hash_table->val_dim();
}
int Lattice::pos_dim(){
    return m_hash_table->pos_dim();
}
int Lattice::capacity(){
    return m_hash_table->capacity();
}
std::string Lattice::name(){
    return m_name;
}
int Lattice::nr_lattice_vertices(){
    //check if the nr_latttice_vertices is dirty which means that a kernel has been executed that might have modified the nr of vertices
    int nr_verts=0;
    if (m_hash_table->m_nr_filled_is_dirty){
        m_hash_table->m_nr_filled_is_dirty=false;
        cudaMemcpy ( &nr_verts,  m_hash_table->m_nr_filled_tensor.data_ptr<int>(), sizeof(int), cudaMemcpyDeviceToHost );
        m_hash_table->m_nr_filled=nr_verts;
    }else{
        // return number lattice vertices that the cpu knows about
        nr_verts=m_hash_table->m_nr_filled;
    }

    CHECK(nr_verts>=0) << "nr vertices cannot be negative. However it is " << nr_verts;
    CHECK(nr_verts<1e+8) << "nr vertices cannot be that high. However it is " << nr_verts;

    return nr_verts;
}
int Lattice::get_filter_extent(const int neighborhood_size) {
    CHECK(neighborhood_size==1) << "At the moment we only have implemented a filter with a neighbourhood size of 1. I haven't yet written the more general formula for more neighbourshood size";
    CHECK(this->pos_dim()!=-1) << "m pos dim is not set. It is -1";

    return 2*(this->pos_dim()+1) + 1; //because we have 2 neighbour for each axis and we have pos_dim+1 axes. Also a +1 for the center vertex
}
int Lattice::get_expected_filter_extent(const int neighborhood_size){
    CHECK(neighborhood_size==1) << "At the moment we only have implemented a filter with a neighbourhood size of 1. I haven't yet written the more general formula for more neighbourshood size";

    return 2*(m_expected_position_dimensions+1) + 1; //because we have 2 neighbour for each axis and we have pos_dim+1 axes. Also a +1 for the center vertex

}
torch::Tensor Lattice::sigmas_tensor(){
    return m_sigmas_tensor;
}
torch::Tensor Lattice::positions(){
    return m_positions;
}
std::shared_ptr<HashTable> Lattice::hash_table(){
    return m_hash_table;
}
torch::Tensor Lattice::values(){
    return  m_hash_table->m_values_tensor;
}



//setters
void Lattice::set_sigma(const float sigma){
    int nr_sigmas=m_sigmas_val_and_extent.size();
    CHECK(nr_sigmas==1) << "We are summing we have onyl one sigma. This method is intended to affect only one and not two sigmas independently";

    for(size_t i=0; i<m_sigmas.size(); i++){
        m_sigmas[i]=sigma;
    }

    m_sigmas_tensor=vec2tensor(m_sigmas);
}
void Lattice::set_name(const std::string name){
    m_name=name;
}
void Lattice::set_values(const torch::Tensor& new_values){
    m_hash_table->set_values(new_values);
    CHECK(new_values.size(0)==nr_lattice_vertices()) << "The nr of rows in the new values does not correspond to the nr_lattice_vertices. Nr of rows is " << new_values.size(0) << " and nr lattice vertices is " << nr_lattice_vertices();
}
void Lattice::set_positions( const torch::Tensor& positions_raw ){
    m_positions=positions_raw;
}
