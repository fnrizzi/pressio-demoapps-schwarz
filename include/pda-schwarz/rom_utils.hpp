
/*
    These are largely copied from pressio-tutorials, simplified for my own reading comprehension
*/

#ifndef PRESSIODEMOAPPS_SCHWARZ_ROMUTILS_
#define PRESSIODEMOAPPS_SCHWARZ_ROMUTILS_

#include <string>
#include <fstream>
#include <iostream>
#include "Eigen/Dense"



namespace pdaschwarz {

void checkfile(const std::string & fileIn){
    std::ifstream infile(fileIn);
    if (infile.good() == 0) {
        throw std::runtime_error("Cannot find file: " + fileIn);
    }
}

template<class ScalarType>
auto read_matrix_from_binary(const std::string & fileName, int numColsToRead) {

    using matrix_type = Eigen::Matrix<ScalarType, -1, -1, Eigen::ColMajor>;
    using sc_t  = typename matrix_type::Scalar;

    checkfile(fileName);

    matrix_type M;

    std::ifstream fin(fileName, std::ios::in | std::ios::binary);
    fin.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    // read 2 8-byte integer header, size matrix accordingly
    size_t rows = {};
    size_t cols = {};
    fin.read((char*) (&rows), sizeof(size_t));
    fin.read((char*) (&cols), sizeof(size_t));
    M.resize(rows, numColsToRead);

    // read matrix
    // assumed that binary file has same record length as intended matrix (float, double, etc.)
    const auto nBytes = rows * numColsToRead * sizeof(sc_t);
    fin.read( (char *) M.data(), nBytes );

    if (!fin){
        std::cerr << strerror(errno) << std::endl;
        throw std::runtime_error("ERROR READING binary file");
    }
    else{
        std::cerr << fin.gcount() << " bytes read\n";
    }
    fin.close();
    return M;

}

template<class ScalarType>
auto read_vector_from_binary(const std::string & fileName) {
    auto Vmat = read_matrix_from_binary<ScalarType>(fileName, 1);

    using vector_type = Eigen::Vector<ScalarType, -1>;
    vector_type V(Vmat.rows());
    for (std::size_t i = 0; i < Vmat.rows(); ++i) {
        V(i) = Vmat(i, 0);
    }

    return V;
}

// TODO: adjust I/O naming conventions to reflect return type
template<class ScalarType>
auto read_vector_from_ascii(const std::string & fileName)
{

    checkfile(fileName);

    std::ifstream source;
    source.open(fileName, std::ios_base::in);
    std::string line, value;
    std::vector<ScalarType> v;
    while (std::getline(source, line) ) {
        std::istringstream in(line);
        in >> value;
        if (std::is_floating_point<ScalarType>::value) {
            v.push_back(std::atof(value.c_str()));
        }
        else {
            v.push_back(std::atoi(value.c_str()));
        }
    }
    source.close();

    return v;
}

// pretty much everything below this is ripped directly from pressio-tutorials
template <typename T = int32_t>
auto create_cell_gids_vector_and_fill_from_ascii(const std::string & fileName)
{
    using vector_type = Eigen::Matrix<T, -1, 1>;

    const auto v = read_vector_from_ascii<T>(fileName);
    vector_type result(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i];
    }
    return result;
}

template<class IntType>
IntType find_index(const Eigen::Matrix<IntType, -1, 1> & vector, IntType target)
{
    for (std::size_t j = 0; j < vector.size(); ++j) {
        if (vector[j] == target) {
            return j;
        }
    }

    return std::numeric_limits<IntType>::max();
}

// class required to pass to LSPG hyper-reduction problem
// provides a sort of BLAS interface for vector and matrix addition
// also supplies mapping from sample mesh indices to stencil mesh indices
template<class ScalarType>
struct HypRedUpdater
{
    using vec_operand_type = Eigen::Matrix<ScalarType, -1, 1>;
    using mat_ll_operand_type = Eigen::Matrix<ScalarType, -1, -1>;
    std::vector<int> indices_;
    int numDofsPerCell_ = {};

    explicit HypRedUpdater(const int numDofsPerCell, const std::string & stfile, const std::string & safile)
        : numDofsPerCell_(numDofsPerCell)
    {
        const auto stencilMeshGids = create_cell_gids_vector_and_fill_from_ascii(stfile);
        const auto sampleMeshGids  = create_cell_gids_vector_and_fill_from_ascii(safile);

        indices_.resize(sampleMeshGids.size());
        for (std::size_t i = 0; i < indices_.size(); ++i) {
            const auto index = find_index<int>(stencilMeshGids, sampleMeshGids[i]);
            assert(index != std::numeric_limits<int>::max());
            indices_[i] = index;
        }
    }

    void updateSampleMeshOperandWithStencilMeshOne(
        vec_operand_type & a,
        ScalarType alpha,
        const vec_operand_type & b,
        ScalarType beta) const
    {
        for (std::size_t i = 0; i < indices_.size(); ++i) {
            const std::size_t r = i * numDofsPerCell_;
            const std::size_t g = indices_[i] * numDofsPerCell_;
            for (std::size_t k = 0; k < numDofsPerCell_; ++k) {
                a(r+k) = alpha * a(r+k) + beta * b(g+k);
            }
        }
    }

    void updateSampleMeshOperandWithStencilMeshOne(
        mat_ll_operand_type & a,
        ScalarType alpha,
        const mat_ll_operand_type & b,
        ScalarType beta) const
    {
        for (std::size_t j = 0; j < b.cols(); ++j) {
            for (std::size_t i = 0; i < indices_.size(); ++i) {
                const std::size_t r = i * numDofsPerCell_;
                const std::size_t g = indices_[i] * numDofsPerCell_;
                for (std::size_t k = 0; k < numDofsPerCell_; ++k) {
                    a(r+k, j) = alpha * a(r+k, j) + beta * b(g+k, j);
                }
            }
        }
    }
};

template<class mesh_t>
auto create_hyper_updater(
    const int numDofsPerCell,
    const std::string & stfile,
    const std::string & safile)
{
    using scalar_type = typename mesh_t::scalar_t;

    checkfile(stfile);
    checkfile(safile);

    using return_type = HypRedUpdater<scalar_type>;
    return return_type(numDofsPerCell, stfile, safile);

}

// extract stencil mesh values from full-order matrix
// rows are unrolled state vector, columns are basis/snapshot/etc. vectors
// OperandType should really be an Eigen::Vector
template<class OperandType, class CellGidsVectorType>
auto reduce_matrix_on_stencil_mesh(
    const OperandType & operand,
    const CellGidsVectorType & stencilMeshGids,
    const int numDofsPerCell)
{

    const auto totStencilDofs = stencilMeshGids.size() * numDofsPerCell;
    OperandType result(totStencilDofs, operand.cols());
    for (int i = 0; i < stencilMeshGids.size(); ++i) {
        for (int k = 0; k < numDofsPerCell; ++k){
            const int row = i * numDofsPerCell + k;
            const int ind = stencilMeshGids[i] * numDofsPerCell + k;
            for (int j = 0; j < operand.cols(); ++j){
                result(row, j) = operand(ind, j);
            }
        }
    }
    return result;
}

// extract stencil mesh values from full-order vector
// OperandType should really be an Eigen::Vector
template<class OperandType, class CellGidsVectorType>
auto reduce_vector_on_stencil_mesh(
    const OperandType & operand,
    const CellGidsVectorType & stencilMeshGids,
    const int numDofsPerCell)
{

    const auto totStencilDofs = stencilMeshGids.size() * numDofsPerCell;
    OperandType result(totStencilDofs);
    for (int i = 0; i < stencilMeshGids.size(); ++i) {
        for (int k = 0; k < numDofsPerCell; ++k) {
            const int row = i * numDofsPerCell + k;
            const int ind = stencilMeshGids[i] * numDofsPerCell + k;
            result(row) = operand(ind);
        }
    }
    return result;
}

}

#endif
