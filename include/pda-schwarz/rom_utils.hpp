
/*
    These are largely copied from pressio-tutorials, simplified for my own reading comprehension
*/

#ifndef PRESSIODEMOAPPS_SCHWARZ_ROMUTILS_
#define PRESSIODEMOAPPS_SCHWARZ_ROMUTILS_

#include <string>
#include <fstream>
#include <iostream>
#include "Eigen/Dense"

using namespace std;

namespace pdaschwarz {

void checkfile(const string & fileIn){
    ifstream infile(fileIn);
    if (infile.good() == 0) {
        throw runtime_error("Cannot find file: " + fileIn);
    }
}

template<class ScalarType>
auto read_matrix_from_binary(const string & fileName, int numColsToRead) {

    using matrix_type = Eigen::Matrix<double, -1, -1, Eigen::ColMajor>;
    using sc_t  = typename matrix_type::Scalar;

    checkfile(fileName);

    matrix_type M;

    ifstream fin(fileName, ios::in | ios::binary);
    fin.exceptions(ifstream::failbit | ifstream::badbit);

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
        cerr << strerror(errno) << endl;
        throw runtime_error("ERROR READING binary file");
    }
    else{
        cerr << fin.gcount() << " bytes read\n";
    }
    fin.close();
    return M;

}

template<class ScalarType>
auto read_vector_from_binary(const string & fileName) {
    auto Vmat = read_matrix_from_binary<ScalarType>(fileName, 1);
    Eigen::VectorXd V = Vmat.col(0);
    return V;
}

}

#endif