

#ifndef PRESSIODEMOAPPS_SCHWARZ_TILING_HPP_
#define PRESSIODEMOAPPS_SCHWARZ_TILING_HPP_


#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>



namespace pdaschwarz{

//
// struct to hold topology information about tiling/partitions
//
struct Tiling
{
    explicit Tiling(const std::string & meshRoot){
        read_domain_info(meshRoot);
        calc_neighbor_dims();
    }

    void describe(){
        std::cout << " Tiling info: "
            << " ndomX = " << m_ndomX
            << " ndomY = " << m_ndomY
            << " ndomZ = " << m_ndomZ
            << " ndomains = " << m_ndomains
            << " overlap = " << m_overlap
            << '\n';
    }

    int dim() const{ return m_dim; }
    int overlap() const { return m_overlap; }
    auto exchDomIdVec() const{ return m_exchDomIdVec; }
    int count() const { return m_ndomains; }
    int countX() const { return m_ndomX; }
    int countY() const { return m_ndomY; }
    int countZ() const { return m_ndomZ; }

private:

    void read_domain_info(const std::string & meshRoot)
    {
        const auto inFile = meshRoot + "/info_domain.dat";
        std::ifstream foundFile(inFile);
        if (!foundFile) {
            std::cout << "file not found " << inFile << std::endl;
            exit(EXIT_FAILURE);
        }

        // defaults
        m_ndomX = 1;
        m_ndomY = 1;
        m_ndomZ = 1;
        std::ifstream source( inFile, std::ios_base::in);
        std::string line;
        while (getline(source, line)) {
	    std::istringstream ss(line);
            std::string colVal;
            ss >> colVal;

            if (colVal == "dim"){
                ss >> colVal;
                m_dim = stoi(colVal);
                if (m_dim < 1) throw std::runtime_error("dim must be >= 1");
            }
            else if (colVal == "ndomX"){
                ss >> colVal;
                m_ndomX = stoi(colVal);
                if (m_ndomX < 1) throw std::runtime_error("ndomX must be >= 1");
            }

            else if (colVal == "ndomY"){
                ss >> colVal;
                m_ndomY = stoi(colVal);
                if (m_ndomY < 1) throw std::runtime_error("ndomY must be >= 1");
            }
            else if (colVal == "ndomZ"){
                ss >> colVal;
                m_ndomZ = stoi(colVal);
                if (m_ndomZ < 1) throw std::runtime_error("ndomZ must be >= 1");
            }

            else if (colVal == "overlap"){
                ss >> colVal;
                m_overlap = stoi(colVal);
                if (m_overlap < 0) throw std::runtime_error("overlap must be >= 0");
            }
        }
        source.close();
        m_ndomains = m_ndomX * m_ndomY * m_ndomZ;
    }

    void calc_neighbor_dims()
    {
        // determine neighboring domain IDs
        int maxDomNeighbors = 2 * m_dim;
        m_exchDomIdVec.resize(m_ndomains, std::vector<int>(maxDomNeighbors, -1));

        for (int domIdx = 0; domIdx < m_ndomains; ++domIdx) {
            // subdomain indices
            int i = {};
            int j = {};
            int k = {};
            i = domIdx % m_ndomX;
            if (m_dim > 1) {
                j = domIdx / m_ndomX;
            }
            if (m_dim == 2) {
                k = domIdx / (m_ndomX * m_ndomY);
            }

            // 1D, 2D, and 3D
            // left boundary
            if (i != 0) {
                m_exchDomIdVec[domIdx][0] = domIdx - 1;
            }

            // right boundary
            if (i != (m_ndomX - 1)) {
                // ordering change for 1D vs. 2D/3D faces
                if (m_dim == 1) {
                    m_exchDomIdVec[domIdx][1] = domIdx + 1;
                } else {
                    m_exchDomIdVec[domIdx][2] = domIdx + 1;
                }
            }

            // 2D and 3D
            if (m_dim > 1) {
                // front boundary
                if (j != (m_ndomY - 1)) {
                    m_exchDomIdVec[domIdx][1] = domIdx + m_ndomX;
                }
                // back boundary
                if (j != 0) {
                    m_exchDomIdVec[domIdx][3] = domIdx - m_ndomX;
                }
            }

            // 3D
            if (m_dim > 2) {
                // bottom boundary
                if (k != 0) {
                    m_exchDomIdVec[domIdx][4] = domIdx - (m_ndomX * m_ndomY);
                }
                // top boundary
                if (k != (m_ndomZ - 1)) {
                    m_exchDomIdVec[domIdx][5] = domIdx + (m_ndomX * m_ndomY);
                }
            }
        }
    }

private:

    int m_dim = {};
    int m_ndomX = {};
    int m_ndomY = {};
    int m_ndomZ = {};
    int m_overlap = {};
    int m_ndomains = {};

    std::vector<std::vector<int>> m_exchDomIdVec;
};

}

#endif
