#include <string>
#include <stdexcept>

#include "Conv2D/Mesh.hpp"

int main(int argc, char **argv)
{
    std::string inFile;
    std::string outFile;
    std::string matFile;

    if (argc == 1)
    {
        inFile  = "test.gri";
        outFile = "output.gri";
        matFile = "matrix.txt";
    }
    else if (argc == 2)
    {
        inFile  = argv[1];
        outFile = "output.gri";
        matFile = "matrix.txt";
    }
    else if (argc == 3)
    {
        inFile  = argv[1];
        outFile = argv[2];
        matFile = "matrix.txt";
    }
    else if (argc == 4)
    {
        inFile  = argv[1];
        outFile = argv[2];
        matFile = argv[3];
    }
    else
    {
        throw std::runtime_error("Error in parsing command line arguments");
    }

    Mesh mesh;

    mesh.readFromFile(inFile);
    mesh.writeToFile(outFile);

    mesh.computeMatrices();
    mesh.writeMatricesToFile(matFile);

    return 0;
}
