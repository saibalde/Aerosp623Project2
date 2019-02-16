#include <string>
#include <stdexcept>

#include "mesh2d.hpp"

int main(int argc, char **argv)
{
    std::string inFile;
    std::string outFile;

    if (argc == 1)
    {
        inFile  = "test.gri";
        outFile = "output.gri";
    }
    else if (argc == 2)
    {
        inFile  = argv[1];
        outFile = "output.gri";
    }
    else if (argc == 3)
    {
        inFile  = argv[1];
        outFile = argv[2];
    }
    else
    {
        throw std::runtime_error("Error in parsing command line arguments");
    }

    Mesh2d mesh(inFile);
    mesh.output(outFile);

    return 0;
}
