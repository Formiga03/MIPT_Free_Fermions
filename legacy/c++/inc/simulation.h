#include <iostream>
#include <vector>
#include <complex>
using namespace std;

class Simulation {
    private:
        int size;
        bool exp;
        double Tstep;
        double Tmax;
        double MPstep;
        double MPmin;
        double MPmax;
        int Iters;
        vector<vector<double>> Data;
        
    public:

        Simulation();
        Simulation(int size, 
                   double Time,
                   double Tstep, 
                   double MPmin, 
                   double MPmax,
                   double MPstep, 
                   int Iters,
                   bool exp=false);

        void hamiltonian_creator_2D(int size, double j=1, bool period=true);
        void neel_state_creator_2D(int size);

        double entanglement_entropy_calc(complex<double>** C);
        
}